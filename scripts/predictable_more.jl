#!/usr/bin/env julia
# The rest of the model classes on the predictable-response test of predictable_response.jl (R1, R4 partial
# observation): the same kicked SLIP walks, training windows, validation walk, test starts, warm-up and scoring against
# the true future. Each model is chosen by the same criterion as DMD and the ESN there (multistart one-stride rollout
# error on the independent validation walk):
#   FIP    phase-varying linear, whole signal ("Fourier in phase", affine, maus_ladder.jl's fit_ltp_aff): the leading r
#          components of the last d samples evolve as z₊ = A(θ) z + b(θ), A and b Fourier series in the step phase θ
#          (known: stride-normalised time); d ∈ {8, 25, 50, 100} (up to a stride), r ∈ {8, 16, 32, 64}, harmonics ∈ {2, 4, 8}, ridge ∈ {1e-4, 1e-3, 1e-2}
#   NODE   neural ODE given the step phase (cos θ, sin θ; known in stride-normalised time, as for Fourier in phase): MLP
#          2 × 64 tanh on the channels (d = 1) or the leading components of the last 8 or 25 samples, RK4 at the sampling
#          step, multiple shooting on 25-sample segments, Adam (1e-3 decaying to 1e-4 over 10000 updates); early stopping
#          on the validation criterion (patience 2000); the state chosen on it; 2 seeds. (Without the phase, on the CoM
#          state or 8 samples, its forecasts were no better than the mean cycle over the first stride and 3–9 times worse
#          over strides 1–3: the hybrid walker's state is not determined by those variables.)
#   LSTM   Winner et al. (2023) training (lstm_model.jl, as predictable_response.jl): 128 units, the training walk cut
#          into 60-step trials; patience and epoch cap in updates as for the running data (maus_learning_all.jl: about
#          2000 and at most 40000; 8 segments per epoch); 2 seeds; GPU. A subset of configurations can be run per process
#          (`lstm 1,2`, → more_lstm_cfg<i>.jls), merged with mode `lstmmerge`
#   SSM    SSMLearnPy (predictable_ssm.py, settings grid of maus_ssm.py chosen on the validation criterion); this
#          script exports the data ("export") and scores the returned forecasts ("ssm")
# R2 on the same walker: from the first test start of each held-out walk, every model also runs freely for 30 strides;
# its fate is classified as in the free-running benchmark (amplitude of the last 10 strides relative to the training
# data's, period relative to the step's: ok, decays, collapse, blow-up, wrong). Mode "dmdesn" refits Hankel-DMD and the
# ESN exactly as predictable_response.jl (its cache predates the free runs) for these fates.
# Usage: julia -t 8 --project=. scripts/predictable_more.jl fip | node | dmdesn | export | ssm
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia --project=. scripts/predictable_more.jl lstm
#        → cache/predictable/more_<mode>.jls (merged by Figure_06.jl)
const PM_MODE = get(ARGS, 1, "fip")
const PM_CFGS = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : nothing     # lstm: configurations for this process
empty!(ARGS)
if PM_MODE == "lstm"
    using CUDA, cuDNN
end
include("predictable_response.jl")
using Zygote, Optimisers, NPZ

const CONFIGS = [(κ, NTR, obs) for κ in (0.1, 0.0), NTR in (60, 300), obs in (:full, :height)] |> vec
const CONT = deserialize(joinpath(ROOT, "cache", "predictable", "slip_truth_continuations.jls"))

"Training data, validation starts and the noisy/true series of one configuration (as run_config)."
function prep(κ, NTR, obs)
    ch = obs == :full ? (1:3) : (1:1)
    w1 = deserialize(WALK(κ, 1)); w2 = deserialize(WALK(κ, 2))
    Xtr = series(w1.S, 101:100+NTR)[:, ch]
    μ = mean(Xtr, dims = 1); σ = std(Xtr, dims = 1)
    z(X) = (X[:, ch] .- μ) ./ σ
    noisy(X, seed) = z(X) .+ NOISE .* randn(MersenneTwister(seed), size(X, 1), length(ch))
    Ytr = noisy(series(w1.S, 1:size(w1.S, 1)), 7)[idx(101, 1):idx(100 + NTR, NPH), :]
    mstep = reduce(vcat, [mean(Ytr[s:NPH:end, :], dims = 1) for s in 1:NPH])
    Yv = noisy(series(w2.S, 1:size(w2.S, 1)), 8)
    vrng = MersenneTwister(5); vstarts = [(101 + 2i, rand(vrng, 1:NPH)) for i in 0:19]
    (; ch, z, noisy, Ytr, mstep, Yv, vstarts)
end
"Validation error of a forecaster f(Yh, n, j) (j: 1-based phase of the last history sample), as run_config."
function val_error(P, f)
    e = 0.0
    for (k, j) in P.vstarts
        i0 = idx(k, j); F = f(P.Yv[i0-HIST+1:i0, :], 2NPH, j)
        any(!isfinite, F) && return Inf
        e += mean(abs2, F .- P.Yv[i0+1:i0+2NPH, :])
    end
    e / length(P.vstarts)
end
const NRUNL = 60NPH                                           # 30 strides of two steps
function period_ac(x; lo = 10)
    x = x .- mean(x); n = length(x)
    ac = [sum(x[1:n-k] .* x[1+k:n]) for k in 0:n÷2]; ac ./= ac[1]
    pk = [i for i in lo:length(ac)-1 if ac[i] > ac[i-1] && ac[i] >= ac[i+1] && ac[i] > 0.3]
    isempty(pk) && return NaN
    i = pk[1]; a, b, c = ac[i-1], ac[i], ac[i+1]; (i - 1) + 0.5 * (a - c) / (a - 2b + c)
end
"Fate of a 30-stride free run F against the training data (as the free-running benchmark; the step is the period)."
function slipfate(F, Ytr)
    any(!isfinite, F) && return "blow-up"
    tail = F[end-20NPH+1:end, :]; amp = mean(std(tail, dims = 1)) / mean(std(Ytr, dims = 1))
    per = amp > 0.05 ? period_ac(tail[:, 1]) / NPH : NaN
    rhythm = isfinite(per) && abs(per - 1) < 0.1
    amp > 4 ? "blow-up" : amp < 0.25 ? "collapse" : (!rhythm || amp > 1.33) ? "wrong" : amp >= 0.75 ? "ok" : "decays"
end
"Score forecasters [(name, seed, f)] on the test starts exactly as run_config (MSE against the true future)."
function test_models(κ, NTR, obs, P, models; longruns = true)
    names = vcat(["mean cycle"], ["$(m[1]) $(m[2])" for m in models])
    SE = Dict(nm => zeros(HOR) for nm in names); SEy = Dict(nm => zeros(HOR) for nm in names)
    nstart = 0; diverged = Dict(nm => 0 for nm in names)
    for seed in 11:20
        w = deserialize(WALK(κ, seed)); c = CONT["k$(κ)_s$(seed)"]
        Yo = P.noisy(series(w.S, 1:size(w.S, 1)), 100 + seed); Xt = P.z(series(w.S, 1:size(w.S, 1)))
        for (s, (k, j)) in enumerate(zip(c.ks, c.js))
            c.C[s] === nothing && continue
            i0 = idx(k, j); fut = Xt[i0+1:i0+HOR, :]; ph = [mod1(j + h, NPH) for h in 1:HOR]
            fc = Dict{String, Matrix{Float64}}("mean cycle" => P.mstep[ph, :])
            for (nm, sd, f) in models; fc["$nm $sd"] = f(Yo[i0-HIST+1:i0, :], HOR, j); end
            nstart += 1
            for (nm, F) in fc
                if any(!isfinite, F); diverged[nm] += 1; F = fill(1e3, size(F)); end
                SE[nm] .+= vec(mean(abs2, F .- fut, dims = 2)); SEy[nm] .+= abs2.(F[:, 1] .- fut[:, 1])
            end
        end
    end
    MSE = Dict(nm => v ./ nstart for (nm, v) in SE); MSEy = Dict(nm => v ./ nstart for (nm, v) in SEy)
    # R2: 30-stride free runs from the first test start of each held-out walk
    fates = Dict{String, Vector{String}}()
    if longruns
        for seed in 11:20
            w = deserialize(WALK(κ, seed)); c = CONT["k$(κ)_s$(seed)"]; Yo = P.noisy(series(w.S, 1:size(w.S, 1)), 100 + seed)
            s = findfirst(!isnothing, c.C); s === nothing && continue
            k, j = c.ks[s], c.js[s]; i0 = idx(k, j)
            for (nm, sd, f) in models
                push!(get!(fates, "$nm $sd", String[]), slipfate(f(Yo[i0-HIST+1:i0, :], NRUNL, j), P.Ytr))
            end
        end
    end
    (; κ, NTR, obs, names, MSE, MSEy, skill = Dict(nm => 1 .- MSE[nm] ./ MSE["mean cycle"] for nm in names),
       skilly = Dict(nm => 1 .- MSEy[nm] ./ MSEy["mean cycle"] for nm in names), nstart, diverged, fates)
end

# ── Fourier in phase (affine, whole signal) ──
fphase50(k, Hh) = (θ = 2π * k / NPH; vcat(1.0, [f(h * θ) for h in 1:Hh for f in (cos, sin)]))
function fit_fip(Y; d, r, Hh, λ)
    p = size(Y, 2); μ = mean(Y, dims = 1); H = hankel(Y .- μ, d - 1); U = svd(H).U[:, 1:min(r, size(H, 1))]; rr = size(U, 2)
    Z = U' * H; nc = size(Z, 2); ph = [mod(c + d - 2, NPH) for c in 1:nc]       # Ytr starts at phase 0
    Φ = reduce(hcat, [(f = fphase50(ph[c], Hh); vcat(kron(f, Z[:, c]), f)) for c in 1:nc-1])
    G = Φ * Φ'; W = Z[:, 2:end] * Φ' / (G + λ * mean(diag(G)) * I); nf = 2Hh + 1
    Aph = [sum(fphase50(k, Hh)[i] .* W[:, (i-1)*rr+1:i*rr] for i in 1:nf) for k in 0:NPH-1]
    bph = [W[:, nf*rr+1:end] * fphase50(k, Hh) for k in 0:NPH-1]
    (Yh, n, j) -> begin
        zt = U' * vec(permutedims(Yh[end-d+1:end, :] .- μ)); out = zeros(n, p)
        for s in 1:n
            k = mod(j - 1 + s - 1, NPH) + 1; zt = Aph[k] * zt .+ bph[k]; out[s, :] = (U * zt)[end-p+1:end] .+ vec(μ)
        end
        out
    end
end
const FIP_GRID = [(; d, r, Hh, λ) for d in (8, 25, 50, 100), r in (8, 16, 32, 64), Hh in (2, 4, 8), λ in (1e-4, 1e-3, 1e-2)] |> vec

# ── neural ODE without phase input (single model, CPU) ──
const NH = 64; const SEGL = 25; const BATCH = 256; const DTN = 1f0 / NPH
phx(τ) = vcat(cos.(2f0π .* τ'), sin.(2f0π .* τ'))                           # τ in steps
vfn(p, Z, τ, nrm) = nrm.σd .* (p.W3 * tanh.(p.W2 * tanh.(p.W1 * vcat(Z ./ nrm.σz, phx(τ)) .+ p.b1) .+ p.b2) .+ p.b3)
function rk4n(p, Z, τ, nrm)
    k1 = vfn(p, Z, τ, nrm); k2 = vfn(p, Z .+ (DTN / 2) .* k1, τ .+ DTN / 2, nrm)
    k3 = vfn(p, Z .+ (DTN / 2) .* k2, τ .+ DTN / 2, nrm); k4 = vfn(p, Z .+ DTN .* k3, τ .+ DTN, nrm)
    Z .+ (DTN / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end
function fit_node(P; d, r, seed, maxup = 10000, patience = 2000, every = 200)
    Y = P.Ytr; p = size(Y, 2); μ = vec(mean(Y, dims = 1)); μv = Float32.(repeat(μ, d))
    U = d == 1 ? Matrix{Float32}(I, p, p) : Float32.(svd(hankel(Y .- μ', d - 1)).U[:, 1:min(r, p * d)])
    win(Yh) = Float32.(vec(permutedims(Yh[end-d+1:end, :])))
    Z = U' * (Float32.(reduce(hcat, [vec(permutedims(Y[c:c+d-1, :])) for c in 1:size(Y, 1)-d+1])) .- μv)
    dZ = (Z[:, 2:end] .- Z[:, 1:end-1]) ./ DTN; sz = vec(std(Z, dims = 2))
    nrm = (; σz = max.(sz, 1f-3 * maximum(sz)), σd = vec(std(dZ, dims = 2)))
    rng = MersenneTwister(seed); g(o, i, s = 1.0) = Float32.(randn(rng, o, i) .* (s * sqrt(1 / i))); rr = size(Z, 1)
    τall = Float32.([mod(c + d - 2, NPH) for c in 1:size(Z, 2)] ./ NPH)          # phase (in steps) of each window's last sample
    pr = (; W1 = g(NH, rr + 2), b1 = zeros(Float32, NH), W2 = g(NH, NH), b2 = zeros(Float32, NH), W3 = g(rr, NH, 0.1), b3 = zeros(Float32, rr))
    opt = Optimisers.setup(Optimisers.Adam(1f-3), pr)
    fc(q) = (Yh, n, j) -> begin
        X = U' * (win(Yh) .- μv); out = zeros(n, p); Ur = U[end-p+1:end, :]; μr = μv[end-p+1:end]; τ = Float32[(j - 1) / NPH]
        for s in 1:n; X = rk4n(q, X, τ, nrm); τ = τ .+ DTN; out[s, :] = Ur * X .+ μr; end
        out
    end
    loss(q, cols) = begin
        X = Z[:, cols]; τ = τall[cols]; tot = 0f0
        for k in 1:SEGL; X = rk4n(q, X, τ, nrm); τ = τ .+ DTN; tot += sum(abs2, X .- Z[:, cols .+ k]); end
        tot / (SEGL * length(cols) * rr)
    end
    best = (Inf, pr); since = 0; ncols = size(Z, 2) - SEGL
    for u in 1:maxup
        cols = rand(rng, 1:ncols, BATCH)
        gr = Zygote.gradient(q -> loss(q, cols), pr)[1]; opt, pr = Optimisers.update(opt, pr, gr)
        u % every == 0 || continue
        Optimisers.adjust!(opt, Float32(1e-3 * 0.1^(u / maxup)))
        v = val_error(P, fc(pr))
        if v < best[1]; best = (v, deepcopy(pr)); since = 0; else since += every; end
        since >= patience && break
    end
    (fc(best[2]), best[1])
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    out = Vector{Any}(undef, length(CONFIGS))
    if PM_MODE == "fip"
        Threads.@threads for i in eachindex(CONFIGS)
            κ, NTR, obs = CONFIGS[i]; P = prep(κ, NTR, obs)
            ev = [(gs, something(safe(() -> val_error(P, fit_fip(P.Ytr; gs...))), Inf)) for gs in FIP_GRID]
            gsel = ev[argmin(last.(ev))][1]
            out[i] = merge(test_models(κ, NTR, obs, P, [("FIP", 0, fit_fip(P.Ytr; gsel...))]), (; setting = gsel))
        end
    elseif PM_MODE == "node"
        Threads.@threads for i in eachindex(CONFIGS)
            κ, NTR, obs = CONFIGS[i]; P = prep(κ, NTR, obs); t0 = time()
            models = map(1:2) do seed
                fits = [fit_node(P; d, r, seed) for (d, r) in ((1, 3), (8, 16), (25, 32))]
                ("NODE", seed, fits[argmin(last.(fits))][1])
            end
            out[i] = merge(test_models(κ, NTR, obs, P, models), (; secs = time() - t0))
        end
    elseif PM_MODE == "lstm"
        dev = gpu_device()
        for i in something(PM_CFGS, eachindex(CONFIGS))
            κ, NTR, obs = CONFIGS[i]; P = prep(κ, NTR, obs); t0 = time()
            ltr = [P.Ytr[(q-1)*60NPH+1:q*60NPH, :] for q in 1:NTR÷60]
            models = map(1:2) do sd
                m = fit_lstm_winner(ltr; H = 128, lookback = 299, trainseg = 8, valseg = 1, seed = sd, device = dev,
                                    patience = cld(2000, 8), max_epochs = cld(40000, 8))
                ("LSTM", sd, (Yh, n, j) -> forecast_lstm(m, Yh, n))
            end
            out[i] = merge(test_models(κ, NTR, obs, P, models), (; secs = time() - t0))
            @printf("lstm config %d done, %.1f min\n", i, (time() - t0) / 60); flush(stdout)
            PM_CFGS === nothing || serialize(joinpath(ROOT, "cache", "predictable", "more_lstm_cfg$i.jls"), out[i])
        end
        PM_CFGS === nothing || exit()
    elseif PM_MODE == "lstmmerge"
        for i in eachindex(CONFIGS); out[i] = deserialize(joinpath(ROOT, "cache", "predictable", "more_lstm_cfg$i.jls")); end
    elseif PM_MODE == "dmdesn"
        Threads.@threads for i in eachindex(CONFIGS)
            κ, NTR, obs = CONFIGS[i]; P = prep(κ, NTR, obs)
            dv = Tuple{NamedTuple, Float64}[]
            for τ in DMD_TAUS
                Ub = svd(hankel(P.Ytr, τ)).U
                for g in DMD_GRID
                    push!(dv, ((; τ, g...), something(safe(() -> (m = fit_dmd(P.Ytr; τ, g..., basis = Ub); val_error(P, (Yh, n, j) -> forecast(m, Yh, n)))), Inf)))
                end
            end
            gd = dv[argmin(last.(dv))][1]; md = fit_dmd(P.Ytr; gd...)
            ev = [(g, mean(something(safe(() -> (m = fit_esn(P.Ytr; g..., seed = sd); val_error(P, (Yh, n, j) -> forecast(m, Yh, n)))), Inf) for sd in 1:2)) for g in ESN_GRID]
            ge = ev[argmin(last.(ev))][1]
            models = vcat([("DMD", 0, (Yh, n, j) -> forecast(md, Yh, n))],
                          [(m = fit_esn(P.Ytr; ge..., seed = sd); ("ESN", sd, (Yh, n, j) -> forecast(m, Yh, n))) for sd in 1:4])
            out[i] = merge(test_models(κ, NTR, obs, P, models), (; gd, ge))
        end
    elseif PM_MODE == "export"
        dir = joinpath(ROOT, "cache", "predictable", "ssm_io"); mkpath(dir)
        for (i, (κ, NTR, obs)) in enumerate(CONFIGS)
            P = prep(κ, NTR, obs)
            VH = cat([P.Yv[idx(k, j)-HIST+1:idx(k, j), :] for (k, j) in P.vstarts]...; dims = 3)
            VF = cat([P.Yv[idx(k, j)+1:idx(k, j)+2NPH, :] for (k, j) in P.vstarts]...; dims = 3)
            TH = Array{Float64, 3}[]; FIRST = Int[]
            for seed in 11:20
                w = deserialize(WALK(κ, seed)); c = CONT["k$(κ)_s$(seed)"]; Yo = P.noisy(series(w.S, 1:size(w.S, 1)), 100 + seed)
                first = true
                for (s, (k, j)) in enumerate(zip(c.ks, c.js))
                    c.C[s] === nothing && continue
                    push!(TH, reshape(Yo[idx(k, j)-HIST+1:idx(k, j), :], HIST, :, 1))
                    first && (push!(FIRST, length(TH)); first = false)
                end
            end
            npzwrite(joinpath(dir, "config_$i.npz"), Dict("Ytr" => P.Ytr, "VH" => VH, "VF" => VF, "TH" => cat(TH...; dims = 3), "hor" => HOR,
                                                       "first" => FIRST, "nrunl" => NRUNL))
        end
        println("exported"); exit()
    elseif PM_MODE == "ssm"
        dir = joinpath(ROOT, "cache", "predictable", "ssm_io")
        for (i, (κ, NTR, obs)) in enumerate(CONFIGS)
            P = prep(κ, NTR, obs); F = npzread(joinpath(dir, "forecast_$i.npz"))["F"]       # HOR × d × starts
            q = Ref(0)
            f = (Yh, n, j) -> (q[] += 1; F[:, :, q[]])
            out[i] = test_models(κ, NTR, obs, P, [("SSM", 0, f)]; longruns = false)
            FL = npzread(joinpath(dir, "forecast_$i.npz"))["FL"]                                      # NRUNL × d × walks
            out[i].fates["SSM 0"] = [slipfate(replace(FL[:, :, k], NaN => Inf), P.Ytr) for k in axes(FL, 3)]
        end
    end
    serialize(joinpath(ROOT, "cache", "predictable", "more_$(PM_MODE == "lstmmerge" ? "lstm" : PM_MODE).jls"), out)
    bin(v, a, b) = mean(v[round(Int, a * 2NPH)+1:round(Int, b * 2NPH)])
    for o in out
        @printf("κ %.1f %3d %-6s |", o.κ, o.NTR, o.obs)
        for nm in o.names[2:end]
            @printf(" %s: RRV first stride %.2f, strides 1–3 %.2f, diverged %d |", nm, 1 - bin(o.skill[nm], 0, 1), 1 - bin(o.skill[nm], 1, 3), o.diverged[nm])
        end
        println()
    end
end
