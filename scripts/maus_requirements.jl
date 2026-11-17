#!/usr/bin/env julia
# Requirements R2, R3 and R6 for every model class on the running deviations (150 training strides, no added noise;
# data, splits, settings selection and test starts as maus_learning_all.jl and maus_node.jl):
#   R3  reproducibility: models that depend on a random seed (ESN reservoir, neural ODE and LSTM initialisation and
#       minibatches) are fitted with seeds 1–5; their forecasts are compared between seeds. Linear models and
#       SSMLearnPy (maus_ssm_req.py) have no seed: the same data and settings give the same fit
#   R2  sustained behaviour of the deviation models: free runs of 30 strides from 40 test starts (the deviation's
#       RMS at the end of each stride, relative to its SD over the training strides); for the linear models also the
#       per-stride Floquet multipliers (eigenvalues of the monodromy matrix, the product of the per-sample maps over a
#       stride). The phase-binned model is a set of direct maps per horizon and cannot be run freely
#   R6  comparability: per-trial quantities (written here, analysed in Figure_S13.jl) for the 4 runners
# Each model's forecasts at ⅛ and ¼ stride from every test start are stored (cache/maus/req/, not tracked; the
# summaries Figure_S13.jl computes are written to cache/maus/requirements.jls).
# Usage: julia -t 16 --project=. scripts/maus_requirements.jl cpu      (DMD, Fourier in phase, phase-binned, ESN seeds)
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 16 --project=. scripts/maus_requirements.jl node   (GPU)
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 16 --project=. scripts/maus_requirements.jl lstm   (GPU)
const RQ_MODE = get(ARGS, 1, "cpu")
const RQ_TEST = get(ARGS, 2, "") == "test"
empty!(ARGS)
if RQ_MODE in ("node", "lstm")
    using CUDA, cuDNN
end
include("maus_node.jl")
using SparseArrays

const NTR = 150; const SEEDS = 1:5; const NLONG = 30; const NSTART_LONG = 40
const OUTDIR = joinpath(ROOT, "cache", "maus", "req"); mkpath(OUTDIR)
const LONGSTEPS = collect(NPH:NPH:NLONG*NPH)
const HQ = (2, 4)                                   # indices into HZ: 12 and 25 samples (⅛ and ¼ stride)

"Per-trial data: deviation, test starts, long-run starts, training rows, deviation SD."
function trial_data(S)
    n = size(S, 1); nh = n ÷ 2; D, _ = deviations(S, 0.0)
    te = nh*NPH+1:n*NPH; ts = collect((first(te) + HISTL):STEP:(last(te) - HZ[end]))
    tl = ts[round.(Int, range(1, length(ts); length = NSTART_LONG))]
    rows = 1:NTR*NPH; sd = sqrt(mean(abs2, D[rows, :]))
    (; D, ts, tl, rows, sd, n)
end
"Forecasts at every horizon in HZ from every start (pred(t, steps) → length(steps) × p), long runs and scores."
function collect_pred(pred, T; long = true)
    P = [pred(t, HZ) for t in T.ts]
    r2 = [1 - sum(sum(abs2, P[i][k, :] .- T.D[t+HZ[k], :]) for (i, t) in enumerate(T.ts)) / sum(sum(abs2, T.D[t+HZ[k], :]) for t in T.ts)
          for k in eachindex(HZ)]
    fc = Dict(HZ[k] => Float32.(reduce(hcat, [P[i][k, :] for i in eachindex(T.ts)])) for k in HQ)
    L = long ? [begin Q = pred(t, LONGSTEPS); sqrt.(vec(mean(abs2, Q, dims = 2))) ./ T.sd end for t in T.tl] : nothing
    (; r2, fc, long = long ? reduce(hcat, L)' : nothing)
end
"Choose a setting of family (settings, fit) on the inner split of the training strides (maus_learning_all.jl)."
function choose(settings, fit, D)
    inner = 1:(NTR÷2)*NPH; va = (NTR÷2)*NPH+1:NTR*NPH
    vs = map(settings) do s
        f = try fit(inner, s) catch; nothing end
        f === nothing ? -Inf : mean(score_at(f(HV), D, vstarts(va), HV))
    end
    settings[argmax(vs)]
end

# ── linear models ──
function linear_models(T)
    D = T.D; F = families(D); out = Dict{Symbol, Any}(); p = size(D, 2)
    # Hankel-DMD on the deviation
    s = choose(F[:dmd]..., D); A, U = opfit(D[T.rows, :]; d = s[1], r = s[2]); μ = mean(D[T.rows, :], dims = 1)
    pred = (t, steps) -> roll(A, U, μ, D, t, s[1], steps[end])[steps, :]
    out[:dmd] = merge(collect_pred(pred, T), (; setting = s, mult = eigvals(A^NPH)))
    # Fourier in phase
    s = choose(F[:ltp]..., D); M = fit_ltp(D, T.rows, s[1], s[2], s[3], [s[4]])[1]
    pred = (t, steps) -> begin
        z = M.U' * vec(permutedims(D[t-M.d+1:t, :] .- M.μ)); Q = zeros(length(steps), p); k = 1
        for st in 1:steps[end]
            z = M.Aph[mod(t + st - 2, NPH)+1] * z
            if st == steps[k]; Q[k, :] = (M.U * z)[end-p+1:end] .+ vec(M.μ); k += 1; end
        end
        Q
    end
    out[:ltp] = merge(collect_pred(pred, T), (; setting = s, mult = eigvals(monodromy(M))))
    # phase-binned direct maps (one map per horizon in HZ; no free run)
    s = choose(F[:lpv]..., D); f = F[:lpv][2](T.rows, s)(1:length(HZ))
    out[:lpv] = merge(collect_pred((t, steps) -> f(t), T; long = false), (; setting = s, mult = nothing))
    out
end

# ── ESN with a seed (esn_family of maus_learning_all.jl, reservoir drawn with the given seed) ──
function esn_seeded(D, seed)
    n, p = size(D)
    settings = [(; N = 300, ρ, α, λ) for ρ in (0.5, 0.9) for α in (0.3, 1.0) for λ in (1e-4, 1e-2)]
    cache = Dict{Tuple{Float64, Float64}, Any}()
    states(s) = get!(cache, (s.ρ, s.α)) do
        rng = MersenneTwister(seed); N = s.N
        W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= s.ρ / maximum(abs, eigvals(W))
        Win = 0.5 .* (2 .* rand(rng, N, p + 2) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
        R = zeros(N, n); r = zeros(N)
        for t in 1:n; c, si = phase_cs(t); r = (1 - s.α) .* r .+ s.α .* tanh.(W * r .+ Win * vcat(D[t, :], c, si) .+ b); R[:, t] = r; end
        (; W, Win, b, R)
    end
    predictor(rows, s) = begin
        E = states(s); ix = collect(first(rows)+50:last(rows)-1)
        F = vcat(E.R[:, ix], D[ix, :]', ones(1, length(ix))); Tg = D[ix.+1, :]'
        G = F * F'; Wout = Tg * F' / (G + s.λ * tr(G) / size(G, 1) * I)
        (t, steps) -> begin
            r = E.R[:, t]; u = D[t, :]; Q = fill(NaN, length(steps), p); k = 1
            for st in 1:steps[end]
                y = Wout * vcat(r, u, 1.0)
                (any(!isfinite, y) || maximum(abs, y) > 1e6) && break
                if st == steps[k]; Q[k, :] = y; k += 1; end
                k > length(steps) && break
                c, si = phase_cs(t + st); r = (1 - s.α) .* r .+ s.α .* tanh.(E.W * r .+ E.Win * vcat(y, c, si) .+ E.b); u = y
            end
            Q
        end
    end
    fit = (rows, s) -> (pr = predictor(rows, s); hidx -> (t -> pr(t, HZ[hidx])))
    s = choose(settings, fit, D)
    (s, predictor(T_ROWS, s))
end
const T_ROWS = 1:NTR*NPH

# ── neural ODE: unclipped long runs of a fitted model ──
function node_long(f, Yv, ts, sd)
    X = states(f.M, Yv, ts); τ = Float32.((collect(ts) .- 1) ./ NPH); Ur = f.M.U[end-f.M.p+1:end, :]; μr = f.M.μv[end-f.M.p+1:end]
    out = zeros(length(ts), NLONG); k = 1
    for st in 1:LONGSTEPS[end]
        X = rk4(f.p, X, τ, f.nrm, true); τ = τ .+ DT
        if st == LONGSTEPS[k]
            P = Float64.(Ur * X .+ μr); out[:, k] = [all(isfinite, c) ? sqrt(mean(abs2, c)) / sd : Inf for c in eachcol(P)]; k += 1
        end
    end
    out
end

# ── LSTM forecasts at given steps from start t (states teacher-forced through t) ──
function lstm_predictor(m, Dn, sd)
    n, p = size(Dn)
    X = Float32.(vcat((Dn ./ sd')', reduce(hcat, [collect(phase_cs(t)) for t in 1:n])))
    Hs, Cs = lstm_states(m, X)
    (t, steps) -> begin
        carry = (Hs[:, t:t], Cs[:, t:t]); u, _ = m.head(carry[1], m.ps.h, m.sh); Q = fill(NaN, length(steps), p); k = 1
        for st in 1:steps[end]
            if st == steps[k]; Q[k, :] = Float64.(u[1:p]) .* sd; k += 1; end
            k > length(steps) && break
            c, si = phase_cs(t + st); x = vcat(u[1:p], Float32[c, si])
            (o, carry), _ = m.cell((reshape(x, :, 1), carry), m.ps.c, m.sc); u, _ = m.head(o, m.ps.h, m.sh)
        end
        Q
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    SS = [pair_strides(t.X) for t in A.trials]; keep = qc_run.(SS) .< 5
    SS = SS[keep]; ids = [(t.subj, t.rep) for t in A.trials[keep]]
    RQ_TEST && (SS = SS[1:2]; ids = ids[1:2])
    TD = [trial_data(S) for S in SS]
    @printf("%d trials, mode %s, %d threads\n", length(SS), RQ_MODE, Threads.nthreads()); flush(stdout)
    if RQ_MODE == "cpu"
        serialize(joinpath(OUTDIR, "truth.jls"), (; ids, ts = [T.ts for T in TD], sd = [T.sd for T in TD],
                                                  d0 = [Float32.(T.D[T.ts, :]') for T in TD],
                                                  d = Dict(HZ[k] => [Float32.(T.D[T.ts .+ HZ[k], :]') for T in TD] for k in HQ)))
        t0 = time(); R = tmap(linear_models, TD)
        @printf("linear models: %.1f min\n", (time() - t0) / 60); flush(stdout)
        for m in (:dmd, :ltp, :lpv)
            serialize(joinpath(OUTDIR, "$(m).jls"), [r === nothing ? nothing : r[m] for r in R])
            @printf("%-4s remaining variance ⅛ stride %.2f\n", m, median(1 - r[m].r2[2] for r in R if r !== nothing))
        end
        for seed in SEEDS
            t0 = time()
            R = tmap(T -> (sp = esn_seeded(T.D, seed); merge(collect_pred(sp[2], T), (; setting = sp[1], mult = nothing))), TD)
            serialize(joinpath(OUTDIR, "esn_seed$(seed).jls"), R)
            @printf("esn seed %d: remaining variance ⅛ stride %.2f | %.1f min\n", seed, median(1 - r.r2[2] for r in R if r !== nothing), (time() - t0) / 60)
            flush(stdout)
        end
    elseif RQ_MODE == "node"
        dev = gpu_device(); bidx = [[(s - 1) * NPH + 1 for s in 1:size(S, 1)+1] for S in SS]
        Ys = [T.D for T in TD]; Sp = [node_split(Ys[i], bidx[i], NTR) for i in eachindex(SS)]
        for seed in SEEDS
            t0 = time(); sel = fit_select(Ys, Ys, Sp, true; maxup = RQ_TEST ? 300 : 8000, patience = 1500, dev, seed)
            R = Vector{Any}(undef, length(SS))
            Threads.@threads for i in eachindex(SS)
                f = sel[i].f; T = TD[i]
                Fc = node_forecast(f.p, f.nrm, f.M, T.D, T.ts, 1:length(HZ), true, Sp[i].lo, Sp[i].hi)
                r2 = skill(Fc, T.D, T.D, T.ts, 1:length(HZ))
                R[i] = (; r2, fc = Dict(HZ[k] => Float32.(Fc[k]) for k in HQ), long = node_long(f, T.D, T.tl, T.sd),
                        setting = (f.d, f.r), mult = nothing)
            end
            serialize(joinpath(OUTDIR, "node_seed$(seed).jls"), R)
            @printf("node seed %d: remaining variance ⅛ stride %.2f | %.1f min\n", seed, median(1 - r.r2[2] for r in R), (time() - t0) / 60)
            flush(stdout)
        end
    elseif RQ_MODE == "lstm"
        dev = gpu_device(); L = 99; nr = NTR * NPH; trainseg = (nr - 1 - L) ÷ (L + 1)
        sds = [vec(std(T.D[1:nr, :], dims = 1)) for T in TD]
        Xs = [Float32.(hcat(TD[j].D[1:nr, :] ./ sds[j]', reduce(vcat, [collect(phase_cs(t))' for t in 1:nr]))) for j in eachindex(TD)]
        for seed in SEEDS
            t0 = time()
            ms = fit_lstm_winner_many(Xs; H = 128, lookback = L, trainseg, valseg = 1, seed, device = dev, verbose = false,
                                      patience = cld(2000, trainseg), max_epochs = RQ_TEST ? 3 : cld(40000, trainseg))
            R = Vector{Any}(undef, length(SS))
            Threads.@threads for j in eachindex(SS)
                R[j] = merge(collect_pred(lstm_predictor(ms[j], TD[j].D, sds[j]), TD[j]), (; setting = nothing, mult = nothing))
            end
            serialize(joinpath(OUTDIR, "lstm_seed$(seed).jls"), R)
            @printf("lstm seed %d: remaining variance ⅛ stride %.2f | %.1f min\n", seed, median(1 - r.r2[2] for r in R), (time() - t0) / 60)
            flush(stdout)
        end
    end
end
