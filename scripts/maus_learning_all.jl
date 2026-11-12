#!/usr/bin/env julia
# How much data, and how clean, does each model class need to forecast the deviations of human running?
# Long treadmill running trials (Maus et al. 2015; 4 people, 23 trials × ~320 strides; maus_extract.jl), 6 sagittal
# joint angles. Same design as maus_learning.jl, with three changes:
#   strides are cut at the phase of the leading oscillatory pair of all angles (pair_phase, as human_ltp_lowdim.jl),
#   not at the right hip's own phase, which pins that hip's deviation to zero at every cut;
#   nonlinear models are added: an ESN and a Winner et al. (2023) LSTM, both given the stride phase as input
#   (cos, sin of the sample's position in the stride), as the phase-varying linear models are;
#   measurement noise: white noise of η × each channel's deviation SD (η ∈ NOISE) is added to the stride-normalised
#   angles before the baseline is removed; models are fitted, selected and warmed up on the noisy deviations and
#   scored against the original ones.
# Deviation = stride-normalised, channel-standardised angles minus their 61-stride moving average. Training: the
# first NTR strides (NTR ∈ SIZES); test: the second half of the strides, starts every STEP samples, R² at horizons
# HZ (6–100% of a stride). Settings chosen per size on an inner split of the training strides (as maus_learning.jl).
#   dmd, lpv, ltp, lattice   maus_learning.jl's families
#   esn     300 units, phase input, ρ ∈ {0.5, 0.9} × leak ∈ {0.3, 1} × ridge ∈ {1e-4, 1e-2}, readout on the
#           deviation and the reservoir; free run feeds back the predicted deviation with the exact phase
#   lstm    (GPU mode) 128 units, Winner et al. training on the trial's training strides (one-stride segments, the
#           last one validates), all trials of a size trained at once (fit_lstm_winner_many); no inner selection;
#           patience ≈ 2000 updates and at most 40000 (an epoch holds one update per training segment); deviation
#           channels scaled to unit SD over the training rows (the phase inputs would otherwise dominate the loss)
# Usage: julia -t 16 --project=. scripts/maus_learning_all.jl cpu <η>       → cache/maus/learning_all_cpu_<η>.jls
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 8 --project=. scripts/maus_learning_all.jl lstm <η>
const MODE = get(ARGS, 1, "cpu")
const ETA = parse(Float64, get(ARGS, 2, "0"))
if MODE == "lstm"
    using CUDA, cuDNN
end
include("maus_learning.jl")
include("lstm_model.jl")
using SparseArrays

# ── strides cut at the leading oscillatory pair's phase (human_ltp_lowdim.jl) ──
interp(v, t) = (i = clamp(floor(Int, t), 1, length(v) - 1); a = t - i; (1 - a) * v[i] + a * v[i+1])
hip_phase(X) = (; ϕ = unwrap(angle.(analytic(X[:, 1] .- mean(X[:, 1])))) .- π / 2, tc = collect(1.0:size(X, 1)))
function pair_phase(X; frac = 0.5)
    Xs = (X .- mean(X, dims = 1)) ./ std(X, dims = 1); h = hip_phase(X)
    per = 2π * (size(X, 1) - 1) / abs(h.ϕ[end] - h.ϕ[1]); L = max(4, round(Int, frac * per))
    H = hankel(Xs, L); F = eigen(Symmetric(H * H')); V = F.vectors[:, end:-1:end-1]; Z = V' * H
    ψ = unwrap(atan.(Z[2, :], Z[1, :])); ψ[end] < ψ[1] && (ψ = -ψ)
    tc = collect(1:size(H, 2)) .+ (L - 1) / 2
    off = angle(mean(cis.(ψ .- [interp(h.ϕ, t) for t in tc])))
    (; ϕ = ψ .- off, tc)
end
function cut_strides(X, ϕ, tc; nph = NPH)
    b = Float64[]
    for k in ceil(Int, ϕ[1] / 2π)+1:floor(Int, ϕ[end] / 2π)
        i = findfirst(>=(2π * k), ϕ); (i === nothing || i == 1) && continue
        push!(b, tc[i-1] + (tc[i] - tc[i-1]) * (2π * k - ϕ[i-1]) / (ϕ[i] - ϕ[i-1]))
    end
    S = zeros(max(length(b) - 1, 0), nph, size(X, 2))
    for s in 1:length(b)-1, (j, t) in enumerate(range(b[s], b[s+1]; length = nph + 1)[1:nph])
        i = clamp(floor(Int, t), 1, size(X, 1) - 1); a = t - i
        S[s, j, :] = (1 - a) .* X[i, :] .+ a .* X[i+1, :]
    end
    S
end
pair_strides(X) = (P = pair_phase(X); cut_strides(X, P.ϕ, P.tc))

const NOISE = [0.0, 0.5, 1.0]
const SIZES_ALL = [4, 8, 16, 32, 64, 150]                 # 150: every trial has ≥ 300 strides
"Clean and noisy deviations (rows = samples) of strides S; noise η × each channel's clean deviation SD."
function deviations(S, η; seed = 1)
    _, _, D = split_baseline(S, :ma)
    η == 0 && return D, D
    n, _, p = size(S); sd = std(flat(S), dims = 1)                      # split_baseline's channel scaling
    sdD = std(D, dims = 1)
    noise = η .* reshape(sdD .* sd, 1, 1, p) .* randn(MersenneTwister(seed), size(S))
    _, _, Dn = split_baseline(S .+ noise, :ma)
    D, Dn
end
phase_cs(t) = (θ = 2π * mod(t - 1, NPH) / NPH; (cos(θ), sin(θ)))

# ── ESN with the stride phase as input ──
function esn_family(Dn)
    n, p = size(Dn)
    settings = [(; N = 300, ρ, α, λ) for ρ in (0.5, 0.9) for α in (0.3, 1.0) for λ in (1e-4, 1e-2)]
    cache = Dict{Tuple{Float64, Float64}, Any}()
    lk = ReentrantLock()
    function states(s)
        lock(lk) do
            get!(cache, (s.ρ, s.α)) do
                rng = MersenneTwister(1); N = s.N
                W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= s.ρ / maximum(abs, eigvals(W))
                Win = 0.5 .* (2 .* rand(rng, N, p + 2) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
                R = zeros(N, n); r = zeros(N)
                for t in 1:n
                    c, si = phase_cs(t); r = (1 - s.α) .* r .+ s.α .* tanh.(W * r .+ Win * vcat(Dn[t, :], c, si) .+ b); R[:, t] = r
                end
                (; W, Win, b, R)
            end
        end
    end
    fit = (rows, s) -> begin
        E = states(s); ix = collect(first(rows)+50:last(rows)-1)
        F = vcat(E.R[:, ix], Dn[ix, :]', ones(1, length(ix))); T = Dn[ix.+1, :]'
        G = F * F'; Wout = T * F' / (G + s.λ * tr(G) / size(G, 1) * I)
        hidx -> (t -> begin
            r = E.R[:, t]; u = Dn[t, :]; out = zeros(length(hidx), p); k = 1
            for st in 1:HZ[hidx[end]]
                y = Wout * vcat(r, u, 1.0)
                if st == HZ[hidx[k]]; out[k, :] = y; k += 1; end
                k > length(hidx) && break
                c, si = phase_cs(t + st); r = (1 - s.α) .* r .+ s.α .* tanh.(E.W * r .+ E.Win * vcat(y, c, si) .+ E.b); u = y
            end
            out
        end)
    end
    (settings, fit)
end

"maus_learning.jl's trial_learning on noisy deviations, scored on clean ones, with the ESN added."
function trial_learning_all(S_, η)
    n = size(S_, 1); nh = n ÷ 2; D, Dn = deviations(S_, η)
    te = nh*NPH+1:n*NPH; ts = (first(te) + HISTL):STEP:(last(te) - HZ[end])
    F = families(Dn); F[:esn] = esn_family(Dn); out = Dict{Tuple{Symbol, Int}, Any}()
    for ntr in SIZES_ALL
        ntr > nh && continue
        rows = 1:ntr*NPH; inner = 1:(ntr÷2)*NPH; va = (ntr÷2)*NPH+1:ntr*NPH
        for (name, (settings, fit)) in F
            name in (:ltp_phase, :ltp_nophase) && continue
            η > 0 && name == :lattice && continue
            t0 = time()
            vs = map(settings) do s
                f = try fit(inner, s) catch; nothing end
                f === nothing ? -Inf : mean(score_at(f(HV), Dn, vstarts(va), HV))       # selection on noisy data
            end
            all(isinf, vs) && continue
            s = settings[argmax(vs)]; t1 = time(); f = fit(rows, s); tf = time() - t1
            out[(name, ntr)] = (; setting = s, r2 = score_at(f(1:length(HZ)), D, ts, 1:length(HZ)), fit_secs = tf,
                                select_secs = t1 - t0)
        end
    end
    out
end

# ── LSTM (GPU): all trials of a size at once ──
"LSTM states (h, c) after teacher-forcing through every sample of X (d × n), on the CPU."
function lstm_states(m, X)
    n = size(X, 2); Hs = zeros(Float32, m.H, n); Cs = zeros(Float32, m.H, n)
    (y, carry), _ = m.cell(X[:, 1:1], m.ps.c, m.sc); Hs[:, 1] = carry[1]; Cs[:, 1] = carry[2]
    for t in 2:n
        (y, carry), _ = m.cell((X[:, t:t], carry), m.ps.c, m.sc); Hs[:, t] = carry[1]; Cs[:, t] = carry[2]
    end
    Hs, Cs
end
"Free-run R² of LSTM m from the test starts ts; the model works on deviations scaled by sd (channel SDs of the training rows)."
function lstm_eval(m, D, Dn, ts, sd)
    n, p = size(Dn)
    X = Float32.(vcat((Dn ./ sd')', reduce(hcat, [collect(phase_cs(t)) for t in 1:n])))
    Hs, Cs = lstm_states(m, X)
    E = zeros(length(HZ)); R = zeros(length(HZ))
    for t in ts
        carry = (Hs[:, t:t], Cs[:, t:t]); u, _ = m.head(carry[1], m.ps.h, m.sh); k = 1
        for st in 1:HZ[end]
            if st == HZ[k]; E[k] += sum(abs2, u[1:p] .* sd .- D[t+st, :]); R[k] += sum(abs2, D[t+st, :]); k += 1; end
            k > length(HZ) && break
            c, si = phase_cs(t + st); x = vcat(u[1:p], Float32[c, si])
            (o, carry), _ = m.cell((reshape(x, :, 1), carry), m.ps.c, m.sc); u, _ = m.head(o, m.ps.h, m.sh)
        end
    end
    1 .- E ./ R
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    SS = [pair_strides(t.X) for t in A.trials]; keep = qc_run.(SS) .< 5
    SS = SS[keep]; ids = [(t.subj, t.rep) for t in A.trials[keep]]
    @printf("%d trials kept, strides per trial %d–%d; noise η = %.1f\n", length(SS), extrema(size.(SS, 1))..., ETA)
    if MODE == "cpu"
        R = tmap(S -> trial_learning_all(S, ETA), SS)
        bad = count(isnothing, R)
        names = (:dmd, :lpv, :ltp, :lattice, :esn)
        @printf("%d trials (%d failed): median test R² at 12.5%% / 25%% / 50%% of a stride, by training strides\n", length(R) - bad, bad)
        @printf("%-9s", "strides"); for n in SIZES_ALL; @printf("%19d", n); end; println()
        for name in names
            @printf("%-9s", name)
            for n in SIZES_ALL
                v = [r[(name, n)].r2 for r in R if r !== nothing && haskey(r, (name, n))]
                isempty(v) ? @printf("%19s", "—") : @printf("%19s", @sprintf("%.2f / %.2f / %.2f", median(x[2] for x in v), median(x[4] for x in v), median(x[6] for x in v)))
            end
            println()
        end
        serialize(joinpath(ROOT, "cache", "maus", "learning_all_cpu_$(ETA).jls"), (; R, ids, SIZES = SIZES_ALL, HZ, ETA))
    else
        dev = gpu_device(); out = Dict{Int, Any}()
        DD = [deviations(S, ETA) for S in SS]
        for ntr in SIZES_ALL
            ok = [i for i in eachindex(SS) if ntr <= size(SS[i], 1) ÷ 2]
            isempty(ok) && continue
            L = 99; nr = ntr * NPH; valseg = 1; trainseg = (nr - 1 - valseg * L) ÷ (L + 1)
            trainseg < 1 && continue
            # each deviation channel scaled to unit SD over the training rows: unscaled, the deviations (SD ≈ 0.1) are
            # swamped in the loss by the phase channels (cos² + sin² = 1), which the model then spends itself predicting
            sds = [vec(std(DD[i][2][1:nr, :], dims = 1)) for i in ok]
            Xs = [Float32.(hcat(DD[i][2][1:nr, :] ./ sds[j]', reduce(vcat, [collect(phase_cs(t))' for t in 1:nr]))) for (j, i) in enumerate(ok)]
            t0 = time()
            # patience and the epoch cap in updates (Winner et al.: 500 epochs × 4 segments ≈ 2000 updates), so that every
            # training size gets a comparable budget; an epoch here holds `trainseg` updates
            ms = fit_lstm_winner_many(Xs; H = 128, lookback = L, trainseg, valseg, seed = 1, device = dev, verbose = false,
                                      patience = cld(2000, trainseg), max_epochs = cld(40000, trainseg))
            tfit = time() - t0
            r2 = Vector{Any}(undef, length(ok))
            Threads.@threads for j in eachindex(ok)
                S = SS[ok[j]]; n = size(S, 1); nh = n ÷ 2; D, Dn = DD[ok[j]]
                te = nh*NPH+1:n*NPH; ts = (first(te) + HISTL):STEP:(last(te) - HZ[end])
                r2[j] = (; trial = ok[j], r2 = lstm_eval(ms[j], D, Dn, ts, sds[j]), best_epoch = ms[j].best_epoch)
            end
            out[ntr] = (; rows = r2, fit_secs = tfit, nmodels = length(ok))
            v = [x.r2 for x in r2]
            @printf("lstm %4d strides: R² at 12.5%% / 25%% / 50%%: %.2f / %.2f / %.2f | %.1f min for %d models\n", ntr,
                    median(x[2] for x in v), median(x[4] for x in v), median(x[6] for x in v), tfit / 60, length(ok))
            flush(stdout)
            serialize(joinpath(ROOT, "cache", "maus", "learning_all_lstm_$(ETA).jls"), (; out, ids, SIZES = SIZES_ALL, HZ, ETA))
        end
    end
end
