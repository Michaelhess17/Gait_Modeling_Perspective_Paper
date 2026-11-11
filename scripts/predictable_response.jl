#!/usr/bin/env julia
# How much of the PREDICTABLE short-time response do fitted models forecast? (R1; replaces the large-kick figure.)
#
# Kicked SLIP walks (slip_truth.jl; κ = 0.1 speed feedback and κ = 0.0 open loop): one disturbance per step at VLO,
# each step resampled onto 50 points of its own duration (stride-normalised time; 100 samples per stride).
# Observed: centre-of-mass (y, ẋ, ẏ) or height y only, z-scored on the training data, with 1% measurement noise.
#
#   training     walk seed 1, steps 101:100+NTR (NTR = 60 ≈ 30 strides, or 300)
#   validation   walk seed 2, 20 starts; settings chosen by multistart 1-stride rollout error (the observed, noisy
#                future), as recommended by FigS7_validation
#   test         walks 11–20, 30 starts each at random phases (slip_predictable_truth.jl); each forecast is warmed up on
#                the 400 observed samples before the start and run freely for 3 strides
# Scored against the TRUE future (noise-free; future disturbances included), per horizon sample h:
#   mean cycle   training mean step, phase-aligned        → skill 0
#   persistence  the last observed step, repeated
#   ceiling      noise-free continuation of the true state (no future disturbances): the best any forecast can do
#   skill(h) = 1 − MSE_model(h) / MSE_mean-cycle(h), MSE over starts and observed channels (and over height alone)
# Models: Hankel-DMD (one ridge fit; τ × rank × variant grid), ESN (N = 400; 24 settings; 4 seeds at the chosen one) and
# LSTM (Winner et al. 2023 training, lstm_model.jl: the training walk cut into 60-step trials, 128 units, 4 seeds).
# Squared errors are also kept per held-out walk (SEw) for bootstrap intervals.
#
# Usage: julia -t 8 --project=. scripts/predictable_response.jl   → cache/predictable/slip_results.jls

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using Serialization, LinearAlgebra, Statistics, Random, SparseArrays, Printf
BLAS.set_num_threads(1)

const ROOT = dirname(@__DIR__)
const NPH = 50                    # samples per step
const HIST = 400                  # warm-up samples (4 strides)
const HOR = 300                   # forecast horizon (3 strides)
const NOISE = 0.01                # measurement noise, z units
const WALK(κ, s) = joinpath(ROOT, "cache", "truth", "slip_walks", "walk_k$(κ)_s$(s).jls")
series(S, steps) = reduce(vcat, [S[k, :, :] for k in steps])           # (nsteps·NPH) × 3
idx(k, j) = (k - 1) * NPH + j                                            # series index of step k, phase j

# ── models ──────────────────────────────────────────────────────────────────
function hankel(Y, τ)
    n, d = size(Y); H = zeros((τ + 1) * d, n - τ)
    for i in 1:n-τ, l in 0:τ; H[l*d+1:(l+1)*d, i] = Y[i+l, :]; end
    H
end
function stabilise(A; unit = false, P = NPH)
    F = schur(complex(A)); T = copy(F.T)
    for i in axes(T, 1)
        m = abs(T[i, i]); (m > 1 || (unit && m^P > 0.5)) && (T[i, i] /= m)
    end
    real(F.Z * T * F.Z')
end
function fit_dmd(Y; τ, r, variant, ridge = 1e-8, basis = nothing)
    d = size(Y, 2); H = hankel(Y, τ); Uall = basis === nothing ? svd(H).U : basis
    r = min(r, size(H, 1) - 1, size(Uall, 2))
    U = Uall[:, 1:r]; a = U' * H; X, Z = a[:, 1:end-1], a[:, 2:end]
    reg(M) = M + ridge * tr(M) / r * I
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    A = stabilise(A; unit = variant == "unit")
    (; kind = :dmd, A, U, τ, d)
end
function forecast(m::NamedTuple{(:kind, :A, :U, :τ, :d)}, Yh, n)
    α = m.U' * vec(permutedims(Yh[end-m.τ:end, :])); out = zeros(n, m.d)
    for k in 1:n; α = m.A * α; out[k, :] = (m.U * α)[m.τ*m.d+1:end]; end
    out
end
function fit_esn(Y; N = 400, ρ, α, σin, λ, seed, washout = 100)
    rng = MersenneTwister(seed); d = size(Y, 2)
    W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= ρ / maximum(abs.(eigvals(W)))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    n = size(Y, 1); Rs = zeros(N, n); r = zeros(N)
    for t in 1:n; r = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * Y[t, :] .+ b); Rs[:, t] = r; end
    F = vcat(Rs[:, washout:n-1], Y[washout:n-1, :]', ones(1, n - washout))
    Wout = Y[washout+1:n, :]' * F' / (F * F' + λ * I)
    (; kind = :esn, W, Win, b, α, N, d, Wout)
end
function forecast(m::NamedTuple{(:kind, :W, :Win, :b, :α, :N, :d, :Wout)}, Yh, n)
    r = zeros(m.N)
    for t in 1:size(Yh, 1)-1                                             # teacher-forced warm-up
        r = (1 - m.α) .* r .+ m.α .* tanh.(m.W * r .+ m.Win * Yh[t, :] .+ m.b)
    end
    u = Yh[end, :]; out = zeros(n, m.d)
    for k in 1:n
        r = (1 - m.α) .* r .+ m.α .* tanh.(m.W * r .+ m.Win * u .+ m.b)
        u = m.Wout * vcat(r, u, 1.0); out[k, :] = u
        (any(!isfinite, u) || maximum(abs, u) > 1e3) && (out[k:end, :] .= NaN; break)
    end
    out
end
safe(f) = try f() catch; nothing end
forecast(f::Function, Yh, n) = f(Yh, n)
include("lstm_model.jl")

const DMD_TAUS = (25, 50, 100, 199, 299, 399)       # up to HIST − 1 samples (four strides)
const DMD_GRID = [(; r, variant, ridge) for r in (8, 16, 30, 50, 80, 120, 200), variant in ("ls", "fb", "unit"),
                                            ridge in (1e-8, 1e-5, 1e-3)]
const ESN_GRID = [(; ρ, α, σin, λ) for ρ in (0.8, 0.95), α in (0.2, 0.5, 1.0), σin in (0.3, 1.0), λ in (1e-6, 1e-4)]

# ── one configuration ───────────────────────────────────────────────────────
function run_config(κ, NTR, obs, cont)
    ch = obs == :full ? (1:3) : (1:1)
    w1 = deserialize(WALK(κ, 1)); w2 = deserialize(WALK(κ, 2))
    Xtr = series(w1.S, 101:100+NTR)[:, ch]
    μ = mean(Xtr, dims = 1); σ = std(Xtr, dims = 1)
    z(X) = (X[:, ch] .- μ) ./ σ
    noisy(X, seed) = z(X) .+ NOISE .* randn(MersenneTwister(seed), size(X, 1), length(ch))
    Ytr = noisy(series(w1.S, 1:size(w1.S, 1)), 7)[idx(101, 1):idx(100 + NTR, NPH), :]
    mstep = reduce(vcat, [mean(Ytr[s:NPH:end, :], dims = 1) for s in 1:NPH])  # NPH × d mean step (noisy training)
    # validation starts (walk 2)
    Yv = noisy(series(w2.S, 1:size(w2.S, 1)), 8)
    vrng = MersenneTwister(5); vstarts = [(101 + 2i, rand(vrng, 1:NPH)) for i in 0:19]
    function val_error(m)
        e = 0.0
        for (k, j) in vstarts
            i0 = idx(k, j); F = forecast(m, Yv[i0-HIST+1:i0, :], 2NPH)
            any(!isfinite, F) && return Inf
            e += mean(abs2, F .- Yv[i0+1:i0+2NPH, :])
        end
        e / length(vstarts)
    end
    # selection
    dv = Tuple{NamedTuple, Float64}[]
    for τ in DMD_TAUS
        Ub = svd(hankel(Ytr, τ)).U
        for g in DMD_GRID
            push!(dv, ((; τ, g...), something(safe(() -> val_error(fit_dmd(Ytr; τ, g..., basis = Ub))), Inf)))
        end
    end
    gd = dv[argmin(last.(dv))][1]
    ev = [(g, mean(something(safe(() -> val_error(fit_esn(Ytr; g..., seed = sd))), Inf) for sd in 1:2)) for g in ESN_GRID]
    ge = ev[argmin(last.(ev))][1]
    # LSTM (Winner et al. 2023 training): the training walk cut into trials of 60 steps, batched; 128 units; 4 seeds
    ltr = [Ytr[(q-1)*60NPH+1:q*60NPH, :] for q in 1:NTR÷60]
    lstm_secs = Float64[]
    lstms = map(1:4) do sd
        t0 = time(); m = fit_lstm_winner(ltr; H = 128, lookback = 299, trainseg = 8, valseg = 1, seed = sd)
        push!(lstm_secs, time() - t0); ("LSTM", sd, (Yh, n) -> forecast_lstm(m, Yh, n))
    end
    models = vcat([("DMD", 0, fit_dmd(Ytr; gd...))], [("ESN", sd, fit_esn(Ytr; ge..., seed = sd)) for sd in 1:4], lstms)
    # test
    names = vcat(["mean cycle", "persistence", "ceiling"], ["$(m[1]) $(m[2])" for m in models])
    SE = Dict(nm => zeros(HOR) for nm in names); SEy = Dict(nm => zeros(HOR) for nm in names)
    SEw = Dict(nm => zeros(10, HOR) for nm in names); NW = Dict(s => 0 for s in 11:20)          # per held-out walk
    nstart = 0; diverged = Dict(nm => 0 for nm in names); examples = Any[]
    for seed in 11:20
        w = deserialize(WALK(κ, seed)); c = cont["k$(κ)_s$(seed)"]
        Yo = noisy(series(w.S, 1:size(w.S, 1)), 100 + seed)                # observed
        Xt = z(series(w.S, 1:size(w.S, 1)))                                 # true, noise-free
        for (s, (k, j)) in enumerate(zip(c.ks, c.js))
            c.C[s] === nothing && continue
            i0 = idx(k, j); fut = Xt[i0+1:i0+HOR, :]
            Cs = z(reduce(vcat, [c.C[s][q, :, :] for q in axes(c.C[s], 1)]))[j+1:j+HOR, :]
            ph = [mod1(j + h, NPH) for h in 1:HOR]
            pers = reduce(vcat, [Yo[i0+h-NPH*cld(h, NPH):i0+h-NPH*cld(h, NPH), :] for h in 1:HOR])
            fc = Dict("mean cycle" => mstep[ph, :], "persistence" => pers, "ceiling" => Cs)
            for (nm, sd, m) in models
                fc["$nm $sd"] = forecast(m, Yo[i0-HIST+1:i0, :], HOR)
            end
            nstart += 1; NW[seed] += 1
            for (nm, F) in fc
                if any(!isfinite, F); diverged[nm] += 1; F = fill(1e3, size(F)); end
                SE[nm] .+= vec(mean(abs2, F .- fut, dims = 2)); SEy[nm] .+= abs2.(F[:, 1] .- fut[:, 1])
                SEw[nm][seed-10, :] .+= vec(mean(abs2, F .- fut, dims = 2))
            end
            length(examples) < 6 && seed == 11 && push!(examples, (; k, j, i0, hist = Yo[i0-HIST+1:i0, :],
                                                                  fut, fc = Dict(k_ => Float32.(v) for (k_, v) in fc),
                                                                  mstep_hist = mstep[[mod1(j - HIST + h, NPH) for h in 1:HIST], :]))
        end
    end
    MSE = Dict(nm => v ./ nstart for (nm, v) in SE); MSEy = Dict(nm => v ./ nstart for (nm, v) in SEy)
    skill = Dict(nm => 1 .- MSE[nm] ./ MSE["mean cycle"] for nm in names)
    skilly = Dict(nm => 1 .- MSEy[nm] ./ MSEy["mean cycle"] for nm in names)
    (; κ, NTR, obs, gd, ge, dmd_val = dv, esn_val = ev, names, MSE, MSEy, skill, skilly, nstart, diverged, examples,
       μ, σ, models = [(m[1], m[2]) for m in models], SEw, NW = [NW[s] for s in 11:20], lstm_secs)
end

cont = deserialize(joinpath(ROOT, "cache", "predictable", "slip_truth_continuations.jls"))
configs = [(κ, NTR, obs) for κ in (0.1, 0.0), NTR in (60, 300), obs in (:full, :height)] |> vec
out = Vector{Any}(undef, length(configs))
Threads.@threads for i in eachindex(configs)
    out[i] = run_config(configs[i]..., cont)
end
serialize(joinpath(ROOT, "cache", "predictable", "slip_results.jls"), out)

bin(v, a, b) = mean(v[round(Int, a * 2NPH)+1:round(Int, b * 2NPH)])
for o in out
    @printf("\nκ %.1f  %3d training steps  %-6s  starts %d | DMD %s | ESN %s\n", o.κ, o.NTR, o.obs, o.nstart,
            string(o.gd), string(o.ge))
    @printf("  %-14s %8s %8s %8s %8s   (skill vs mean cycle over horizon bins, strides)\n", "", "0–0.5", "0.5–1", "1–2", "2–3")
    for nm in o.names
        sk = o.skill[nm]
        @printf("  %-14s %8.2f %8.2f %8.2f %8.2f  diverged %d\n", nm, bin(sk, 0, 0.5), bin(sk, 0.5, 1), bin(sk, 1, 2),
                bin(sk, 2, 3), o.diverged[nm])
    end
end
