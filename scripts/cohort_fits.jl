#!/usr/bin/env julia
# Individual against group fits on simulated cohorts with known truth (R1, R4, R6): how much of each individual's
# predictable short-time response does a model fitted to that individual, or to the whole cohort, forecast?
#
# Cohorts (10 individuals × 8 trials of ~15 s; trials 1–6 train, 7–8 test):
#   slip   kicked SLIP walkers (slip_family_data.jl), centre-of-mass (y, ẋ, ẏ), 50 samples per step; horizon 3 strides
#   cfhn   coupled FHN with process noise (cfhn_family_data.jl), (v₁, w₁, v₂, w₂), 50 samples per cycle; horizon 3 cycles
# Observations carry 1% measurement noise (of each individual's channel SD). Modes:
#   individual   one model per individual, fitted to its 6 training trials, z-scored with its own training data
#   group        one model for all 60 training trials, z-scored with the pooled training data
# Forecasts are scored in each individual's own z-units against the true future (noise-free, future disturbances
# included) from every test start; the noise-free continuation of the true state is the ceiling, the phase-aligned mean
# cycle the zero-skill reference, the last observed cycle repeated a second reference. Warm-up: the whole observed
# trial up to the start (DMD uses its delay window of it).
#   DMD    multi-trajectory Hankel-DMD; delay window, rank, variant chosen by 1-cycle rollout error from starts in the
#          held-back sixth training trial (fitted on trials 1–5), then refitted on trials 1–6
#   ESN    one reservoir, readout trained on all training trials; esn_tuning grid, chosen the same way; 3 seeds
#   LSTM   Winner et al. (2023) training (lstm_model.jl): batch = trials, stateful segments, teacher forcing, validation on
#          each trial's last segment, early stopping; 128 units individually, 512 for the group (GPU); 3 seeds
# Wall-clock fit times are recorded.
#
# Usage: julia -t 10 --project=. scripts/cohort_fits.jl <slip|cfhn> <dmd|esn|lstm_ind|lstm_grp>
#        (lstm_grp: NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib for the GPU)   → cache/cohort/<cohort>_<model>.jls

const COH, MODEL = ARGS[1], ARGS[2]
if MODEL == "lstm_grp"
    using CUDA, cuDNN
end
include("dmd_tuning.jl")                     # stabilise, build_hankel_multi, fit_template
include("lstm_model.jl")
using LinearAlgebra, SparseArrays, Random
BLAS.set_num_threads(1)

const ROOT_ = dirname(@__DIR__)
const NOISE = 0.01
const SPC = 50
const HOR = COH == "slip" ? 300 : 150
const VSTEP = COH == "slip" ? 100 : 50                     # validation horizon: one stride / cycle
const TAUS = COH == "slip" ? (25, 50, 100, 199, 299, 399) : (12, 25, 50, 100, 149)
const LOOKBACK, TRAINSEG = COH == "slip" ? (299, 4) : (149, 3)

# ── cohort ──────────────────────────────────────────────────────────────────
raw = deserialize(joinpath(ROOT_, "cache", COH == "slip" ? "slip_family" : "cfhn_family", "data.jls"))
series(tr) = COH == "slip" ? reduce(vcat, [tr.S[k, :, :] for k in axes(tr.S, 1)]) : Matrix{Float64}(tr)
cohort = map(raw) do c
    X = [series(tr) for tr in c.trials]                                   # noise-free
    μ = mean(reduce(vcat, X[1:6]), dims = 1); σ = std(reduce(vcat, X[1:6]), dims = 1)
    Xo = [x .+ NOISE .* σ .* randn(MersenneTwister(1000c.id + j), size(x)) for (j, x) in enumerate(X)]   # observed
    tests = map(c.tests) do t
        if COH == "slip"
            s = (t.k - 1) * SPC + t.j_phase
            Cc = reduce(vcat, [t.C[q, :, :] for q in axes(t.C, 1)])[t.j_phase+1:t.j_phase+HOR, :]
        else
            s = t.s; Cc = t.C[1:HOR, :]
        end
        (; trial = t.trial, s, ceiling = Cc)
    end
    (; id = c.id, X, Xo, μ, σ, tests)
end
const NI = length(cohort)
const μG = mean(reduce(vcat, [x for c in cohort for x in c.X[1:6]]), dims = 1)
const σG = std(reduce(vcat, [x for c in cohort for x in c.X[1:6]]), dims = 1)
zi(x, c) = (x .- c.μ) ./ c.σ
zg(x) = (x .- μG) ./ σG
g2i(F, c) = ((F .* σG .+ μG) .- c.μ) ./ c.σ

# ── models ──────────────────────────────────────────────────────────────────
function fit_dmd_multi(Ys; τ, r, variant, ridge = 1e-8, basis = nothing)
    Hs = [build_hankel_multi(Y, τ) for Y in Ys]
    Uall = basis === nothing ? svd(reduce(hcat, Hs)).U : basis
    r = min(r, size(Uall, 2), size(Hs[1], 1) - 1); U = Uall[:, 1:r]
    X = reduce(hcat, [U' * H[:, 1:end-1] for H in Hs]); Z = reduce(hcat, [U' * H[:, 2:end] for H in Hs])
    reg(M) = M + ridge * tr(M) / r * I
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    A = stabilise(A, SPC; mode = variant == "unit" ? :unit : :clip)
    d = size(Ys[1], 2)
    (Yh, n) -> begin
        α = U' * vec(permutedims(Yh[end-τ:end, :])); out = zeros(n, d)
        for k in 1:n; α = A * α; out[k, :] = (U * α)[τ*d+1:end]; end
        out
    end
end
function fit_esn_multi(Ys; N, ρ, α, λ, seed, σin = 0.5, washout = 50)
    rng = MersenneTwister(seed); d = size(Ys[1], 2)
    W = sprandn(rng, N, N, 10 / N); W .*= ρ / maximum(abs.(eigvals(Matrix(W))))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    step(r, u) = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * u .+ b)
    G = zeros(N + d + 1, N + d + 1); C = zeros(d, N + d + 1)
    for Y in Ys
        n = size(Y, 1); R = zeros(N, n - 1); r = zeros(N)
        for t in 1:n-1; r = step(r, Y[t, :]); R[:, t] = r; end
        F = vcat(R[:, washout:end], Y[washout:n-1, :]', ones(1, n - washout))
        G .+= F * F'; C .+= Y[washout+1:n, :]' * F'
    end
    Wout = C / (G + λ * I)
    (Yh, n) -> begin
        r = zeros(N)
        for t in 1:size(Yh, 1)-1; r = step(r, Yh[t, :]); end
        u = Yh[end, :]; out = zeros(n, d)
        for k in 1:n
            r = step(r, u); u = Wout * vcat(r, u, 1.0); out[k, :] = u
            (any(!isfinite, u) || maximum(abs, u) > 1e6) && (out[k:end, :] .= NaN; break)
        end
        out
    end
end
const DMD_GRID = [(; r, variant) for r in (16, 30, 50, 80, 120, 200, 300), variant in ("ls", "fb", "unit")]
const ESN_GRID = [(; ρ, α, N, λ) for ρ in (0.5, 0.9, 1.2), α in (0.2, 0.5, 1.0), N in (200, 800), λ in (1e-6, 1e-3)]

# validation: starts in the sixth training trial, one cycle/stride ahead, observed (noisy) data
vstarts(Y) = [s for s in (maximum(TAUS) + 1):(2SPC + 7):(size(Y, 1) - VSTEP)]
function val_error(fc, Vs)        # Vs: [(Y, starts, unit-converter)]
    e = 0.0; n = 0
    for (Y, ss, conv) in Vs, s in ss
        F = conv(fc(Y[1:s, :], VSTEP)); any(!isfinite, F) && return Inf
        e += mean(abs2, F .- conv(Y[s+1:s+VSTEP, :])); n += 1
    end
    e / n
end

"Choose DMD settings on trial 6 (fit 1–5), refit on 1–6; returns forecaster and record."
function select_dmd(fitY, valsets, allY)
    best = (Inf, nothing)
    for τ in TAUS
        Ub = svd(reduce(hcat, [build_hankel_multi(Y, τ) for Y in fitY])).U
        for g in DMD_GRID
            v = try val_error(fit_dmd_multi(fitY; τ, g..., basis = Ub), valsets) catch; Inf end
            v < best[1] && (best = (v, (; τ, g...)))
        end
    end
    t0 = time(); f = fit_dmd_multi(allY; best[2]...); (f, best[2], best[1], time() - t0)
end
function select_esn(fitY, valsets, allY; seeds = 1:3)
    vals = [median([try val_error(fit_esn_multi(fitY; g..., seed), valsets) catch; Inf end for seed in 1:2]) for g in ESN_GRID]
    g = ESN_GRID[argmin(vals)]
    fs = map(seeds) do seed
        t0 = time(); f = fit_esn_multi(allY; g..., seed); (f, time() - t0)
    end
    (fs, g, minimum(vals))
end

# ── evaluation ──────────────────────────────────────────────────────────────
"Squared errors per horizon sample, summed over the individual's test starts (its own z-units, all channels)."
function evaluate(c, forecasters)    # forecasters: Dict(name => (Yobs_trial_upto_s) -> HOR × d in the individual's z-units)
    SE = Dict{String, Vector{Float64}}(); nst = 0; div = Dict{String, Int}(); FC = Dict{String, Vector{Matrix{Float32}}}()
    mstep = COH == "slip" ? reduce(vcat, [mean(reduce(vcat, [zi(x, c)[q:SPC:end, :] for x in c.Xo[1:6]]), dims = 1) for q in 1:SPC]) : nothing
    for t in c.tests
        Yo = c.Xo[t.trial]; fut = zi(c.X[t.trial][t.s+1:t.s+HOR, :], c)
        hist_i = zi(Yo[1:t.s, :], c)
        ref = COH == "slip" ? mstep[[mod1(t.s + h, SPC) for h in 1:HOR], :] : fit_template(hist_i, SPC)(hist_i, HOR)
        pers = reduce(vcat, [hist_i[end-SPC+mod1(h, SPC):end-SPC+mod1(h, SPC), :] for h in 1:HOR])
        fc = Dict("mean cycle" => ref, "persistence" => pers, "ceiling" => zi(t.ceiling, c))
        for (nm, f) in forecasters; fc[nm] = f(Yo[1:t.s, :]); end
        nst += 1
        push!(get!(FC, "truth", Matrix{Float32}[]), Float32.(fut))
        for (nm, F) in fc
            push!(get!(FC, nm, Matrix{Float32}[]), Float32.(F))
            bad = any(!isfinite, F); bad && (div[nm] = get(div, nm, 0) + 1; F = fill(10.0, size(F)))
            SE[nm] = get(SE, nm, zeros(HOR)) .+ vec(mean(abs2, F .- fut, dims = 2))
        end
    end
    (; SE, nst, diverged = div, FC, starts = [(t.trial, t.s) for t in c.tests])
end

out = Dict{String, Any}("cohort" => COH, "model" => MODEL, "HOR" => HOR, "SPC" => SPC)
if MODEL in ("dmd", "esn")
    # individual
    ind = Vector{Any}(undef, NI)
    Threads.@threads for i in 1:NI
        c = cohort[i]; Yz = [zi(x, c) for x in c.Xo]
        vs = [(Yz[6], vstarts(Yz[6]), identity)]
        if MODEL == "dmd"
            f, g, v, tfit = select_dmd(Yz[1:5], vs, Yz[1:6])
            ev = evaluate(c, Dict("model" => Yh -> f(zi(Yh, c), HOR)))
            ind[i] = (; id = c.id, chosen = g, val = v, fit_secs = [tfit], ev...)
        else
            fs, g, v = select_esn(Yz[1:5], vs, Yz[1:6])
            ev = evaluate(c, Dict("model $k" => (Yh -> f(zi(Yh, c), HOR)) for (k, (f, _)) in enumerate(fs)))
            ind[i] = (; id = c.id, chosen = g, val = v, fit_secs = [t for (_, t) in fs], ev...)
        end
        println("$COH $MODEL individual $i done"); flush(stdout)
    end
    out["individual"] = ind
    # group
    Yg = [[zg(x) for x in c.Xo] for c in cohort]
    fitY = [y for Y in Yg for y in Y[1:5]]; allY = [y for Y in Yg for y in Y[1:6]]
    vsi = [(Yg[i][6], vstarts(Yg[i][6]), F -> ((F .* σG .+ μG) .- cohort[i].μ) ./ cohort[i].σ) for i in 1:NI]
    if MODEL == "dmd"
        f, g, v, tfit = select_dmd(fitY, vsi, allY)
        grp = [(; id = c.id, evaluate(c, Dict("model" => Yh -> g2i(f(zg(Yh), HOR), c)))...) for c in cohort]
        out["group"] = (; rows = grp, chosen = g, val = v, fit_secs = [tfit])
    else
        fs, g, v = select_esn(fitY, vsi, allY)
        grp = [(; id = c.id, evaluate(c, Dict("model $k" => (Yh -> g2i(f(zg(Yh), HOR), c)) for (k, (f, _)) in enumerate(fs)))...) for c in cohort]
        out["group"] = (; rows = grp, chosen = g, val = v, fit_secs = [t for (_, t) in fs])
    end
elseif MODEL == "lstm_ind"
    ind = Vector{Any}(undef, NI)
    Threads.@threads for i in 1:NI
        c = cohort[i]; Yz = [zi(x, c) for x in c.Xo]
        ms = map(1:3) do seed
            t0 = time(); m = fit_lstm_winner(Yz[1:6]; H = 128, lookback = LOOKBACK, trainseg = TRAINSEG, valseg = 1, seed)
            (m, time() - t0)
        end
        ev = evaluate(c, Dict("model $k" => (Yh -> forecast_lstm(m, zi(Yh, c), HOR)) for (k, (m, _)) in enumerate(ms)))
        ind[i] = (; id = c.id, fit_secs = [t for (_, t) in ms], best_epochs = [m.best_epoch for (m, _) in ms], ev...)
        println("$COH lstm individual $i: epochs $([m.best_epoch for (m, _) in ms])"); flush(stdout)
    end
    out["individual"] = ind
elseif MODEL == "lstm_grp"
    Yg = [zg(x) for c in cohort for x in c.Xo[1:6]]
    ms = map(1:3) do seed
        t0 = time(); m = fit_lstm_winner(Yg; H = 512, lookback = LOOKBACK, trainseg = TRAINSEG, valseg = 1, seed,
                                         device = gpu_device(), verbose = true)
        println("group seed $seed: epoch $(m.best_epoch), $(round((time() - t0) / 60, digits = 1)) min"); flush(stdout)
        (m, time() - t0)
    end
    grp = map(cohort) do c
        (; id = c.id, evaluate(c, Dict("model $k" => (Yh -> g2i(forecast_lstm(m, zg(Yh), HOR), c)) for (k, (m, _)) in enumerate(ms)))...)
    end
    out["group"] = (; rows = grp, fit_secs = [t for (_, t) in ms], best_epochs = [m.best_epoch for (m, _) in ms], device = "GPU")
end
mkpath(joinpath(ROOT_, "cache", "cohort"))
serialize(joinpath(ROOT_, "cache", "cohort", "$(COH)_$(MODEL).jls"), out)

# summary: share of attainable skill over the first stride / cycle and the next two, median over individuals
function share(r, nm, a, b)
    sk(x) = 1 - sum(x[a:b]) / sum(r.SE["mean cycle"][a:b])
    sk(r.SE[nm]) / sk(r.SE["ceiling"])
end
for mode in ("individual", "group")
    haskey(out, mode) || continue
    rows = mode == "individual" ? out[mode] : out[mode].rows
    nms = sort([k for k in keys(rows[1].SE) if startswith(k, "model")])
    s1 = [median([share(r, nm, 1, HOR ÷ 3) for nm in nms]) for r in rows]
    s2 = [median([share(r, nm, HOR ÷ 3 + 1, HOR) for nm in nms]) for r in rows]
    @printf("%s %s %-10s share of attainable skill, first stride: median %.2f (range %.2f–%.2f); strides 1–3: %.2f (%.2f–%.2f)\n",
            COH, MODEL, mode, median(s1), extrema(s1)..., median(s2), extrema(s2)...)
end
