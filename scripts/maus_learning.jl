#!/usr/bin/env julia
# Learning curves: which models of running deviations benefit from more training strides?
#
# Running trials (maus_extract.jl), 6 joint angles, deviations from the 61-stride moving-average stride (maus_lattice.jl
# :ma). Test set fixed: the second half of each trial's strides, starts every 5th sample, R² at horizons 6–100% of a
# stride (maus_horizon.jl). Training: the first NTR strides (NTR ∈ SIZES). Each model's settings are chosen at every
# size on an inner split of its own training strides (fit on the first half, score on the second, horizons 6–50%), then
# refitted on all NTR strides:
#   deviation DMD   delay d ∈ {16, 32}, rank ∈ {4 … 128}, free rollout (opfit: ridge 1e-4, Schur-stabilised)
#   LPV (binned)    Rosenberg-style direct maps, ⅓-stride delay window, 20 phase bins, ridge ∈ {1e-3, 1e-1, 10}
#   LTP             Fourier-in-phase operator, d = 8, rank ∈ {8, 16, 32, 64}, harmonics ∈ {0, 2, 4, 8, 12}, ridge ∈ {1e-3, 1e-2, 1e-1}
#   lattice         one Floquet mode, μ on ρ 0.05:0.1:0.95 × rotation 0:30:180°, window 50 samples
#   ltp_phase / ltp_nophase   LTP at a fixed setting, no selection: d 8, rank 32, ridge 1e-2, 8 harmonics / none
#                   (does phase dependence pay off at small sizes when the inner split does not have to find it?)
#
# Usage: julia --project=. -t 14 scripts/maus_learning.jl      (writes cache/maus/learning.jls)

include("maus_floquet_models.jl")

const SIZES = [4, 8, 16, 32, 64, 160]
const HV = [1, 2, 4, 6]                         # indices into HZ used for selection: 6, 12, 25, 50 samples
const HISTL = 50

"R² by horizon (indices hidx into HZ) of predh(t) (length(hidx) × p) over starts ts."
function score_at(predh, D, ts, hidx)
    E = zeros(length(hidx)); R = zeros(length(hidx))
    for t in ts
        P = predh(t)
        for (j, k) in enumerate(hidx); E[j] += sum(abs2, P[j, :] .- D[t+HZ[k], :]); R[j] += sum(abs2, D[t+HZ[k], :]); end
    end
    1 .- E ./ R
end
traj(pred) = (hidx -> (t -> pred(t)[HZ[hidx], :]))
vstarts(r) = (first(r) + HISTL):5:(last(r) - HZ[HV[end]])

"Direct phase-binned maps from X(t) to D(t+HZ[k]), trained on starts in tr."
function lpv_fit(D, X, tr; λ)
    W = Dict{Tuple{Int, Int}, Matrix{Float64}}()
    for (k, h) in enumerate(HZ), b in 1:NBIN
        T = [t for t in first(tr)+maximum(LAGS):last(tr)-h if phasebin(t) == b]
        isempty(T) && continue
        Z = hcat(ones(length(T)), reduce(vcat, [X(t)' for t in T])); Yt = reduce(vcat, [D[t+h, :]' for t in T])
        W[(k, b)] = (Z' * Z + λ * max(size(Z, 1), 1) * Diagonal([0; ones(size(Z, 2) - 1)])) \ (Z' * Yt)
    end
    hidx -> (t -> reduce(vcat, [(haskey(W, (k, phasebin(t))) ? W[(k, phasebin(t))]' * vcat(1.0, X(t)) : zeros(size(D, 2)))' for k in hidx]))
end

"Model families: name => (settings, fit(D, rows, setting) -> hidx -> predh)."
function families(D)
    dly(t) = reduce(vcat, [D[t-l, :] for l in LAGS])
    Dict(
      :dmd => ([(d, r) for d in (16, 32) for r in (4, 8, 16, 32, 64, 128)],
               (rows, s) -> (ncol = length(rows) - s[1]; ncol < 2s[2] ? nothing :
                             (F = opfit(D[rows, :]; d = s[1], r = s[2]); μ = mean(D[rows, :], dims = 1);
                              traj(t -> roll(F..., μ, D, t, s[1], HZ[end]))))),
      :lpv => ([1e-3, 1e-1, 10.0], (rows, s) -> lpv_fit(D, dly, rows; λ = s)),
      :ltp => ([(8, r, h, λ) for r in (8, 16, 32, 64) for h in (0, 2, 4, 8, 12) for λ in (1e-3, 1e-2, 1e-1)],
               (rows, s) -> (M = fit_ltp(D, rows, s[1], s[2], s[3], [s[4]])[1]; traj(ltp_predictor(D, M)))),
      :ltp_phase => ([(8, 32, 8, 1e-2)], (rows, s) -> (M = fit_ltp(D, rows, s[1], s[2], s[3], [s[4]])[1]; traj(ltp_predictor(D, M)))),
      :ltp_nophase => ([(8, 32, 0, 1e-2)], (rows, s) -> (M = fit_ltp(D, rows, s[1], s[2], s[3], [s[4]])[1]; traj(ltp_predictor(D, M)))),
      :lattice => ([μof(ρ, rot) for ρ in 0.05:0.1:0.95 for rot in 0:30:180],
                   (rows, s) -> (k1 = last(rows) ÷ NPH; segs = segments(D, 0, k1); length(segs) < 1 ? nothing :
                                 traj(modes_predictor(D, fit_modes(segs, [s]), [s])))),
    )
end

function trial_learning(S_)
    n = size(S_, 1); nh = n ÷ 2; _, _, D = split_baseline(S_, :ma)
    te = nh*NPH+1:n*NPH; ts = (first(te) + HISTL):STEP:(last(te) - HZ[end])
    F = families(D); out = Dict{Tuple{Symbol, Int}, Any}()
    for ntr in SIZES
        ntr > nh && continue
        rows = 1:ntr*NPH; inner = 1:(ntr÷2)*NPH; va = (ntr÷2)*NPH+1:ntr*NPH
        for (name, (settings, fit)) in F
            vs = map(settings) do s
                f = try fit(inner, s) catch; nothing end
                f === nothing ? -Inf : mean(score_at(f(HV), D, vstarts(va), HV))
            end
            all(isinf, vs) && continue
            s = settings[argmax(vs)]; f = fit(rows, s)
            out[(name, ntr)] = (; setting = s, r2 = score_at(f(1:length(HZ)), D, ts, 1:length(HZ)))
            name == :ltp && (out[(:ltp_harmonics, ntr)] = s[3])
        end
    end
    out
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls")); T = A.trials[qc_run.([strides(t.X) for t in A.trials]) .< 5]
    R = tmap(t -> trial_learning(strides(t.X)), T); bad = count(isnothing, R); R = filter(!isnothing, R)
    @printf("joint angles, baseline ma, %d trials (%d failed): median test R² at 12.5%% / 25%% / 50%% of a stride, by training strides\n", length(R), bad)
    @printf("%-12s", "strides"); for n in SIZES; @printf("%20d", n); end; println()
    for name in (:dmd, :lpv, :ltp, :ltp_phase, :ltp_nophase, :lattice)
        @printf("%-12s", name)
        for n in SIZES
            v = [r[(name, n)].r2 for r in R if haskey(r, (name, n))]
            isempty(v) ? @printf("%20s", "—") : @printf("%20s", @sprintf("%.2f / %.2f / %.2f", median(x[2] for x in v), median(x[4] for x in v), median(x[6] for x in v)))
        end
        println()
    end
    serialize(joinpath(ROOT, "cache", "maus", "learning.jls"), (; R, SIZES, HZ))
    println("wrote cache/maus/learning.jls")
end
