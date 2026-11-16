#!/usr/bin/env julia
# How many directions do the cycle and the departures from it occupy in human walking? (Figure 1A; text)
# 299 treadmill trials of 15 s (data/all_human_data.npy, 100 Hz, six sagittal joint angles). Strides cut at the phase of
# the leading oscillatory pair of all angles (pair_strides, maus_learning_all.jl) and resampled to 100 samples; each
# angle scaled by its SD over the trial; departures = each stride minus the trial's average stride. Principal
# directions of the delay vectors (the last L samples of all six angles; L = 13, 25, 50 ≈ ⅛, ¼, ½ stride), for the
# whole signal (centred) and for the departures: cumulative share of variance and participation ratio (Σλ)²/Σλ².
# Also each trial's departure share of the variance. Trials with fewer than 6 strides are skipped.
# Usage: julia --project=. scripts/gait_dimensionality.jl   → cache/fig5/dims.jls
empty!(ARGS)
include("maus_learning_all.jl")
using NPZ

"Cumulative share of variance of the principal directions of the L-sample delay vectors of Y, and the participation ratio."
function directions(Y, L; kmax = 60)
    s = svdvals(hankel(Y .- mean(Y, dims = 1), L)') .^ 2
    c = cumsum(s) ./ sum(s)
    (; cum = c[1:min(kmax, length(c))], pr = sum(s)^2 / sum(s .^ 2), k90 = searchsortedfirst(c, 0.9), k99 = searchsortedfirst(c, 0.99))
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    X = npzread(joinpath(ROOT, "data", "all_human_data.npy"))
    WIN = (13, 25, 50); rows = NamedTuple[]
    for i in axes(X, 1)
        S = try pair_strides(Float64.(X[i, :, :])) catch; nothing end
        (S === nothing || size(S, 1) < 6) && continue
        Sn = S ./ reshape(std(flat(S), dims = 1), 1, 1, :); m = mean(Sn, dims = 1)
        Y = flat(Sn); D = flat(Sn .- m)
        share = sum(abs2, D) / sum(abs2, Y .- mean(Y, dims = 1))
        push!(rows, (; trial = i, nstr = size(S, 1), share, whole = Dict(L => directions(Y, L) for L in WIN),
                       dev = Dict(L => directions(D, L) for L in WIN)))
    end
    for L in WIN
        @printf("window %2d samples: %d trials | whole: 90%% in %.0f, 99%% in %.0f directions, PR %.1f | departures: 90%% in %.0f, 99%% in %.0f, PR %.1f\n",
                L, length(rows), median(r.whole[L].k90 for r in rows), median(r.whole[L].k99 for r in rows), median(r.whole[L].pr for r in rows),
                median(r.dev[L].k90 for r in rows), median(r.dev[L].k99 for r in rows), median(r.dev[L].pr for r in rows))
    end
    @printf("departure share of the variance: median %.1f%% (IQR %.1f–%.1f%%)\n", 100median(r.share for r in rows),
            100quantile([r.share for r in rows], 0.25), 100quantile([r.share for r in rows], 0.75))
    serialize(joinpath(ROOT, "cache", "fig5", "dims.jls"), (; rows, WIN))
end
