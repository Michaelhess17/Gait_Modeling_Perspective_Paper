#!/usr/bin/env julia
# Instrumental-variable stride maps on human gait (OUTLINE.md §5 ¶5, §6 ¶4). Result: not identifiable from 15-s trials.
#
# State at a stride section: six joint-angle deviations and their phase derivatives (12-D), z-scored per trial;
# stride triplets (k−1, k, k+1) pooled within trials over each group (AB, HF, LF); bootstrap over people.
#   LS  A = C₁C₀⁻¹ (stride k+1 on stride k)
#   IV  A = C(k+1, k−1) C(k, k−1)⁻¹ (previous stride as the instrument; Fig 6)
# Printed: full 12-D and rank-r (leading section PCs) spectral radii at 10 phases, and the pooled lag-1/lag-2
# autocorrelations of the leading PC. Finding (2026-10-27): lag-1 −0.33 to +0.16, lag-2 −0.33 to −0.02, comparable to
# the −1/n bias of removing each trial's mean over 6–12 strides; the IV ratio is then unstable (0.06 to >40).
#
# Usage: julia --project=. scripts/iv_gait_stride.jl

include("fig5_gait_data.jl")

data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
subs = String.(meta.subject); grp = String.(meta.lf_or_hf)
const PH = 1:10:100
trials = []
for i in axes(data, 1)
    S = strides(data[i, :, :]); size(S, 1) < 4 && continue
    D = S .- mean(S, dims = 1)
    V = (circshift(S, (0, -1, 0)) .- circshift(S, (0, 1, 0))) ./ 2; V = V .- mean(V, dims = 1)
    X = Dict(φ => (Z = hcat(D[:, φ, :], V[:, φ, :]); Z ./ max.(std(Z, dims = 1), eps())) for φ in PH)
    push!(trials, (; i, sub = subs[i], grp = grp[i], X))
end
function covs(T, φ)
    C = [zeros(12, 12) for _ in 1:4]                 # C(k,k), C(k+1,k), C(k+1,k−1), C(k,k−1)
    for t in T, k in 2:size(t.X[φ], 1)-1
        X = t.X[φ]; xm, x0, x1 = X[k-1, :], X[k, :], X[k+1, :]
        C[1] .+= x0 * x0'; C[2] .+= x1 * x0'; C[3] .+= x1 * xm'; C[4] .+= x0 * xm'
    end
    C
end
function est(T, φ, r)
    C00, C10, C1m, C0m = covs(T, φ)
    P = r == 12 ? Matrix(1.0I, 12, 12) : eigen(Symmetric(C00)).vectors[:, end:-1:end-r+1]
    pr(M) = P' * M * P
    maximum(abs, eigvals(pr(C10) / pr(C00))), maximum(abs, eigvals(pr(C1m) / pr(C0m)))
end
rng = MersenneTwister(1)
for g in ("AB", "HF", "LF"), r in (12, 2, 1)
    T = [t for t in trials if t.grp == g]; us = unique([t.sub for t in T])
    res = [est(T, φ, r) for φ in PH]
    bs = [begin pick = rand(rng, us, length(us)); TT = reduce(vcat, [[t for t in T if t.sub == s] for s in pick]); [est(TT, φ, r) for φ in PH] end
          for _ in 1:200]
    @printf("%s r=%2d  median over phases: LS %.2f [%.2f–%.2f]  IV %.2f [%.2f–%.2f]\n", g, r,
            median(first.(res)), quantile([median(first.(b)) for b in bs], [0.05, 0.95])...,
            median(last.(res)), quantile([median(last.(b)) for b in bs], [0.05, 0.95])...)
end
for g in ("AB", "HF", "LF"), φ in (1, 51)
    C00, C10, C1m, C0m = covs([t for t in trials if t.grp == g], φ); P = eigen(Symmetric(C00)).vectors[:, end]
    @printf("%s phase %2d%% PC1: lag-1 %.3f  lag-2 %.3f\n", g, φ - 1, P' * C10 * P / (P' * C00 * P), P' * C1m * P / (P' * C00 * P))
end
