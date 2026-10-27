#!/usr/bin/env julia
# Instrumental-variable version of the companion paper's operator (OUTLINE.md §5, §6 ¶4).
#
# Operator as in fig5_gait_data.jl (delay window d, rank 70, ridge 1e-4, Schur stabilisation, TG = max over one
# stride of ‖Ãᵏ‖). LS: the companion paper's ridge least squares. IV: two-stage least squares with the delay window
# that ends g samples before the fitted window as the instrument (rank 70), then the same ridge step. IV discards
# whatever in the fitted window is not predictable from more than g samples back: white or short-correlated
# measurement noise, but also any genuine deviation structure that decorrelates faster than d + g samples.
#
#   calibration   the trial's own mean cycle (truth: its noise-free TG) plus white noise of σ × the trial's
#                 deviation SD, σ ∈ {0, 0.5, 1, 2}; d = 32, g = 10
#   recorded      all trials, d ∈ {32, 100}, g ∈ {10, 30}
#
# Usage: julia --project=. scripts/iv_gait_operator.jl      (writes cache/fig5/iv_operator.jls)

include("fig5_gait_data.jl")

"LS (companion) and IV (2SLS) ridge operators from the same fitted window."
function ls_iv(Y; d = 32, r = 70, α = 1e-4, g = 10)
    H = hankel(Y .- mean(Y, dims = 1), d); s = d + g; N = size(H, 2)
    X = H[:, s+1:N-1]; Yp = H[:, s+2:N]; Z = H[:, 1:N-s-1]
    F = svd(X); U = F.U[:, 1:r]; s1 = F.S[1]
    Xr = U' * X; Yr = U' * Yp
    ridge(Xin) = stabilize(Yr * Xin' / (Xin * Xin' + α * s1^2 * I))
    Zr = svd(Z).U[:, 1:r]' * Z
    ridge(Xr), ridge((Xr * Zr') / (Zr * Zr') * Zr)
end

if abspath(PROGRAM_FILE) == @__FILE__
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    grp = String.(meta.lf_or_hf); rng = MersenneTwister(3)
    calib = NamedTuple[]; rec = NamedTuple[]
    for i in axes(data, 1)
        Y = stride_rescale(data[i, :, :]); M = meancycle_only(Y); sd = std(Y .- M)
        truth = log10(tg(ls_iv(M)[1]))
        for σ in (0.0, 0.5, 1.0, 2.0)
            a, b = ls_iv(M .+ σ * sd .* randn(rng, size(M)))
            push!(calib, (; i, grp = grp[i], σ, truth, ls = log10(tg(a)), iv = log10(tg(b))))
        end
        for d in (32, 100), g in (10, 30)
            a, b = ls_iv(Y; d, g)
            push!(rec, (; i, grp = grp[i], d, g, ls = log10(tg(a)), iv = log10(tg(b))))
        end
        i % 50 == 0 && println("trial $i")
    end
    serialize(joinpath(ROOT, "cache", "fig5", "iv_operator.jls"), (; calib, rec))
    for σ in (0.0, 0.5, 1.0, 2.0)
        c = [x for x in calib if x.σ == σ]
        @printf("calibration σ = %.1f: median error LS %+.3f  IV %+.3f (truth median %.3f)\n", σ,
                median([x.ls - x.truth for x in c]), median([x.iv - x.truth for x in c]), median([x.truth for x in c]))
    end
    for d in (32, 100), g in (10, 30), G in ("AB", "HF", "LF")
        c = [x for x in rec if x.d == d && x.g == g && x.grp == G]
        @printf("recorded d=%3d g=%d %s: LS %.3f  IV %.3f\n", d, g, G, median([x.ls for x in c]), median([x.iv for x in c]))
    end
end
