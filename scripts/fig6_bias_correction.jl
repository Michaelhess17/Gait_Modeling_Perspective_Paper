#!/usr/bin/env julia
# Bias of a linear return-map fit under measurement noise, and its correction (OUTLINE.md §6 ¶4; text only,
# formerly Fig 6B–D; the IV line is in Fig 3B, C via fig3_slip_maps.jl).
#
# Same data and noise model as Fig 3 (scripts/fig3_slip_maps.jl): kicked SLIP walks (stride-level kicks to
# (y, θ) at VLO = process noise), VLO-section states observed with i.i.d. measurement noise of SD ratio × the
# stride-to-stride SD of each variable, 2-D section map fitted in the top-2 principal components.
#
# Estimators of the one-step map A from w_k = z_k + e_k (rows = strides, centred):
#   LS    least squares w_{k+1} ≈ A w_k. Treats w_k as exact: measurement noise in the regressor attenuates A.
#   FB    forward–backward (Dawson et al. 2016): A = (A_f A_b⁻¹)^{1/2}. Consistent for measurement noise on a
#         deterministic map; the stride kicks (process noise) make the backward fit biased.
#   TLS   total least squares on [w_k  w_{k+1}] (Hemati et al. 2017). Assumes equal noise in both and no
#         process noise.
#   IV    instrumental variables: w_{k−1} as the instrument, A = (W₊ᵀW₋)(WᵀW₋)⁻¹ — consistent under white
#         measurement AND process noise, because w_{k−1} is correlated with z_k but not with e_k, e_{k+1} or the
#         kick at k. Needs no knowledge of the noise.
#
# Usage: julia --project=. scripts/fig6_bias_correction.jl     (writes cache/fig6/bias.jls)

include("fig3_slip_maps.jl")      # pca2, noisy, WALKDIR (its main block does not run when included)

const NS6 = [30, 60, 120, 250, 500, 1000, 2000]
const NOISE6 = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

function maps(S)
    W, _ = pca2(S[:, 1, :]); W = W .- mean(W, dims = 1)
    Wm, W0, Wp = W[1:end-2, :], W[2:end-1, :], W[3:end, :]       # w_{k-1}, w_k, w_{k+1}
    Xa, Xb = W[1:end-1, :], W[2:end, :]
    ls = (Xa \ Xb)'
    ab = (Xb \ Xa)'
    fb = real(sqrt(complex(ls * inv(ab))))
    V = svd(hcat(Xa, Xb)).V; tls = (-V[1:2, 3:4] / V[3:4, 3:4])'
    iv = ((Wm' * W0) \ (Wm' * Wp))'
    (; LS = ls, FB = fb, TLS = tls, IV = iv)
end
specrad(A) = all(isfinite, A) ? maximum(abs, eigvals(A)) : NaN

if abspath(PROGRAM_FILE) == @__FILE__
    rows = NamedTuple[]
    for f in sort(readdir(WALKDIR))
        w = deserialize(joinpath(WALKDIR, f)); w.fallen && continue
        truth = maximum(abs, eigvals(w.J))
        for N in NS6, r in NOISE6
            N > size(w.S, 1) && continue
            S = noisy(w.S[1:N, :, :], r, MersenneTwister(hash((w.seed, N, r, :fig6))))
            for (m, A) in pairs(maps(S))
                push!(rows, (; kappa = w.kappa, seed = w.seed, N, noise = r, truth, method = String(m), ρ = specrad(A)))
            end
        end
    end
    mkpath(joinpath(ROOT, "cache", "fig6"))
    serialize(joinpath(ROOT, "cache", "fig6", "bias.jls"), rows)
    med(sel) = (v = filter(isfinite, [x.ρ for x in rows if sel(x)]); isempty(v) ? NaN : median(v))
    iqr(sel) = (v = filter(isfinite, [x.ρ for x in rows if sel(x)]); quantile(v, 0.75) - quantile(v, 0.25))
    for κ in (0.0, 0.1)
        @printf("κ = %.1f, truth %.3f\n", κ, [x.truth for x in rows if x.kappa == κ][1])
        for m in ("LS", "FB", "TLS", "IV")
            @printf("  %-4s N=2000: noise 0 %.3f | .25 %.3f | .5 %.3f | 1 %.3f   noise .25: N=60 %.3f (IQR %.3f) N=250 %.3f (IQR %.3f)\n", m,
                    [med(x -> x.kappa == κ && x.method == m && x.N == 2000 && x.noise == r) for r in (0.0, 0.25, 0.5, 1.0)]...,
                    med(x -> x.kappa == κ && x.method == m && x.N == 60 && x.noise == 0.25), iqr(x -> x.kappa == κ && x.method == m && x.N == 60 && x.noise == 0.25),
                    med(x -> x.kappa == κ && x.method == m && x.N == 250 && x.noise == 0.25), iqr(x -> x.kappa == κ && x.method == m && x.N == 250 && x.noise == 0.25))
        end
    end
end
