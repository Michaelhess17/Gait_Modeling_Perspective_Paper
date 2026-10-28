#!/usr/bin/env julia
# Lattice-constrained Floquet model on noise-driven, stride-normalised data: kicked SLIP walks (validation, truth
# 0.769 ∠ 13.6° per step) and the human gait superset (OUTLINE.md §6 ¶6).
#
# Data: strides re-anchored at a fixed phase and resampled to 100 points each (gait: fig5_gait_data.jl strides(),
# hip Hilbert phase; SLIP: VLO to VLO). This re-estimates the phase every stride, which removes phase diffusion.
# Channels standardised by their SD; deviation = strides minus the trial's mean stride (all strides: a mean from
# the first half only leaves an offset in the second half that a near-neutral mode then 'predicts').
# Model for the deviation over a 3-stride segment that starts at the stride origin (t = 0 … 299):
#   δ(t) = Re[ψ e^{μt} b(θ_t)],  θ_t = 2πt/100,  b(θ) = Σ_{|k|≤Ks} b_k e^{ikθ}   (one Floquet mode)
# μ is the only spectral parameter (per stride: e^{100μ} = ρ e^{i·rot}); b is shared by all segments of a trial,
# ψ is per segment. Noise-driven data have no decays, so μ is chosen by forward prediction: b fitted on segments
# from the first half of the trial's strides; on segments from the second half, ψ is fitted to the first W_WIN
# samples (half a stride) and the next 2.5 strides are predicted. Skill: R² of the deviation 1 and 2 strides
# ahead (0 = mean cycle alone). Comparison: the companion paper's delay DMD (d 32, rank 70, ridge 1e-4) fitted
# to the first half and run freely from the same windows.
#
# Usage: julia --project=. -t 12 scripts/gait_lattice.jl      (writes cache/fig5/gait_lattice.jls)

include("fig5_gait_data.jl")

const T_SEG, W_WIN, KS = 300, 50, 12
const RHOS = 0.05:0.05:0.95
const ROTS = 0:15:180

basis(μ, Ks = KS; T = T_SEG) = [exp(μ * t + im * k * 2π * t / NPH) for t in 0:T-1, k in -Ks:Ks]
μof(ρ, rot) = (log(ρ) + im * deg2rad(rot)) / NPH

"Shared mode shape b and per-segment ψ (rank-1 start from free per-segment fits, then alternating LS)."
function fit_mode(segs, μ; Ks = KS, iters = 15)
    G = basis(μ, Ks); nb = size(G, 2); p = size(segs[1], 2)
    Rp = pinv(hcat(real(G), -imag(G)))
    Z = reduce(hcat, [(C = Rp * S; vec(C[1:nb, :] .- im .* C[nb+1:end, :])) for S in segs])
    F = svd(Z); b = reshape(F.U[:, 1] .* F.S[1], nb, p); ψ = conj.(F.V[:, 1])
    for _ in 1:iters
        Q = G * b; Dq = hcat(vec(real(Q)), vec(-imag(Q)))
        ψ = [(c = Dq \ vec(S); c[1] + im * c[2]) for S in segs]
        A = reduce(vcat, [hcat(real(ψ[s] .* G), -imag(ψ[s] .* G)) for s in eachindex(segs)])
        Bc = A \ reduce(vcat, segs); b = Bc[1:nb, :] .+ im .* Bc[nb+1:end, :]
    end
    (; G, b)
end
"Deviation predicted over the segment from ψ fitted to its first W samples."
function predict_mode(M, S; W = W_WIN)
    Q = M.G * M.b; c = hcat(vec(real(Q[1:W, :])), vec(-imag(Q[1:W, :]))) \ vec(S[1:W, :])
    real.((c[1] + im * c[2]) .* Q)
end
skill(err, ref) = 1 - err / ref

"Segments (T_SEG × p deviations) starting at each stride origin in [k0, k1)."
segments(D, k0, k1) = [D[k*NPH+1:k*NPH+T_SEG, :] for k in k0:k1-1 if (k + 3) * NPH <= size(D, 1) && k + 3 <= k1]

"Companion delay DMD fitted to Ytr, run from the window ending at row t0 + W of Y; deviation from the mean stride."
function dmd_predict(A, U, μY, Y, t0, M; d = 32, W = W_WIN)
    tw = t0 + W; h = vec(permutedims(Y[tw-d+1:tw, :] .- μY)); α = U' * h; p = size(Y, 2)
    out = zeros(T_SEG - W, p)
    for k in 1:T_SEG-W; α = A * α; out[k, :] = (U * α)[end-p+1:end] .+ vec(μY); end
    out .- M[mod1.(tw .+ (1:T_SEG-W), NPH), :]
end

"Fit and score one trial: S is n × 100 × p strides. mean_all: mean stride from all strides (else the first half)."
function trial_lattice(S; Ks = KS, W = W_WIN, mean_all = true)
    n, _, p = size(S); Y = reduce(vcat, [S[k, :, :] for k in 1:n]); Y = Y ./ std(Y, dims = 1)
    nh = n ÷ 2; Sy = permutedims(reshape(Y, NPH, n, p), (2, 1, 3))
    M = dropdims(mean(Sy[1:(mean_all ? n : nh), :, :], dims = 1), dims = 1)
    D = Y .- reduce(vcat, fill(M, n))
    tr = segments(D, 0, nh); te = segments(D, nh, n)
    (length(tr) < 2 || length(te) < 1) && return nothing
    ref1 = sum(sum(abs2, S_[NPH+1:2NPH, :]) for S_ in te); ref2 = sum(sum(abs2, S_[2NPH+1:3NPH, :]) for S_ in te)
    prof = fill(Inf, length(RHOS), length(ROTS)); best = nothing
    for (i, ρ) in enumerate(RHOS), (j, rot) in enumerate(ROTS)
        Mm = fit_mode(tr, μof(ρ, rot); Ks)
        e = sum(sum(abs2, predict_mode(Mm, S_; W)[W+1:end, :] .- S_[W+1:end, :]) for S_ in te); prof[i, j] = e
        e <= minimum(prof) && (best = (; ρ, rot, M = Mm))
    end
    P = [predict_mode(best.M, S_; W) for S_ in te]
    r1 = skill(sum(sum(abs2, Pp[NPH+1:2NPH, :] .- S_[NPH+1:2NPH, :]) for (Pp, S_) in zip(P, te)), ref1)
    r2 = skill(sum(sum(abs2, Pp[2NPH+1:3NPH, :] .- S_[2NPH+1:3NPH, :]) for (Pp, S_) in zip(P, te)), ref2)
    A, U = opfit(Y[1:nh*NPH, :]); μY = mean(Y[1:nh*NPH, :], dims = 1)
    starts = [k * NPH for k in nh:n-1 if (k + 3) * NPH <= size(Y, 1)]
    Q = [dmd_predict(A, U, μY, Y, t0, M; W) for t0 in starts]
    d1 = skill(sum(sum(abs2, q[NPH-W+1:2NPH-W, :] .- S_[NPH+1:2NPH, :]) for (q, S_) in zip(Q, te)), ref1)
    d2 = skill(sum(sum(abs2, q[2NPH-W+1:3NPH-W, :] .- S_[2NPH+1:3NPH, :]) for (q, S_) in zip(Q, te)), ref2)
    θ = range(0, 2π; length = NPH + 1)[1:NPH]; bθ = [sum(best.M.b[k+Ks+1, :] .* exp(im * k * t) for k in -Ks:Ks) for t in θ]
    shape = reduce(hcat, bθ)'                                        # 100 × p complex mode shape over the stride
    (; ρ = best.ρ, rot = best.rot, prof, r1, r2, d1, d2, shape, joint_share = vec(sum(abs2, shape, dims = 1)) ./ sum(abs2, shape),
       ntr = length(tr), nte = length(te), n)
end

function summarise(lab, R)
    R = filter(!isnothing, R)
    @printf("%-14s n=%3d  multiplier per stride: ρ median %.2f [IQR %.2f–%.2f], rotation median %.0f° [%.0f–%.0f], complex in %d%% | ",
            lab, length(R), median(r.ρ for r in R), quantile([r.ρ for r in R], 0.25), quantile([r.ρ for r in R], 0.75),
            median(r.rot for r in R), quantile([r.rot for r in R], 0.25), quantile([r.rot for r in R], 0.75),
            round(Int, 100 * mean(r.rot > 0 && r.rot < 180 for r in R)))
    @printf("R² 1 / 2 strides ahead: lattice %.2f / %.2f, companion DMD %.2f / %.2f\n", median(r.r1 for r in R), median(r.r2 for r in R),
            median(r.d1 for r in R), median(r.d2 for r in R))
end

if abspath(PROGRAM_FILE) == @__FILE__
    K = deserialize(joinpath(ROOT, "cache", "truth", "slip_kicked_walks.jls"))
    @printf("SLIP truth: %.3f ∠ %.1f° per step\n", maximum(abs, K.λ), rad2deg(maximum(angle.(K.λ))))
    Rs = [trial_lattice(W) for W in K.walks]; summarise("SLIP kicked", Rs)
    Pp = sum(r.prof for r in Rs); i, j = Tuple(argmin(Pp)); @printf("  pooled over walks: ρ %.2f, rotation %d°\n", RHOS[i], ROTS[j]); flush(stdout)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame); grp = String.(meta.lf_or_hf)
    Rg = Vector{Any}(undef, size(data, 1)); BLAS.set_num_threads(1)
    Threads.@threads for i in axes(data, 1)
        Rg[i] = try trial_lattice(strides(data[i, :, :])) catch; nothing end
    end
    for G in ("AB", "HF", "LF"); summarise("gait " * G, Rg[grp .== G]); end
    serialize(joinpath(ROOT, "cache", "fig5", "gait_lattice.jls"), (; slip = Rs, gait = Rg, grp, subs = String.(meta.subject)))
end
