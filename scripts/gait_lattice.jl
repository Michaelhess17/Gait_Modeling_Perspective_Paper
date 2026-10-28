#!/usr/bin/env julia
# Lattice-constrained Floquet model on noise-driven, stride-normalised data: kicked SLIP walks (validation, truth
# 0.769 ∠ 13.6° per step) and the human gait superset (OUTLINE.md §6 ¶6; Fig S6).
#
# Data: strides re-anchored at a fixed phase and resampled to 100 points each (gait: fig5_gait_data.jl strides(),
# hip Hilbert phase; SLIP: VLO to VLO). This re-estimates the phase every stride, which removes phase diffusion.
# Channels standardised by their SD; deviation = strides minus the trial's mean stride (all strides: a mean from
# the first half only leaves an offset in the second half that a near-neutral mode then 'predicts').
# Model for the deviation over a 3-stride segment that starts at the stride origin (t = 0 … 299):
#   δ(t) = Re[ψ e^{μt} b(θ_t)],  θ_t = 2πt/100,  b(θ) = Σ_{|k|≤Ks} b_k e^{ikθ}   (one Floquet mode)
# μ is the only spectral parameter (per stride: e^{100μ} = ρ e^{i·rot}); b is shared by all segments of a trial,
# ψ is per segment. b is fitted on segments from the first half of the trial's strides; on segments from the
# second half, ψ is fitted to the first W_WIN samples (half a stride) and the next 2.5 strides are predicted.
# Squared errors are kept per block (rest of the current stride; 1 and 2 strides ahead) for every μ on the grid.
#
# Selection never uses the scored strides: a trial's μ is the grid point that best predicts strides +1 and +2 on
# the SAME PERSON's other trials (each trial's error normalised by its own deviation), or, when the person has no
# other usable trial, on the other people of its group. Skill = R² of the deviation 1 and 2 strides ahead
# (0 = mean stride alone). The per-person multiplier is the minimum of the summed profile over all their trials.
# (Choosing μ on the scored strides themselves, as the first version of this script did, inflates the skill.)
# Baselines, scored on the same windows and strides:
#   deviation DMD  delay DMD (d 32, ridge 1e-4, Schur-stabilised) fitted to the first half's deviations and run
#                  freely from the window; rank chosen out of trial like μ, from RANKS
#   companion DMD  the companion paper's operator (d 32, rank 70) fitted to the first half's full signal (cycle
#                  included) and run freely; scored on its deviation from the mean stride
# Stride-matched check: every trial cut to its first N_MATCH strides (AB trials are longer than stroke trials).
# Mode shape: b(θ) refitted on all of a trial's segments at its person's multiplier.
#
# Usage: julia --project=. -t 12 scripts/gait_lattice.jl      (writes cache/fig5/gait_lattice.jls)

include("fig5_gait_data.jl")

const T_SEG, W_WIN, KS = 300, 50, 12
const RHOS = 0.05:0.05:0.95
const ROTS = 0:15:180
const RANKS = (2, 4, 8, 16, 32, 70)
const N_MATCH = 10

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
"Squared error of P (rows = segment samples W+1 … T_SEG) on segment S: rest of the stride, stride +1, stride +2."
blocks(P, S; W = W_WIN) = [sum(abs2, P[a-W:b-W, :] .- S[a:b, :]) for (a, b) in ((W + 1, NPH), (NPH + 1, 2NPH), (2NPH + 1, 3NPH))]

"Segments (T_SEG × p deviations) starting at each stride origin in [k0, k1)."
segments(D, k0, k1) = [D[k*NPH+1:k*NPH+T_SEG, :] for k in k0:k1-1 if (k + 3) * NPH <= size(D, 1) && k + 3 <= k1]

"Delay DMD (A, U, mean) fitted to Ytr, run freely from the window ending at row t0 + W of Y; minus the mean stride M."
function dmd_predict(A, U, μY, Y, t0, M; d = 32, W = W_WIN)
    tw = t0 + W; h = vec(permutedims(Y[tw-d+1:tw, :] .- μY)); α = U' * h; p = size(Y, 2)
    out = zeros(T_SEG - W, p)
    for k in 1:T_SEG-W; α = A * α; out[k, :] = (U * α)[end-p+1:end] .+ vec(μY); end
    out .- M[mod1.(tw .+ (1:T_SEG-W), NPH), :]
end

"Standardised signal Y, mean stride M (all strides), deviation D."
function deviations(S)
    n, _, p = size(S); Y = reduce(vcat, [S[k, :, :] for k in 1:n]); Y = Y ./ std(Y, dims = 1)
    M = dropdims(mean(permutedims(reshape(Y, NPH, n, p), (2, 1, 3)), dims = 1), dims = 1)
    Y, M, Y .- reduce(vcat, fill(M, n))
end

"Errors of every grid μ, every deviation-DMD rank and the companion DMD on one trial (S: n × 100 × p strides)."
function trial_profiles(S; Ks = KS, W = W_WIN)
    n, _, p = size(S); Y, M, D = deviations(S); nh = n ÷ 2
    tr = segments(D, 0, nh); te = segments(D, nh, n)
    (length(tr) < 2 || length(te) < 1) && return nothing
    ref = sum(blocks(zeros(T_SEG - W, p), S_) for S_ in te)
    prof = zeros(length(RHOS), length(ROTS), 3)
    for (i, ρ) in enumerate(RHOS), (j, rot) in enumerate(ROTS)
        Mm = fit_mode(tr, μof(ρ, rot); Ks)
        prof[i, j, :] = sum(blocks(predict_mode(Mm, S_; W)[W+1:end, :], S_) for S_ in te)
    end
    starts = [k * NPH for k in nh:n-1 if (k + 3) * NPH <= size(Y, 1) && k + 3 <= n]
    A, U = opfit(Y[1:nh*NPH, :]); μY = mean(Y[1:nh*NPH, :], dims = 1)
    cdmd = sum(blocks(dmd_predict(A, U, μY, Y, t0, M; W), S_) for (t0, S_) in zip(starts, te))
    Dtr = D[1:nh*NPH, :]; μD = mean(Dtr, dims = 1); Z = zero(M)
    derr(r) = (F = opfit(Dtr; r); sum(blocks(dmd_predict(F[1], F[2], μD, D, t0, Z; W), S_) for (t0, S_) in zip(starts, te)))
    ddmd = reduce(vcat, [derr(r)' for r in RANKS])
    (; prof, ref, cdmd, ddmd, n, ntr = length(tr), nte = length(te))
end

"Mode shape over the stride (100 × p complex), refitted on all segments of the trial at μ."
function trial_shape(S, μ; Ks = KS)
    _, _, D = deviations(S); segs = segments(D, 0, size(S, 1)); length(segs) < 2 && return nothing
    b = fit_mode(segs, μ; Ks).b; θ = range(0, 2π; length = NPH + 1)[1:NPH]
    permutedims(reduce(hcat, [sum(b[k+Ks+1, :] .* exp(im * k * t) for k in -Ks:Ks) for t in θ]))
end

crit(r) = (r.prof[:, :, 2] .+ r.prof[:, :, 3]) ./ (r.ref[2] + r.ref[3])           # strides +1 and +2
critd(r) = (r.ddmd[:, 2] .+ r.ddmd[:, 3]) ./ (r.ref[2] + r.ref[3])

"Out-of-trial choice of μ and deviation-DMD rank for each trial, and the skill it gives on that trial."
function heldout(R, subs, grp)
    out = Vector{Any}(nothing, length(R))
    for i in eachindex(R)
        r = R[i]; r === nothing && continue
        js = [j for j in eachindex(R) if j != i && R[j] !== nothing && subs[j] == subs[i]]; lvl = :person
        isempty(js) && (js = [j for j in eachindex(R) if R[j] !== nothing && subs[j] != subs[i] && grp[j] == grp[i]]; lvl = :group)
        ij = argmin(sum(crit(R[j]) for j in js)); kd = argmin(sum(critd(R[j]) for j in js)); io = argmin(crit(r))
        out[i] = (; ρ = RHOS[ij[1]], rot = ROTS[ij[2]], lvl, rank = RANKS[kd],
                  lat = [1 - r.prof[ij, b] / r.ref[b] for b in 1:3], ddmd = [1 - r.ddmd[kd, b] / r.ref[b] for b in 1:3],
                  cdmd = [1 - r.cdmd[b] / r.ref[b] for b in 1:3], insample = [1 - r.prof[io, b] / r.ref[b] for b in 1:3])
    end
    out
end
"Per-person multiplier (ρ, rotation): grid minimum of the summed normalised profile over the person's trials."
function person_mu(R, subs)
    us = unique(subs[[r !== nothing for r in R]])
    Dict(s => (ij = argmin(sum(crit(R[j]) for j in findall(subs .== s) if R[j] !== nothing)); (RHOS[ij[1]], ROTS[ij[2]])) for s in us)
end

med_iqr(v) = @sprintf("%.2f [%.2f, %.2f]", median(v), quantile(v, 0.25), quantile(v, 0.75))
function summarise(lab, H, μp, subs)
    ok = [h !== nothing for h in H]; H = H[ok]; s = unique(subs[ok])
    ρs = [μp[x][1] for x in s]; rs = [μp[x][2] for x in s]
    @printf("%-16s trials %3d (μ from person %d, group %d), people %2d | per-person ρ %s, rotation %s\n", lab, length(H),
            count(h.lvl == :person for h in H), count(h.lvl == :group for h in H), length(s), med_iqr(ρs), med_iqr(Float64.(rs)))
    for (b, nm) in ((2, "+1 stride"), (3, "+2 strides"))
        @printf("    R² %-10s lattice %s (> 0 in %d%%; on-test selection would give %.2f) | deviation DMD %s | companion DMD %s | lattice > dev. DMD in %d%%\n",
                nm, med_iqr([h.lat[b] for h in H]), round(Int, 100 * mean(h.lat[b] > 0 for h in H)), median(h.insample[b] for h in H),
                med_iqr([h.ddmd[b] for h in H]), med_iqr([h.cdmd[b] for h in H]), round(Int, 100 * mean(h.lat[b] > h.ddmd[b] for h in H)))
    end
    @printf("    rest of the current stride: lattice %.2f, deviation DMD %.2f | deviation-DMD ranks chosen: %s\n",
            median(h.lat[1] for h in H), median(h.ddmd[1] for h in H), join(sort(unique(h.rank for h in H)), " "))
end
tmap(f, xs) = (out = Vector{Any}(undef, length(xs)); Threads.@threads for i in eachindex(xs); out[i] = try f(xs[i]) catch; nothing end; end; out)

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    K = deserialize(joinpath(ROOT, "cache", "truth", "slip_kicked_walks.jls"))
    @printf("SLIP truth: %.3f ∠ %.1f° per step\n", maximum(abs, K.λ), rad2deg(maximum(angle.(K.λ))))
    Rs = tmap(trial_profiles, K.walks); ss = fill("slip", length(Rs)); Hs = heldout(Rs, ss, ss); μs = person_mu(Rs, ss)
    summarise("SLIP kicked", Hs, μs, ss); flush(stdout)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    grp = String.(meta.lf_or_hf); subs = String.(meta.subject)
    Ss = tmap(i -> strides(data[i, :, :]), collect(axes(data, 1)))
    Rg = tmap(trial_profiles, Ss); Hg = heldout(Rg, subs, grp); μg = person_mu(Rg, subs)
    for G in ("AB", "HF", "LF"); summarise("gait " * G, Hg[grp .== G], μg, subs[grp .== G]); end; flush(stdout)
    R10 = tmap(S -> trial_profiles(S[1:min(end, N_MATCH), :, :]), Ss); H10 = heldout(R10, subs, grp); μ10 = person_mu(R10, subs)
    for G in ("AB", "HF", "LF"); summarise("≤$N_MATCH strides " * G, H10[grp .== G], μ10, subs[grp .== G]); end; flush(stdout)
    shapes = tmap(i -> Rg[i] === nothing ? nothing : trial_shape(Ss[i], μof(μg[subs[i]]...)), collect(eachindex(Ss)))
    slip_shapes = [trial_shape(W, μof(μs["slip"]...)) for W in K.walks]
    serialize(joinpath(ROOT, "cache", "fig5", "gait_lattice.jls"),
              (; slip = Rs, slip_held = Hs, slip_mu = μs["slip"], slip_shapes, gait = Rg, held = Hg, mu = μg,
                 gait10 = R10, held10 = H10, mu10 = μ10, shapes, grp, subs, speed = Float64.(meta.speed), nstrides = [size(S, 1) for S in Ss]))
end
