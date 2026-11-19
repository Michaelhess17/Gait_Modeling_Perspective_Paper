#!/usr/bin/env julia
# Does the instrumental-variable correction that rescues SLIP's section map (Figure 7 E–F) help on human data? Section maps
# on long running trials (Maus et al. 2015; maus_learning_all.jl's 22 trials, strides cut at the leading oscillatory
# pair's phase, deviation from the 61-stride moving-average stride), with the estimator of fig3_slip_maps.jl:
#   section state  the deviation at stride phase φ₀ in its top-2 principal components (PCs of the fitting strides)
#   LS             affine least-squares map from stride k to k + 1
#   IV             the same with the previous stride (k − 1) as the instrument (consistent under white measurement noise
#                  and stride-level kicks when the instrument is correlated with the regressor)
# Running: fitted on the first NTR strides (NTR ∈ SIZES), scored one stride ahead on the second half of the trial,
# R² of the section state 1 − Σ‖ẑ − z‖² / Σ‖z − z̄‖²; four sections. Spectral radius of the map.
# Instrument strength: correlation of the section state with itself one stride earlier, as the canonical correlations
# between z_{k−1} and z_k (2-D; the smaller one limits the IV map) and the lag-1 autocorrelation of the leading PC.
#   SLIP     kicked walks (cache/truth/slip_walks; κ = 0, open loop), section at VLO, 250 steps, noise 0–1× the
#            stride-to-stride SD (fig3_slip_maps.jl's noise model)
#   running  all strides of a trial, and its first 12 strides (the length of a 15-s walking trial)
#   walking  the 299 15-s treadmill trials of the companion paper (fig5_gait_data.jl), same stride cut, deviation from
#            the trial's mean stride
# Usage: julia -t 8 --project=. scripts/iv_running.jl     → cache/maus/iv_running.jls
include("maus_learning_all.jl")
using NPZ, CSV, DataFrames

const SIZES_IV = [8, 16, 32, 64, 150]
const SECT = (1, 26, 51, 76)
const SECT_W = (1, 26, 51, 76)

pcs2(X) = (μ = mean(X, dims = 1); V = svd(X .- μ).V[:, 1:2]; (μ, V))
function lsmap(Za, Zb)                                  # Zb ≈ A Za + c (rows = strides)
    B = hcat(Za, ones(size(Za, 1))) \ Zb; (B[1:2, :]', B[3, :])
end
function ivmap(Zm, Z0, Zp)                              # instrument Zm (k−1), regressor Z0 (k), target Zp (k+1)
    W = hcat(Zm, ones(size(Zm, 1))); X = hcat(Z0, ones(size(Z0, 1)))
    B = (W' * X) \ (W' * Zp); (B[1:2, :]', B[3, :])
end
specrad(A) = maximum(abs, eigvals(A))
"Canonical correlations between the 2-D states one stride apart (rows = strides), and the lag-1 autocorrelation of PC 1."
function instrument(Z)
    a = Z[1:end-1, :] .- mean(Z[1:end-1, :], dims = 1); b = Z[2:end, :] .- mean(Z[2:end, :], dims = 1)
    isq(C) = (F = eigen(Symmetric(C)); F.vectors * Diagonal(1 ./ sqrt.(max.(F.values, eps()))) * F.vectors')
    cc = svdvals(isq(a' * a) * (a' * b) * isq(b' * b))
    (; cc_min = minimum(cc), cc_max = maximum(cc), lag1 = cor(a[:, 1], b[:, 1]))
end

"Running trial: LS and IV section maps by training size and section, and instrument strength."
function trial_iv(S)
    _, _, D = split_baseline(S, :ma); n = size(S, 1); nh = n ÷ 2
    Ds = permutedims(reshape(D, NPH, n, :), (2, 1, 3))              # stride × phase × channel
    out = Dict{Any, Any}()
    for φ in SECT
        X = Ds[:, φ, :]
        μ, V = pcs2(X); Zall = (X .- μ) * V
        out[(:inst, φ, :all)] = instrument(Zall)
        μ12, V12 = pcs2(X[1:12, :]); out[(:inst, φ, 12)] = instrument((X[1:12, :] .- μ12) * V12)
        for ntr in SIZES_IV
            ntr > nh && continue
            μt, Vt = pcs2(X[1:ntr, :]); Z = (X .- μt) * Vt
            Zt = Z[nh:n, :]; T = Zt[2:end, :]; den = sum(abs2, T .- mean(T, dims = 1))
            for (m, (A, c)) in ((:ls, lsmap(Z[1:ntr-1, :], Z[2:ntr, :])), (:iv, ivmap(Z[1:ntr-2, :], Z[2:ntr-1, :], Z[3:ntr, :])))
                P = Zt[1:end-1, :] * A' .+ c'
                out[(m, φ, ntr)] = (; ρ = specrad(A), r2 = 1 - sum(abs2, P .- T) / den)
            end
        end
    end
    out
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    # running
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    SS = [pair_strides(t.X) for t in A.trials]; SS = SS[qc_run.(SS) .< 5]
    RUN = tmap(trial_iv, SS)
    # SLIP (open loop), section at VLO, 250 steps
    WALKDIR = joinpath(ROOT, "cache", "truth", "slip_walks")
    slip = Dict{Float64, Vector{Any}}()
    for f in sort(readdir(WALKDIR))
        w = deserialize(joinpath(WALKDIR, f)); (w.fallen || w.kappa != 0.0) && continue
        for r in (0.0, 0.25, 0.5, 1.0)
            S = w.S[1:250, :, :]; sd = vec(mean(std(S, dims = 1), dims = 2))
            r > 0 && (S = S .+ r .* reshape(sd, 1, 1, :) .* randn(MersenneTwister(hash((w.seed, 250, r))), size(S)))
            μ, V = pcs2(S[:, 1, :]); push!(get!(slip, r, Any[]), instrument((S[:, 1, :] .- μ) * V))
        end
    end
    # walking (15-s trials)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    walk = tmap(1:size(data, 1)) do i
        S = try pair_strides(data[i, :, :]) catch; nothing end
        (S === nothing || size(S, 1) < 6) && return nothing
        D = S .- mean(S, dims = 1)
        inst(φ) = begin X = D[:, φ, :]; μ, V = pcs2(X); merge(instrument((X .- μ) * V), (; n = size(S, 1))) end
        Dict(φ => inst(φ) for φ in SECT_W)
    end
    walk = filter(!isnothing, walk)

    med(v) = isempty(v) ? NaN : median(v)
    println("Instrument strength: smallest / largest canonical correlation of the section state one stride apart; lag-1 autocorrelation of PC 1")
    for r in (0.0, 0.25, 0.5, 1.0)
        v = slip[r]; @printf("  SLIP, noise %.2f×     %.2f / %.2f | %.2f   (%d walks)\n", r, med([x.cc_min for x in v]), med([x.cc_max for x in v]), med([x.lag1 for x in v]), length(v))
    end
    for (lab, k) in (("running, all strides", :all), ("running, 12 strides ", 12))
        v = [r[(:inst, φ, k)] for r in RUN for φ in SECT]
        @printf("  %s  %.2f / %.2f | %.2f   (%d trials × %d sections)\n", lab, med([x.cc_min for x in v]), med([x.cc_max for x in v]), med([x.lag1 for x in v]), length(RUN), length(SECT))
    end
    v = [w[φ] for w in walk for φ in SECT_W]
    @printf("  walking, 15-s trials  %.2f / %.2f | %.2f   (%d trials, median %d strides)\n", med([x.cc_min for x in v]), med([x.cc_max for x in v]), med([x.lag1 for x in v]), length(walk), med([w[1].n for w in walk]))
    println("Running section maps: median spectral radius [IQR] and R² one stride ahead, over trials × sections")
    for m in (:ls, :iv), ntr in SIZES_IV
        v = [r[(m, φ, ntr)] for r in RUN for φ in SECT if haskey(r, (m, φ, ntr))]
        ρ = [x.ρ for x in v]; r2 = [x.r2 for x in v]
        @printf("  %-2s %3d strides: ρ %.2f [%.2f–%.2f], R² %.2f [%.2f–%.2f]\n", m, ntr, median(ρ), quantile(ρ, 0.25), quantile(ρ, 0.75), median(r2), quantile(r2, 0.25), quantile(r2, 0.75))
    end
    serialize(joinpath(ROOT, "cache", "maus", "iv_running.jls"), (; RUN, slip, walk, SIZES = SIZES_IV, SECT))
end
