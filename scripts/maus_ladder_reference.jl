#!/usr/bin/env julia
# Reference forecasts for maus_ladder.jl's whole-signal representations, and how good whole-signal forecasts are on the
# raw signal. The ladder scores every forecast as R² of the deviation from the moving-average stride at each sample's
# ACTUAL phase; in real time that zero line uses the true future stride boundaries, which no real-time forecast knows.
#   causal reference   the average of the training strides at the phase extrapolated from the forecast start (rt: at the
#                      previous stride's rate; sn: one sample per sample), scored the same way
#   raw-signal R²      1 − Σ‖ŷ − y‖² / Σ‖y − ȳ‖² of whole-signal Hankel-DMD (window 50, rank 64), and the deviation's
#                      share of the signal's variance
# 150 training strides; horizons HZ. Usage: julia -t 12 --project=. scripts/maus_ladder_reference.jl
#   → cache/maus/ladder_reference.jls
include("maus_ladder.jl")

function trial_reference(X)
    R = rung_data(X); nh = R.n ÷ 2; out = Dict{Symbol, Any}()
    Sbar = reduce(vcat, [mean(reduce(vcat, [R.Y[(s-1)*NPH+k:(s-1)*NPH+k, :] for s in 1:150]), dims = 1) for k in 1:NPH])
    for rung in (:rt, :sn)
        Yv, Dv, bidx = rung == :rt ? (R.Yrt, R.Drt, R.bidx_rt) : (R.Y, R.D, R.bidx_sn)
        rows = 1:bidx[151]-1; ts = (bidx[nh+1] + HISTW):STEP:(size(Yv, 1) - HZ[end])
        Am, U = opfit(Yv[rows, :]; d = 50, r = 64); μ = mean(Yv[rows, :], dims = 1)
        Ed = zeros(length(HZ)); Er = zeros(length(HZ)); Dn = zeros(length(HZ)); Sn = zeros(length(HZ))
        for t in ts
            P = roll(Am, U, μ, Yv, t, 50, HZ[end])
            for (j, h) in enumerate(HZ)
                ph = rung == :rt ? mod(floor(Int, R.ph_rt[t] + NPH * R.rate_rt[t] * h), NPH) : mod(t - 1 + h, NPH)
                Ed[j] += sum(abs2, P[h, :] .- Yv[t+h, :]); Er[j] += sum(abs2, Sbar[ph+1, :] .- Yv[t+h, :])
                Dn[j] += sum(abs2, Dv[t+h, :]); Sn[j] += sum(abs2, Yv[t+h, :] .- vec(μ))
            end
        end
        out[rung] = (; ref_causal = 1 .- Er ./ Dn, dmd_dev = 1 .- Ed ./ Dn, dmd_signal = 1 .- Ed ./ Sn, devfrac = Dn ./ Sn)
    end
    out
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    keep = [qc_run(pair_strides(t.X)) < 5 for t in A.trials]
    Rs = tmap(t -> trial_reference(t.X), A.trials[keep])
    for rung in (:rt, :sn)
        v(f) = median([f(r[rung]) for r in Rs if r !== nothing])
        @printf("%s ⅛ / ¼ stride: causal reference R² of the deviation %.2f / %.2f | whole-signal DMD: deviation %.2f / %.2f, raw signal %.3f / %.3f | deviation %.1f%% of the signal's variance\n",
                rung, v(r -> r.ref_causal[2]), v(r -> r.ref_causal[4]), v(r -> r.dmd_dev[2]), v(r -> r.dmd_dev[4]),
                v(r -> r.dmd_signal[2]), v(r -> r.dmd_signal[4]), 100v(r -> r.devfrac[2]))
    end
    serialize(joinpath(ROOT, "cache", "maus", "ladder_reference.jls"), (; R = Rs, HZ))
end
