#!/usr/bin/env julia
# What structure does a Fourier-series-in-phase (LTP) operator find in short walking trials, and does it differ by
# group (able-bodied, higher- and lower-functioning stroke) or speed? (maus_floquet_models.jl for the operator.)
#
# Data: the companion superset (../Gait_NonNormality_Paper3), 299 treadmill trials × 15 s, 6 sagittal angles (limb 1 =
# paretic in stroke). Strides at the limb-1 hip Hilbert phase (fig5_gait_data.jl strides(); phase 0 = that hip's
# descending zero crossing), 100 samples each, channels standardised, deviation from the trial's mean stride. Every
# trial is cut to its first NSTR strides (shorter trials dropped) so the amount of data is the same at every speed.
#
# One setting for all trials, chosen without groups: each trial fitted on its first half of strides and scored on the
# second (R² of the deviation 6–50% of a stride ahead, every 2nd sample), median over all trials, over the grid
# GD × GR × GH × GL. With it, per trial (fit on all NSTR strides):
#   mult1, mult2  the largest moduli of the one-stride monodromy Π A(θ_k) (Floquet multipliers per stride)
#   growth(φ)     largest singular value of Π_{k=φ}^{φ+24} A(θ_k), in the output angles (the latest delay block of the
#                 POD basis): how much a deviation present at phase φ can grow over the next quarter stride; also the
#                 same into limb-1 and into limb-2 angles only (paretic / non-paretic in stroke)
#   skill, phase  held-out R² at 12.5% of a stride with phase harmonics, and its gain over the same operator without
#                 (Hh = 0): how much the dynamics depend on the phase of the stride
# Reliability first: the same pipeline on the long running trials (maus_extract.jl) cut into NSTR-stride chunks:
# chunk-to-chunk agreement within a trial against the spread across trials and people.
#
# Usage: julia --project=. -t 16 scripts/human_ltp.jl      (writes cache/fig5/human_ltp.jls)

include("maus_floquet_models.jl")

const NSTR = 8
const GD, GR, GH, GL = [8, 16], [6, 12, 24], [0, 1, 2, 4], [1e-3, 1e-2, 1e-1]
const HZH = [6, 12, 25, 50]
const QS = 25

function score_h(pred, D, ts)
    E = zeros(length(HZH)); R = zeros(length(HZH))
    for t in ts
        P = pred(t)
        for (k, h) in enumerate(HZH); E[k] += sum(abs2, P[h, :] .- D[t+h, :]); R[k] += sum(abs2, D[t+h, :]); end
    end
    1 .- E ./ R
end
"Held-out skill (horizons HZH) for every grid setting: fit on the first half of the strides, score on the second."
function grid_skill(D, n)
    nh = n ÷ 2; tr = 1:nh*NPH; te = nh*NPH+1:n*NPH; ts = (first(te) + 20):2:(last(te) - maximum(HZH))
    Dict((d, r, h, λ) => score_h(ltp_predictor(D, M), D, ts) for d in GD, r in GR, h in GH
         for (λ, M) in zip(GL, fit_ltp(D, tr, d, r, h, GL)))
end
"Structure of the operator fitted on all strides with setting s."
function structure(D, n, s)
    d, r, h, λ = s; M = fit_ltp(D, 1:n*NPH, d, r, h, [λ])[1]; p = size(D, 2)
    m = sort(abs.(eigvals(monodromy(M))), rev = true)
    C = M.U[end-p+1:end, :]; L1, L2 = C[1:p÷2, :], C[p÷2+1:end, :]
    g = zeros(NPH, 3); Cp = pinv(C)                 # a unit deviation in the output angles, lifted into the state
    for φ in 0:NPH-1
        P = foldl((X, k) -> M.Aph[mod(k, NPH)+1] * X, φ:φ+QS-1; init = Matrix(1.0I, size(M.Aph[1])...))
        g[φ+1, :] = [opnorm(X * P * Cp) for X in (C, L1, L2)]
    end
    (; mult1 = m[1], mult2 = length(m) > 1 ? m[2] : NaN, growth = g)
end
prep(S) = (Sn = S[1:NSTR, :, :]; split_baseline(Sn, :mean)[3])

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    Sh = [strides(data[i, :, :]) for i in axes(data, 1)]; ok = findall(S -> size(S, 1) >= NSTR, Sh)
    @printf("human: %d of %d trials have ≥ %d strides (dropped by group: %s)\n", length(ok), length(Sh), NSTR,
            join(["$g $(count(i -> !(i in ok) && meta.lf_or_hf[i] == g, eachindex(Sh)))" for g in ("AB", "HF", "LF")], ", "))
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls")); T = A.trials[qc_run.([strides(t.X) for t in A.trials]) .< 5]
    chunks = [(t.subj, t.rep, k, strides(t.X)[(k-1)*NSTR+1:k*NSTR, :, :]) for t in T for k in 1:10]

    # 1. setting, chosen on all human trials (no groups)
    G = tmap(i -> grid_skill(prep(Sh[i]), NSTR), ok); G = filter(!isnothing, G)
    keys_ = collect(keys(G[1])); med = [median(g[k][2] for g in G) for k in keys_]
    s = keys_[argmax(med)]; s0 = keys_[argmax([k[3] == 0 ? m : -Inf for (k, m) in zip(keys_, med)])]
    @printf("setting (d, r, Hh, ridge) = %s: median R² at 12.5%% %.3f; best without phase %s: %.3f\n", string(s), maximum(med), string(s0),
            median(g[s0][2] for g in G))

    # 2. per-trial structure and skill, humans and running chunks
    function trial(S)
        D = prep(S); gs = grid_skill(D, NSTR); st = structure(D, NSTR, s)
        (; st..., skill = gs[s][2], phase = gs[s][2] - gs[s0][2], skill25 = gs[s][3])
    end
    H = tmap(i -> trial(Sh[i]), ok); Rc = tmap(c -> trial(c[4]), chunks)
    serialize(joinpath(ROOT, "cache", "fig5", "human_ltp.jls"),
              (; setting = s, setting0 = s0, human = H, idx = ok, subject = String.(meta.subject[ok]), group = String.(meta.lf_or_hf[ok]),
                 speed = Float64.(meta.speed[ok]), running = Rc, chunks = [(c[1], c[2], c[3]) for c in chunks],
                 mean_cycles = [mean(Sh[i][1:NSTR, :, :], dims = 1)[1, :, :] for i in ok]))
    println("wrote cache/fig5/human_ltp.jls")
end
