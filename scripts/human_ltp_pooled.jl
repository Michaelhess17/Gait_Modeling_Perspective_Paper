#!/usr/bin/env julia
# Group-level Fourier-in-phase (LTP) operators for walking (follow-up to human_ltp.jl, whose per-trial operators from
# 8 strides proved unreliable: on running chunks of 8 strides only 7–14% of each metric's variance lies between trials).
#
# 1. How many strides does the operator need? Running trials (maus_extract.jl), one fixed setting S, chunks of NS
#    strides (≤ 4 per trial): leading multiplier and quarter-stride growth profile against the same trial's full-length
#    estimate (all strides), and chunk-to-chunk profile agreement.
# 2. Walking (companion superset): one operator per group × speed band, pooling the deviations (each trial from its own
#    mean stride, channels standardised per trial) of the cell's people; transitions only within a trial. Each fit
#    sees exactly NPOOL strides: people drawn with replacement, then whole trials in random order (the last cut to
#    whole strides); B draws per cell. Per draw: Floquet multipliers (monodromy), log quarter-stride growth by phase
#    (into all angles, limb-1 angles, limb-2 angles; limb 1 = paretic in stroke).
#    Forecasting, leave one person out: the cell's operator fitted on NPOOL strides of the other people, scored on the
#    held-out person's trials (R² 12.5% of a stride ahead), with S and with the same operator without phase (Hh = 0).
#
# Usage: julia --project=. -t 16 scripts/human_ltp_pooled.jl      (writes cache/fig5/human_ltp_pooled.jls)

include("human_ltp.jl")

const S = (8, 24, 4, 1e-2)                     # d, rank, harmonics, ridge
const NSC = [8, 16, 32, 64, 128]
const BANDS = [(20, 45), (45, 70), (70, 100), (100, 140), (140, 200)]
const NPOOL = 60
const B = 40

"LTP fitted to several trials' deviations at once (each starting at a stride origin); transitions within trials only."
function fit_ltp_multi(Ds, d, r, Hh, λ)
    μ = mean(reduce(vcat, Ds), dims = 1); Hs = [hankel(D .- μ, d) for D in Ds]
    U = svd(reduce(hcat, Hs)).U[:, 1:r]; Φs = Matrix{Float64}[]; Ys = Matrix{Float64}[]
    for H in Hs
        Z = U' * H; nc = size(Z, 2)
        push!(Φs, reduce(hcat, [kron(fphase(mod(c + d - 2, NPH), Hh), Z[:, c]) for c in 1:nc-1])); push!(Ys, Z[:, 2:end])
    end
    Φ = reduce(hcat, Φs); G = Φ * Φ'; W = reduce(hcat, Ys) * Φ' / (G + λ * mean(diag(G)) * I); nf = 2Hh + 1
    (; Aph = [sum(fphase(k, Hh)[i] .* W[:, (i-1)*r+1:i*r] for i in 1:nf) for k in 0:NPH-1], U, μ, d)
end
function op_structure(M, p)
    m = sort(abs.(eigvals(monodromy(M))), rev = true); C = M.U[end-p+1:end, :]; Cp = pinv(C)
    g = zeros(NPH, 3)
    for φ in 0:NPH-1
        P = foldl((X, k) -> M.Aph[mod(k, NPH)+1] * X, φ:φ+QS-1; init = Matrix(1.0I, size(M.Aph[1])...))
        g[φ+1, :] = [opnorm(X * P * Cp) for X in (C, C[1:p÷2, :], C[p÷2+1:end, :])]
    end
    (; mult = m[1:3], lgrowth = log.(g))
end
devs(S_) = split_baseline(S_, :mean)[3]
"Whole trials of people drawn with replacement until exactly n strides; returns deviation matrices."
function pool_draw(rng, people, trials_of, n)
    out = Matrix{Float64}[]; left = n
    while left > 0
        tr = trials_of[rand(rng, people)]; St = tr[rand(rng, eachindex(tr))]; k = min(size(St, 1), left)   # the last piece may be 1–2 strides
        push!(out, devs(St)[1:k*NPH, :]); left -= k                # deviation from the whole trial's mean stride
    end
    out
end
function lopo_skill(Mfit, D)
    ts = 20:2:size(D, 1)-maximum(HZH); score_h(ltp_predictor(D, Mfit), D, ts)[2]
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    # 1. calibration on running
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls")); T = A.trials[qc_run.([strides(t.X) for t in A.trials]) .< 5]
    Sr = [strides(t.X) for t in T]
    ref = tmap(i -> op_structure(fit_ltp_multi([devs(Sr[i])], S...), 6), collect(eachindex(Sr)))
    println("Running, setting $(S): full-trial leading multiplier ", mi([r.mult[1] for r in ref]))
    cal = Dict{Int, Any}()
    for ns in NSC
        jobs = [(i, k) for i in eachindex(Sr) for k in 1:min(4, size(Sr[i], 1) ÷ ns)]
        R = tmap(((i, k),) -> op_structure(fit_ltp_multi([devs(Sr[i][(k-1)*ns+1:k*ns, :, :])], S...), 6), jobs)
        dm = [R[j].mult[1] - ref[jobs[j][1]].mult[1] for j in eachindex(jobs)]
        cr = [cor(vec(R[j].lgrowth[:, 1]), vec(ref[jobs[j][1]].lgrowth[:, 1])) for j in eachindex(jobs)]
        cc = [cor(R[a].lgrowth[:, 1], R[b].lgrowth[:, 1]) for a in eachindex(jobs) for b in a+1:length(jobs) if jobs[a][1] == jobs[b][1]]
        @printf("  %3d strides: multiplier − full-trial %s | growth profile vs full trial r %s | chunk vs chunk r %s\n", ns, mi(dm), mi(cr), isempty(cc) ? "—" : mi(cc))
        cal[ns] = (; dm, cr, cc)
    end

    # 2. pooled walking operators per group × speed band
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    Sh = [strides(data[i, :, :]) for i in axes(data, 1)]; sub = String.(meta.subject); grp = String.(meta.lf_or_hf); spd = Float64.(meta.speed)
    band(v) = findfirst(b -> b[1] <= v < b[2], BANDS)
    cells = [(g, bi) for g in ("AB", "HF", "LF") for bi in eachindex(BANDS)]
    res = Dict{Tuple{String, Int}, Any}()
    for (g, bi) in cells
        I = [i for i in eachindex(Sh) if grp[i] == g && band(spd[i]) == bi && size(Sh[i], 1) >= 3]
        people = unique(sub[I]); trials_of = Dict(p => [Sh[i] for i in I if sub[i] == p] for p in people)
        navail = sum(size(Sh[i], 1) for i in I; init = 0)
        if length(people) < 4 || navail < NPOOL
            @printf("%s %3d–%3d cm/s: %d people, %d strides — skipped\n", g, BANDS[bi]..., length(people), navail); continue
        end
        draws = tmap(b -> op_structure(fit_ltp_multi(pool_draw(MersenneTwister(hash((g, bi, b))), people, trials_of, NPOOL), S...), 6), collect(1:B))
        lopo = tmap(collect(eachindex(people))) do k
            others = people[[j for j in eachindex(people) if j != k]]
            Ds = pool_draw(MersenneTwister(hash((g, bi, k))), others, trials_of, NPOOL)
            M = fit_ltp_multi(Ds, S...); M0 = fit_ltp_multi(Ds, S[1], S[2], 0, S[4])
            held = [devs(St) for St in trials_of[people[k]] if size(St, 1) >= 2]
            (mean(lopo_skill(M, D) for D in held), mean(lopo_skill(M0, D) for D in held))
        end
        draws = filter(!isnothing, draws); lopo = filter(!isnothing, lopo)
        res[(g, bi)] = (; people, ntrials = length(I), navail, draws, lopo, speed = median(spd[I]))
        @printf("%s %3d–%3d cm/s: %2d people, %3d trials, %4d strides | multiplier %s | LOPO R² 12.5%% %.3f, without phase %.3f\n", g, BANDS[bi]...,
                length(people), length(I), navail, mi([d.mult[1] for d in draws]), median(first.(lopo)), median(last.(lopo)))
        flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "fig5", "human_ltp_pooled.jls"), (; S, cal, ref, res, BANDS, NPOOL))
    println("wrote cache/fig5/human_ltp_pooled.jls")
end
