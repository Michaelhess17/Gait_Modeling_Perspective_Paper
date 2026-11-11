#!/usr/bin/env julia
# Control for human_ltp_pooled.jl: does the poor transfer of the phase-dependent group operator in stroke come from the
# phase reference? Phase 0 is the limb-1 hip angle's descending zero crossing (limb 1 = paretic in stroke); if that
# reference falls at different gait events in different people, a phase-locked operator cannot transfer even when the
# dynamics are shared. Here every trial's phase is re-referenced to its own limb-1 hip flexion peak (≈ initial contact,
# from the trial's mean stride): the deviation series is shifted by that many samples so strides start at the peak.
# Leave-one-person-out skill (R² 12.5% of a stride, with and without phase) is recomputed per group × speed band with
# the same setting and pool size, original and re-referenced phase; per-person values are kept for intervals.
# Also: spread across people of the limb-1 and limb-2 hip-peak phases (in the original reference), by group.
#
# Usage: julia --project=. -t 16 scripts/human_ltp_align.jl      (writes cache/fig5/human_ltp_align.jls)

include("human_ltp_pooled.jl")

"Deviation series of a trial with phase re-referenced to its limb-1 hip flexion peak (whole strides only)."
function devs_aligned(St)
    D = devs(St); s = argmax(vec(mean(St[:, :, 1], dims = 1))) - 1      # peak phase index (0-based)
    n = size(St, 1); n < 2 && return nothing
    D[s+1:s+(n-1)*NPH, :]
end
function pool_draw_f(rng, people, trials_of, n, f)
    out = Matrix{Float64}[]; left = n
    while left > 0
        tr = trials_of[rand(rng, people)]; St = tr[rand(rng, eachindex(tr))]; D = f(St); D === nothing && continue
        k = min(size(D, 1) ÷ NPH, left); k < 1 && continue
        push!(out, D[1:k*NPH, :]); left -= k
    end
    out
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    Sh = [strides(data[i, :, :]) for i in axes(data, 1)]; sub = String.(meta.subject); grp = String.(meta.lf_or_hf); spd = Float64.(meta.speed)
    peak(S_, c) = argmax(vec(mean(S_[:, :, c], dims = 1))) - 1
    println("Hip flexion peak phase (% stride, original reference), person medians: median and SD across people")
    for g in ("AB", "HF", "LF")
        for (c, nm) in ((1, "limb 1"), (4, "limb 2"))
            pp = [median(peak(Sh[i], c) for i in eachindex(Sh) if sub[i] == s && size(Sh[i], 1) >= 2) for s in unique(sub[grp .== g])]
            @printf("  %s %s: median %.0f, SD %.1f, range %d–%d\n", g, nm, median(pp), std(pp), extrema(round.(Int, pp))...)
        end
    end
    band(v) = findfirst(b -> b[1] <= v < b[2], BANDS)
    res = Dict{Tuple{String, Int}, Any}()
    for g in ("AB", "HF", "LF"), bi in eachindex(BANDS)
        I = [i for i in eachindex(Sh) if grp[i] == g && band(spd[i]) == bi && size(Sh[i], 1) >= 3]
        people = unique(sub[I]); trials_of = Dict(p => [Sh[i] for i in I if sub[i] == p] for p in people)
        (length(people) < 4 || sum(size(Sh[i], 1) for i in I; init = 0) < NPOOL) && continue
        out = Dict{Symbol, Any}()
        for (lab, f) in ((:original, St -> devs(St)), (:aligned, devs_aligned))
            lopo = tmap(collect(eachindex(people))) do k
                others = people[[j for j in eachindex(people) if j != k]]
                Ds = pool_draw_f(MersenneTwister(hash((g, bi, k))), others, trials_of, NPOOL, f)
                M = fit_ltp_multi(Ds, S...); M0 = fit_ltp_multi(Ds, S[1], S[2], 0, S[4])
                held = filter(!isnothing, [f(St) for St in trials_of[people[k]] if size(St, 1) >= 2])
                (mean(lopo_skill(M, D) for D in held), mean(lopo_skill(M0, D) for D in held))
            end
            out[lab] = filter(!isnothing, lopo)
        end
        res[(g, bi)] = (; people, out)
        @printf("%s %3d–%3d cm/s (%2d people): LOPO R² with / without phase — original %.3f / %.3f | aligned to limb-1 hip peak %.3f / %.3f\n",
                g, BANDS[bi]..., length(people), median(first.(out[:original])), median(last.(out[:original])),
                median(first.(out[:aligned])), median(last.(out[:aligned])))
        flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "fig5", "human_ltp_align.jls"), (; res, BANDS))
    println("wrote cache/fig5/human_ltp_align.jls")
end
