#!/usr/bin/env julia
# Does the pooled walking result survive a phase that pins no channel? Cutting strides at the limb-1 hip's own Hilbert
# phase (strides(), used by human_ltp_pooled.jl / human_ltp_within.jl) forces that hip's deviation through zero at every
# cut, a phase-locked pattern a phase-dependent operator can learn without any dynamics (human_ltp_lowdim.jl: deviation
# SD 0.000 at the cut vs ~0.13 mid-stride). Here both analyses are rerun with strides cut by the leading oscillatory
# pair of all six angles (pair_phase in human_ltp_lowdim.jl; deviation SD flat over the stride), and with the hip cut on
# the same seeds for comparison:
#   within/between  each held-out trial predicted by an operator from the same person's other trials (≤ NPOOL strides,
#                   speeds within ×/÷ 1.5, ≥ 30 strides) or from other people of the group at such speeds (human_ltp_within)
#   LOPO            group × speed-band operator from NPOOL strides of the other people, scored on the held-out person
#                   (human_ltp_pooled)
# Setting S = (8, 24, 4, 1e-2), with and without phase harmonics; R² 12.5% of a stride ahead.
# Usage: julia --project=. -t 10 scripts/human_ltp_pairphase.jl      (writes cache/fig5/human_ltp_pairphase.jls)

include("human_ltp_lowdim.jl")

function pool_draw_g(rng, people, trials_of, n)
    out = Matrix{Float64}[]; left = n
    while left > 0
        tr = trials_of[rand(rng, people)]; St = tr[rand(rng, eachindex(tr))]; k = min(size(St, 1), left)
        push!(out, devs(St)[1:k*NPH, :]); left -= k
    end
    out
end
fit2(Ds) = (fit_ltp_multi(Ds, S...), fit_ltp_multi(Ds, S[1], S[2], 0, S[4]))

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    sub = String.(meta.subject); grp = String.(meta.lf_or_hf); spd = Float64.(meta.speed)
    out = Dict{Symbol, Any}()
    for how in (:hip, :pair)
        Sh = [cut(data[i, :, :], how)[1] for i in axes(data, 1)]
        near(i, j) = max(spd[i], spd[j]) / min(spd[i], spd[j]) <= 1.5
        jobs = [i for i in eachindex(Sh) if size(Sh[i], 1) >= 3]
        W = filter(!isnothing, tmap(jobs) do i
            own = [j for j in jobs if j != i && sub[j] == sub[i] && near(i, j)]
            oth = [j for j in jobs if sub[j] != sub[i] && grp[j] == grp[i] && near(i, j)]
            nown = sum(size(Sh[j], 1) for j in own; init = 0); noth = sum(size(Sh[j], 1) for j in oth; init = 0)
            (nown < 30 || noth < NPOOL) && return nothing
            D = devs(Sh[i]); n = min(NPOOL, nown)
            sc(pool, seed) = (M = fit2(pool_draw_g(MersenneTwister(seed), [1], Dict(1 => [Sh[j] for j in pool]), n)); (lopo_skill(M[1], D), lopo_skill(M[2], D)))
            (; i, subject = sub[i], group = grp[i], within = sc(own, hash((i, :own))), between = sc(oth, hash((i, :oth))))
        end)
        println("── $how cut ── own other trials vs other people (R² 12.5%, with / without phase; medians of person means)")
        for g in ("AB", "HF", "LF")
            x = [r for r in W if r.group == g]; ps = unique(getfield.(x, :subject))
            pm(f) = [mean(f(r) for r in x if r.subject == p) for p in ps]
            w1, w0, b1, b0 = pm(r -> r.within[1]), pm(r -> r.within[2]), pm(r -> r.between[1]), pm(r -> r.between[2])
            rng = MersenneTwister(3); bs = [median(rand(rng, w1 .- b1, length(ps))) for _ in 1:4000]
            gp = [median(rand(rng, w1 .- w0, length(ps))) for _ in 1:4000]
            @printf("  %s (%2d people): own %.3f / %.3f (phase gain %+.3f [%+.3f, %+.3f]) | other people %.3f / %.3f | own − other %+.3f [%+.3f, %+.3f]\n",
                    g, length(ps), median(w1), median(w0), median(w1 .- w0), quantile(gp, 0.025), quantile(gp, 0.975), median(b1), median(b0),
                    median(w1 .- b1), quantile(bs, 0.025), quantile(bs, 0.975))
        end
        band(v) = findfirst(b -> b[1] <= v < b[2], BANDS); lopo = Dict{Tuple{String, Int}, Any}()
        println("  leave one person out, group × speed band: R² with / without phase")
        for g in ("AB", "HF", "LF"), bi in eachindex(BANDS)
            I = [i for i in eachindex(Sh) if grp[i] == g && band(spd[i]) == bi && size(Sh[i], 1) >= 3]
            people = unique(sub[I]); trials_of = Dict(p => [Sh[i] for i in I if sub[i] == p] for p in people)
            (length(people) < 4 || sum(size(Sh[i], 1) for i in I; init = 0) < NPOOL) && continue
            v = filter(!isnothing, tmap(collect(eachindex(people))) do k
                others = people[[j for j in eachindex(people) if j != k]]
                M = fit2(pool_draw_g(MersenneTwister(hash((g, bi, k))), others, trials_of, NPOOL))
                held = [devs(St) for St in trials_of[people[k]] if size(St, 1) >= 2]
                (mean(lopo_skill(M[1], D) for D in held), mean(lopo_skill(M[2], D) for D in held))
            end)
            lopo[(g, bi)] = v
            @printf("    %s %3d–%3d cm/s (%2d people): %.3f / %.3f\n", g, BANDS[bi]..., length(people), median(first.(v)), median(last.(v)))
        end
        out[how] = (; W, lopo); flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "fig5", "human_ltp_pairphase.jls"), (; out, S, NPOOL, BANDS))
    println("wrote cache/fig5/human_ltp_pairphase.jls")
end
