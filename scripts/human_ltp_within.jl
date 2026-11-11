#!/usr/bin/env julia
# Individual or absent? (follow-up to human_ltp_pooled.jl / human_ltp_align.jl). In stroke the phase-dependent operator
# learned from other people transfers poorly. Here each held-out trial is predicted by an operator learned from the SAME
# person's other trials within ×/÷ 1.5 of its speed (stroke survivors' six trials span a narrow range), pooled to at most
# NPOOL strides (whole trials in random order), fitted only if ≥ MINPOOL strides are available; same setting S, with and
# without phase. Compared with the between-person (leave-one-person-out) operator for the same held-out trial: other
# people of the same group, trials within ×/÷ 1.5 of its speed, NPOOL strides.
#
# Usage: julia --project=. -t 16 scripts/human_ltp_within.jl      (writes cache/fig5/human_ltp_within.jls)

include("human_ltp_align.jl")

const MINPOOL = 30
const RATIO = 1.5

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    Sh = [strides(data[i, :, :]) for i in axes(data, 1)]; sub = String.(meta.subject); grp = String.(meta.lf_or_hf); spd = Float64.(meta.speed)
    near(i, j) = max(spd[i], spd[j]) / min(spd[i], spd[j]) <= RATIO
    jobs = [i for i in eachindex(Sh) if size(Sh[i], 1) >= 3]
    R = tmap(jobs) do i
        own = [j for j in jobs if j != i && sub[j] == sub[i] && near(i, j)]
        oth = [j for j in jobs if sub[j] != sub[i] && grp[j] == grp[i] && near(i, j)]
        nown = sum(size(Sh[j], 1) for j in own; init = 0); noth = sum(size(Sh[j], 1) for j in oth; init = 0)
        (nown < MINPOOL || noth < NPOOL) && return nothing
        D = devs(Sh[i]); n = min(NPOOL, nown)
        fitpair(pool, seed) = begin
            Ds = pool_draw_f(MersenneTwister(seed), [1], Dict(1 => [Sh[j] for j in pool]), n, devs)
            M = fit_ltp_multi(Ds, S...); M0 = fit_ltp_multi(Ds, S[1], S[2], 0, S[4])
            (lopo_skill(M, D), lopo_skill(M0, D))
        end
        (; i, subject = sub[i], group = grp[i], speed = spd[i], n, within = fitpair(own, hash((i, :own))), between = fitpair(oth, hash((i, :oth))))
    end
    R = filter(!isnothing, R)
    println("R² 12.5% of a stride on a held-out trial, median over people (person means first): with / without phase")
    for g in ("AB", "HF", "LF")
        x = [r for r in R if r.group == g]; ps = unique(getfield.(x, :subject))
        pm(f) = [mean(f(r) for r in x if r.subject == p) for p in ps]
        w1, w0, b1, b0 = pm(r -> r.within[1]), pm(r -> r.within[2]), pm(r -> r.between[1]), pm(r -> r.between[2])
        rng = MersenneTwister(3); bs = [median(rand(rng, w1 .- b1, length(ps))) for _ in 1:4000]
        @printf("  %s (%2d people, %3d trials, pool %d strides median): own other trials %.3f / %.3f | other people %.3f / %.3f | own − others (with phase) %+.3f [%+.3f, %+.3f]\n",
                g, length(ps), length(x), round(Int, median(getfield.(x, :n))), median(w1), median(w0), median(b1), median(b0),
                median(w1 .- b1), quantile(bs, 0.025), quantile(bs, 0.975))
    end
    serialize(joinpath(ROOT, "cache", "fig5", "human_ltp_within.jls"), (; R, S, NPOOL, MINPOOL, RATIO))
    println("wrote cache/fig5/human_ltp_within.jls")
end
