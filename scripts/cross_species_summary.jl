#!/usr/bin/env julia
# Speed effects on the operator metrics across five datasets, in the rodent decks' terms (33_speed_consistency.py):
# the share of units in which a metric is lower at the unit's fastest level than at its slowest (mean over draws per
# level), with a bootstrap interval over units, and the median within-unit relative change.
#
#   lone pups       pup × age band, LOCOMOTION (QC) bouts, slower / faster half, 1000 transitions, 12 limb angles
#                   ../Rodent_DMD_Analysis/results/matched_size/fits_n1000_speedsplit[_cycle].csv (rank 240)
#   Cntnap2/L7-Tsc1 mouse, slower / faster half, 1000 transitions, paw centroid geometry
#                   ../Rodent_DMD_Analysis/results/original_cohort/fits_n1000_speedsplit[_cycle].csv
#   wheel           mouse, speed terciles (1 vs 3), 1000 transitions, 12 joint angles
#                   ../Rodent_DMD_Analysis/results/wheel/fits_speed_n1000[_stride].csv
#   human walking   person, slowest vs fastest treadmill trial (cross_species_fits.jl human_real / human_stride);
#                   also the within-person Spearman correlation with speed over all of a person's trials
#   fly walking     fly, slower / faster half of its 2-s walking segments, real time only (fly_speed)
# Stride time = each stride resampled to 25 samples (none for flies: their tracks have no stride rhythm).
# Also: human group contrasts at matched speed (AB / HF / LF), and fly strain differences beyond speed (fly_all).
#
# Usage: julia --project=. scripts/cross_species_summary.jl      (writes cache/cross_species/summary.jls)

using CSV, DataFrames, Statistics, Random, Printf, Serialization

const ROOT = dirname(@__DIR__)
const RD = normpath(joinpath(ROOT, "..", "Rodent_DMD_Analysis", "results"))
const CS = joinpath(ROOT, "cache", "cross_species")
const METRICS = [:participation_ratio, :diagonal_dominance, :henrici, :log_tg_stride, :log_res_stride, :spectral_radius]

"Per unit: (slow, fast) means over draws of each metric, from a table with unit and level columns."
function pairs_by(df, unit, level, slow, fast)
    out = Dict{Symbol, Vector{Tuple{Float64, Float64}}}()
    g = combine(groupby(df, [unit, level]), [m => mean => m for m in METRICS if hasproperty(df, m)]...)
    for m in METRICS
        hasproperty(df, m) || continue
        v = Tuple{Float64, Float64}[]
        for u in unique(g[!, unit])
            a = g[(g[!, unit] .== u) .& (g[!, level] .== slow), m]; b = g[(g[!, unit] .== u) .& (g[!, level] .== fast), m]
            (isempty(a) || isempty(b)) && continue
            (isfinite(a[1]) && isfinite(b[1])) && push!(v, (a[1], b[1]))
        end
        out[m] = v
    end
    out
end

function share_lower(v; rng = MersenneTwister(20261104))
    d = [b < a for (a, b) in v]; s = mean(d)
    bs = [mean(rand(rng, d, length(d))) for _ in 1:4000]
    rel = median([(b - a) / abs(a) for (a, b) in v])
    (; share = s, lo = quantile(bs, 0.025), hi = quantile(bs, 0.975), n = length(d), rel)
end

human_levels(df) = (g = combine(groupby(df, [:subject, :speed]), :group => first => :group);
    Dict(s => (minimum(g.speed[g.subject .== s]), maximum(g.speed[g.subject .== s])) for s in unique(g.subject)))
function human_pairs(df)
    L = human_levels(df); df = copy(df)
    df.level = [r.speed == L[r.subject][1] ? "slow" : r.speed == L[r.subject][2] ? "fast" : "mid" for r in eachrow(df)]
    pairs_by(df, :subject, :level, "slow", "fast")
end
rk(v) = invperm(sortperm(v)) .* 1.0
function human_spearman(df, m)
    g = combine(groupby(df, [:subject, :speed, :group]), m => mean => :y)
    [(s, first(g.group[g.subject .== s]), cor(rk(g.speed[g.subject .== s]), rk(g.y[g.subject .== s])))
     for s in unique(g.subject) if count(g.subject .== s) >= 4]
end

function datasets(time)
    st = time == :stride
    lp = CSV.read(joinpath(RD, "matched_size", st ? "fits_n1000_speedsplit_cycle.csv" : "fits_n1000_speedsplit.csv"), DataFrame)
    lp = lp[startswith.(lp.set, "loco_qc_") .& (lp.rank .== 240), :]
    lp.unit = lp.pup .* "_" .* lp.band; lp.level = last.(split.(lp.set, "_"))
    oc = CSV.read(joinpath(RD, "original_cohort", st ? "fits_n1000_speedsplit_cycle.csv" : "fits_n1000_speedsplit.csv"), DataFrame)
    wh = CSV.read(joinpath(RD, "wheel", st ? "fits_speed_n1000_stride.csv" : "fits_speed_n1000.csv"), DataFrame)
    hu = CSV.read(joinpath(CS, st ? "human_stride.csv" : "human_real.csv"), DataFrame)
    out = [("rat pups (development)", pairs_by(lp, :unit, :level, "slow", "fast")),
           ("mice, Cntnap2/L7-Tsc1", pairs_by(oc, :subject, :half, "slow", "fast")),
           ("mice, wheel", pairs_by(wh, :mouse, :tercile, 1, 3)),
           ("human walking", human_pairs(hu))]
    if !st
        fl = CSV.read(joinpath(CS, "fly_speed.csv"), DataFrame)
        push!(out, ("fly walking (body velocity)", pairs_by(fl, :fly, :level, "slow", "fast")))
    end
    out, hu
end

if abspath(PROGRAM_FILE) == @__FILE__
    S = Dict{Symbol, Any}()
    for time in (:real, :stride)
        D, hu = datasets(time)
        println("\n== $(time) time: share of units lower at the faster speed [95% bootstrap], median relative change ==")
        @printf("%-28s", ""); for m in METRICS; @printf("%-26s", m); end; println()
        tab = []
        for (name, P) in D
            @printf("%-28s", name)
            for m in METRICS
                if haskey(P, m) && length(P[m]) >= 3
                    r = share_lower(P[m]); push!(tab, (; dataset = name, metric = m, r...))
                    @printf("%3.0f%% [%2.0f,%3.0f] %+5.1f%% n%-4d", 100r.share, 100r.lo, 100r.hi, 100r.rel, r.n)
                else
                    @printf("%-26s", "   —")
                end
            end
            println()
        end
        println("  human, within-person Spearman ρ(metric, speed) over all trials (median; share < 0):")
        sp = Dict()
        for m in METRICS
            c = human_spearman(hu, m); sp[m] = c
            @printf("    %-20s all %+.2f (%.0f%% < 0, %d people) | AB %+.2f  HF %+.2f  LF %+.2f\n", m, median(last.(c)), 100mean(last.(c) .< 0), length(c),
                    (median([x[3] for x in c if x[2] == g]) for g in ("AB", "HF", "LF"))...)
        end
        S[time] = (; tab, spearman = sp)
    end

    # human groups at matched speed: trials 40–80 cm/s (where AB, HF and LF overlap), person means
    hu = CSV.read(joinpath(CS, "human_real.csv"), DataFrame)
    w = hu[(hu.speed .>= 40) .& (hu.speed .<= 80), :]
    g = combine(groupby(w, [:subject, :group]), [m => mean => m for m in METRICS]..., :speed => mean => :speed)
    println("\nHuman groups, real time, trials at 40–80 cm/s, person means (median):")
    for G in ("AB", "HF", "LF")
        x = g[g.group .== G, :]
        @printf("  %-3s n %2d  speed %3.0f | %s\n", G, nrow(x), median(x.speed), join([@sprintf("%s %.3f", m, median(x[!, m])) for m in METRICS], "  "))
    end

    # fly strains beyond speed: metric residual after a linear fit on the fly's walking speed; between-strain share
    fa = CSV.read(joinpath(CS, "fly_all.csv"), DataFrame)
    f = combine(groupby(fa, [:fly, :strain]), [m => mean => m for m in METRICS]..., :mean_speed => first => :speed)
    println("\nFly strains (one fit per fly, all walking, 1000 transitions × 5 draws; $(nrow(f)) flies, $(length(unique(f.strain))) strains):")
    for m in METRICS
        X = hcat(ones(nrow(f)), f.speed); β = X \ f[!, m]; r = f[!, m] .- X * β
        sb = sum(count(f.strain .== s) * mean(r[f.strain .== s])^2 for s in unique(f.strain)) / sum(abs2, r .- mean(r))
        perm = [(sh = shuffle(MersenneTwister(k), f.strain); sum(count(sh .== s) * mean(r[sh .== s])^2 for s in unique(sh)) / sum(abs2, r .- mean(r))) for k in 1:500]
        @printf("  %-20s slope on speed %+.3f per cm/s (r %+.2f) | between-strain share of the residual %.2f (shuffled strains %.2f, p %.3f)\n",
                m, β[2], cor(f.speed, f[!, m]), sb, mean(perm), mean(perm .>= sb))
    end
    serialize(joinpath(CS, "summary.jls"), (; S, human_groups = g, fly = f))
end
