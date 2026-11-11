#!/usr/bin/env julia
# Is the stride-time speed trend of the operator metrics in human walking (cross_species_fits.jl, human_stride) a
# property of the fit settings or of the mean cycle? The companion paper found Henrici's speed relation flips with
# rank and stabiliser, and that most of its transient growth is reproduced by the mean cycle alone.
#
# Human stride time (25 samples per stride, 20 delays, N = 150 transitions, 5 draws per trial):
#   grid     rank ∈ RANKS (120 = full) × ridge ∈ RIDGES (λ = ridge · s₁²), Schur-stabilised, as the rodent decks
#   surrogate  the trial's own mean stride tiled to the same length (+ white noise at 1% of the signal SD so the
#              Hankel matrix has full rank), fitted at the rodent stride-time settings (full rank, ridge 1e-4)
# Reported per setting: within-person Spearman ρ(metric, speed) (median over people, share < 0) and the share of
# people lower at their fastest than at their slowest trial.
# Mice: the Cntnap2 / L7-Tsc1 cohort's stride-time bouts as cached by the rodent deck (../Rodent_DMD_Analysis/results/
# original_cohort/bouts_centroid_geometry_cycle25.jls; 8 channels, whole strides of 25 samples), each mouse's bouts
# split at their median speed (scripts/28 speedsplit), N = 1000, 2 draws, rank ∈ COH_RANKS × ridge ∈ RIDGES; and the
# mean-cycle surrogate (each bout replaced by its half's mean stride, tiled, + 1% noise) at full rank, ridge 1e-4.
#
#   OPENBLAS_NUM_THREADS=1 julia -t 16 --project=../Rodent_DMD_Analysis scripts/cross_species_robustness.jl
# Writes cache/cross_species/robust_human.csv, robust_cohort.csv

include("cross_species_fits.jl")

const RANKS = [10, 20, 40, 80, 120]
const RIDGES = [0.0, 1e-6, 1e-4, 1e-2]
const NS = 150
const COH_RANKS = [20, 40, 160]

function fit_metrics_rr(bouts, f0, rank, ridge)
    model = DMD(rank = RodentDMDAnalysis.rank_rule(rank, 1e-6), stabilize = true)
    hr = DMDAnalysis.transform(HankelGenerator(n_delays = ND, delay_interval = 1), bouts)
    res = RodentDMDAnalysis.fit_ridge_dmd(model, hr, ridge)
    k = max(1, round(Int, FS / f0)); tg = try DMDAnalysis.transient_growth(res; kmax = k) catch; NaN end
    s2 = Float64.(res.singular_values) .^ 2; s2 ./= sum(s2)
    (; henrici = analyze(res, MetricSet(Henrici())).henrici, log_tg_stride = log10(max(tg, eps())), participation_ratio = 1 / sum(s2 .^ 2))
end

rk(v) = invperm(sortperm(v)) .* 1.0
tile(h) = (S = reshape(h.stride, 25, :, size(h.stride, 2)); M = dropdims(mean(S, dims = 2), dims = 2); repeat(M, size(S, 2)))

function report_human(df)
    for kind in ("real", "meancycle"), r in RANKS, a in RIDGES
        x = df[(df.kind .== kind) .& (df.rank .== r) .& (df.ridge .== a), :]; nrow(x) == 0 && continue
        g = combine(groupby(x, [:subject, :speed]), :henrici => mean => :hen, :log_tg_stride => mean => :tg, :participation_ratio => mean => :pr)
        line = String[]
        for (m, nm) in ((:hen, "Henrici"), (:tg, "log TG"), (:pr, "PR"))
            ρ = [cor(rk(g.speed[g.subject .== s]), rk(g[g.subject .== s, m])) for s in unique(g.subject) if count(g.subject .== s) >= 4]
            low = [(y = g[g.subject .== s, m]; sp = g.speed[g.subject .== s]; y[argmax(sp)] < y[argmin(sp)]) for s in unique(g.subject)]
            push!(line, @sprintf("%-7s ρ %+.2f (%2.0f%% < 0), lower when faster %3.0f%%", nm, median(ρ), 100mean(ρ .< 0), 100mean(low)))
        end
        @printf("%-9s rank %3d ridge %-6s | %s\n", kind, r, a == 0 ? "0" : @sprintf("%.0e", a), join(line, " | "))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    H = deserialize(joinpath(OUT, "inputs.jls")).human
    ok = [i for i in eachindex(H) if size(H[i].stride, 1) >= NS + ND]
    jobs = [(i, d, r, a, :real) for i in ok for d in 1:5 for r in RANKS for a in RIDGES]
    append!(jobs, [(i, d, 120, 1e-4, :meancycle) for i in ok for d in 1:5])
    rows = tmap(jobs) do (i, d, r, a, kind)
        h = H[i]; rng = MersenneTwister(hash((h.subject, h.speed, d, :stride))); s = rand(rng, 0:size(h.stride, 1)-(NS+ND))
        Y = kind == :real ? h.stride : (M = tile(h); M .+ 0.01 .* std(M, dims = 1) .* randn(MersenneTwister(hash((i, d))), size(M)))
        merge((subject = h.subject, group = h.group, speed = h.speed, draw = d, rank = r, ridge = a, kind = String(kind)),
              fit_metrics_rr([centre(Y[s+1:s+NS+ND, :])], FS / 25, r, a))
    end
    df = collect_rows(rows, "robust_human")
    report_human(df)
    # mice, stride time
    mice = deserialize(normpath(joinpath(RODENT, "results", "original_cohort", "bouts_centroid_geometry_cycle25.jls")))
    jobs = []
    for m in mice
        md = median(m.speeds)
        for (lvl, k) in (("slow", m.speeds .<= md), ("fast", m.speeds .> md))
            b = m.bouts[k]; sum(ntrans, b; init = 0) < 1000 && continue
            Mc = mean(reduce(vcat, [permutedims(reshape(x, 25, :, size(x, 2)), (2, 1, 3)) for x in b]), dims = 1)[1, :, :]
            for d in 1:2
                for r in COH_RANKS, a in RIDGES; push!(jobs, (m, lvl, b, d, r, a, :real, Mc)); end
                push!(jobs, (m, lvl, b, d, 160, 1e-4, :meancycle, Mc))
            end
        end
    end
    rows = tmap(jobs) do (m, lvl, b, d, r, a, kind, Mc)
        bb = kind == :real ? b : [(T = repeat(Mc, size(x, 1) ÷ 25); T .+ 0.01 .* std(T, dims = 1) .* randn(MersenneTwister(hash((m.subject, d, size(x, 1)))), size(T))) for x in b]
        sel, _, _ = draw(MersenneTwister(hash((m.subject, lvl, d))), bb, 1000)
        merge((subject = m.subject, group = m.group, level = lvl, draw = d, rank = r, ridge = a, kind = String(kind)),
              fit_metrics_rr(centre.(sel), FS / 25, r, a))
    end
    dc = collect_rows(rows, "robust_cohort")
    for kind in ("real", "meancycle"), r in COH_RANKS, a in RIDGES
        x = dc[(dc.kind .== kind) .& (dc.rank .== r) .& (dc.ridge .== a), :]; nrow(x) == 0 && continue
        g = combine(groupby(x, [:subject, :level]), :henrici => mean => :hen, :log_tg_stride => mean => :tg, :participation_ratio => mean => :pr)
        line = String[]
        for (m, nm) in ((:hen, "Henrici"), (:tg, "log TG"), (:pr, "PR"))
            low = [g[(g.subject .== s) .& (g.level .== "fast"), m][1] < g[(g.subject .== s) .& (g.level .== "slow"), m][1]
                   for s in unique(g.subject) if count(g.subject .== s) == 2]
            push!(line, @sprintf("%-7s lower when faster %3.0f%% of %d", nm, 100mean(low), length(low)))
        end
        @printf("mice %-9s rank %3d ridge %-6s | %s\n", kind, r, a == 0 ? "0" : @sprintf("%.0e", a), join(line, " | "))
    end
end
