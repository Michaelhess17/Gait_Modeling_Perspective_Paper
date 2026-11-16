#!/usr/bin/env julia
# Figure_08.jl — When one model can serve many recordings (manuscript: @fig-heterogeneity). Run: julia --project=. scripts/figures/Figure_08.jl
# When does one model for many recordings beat one model per recording? (heterogeneity_pools.jl, cohort_fits.jl)
#   A  group over individual remaining variance in the first cycle (MSE of the group fit / MSE of the trial's own fit,
#      both relative to the mean-cycle forecast; median over the pool's own trials), against the pool's heterogeneity H
#   B  the same for a trial the group model has not seen (leave-one-out)
#   C  pool size against H, coloured by the ratio of A; where our datasets lie
# The caches hold skill1 = 1 − RMSE / RMSE of the mean cycle, so the relative remaining variance is (1 − skill1)².
# Usage: julia --project=. scripts/figures/Figure_08.jl   → figures/Figure_08.{pdf,svg,png}

include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
D = deserialize(joinpath(ROOT, "cache", "heterogeneity", "gait_pools_dmd.jls"))
pools = D.pools

# cohorts: first-cycle relative remaining variance as for the gait trials, median over individuals
function cohort_point(coh)
    out = deserialize(joinpath(ROOT, "cache", "cohort", "$(coh)_dmd.jls")); n1 = out["HOR"] ÷ 3
    rv(r, nm) = sum(r.SE[nm][1:n1]) / sum(r.SE["mean cycle"][1:n1])
    mdl(r) = median([rv(r, k) for k in keys(r.SE) if startswith(k, "model")])
    d = [mdl(g) / mdl(i) for (i, g) in zip(out["individual"], out["group"].rows)]
    (; Δ = median(d), lo = quantile(d, 0.25), hi = quantile(d, 0.75))
end
const H_COH = Dict("slip" => 1.72, "cfhn" => 1.00)          # heterogeneity.jl on the cohorts (H_check)
coh = Dict(c => cohort_point(c) for c in ("slip", "cfhn"))

ratio(x) = ((1 - x.skill1) / (1 - x.skill1_ind))^2              # group / individual remaining variance
Δseen(p) = median(ratio.(p.rows))
Δloo(p) = median(ratio.(p.loo))
const KINDS = [("nearest", "nearest neighbours", CAT[1], :circle),
               ("subject", "one participant", CAT[2], :diamond),
               ("speed", "one speed band", CAT[3], :rect),
               ("random", "random trials", C_REF, :utriangle),
               ("group", "participant group / all", C_TRUTH, :star5)]
const YLO, YHI = 0.3, 100.0
clampΔ(x) = clamp(x, YLO, YHI)

fig = Figure(size = (COL2, 74MM))
gA = fig[1, 1]; gB = fig[1, 2]; gC = fig[1, 3]
for (g, f, ttl) in ((gA, Δseen, "trials in the pool"), (gB, Δloo, "a trial left out"))
    ax = Axis(g; xscale = log10, yscale = log10, xlabel = "heterogeneity H", ylabel = "remaining variance, group / individual",
              title = ttl, xticks = [1, 2, 4, 8, 16], yticks = ([0.5, 1, 2, 5, 10, 30, 100], ["0.5", "1", "2", "5", "10", "30", "100"]))
    refline!(ax, 1)
    for (kd, lab, col, mk) in KINDS
        ps = filter(p -> p.kind == kd, pools)
        scatter!(ax, [p.H for p in ps], [clampΔ(f(p)) for p in ps]; color = (col, 0.8), marker = mk,
                 markersize = kd == "group" ? 8 : 5, label = lab)
    end
    if f === Δseen
        for (c, lab) in (("cfhn", "FHN"), ("slip", "SLIP"))
            scatter!(ax, [H_COH[c]], [coh[c].Δ]; color = :white, strokecolor = C_TRUTH, strokewidth = 0.9,
                     marker = :hexagon, markersize = 7, label = c == "cfhn" ? "simulated cohort" : nothing)
            text!(ax, H_COH[c], coh[c].Δ; text = lab, fontsize = 6, align = (:center, :bottom), offset = (0, 4))
        end
        allp = only(filter(p -> p.label == "all", pools))
        text!(ax, allp.H, clampΔ(f(allp)); text = "all 299", fontsize = 6, align = (:left, :center), offset = (5, 0))
    else
        text!(ax, 0.85, 95; text = "values above 100 clipped", fontsize = 6, align = (:left, :top), color = C_REF)
        text!(ax, 0.85, 0.35; text = "below 1: the group fit forecasts better", fontsize = 6, align = (:left, :bottom), color = C_REF)
    end
    xlims!(ax, 0.8, 25); ylims!(ax, YLO, YHI * 1.1)
    f === Δseen && axislegend(ax; position = :rb, nbanks = 1, labelsize = 6)
    panel_label!(g, f === Δseen ? "A" : "B")
end
# C: median gain over the pools in each (H, N) cell; grey cells: no such pools in the data
const HE = 2.0 .^ (0:0.5:4.5)                                      # H bin edges, 1 … 22.6
const NE = [1.5, 2.5, 4, 10, 18, 35, 70, 140, 400]               # N bins: 2, 3, 5–9, 12, 25, 50, 100, 180–299
M = fill(NaN, length(HE) - 1, length(NE) - 1)
for i in 1:length(HE)-1, j in 1:length(NE)-1
    ps = filter(p -> HE[i] <= p.H < HE[i+1] && NE[j] <= p.N < NE[j+1], pools)
    isempty(ps) || (M[i, j] = log10(median(Δseen.(ps))))
end
axC = Axis(gC; xscale = log10, yscale = log10, xlabel = "heterogeneity H", ylabel = "recordings pooled N",
           title = "where pooling pays", xticks = [1, 2, 4, 8, 16], yticks = [2, 3, 6, 12, 25, 50, 100, 300])
cm = cgrad([CAT[2], colorant"#c9c7c2", CAT[1]], [0, 0.5, 1])          # orange: group better, blue: individual better
cr = (-1, 1)                                                          # log10 of the ratio: 0.1 to 10
heatmap!(axC, HE, NE, M; colormap = cm, colorrange = cr, lowclip = CAT[2], highclip = CAT[1], nan_color = :transparent)
for (c, lab) in (("cfhn", "FHN"), ("slip", "SLIP"))
    scatter!(axC, [H_COH[c]], [10]; color = :white, marker = :hexagon, markersize = 7, strokecolor = C_TRUTH, strokewidth = 0.9)
    text!(axC, H_COH[c], 10; text = lab, fontsize = 6, align = (:center, :bottom), offset = (0, 4))
end
for (lab, hh, nn) in (("all 299", only(filter(p -> p.label == "all", pools)).H, 299),
                      ("one participant", median([p.H for p in pools if p.kind == "subject"]), 6.5))
    scatter!(axC, [hh], [nn]; color = :white, marker = :star5, markersize = 7, strokecolor = C_TRUTH, strokewidth = 0.9)
    text!(axC, hh, nn; text = lab, fontsize = 6, align = (:right, :center), offset = (-5, 0))
end
text!(axC, 1.05, 330; text = "white: no such\npools in the data", fontsize = 6, color = C_REF, align = (:left, :top))
xlims!(axC, 0.85, 22.6); ylims!(axC, 1.5, 400)
Colorbar(fig[1, 4], limits = cr, colormap = cm, label = "remaining variance, group / individual (median)",
         ticks = ([-1, -0.5, 0, 0.5, 1], ["0.1", "0.3", "1", "3", "10"]), lowclip = CAT[2], highclip = CAT[1])
panel_label!(gC, "C")
colgap!(fig.layout, 8)
save_paper(fig, "Figure_08")

# numbers for the text
for (kd, lab, _, _) in KINDS
    ps = filter(p -> p.kind == kd, pools)
    @printf("%-26s %3d pools | H %5.2f–%5.2f | ratio seen median %.2f | ratio unseen median %.2f\n", lab, length(ps),
            extrema([p.H for p in ps])..., median(Δseen.(ps)), median(Δloo.(ps)))
end
for N in (2, 3, 6, 12, 25, 50, 100)
    for kd in ("nearest", "random")
        ps = filter(p -> p.kind == kd && p.N == N, pools)
        @printf("%-8s N %3d: H %5.2f, ratio seen %.2f, ratio unseen %.2f\n", kd, N, median([p.H for p in ps]), median(Δseen.(ps)), median(Δloo.(ps)))
    end
end
for lim in (1.5, 2, 3)
    lo = filter(p -> p.H < lim, pools); hi = filter(p -> p.H >= lim, pools)
    @printf("H < %.1f: %3d pools, ratio seen %.2f, group better in %3.0f%% | H ≥ %.1f: %3d pools, ratio seen %.2f, group better in %3.0f%%\n",
            lim, length(lo), median(Δseen.(lo)), 100mean(Δseen.(lo) .< 1), lim, length(hi), median(Δseen.(hi)),
            100mean(Δseen.(hi) .< 1))
end
