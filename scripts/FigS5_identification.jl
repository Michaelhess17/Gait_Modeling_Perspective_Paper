#!/usr/bin/env julia
# Figure S5 — Identifying individuals: the mean cycle does as well as the operator signature (OUTLINE.md §5 ¶3).
#
# Data: scripts/fig5_gait_data.jl (cache/fig5/gait.jls). The companion paper's split-half protocol
# (07_individual_routing_signatures.jl): each person's trials split at random into halves, features averaged per
# half and z-scored against the able-bodied half-A distribution; a person is identified when their other half is the
# nearest (cosine) match among all 42 people. Mean and 95% range over 200 random splits; dashed: chance (1/42).
# Caveat: deviation features can carry within-stride timing jitter (∝ mean-cycle velocity) and marker artefacts.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf

G = deserialize(joinpath(dirname(@__DIR__), "cache", "fig5", "gait.jls"))
ID = [(:shape, "mean-cycle shape", CAT[1]), (:rom, "mean-cycle range-of-motion shares", CAT[1]),
      (:map_real, "operator map, recorded trials", CAT[2]), (:map_mean, "operator map, mean cycle only", CAT[2]),
      (:dev_sd, "deviation SD profile", CAT[3]), (:dev_corr, "deviation cross-joint correlation", CAT[3])]
ys = [1, 2, 3.5, 4.5, 6, 7]
fig = Figure(size = (COL1 * 1.3, 62MM))
ax = Axis(fig[1, 1]; xlabel = "People identified from a held-out half of their trials (%)", title = "Who is walking?",
          yticks = (ys, [x[2] for x in ID]), yreversed = true, limits = (0, 100, 0.3, 7.7), yticklabelsize = 6.5)
for (y, (k, _, col)) in zip(ys, ID)
    r = G.ident[k]
    barplot!(ax, [y], [100r.rate]; direction = :x, color = (col, 0.8), width = 0.75)
    rangebars!(ax, [y], [100r.lo], [100r.hi]; direction = :x, color = C_TRUTH, linewidth = 0.8, whiskerwidth = 3)
    text!(ax, 100r.hi + 1.5, y; text = @sprintf("%.0f%%", 100r.rate), fontsize = 6.5, align = (:left, :center))
end
vlines!(ax, 100 / G.ident[:shape].n; color = C_REF, linestyle = :dash, linewidth = 0.8)
text!(ax, 100 / G.ident[:shape].n + 1, 7.6; text = "chance", fontsize = 6.5, color = C_REF, align = (:left, :bottom))
save_paper(fig, "FigS5_identification")
