#!/usr/bin/env julia
# Phase-dependent (Fourier-in-phase) operators of walking deviations: shared in able-bodied walking, individual after
# stroke. Companion superset, 15-s treadmill trials; setting S = (delay 8, rank 24, 4 harmonics, ridge 1e-2).
#   (A) How many strides the operator needs (human_ltp_pooled.jl calibration on running): agreement of the phase-
#       resolved growth profile with the same trial's full-length estimate (r), and leading-multiplier error, by chunk
#       size. A 15-s walking trial has 6–12 strides.
#   (B) Group operators (60 pooled strides per fit), leave-one-person-out: R² of a new person's deviation 12.5% of a
#       stride ahead, with phase (filled) and without (open), by group and speed band (human_ltp_pooled.jl).
#   (C) The same held-out trials predicted by an operator from the person's own other trials (≤ 60 strides, speeds
#       within ×1.5) or from other people of the group at similar speed (human_ltp_within.jl); dots: people (means
#       over their trials), lines join a person's two values; with phase.
# Usage: julia --project=. scripts/Fig_human_group_operators.jl      (writes figures/Fig_human_group_operators.*)

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf, Random

const ROOT = dirname(@__DIR__)
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const P = deserialize(joinpath(ROOT, "cache", "fig5", "human_ltp_pooled.jls"))
const W = deserialize(joinpath(ROOT, "cache", "fig5", "human_ltp_within.jls"))
q3(v) = (quantile(v, 0.25), median(v), quantile(v, 0.75))

fig = Figure(size = (COL2, 70MM))
g = fig[1, 1] = GridLayout()

# A: calibration
aA = Axis(g[1, 1]; xscale = log2, xticks = [8, 16, 32, 64, 128], xlabel = "Strides of data", ylabel = "Growth profile vs full trial (r)",
          title = "The operator needs ≳ 50 strides", limits = ((6, 170), (0, 1)))
vspan!(aA, 6, 12; color = C_SHADE); text!(aA, 6.3, 0.97; text = "15-s\nwalking\ntrial", fontsize = 6, color = C_REF, align = (:left, :top))
ns = sort(collect(keys(P.cal)))
for (k, n) in enumerate(ns)
    q = q3(P.cal[n].cr); lines!(aA, [n, n], [q[1], q[3]]; color = C_TRUTH, linewidth = 1.1); scatter!(aA, [n], [q[2]]; color = C_TRUTH, markersize = 5)
end
lines!(aA, ns, [median(P.cal[n].cr) for n in ns]; color = C_TRUTH, linewidth = 1.0)
text!(aA, 160, 0.05; text = @sprintf("multiplier error at 8 strides:\n%.2f [%.2f, %.2f]", q3(P.cal[8].dm)[2], q3(P.cal[8].dm)[1], q3(P.cal[8].dm)[3]),
      fontsize = 6, align = (:right, :bottom))
plabel!(g[1, 1], "A")

# B: group operators, leave one person out
aB = Axis(g[1, 2]; xlabel = "Treadmill speed (cm/s)", ylabel = "R², new person, 12.5% of a stride",
          title = "A group operator transfers only in AB", limits = ((15, 205), (-0.25, 0.5)), xticks = 0:50:200)
hlines!(aB, 0; color = C_REF, linewidth = 0.6)
for (gi, gr) in enumerate(("AB", "HF", "LF"))
    ks = sort([k for k in keys(P.res) if k[1] == gr], by = last)
    xs = [mean(P.BANDS[k[2]]) + (gi - 2) * 3 for k in ks]
    v1 = [median(first.(P.res[k].lopo)) for k in ks]; v0 = [median(last.(P.res[k].lopo)) for k in ks]
    lines!(aB, xs, v1; color = GROUP_COL[gr], linewidth = 1.3); scatter!(aB, xs, v1; color = GROUP_COL[gr], markersize = 6)
    lines!(aB, xs, v0; color = GROUP_COL[gr], linewidth = 0.8, linestyle = :dash)
    scatter!(aB, xs, v0; color = :white, strokecolor = GROUP_COL[gr], strokewidth = 0.9, markersize = 5)
    text!(aB, xs[end] + 6, v1[end]; text = gr, color = GROUP_COL[gr], fontsize = 7, align = (:left, :center))
end
text!(aB, 20, 0.49; text = "filled: with phase   open: without", fontsize = 6, align = (:left, :top))
plabel!(g[1, 2], "B")

# C: own vs other people
aC = Axis(g[1, 3]; ylabel = "R², held-out trial, 12.5% of a stride", title = "Stroke dynamics are individual",
          xticks = (1:3, ["AB", "HF", "LF"]), limits = ((0.4, 3.6), (-0.6, 0.9)))
hlines!(aC, 0; color = C_REF, linewidth = 0.6)
for (gi, gr) in enumerate(("AB", "HF", "LF"))
    x = [r for r in W.R if r.group == gr]; ps = unique(getfield.(x, :subject))
    own = [mean(r.within[1] for r in x if r.subject == p) for p in ps]; oth = [mean(r.between[1] for r in x if r.subject == p) for p in ps]
    jit = 0.05 .* randn(MersenneTwister(gi), length(ps))
    for k in eachindex(ps)
        lines!(aC, [gi - 0.18 + jit[k], gi + 0.18 + jit[k]], clamp.([own[k], oth[k]], -0.58, 0.88); color = (GROUP_COL[gr], 0.3), linewidth = 0.6)
    end
    scatter!(aC, gi - 0.18 .+ jit, clamp.(own, -0.58, 0.88); color = GROUP_COL[gr], markersize = 4.5)
    scatter!(aC, gi + 0.18 .+ jit, clamp.(oth, -0.58, 0.88); color = :white, strokecolor = GROUP_COL[gr], strokewidth = 0.8, markersize = 4.5)
end
text!(aC, 0.45, 0.88; text = "● own other trials   ○ other people", fontsize = 6, align = (:left, :top))
plabel!(g[1, 3], "C")
colgap!(g, 16)
save_paper(fig, "Fig_human_group_operators")
