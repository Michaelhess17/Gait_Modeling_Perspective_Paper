#!/usr/bin/env julia
# What the Fourier-in-phase operator learns from long running trials (maus_structure.jl; 22 trials, 4 runners).
#   (A) Persistence of a typical deviation over the next quarter stride, by the phase it starts at (the trial's own
#       delay states carried 25 samples by the operator; < 1 = decaying). Thin: trials; thick: runner means. Shading:
#       right and left stance from the treadmill forces (circular mean over runners), white: flight.
#   (B) The leading Floquet mode (multiplier ≈ 0.36 per stride, real) carried through the stride: its size in each joint
#       angle by phase, mean over all trials (right leg solid, left leg dashed).
#   (C) Reproducibility: correlation of the persistence profile and of the leading and second modes' shapes between
#       the two halves of a trial, between trials of the same runner, and between runners (median, IQR).
#   (D) Floquet multipliers per stride, largest and second, by runner and trial.
#   (E) Routing of typical deviations: |joint j now → joint i a quarter stride later|, mean over right stance and over
#       left stance (all trials).
# Usage: julia --project=. scripts/Fig_running_structure.jl      (writes figures/Fig_running_structure.*)

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf, LinearAlgebra, Random

const ROOT = dirname(@__DIR__)
const NPH = 100
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const ST = deserialize(joinpath(ROOT, "cache", "maus", "structure.jls"))
R, subs = ST.R, ST.subs; us = unique(subs); ph = 0:NPH-1
cm(φs) = mod(angle(mean(cis.(2π .* φs))) / 2π, 1) * 100
EV = Dict(k => cm([r.events[k] for r in R]) for k in (:RFS, :RTO, :LFS, :LTO))
"Shade an interval of the stride (%), wrapping at 100."
function stance!(ax, a, b, col)
    a < b ? vspan!(ax, a, b; color = col) : (vspan!(ax, a, 100; color = col); vspan!(ax, 0, b; color = col))
end
shade!(ax) = (stance!(ax, EV[:RFS], EV[:RTO], C_SHADE); stance!(ax, EV[:LFS], EV[:LTO], C_SHADE))
stancelabels!(ax, y) = (text!(ax, (EV[:LFS] + EV[:LTO]) / 2, y; text = "left stance", fontsize = 6, color = C_REF, align = (:center, :top));
                        text!(ax, 99, y; text = "right stance", fontsize = 6, color = C_REF, align = (:right, :top)))
mi3(v) = (quantile(v, 0.25), median(v), quantile(v, 0.75))

fig = Figure(size = (COL2, 125MM))
top = fig[1, 1] = GridLayout(); bot = fig[2, 1] = GridLayout()

# A: persistence
aA = Axis(top[1, 1]; xlabel = "Phase at which the deviation is taken (% stride)", ylabel = "Typical deviation a quarter\nstride later / now",
          title = "Deviations are corrected in stance", limits = ((0, 100), (0.3, 1.32)), xticks = 0:25:100)
shade!(aA); hlines!(aA, 1.0; color = C_REF, linewidth = 0.6)
for (k, s) in enumerate(us)
    I = findall(subs .== s)
    for i in I; lines!(aA, ph, R[i].whole.persist; color = (CAT[k], 0.25), linewidth = 0.5); end
    lines!(aA, ph, mean(R[i].whole.persist for i in I); color = CAT[k], linewidth = 1.4)
    text!(aA, 99, 1.18 - 0.06 * (k - 1); text = "runner $k", color = CAT[k], fontsize = 6.5, align = (:right, :top))
end
for (k, lab, y) in ((:RFS, "R strike", 0.36), (:RTO, "R off", 0.36), (:LFS, "L strike", 0.36), (:LTO, "L off", 0.36))
    text!(aA, EV[k], y; text = lab, fontsize = 5.5, color = C_TRUTH, align = (:center, :bottom), rotation = π / 2)
end
stancelabels!(aA, 1.29)
plabel!(top[1, 1], "A")

# B: leading mode shape
aB = Axis(top[1, 2]; xlabel = "Phase (% stride)", ylabel = "Leading mode, size in each joint", title = "The slowest deviation is in the swing leg",
          limits = ((0, 100), (0, 1.25)), xticks = 0:25:100, yticks = 0:0.25:1)
shade!(aB); stancelabels!(aB, 0.99)
Ms = mean(abs.(r.whole.modes[1]) for r in R)
for (j, nm) in enumerate(("hip", "knee", "ankle"))
    lines!(aB, ph, Ms[:, j]; color = CAT[j], linewidth = 1.3)
    lines!(aB, ph, Ms[:, j+3]; color = CAT[j], linewidth = 1.1, linestyle = :dash)
end
axislegend(aB, [LineElement(color = CAT[1]), LineElement(color = CAT[2]), LineElement(color = CAT[3]),
                LineElement(color = C_REF), LineElement(color = C_REF, linestyle = :dash)],
           ["hip (thigh)", "knee", "ankle", "right leg", "left leg"]; position = :ct, labelsize = 6, orientation = :horizontal, nbanks = 2)
plabel!(top[1, 2], "B")
colgap!(top, 16)

# C: reproducibility
hcor(f) = [cor(f(r.half1), f(r.half2)) for r in R]
pcor(f, same) = [cor(f(R[i].whole), f(R[j].whole)) for i in eachindex(R) for j in i+1:length(R) if (subs[i] == subs[j]) == same]
ITEMS = [("persistence\nprofile", o -> log.(o.persist)), ("leading\nmode", o -> vec(abs.(o.modes[1]))), ("second\nmode", o -> vec(abs.(o.modes[2])))]
aC = Axis(bot[1, 1]; ylabel = "Correlation", title = "Reproducible, partly individual", xticks = (1:3, first.(ITEMS)),
          limits = ((0.5, 3.5), (0, 1)))
for (k, (_, f)) in enumerate(ITEMS)
    for (j, (v, col, lab)) in enumerate(((hcor(f), C_TRUTH, "halves of a trial"), (pcor(f, true), CAT[1], "same runner"), (pcor(f, false), CAT[2], "different runners")))
        x = k + (j - 2) * 0.22; q = mi3(v)
        lines!(aC, [x, x], [q[1], q[3]]; color = col, linewidth = 1.2); scatter!(aC, [x], [q[2]]; color = col, markersize = 5.5)
        k == 3 && text!(aC, 0.55, 0.25 - 0.07 * (j - 1); text = lab, color = col, fontsize = 6, align = (:left, :center))
    end
end
plabel!(bot[1, 1], "C")

# D: multipliers
aD = Axis(bot[1, 2]; ylabel = "|multiplier| per stride", title = "One slow mode", xticks = (1:length(us), ["runner $k" for k in eachindex(us)]),
          limits = ((0.5, length(us) + 0.5), (0, 0.6)), xticklabelrotation = π / 6)
for (k, s) in enumerate(us)
    I = findall(subs .== s); x = k .+ range(-0.25, 0.25; length = length(I))
    scatter!(aD, x, [abs(R[i].whole.λ[1]) for i in I]; color = CAT[k], markersize = 5)
    scatter!(aD, x, [abs(R[i].whole.λ[2]) for i in I]; color = :white, strokecolor = CAT[k], strokewidth = 0.8, markersize = 4.5)
end
text!(aD, 0.6, 0.58; text = "● largest  ○ second", fontsize = 6, align = (:left, :top))
plabel!(bot[1, 2], "D")

# E: routing in stance
JL = ["hip R", "knee R", "ank R", "hip L", "knee L", "ank L"]
gE = bot[1, 3] = GridLayout()
rs = vcat(round(Int, EV[:RFS]):99, 0:round(Int, EV[:RTO])) .+ 1; ls = (round(Int, EV[:LFS]):round(Int, EV[:LTO])) .+ 1
hms = Any[]
for (k, (nm, idx)) in enumerate((("right stance", rs), ("left stance", ls)))
    Rm = mean(mean(abs.(r.whole.route[:, :, idx]), dims = 3)[:, :, 1] for r in R)
    ax = Axis(gE[1, k]; title = nm, xticks = (1:6, JL), yticks = (1:6, JL), xticklabelrotation = π / 2, yreversed = true,
              xlabel = "now", ylabel = k == 1 ? "¼ stride later" : "", yticklabelsvisible = k == 1, aspect = DataAspect(),
              xticklabelsize = 6, yticklabelsize = 6)
    push!(hms, heatmap!(ax, 1:6, 1:6, permutedims(Rm); colormap = cgrad([colorant"#fcfcfb", CAT[1]]), colorrange = (0, 1.5)))
end
Colorbar(gE[1, 3], hms[1]; label = "|gain|", height = Relative(0.6), ticks = 0:0.5:1.5)
Label(gE[0, 1:2], "Routing of typical deviations"; fontsize = 8, halign = :left, tellwidth = false)
colgap!(gE, 4); rowgap!(gE, 2); plabel!(gE[1, 1], "E")
colsize!(bot, 1, Relative(0.3)); colsize!(bot, 2, Relative(0.22)); colgap!(bot, 14); rowgap!(fig.layout, 12)
save_paper(fig, "Fig_running_structure")
