#!/usr/bin/env julia
# Figure_06.jl — Forecasting the predictable short-time response (SLIP, with ceiling) (manuscript: @fig-predictable). Run: julia --project=. scripts/figures/Figure_06.jl
# Figure: how much of the predictable short-time response do fitted models forecast? (R1)
# Data: scripts/slip_predictable_truth.jl (SLIP.jl env) → scripts/predictable_response.jl → cache/predictable/slip_results.jls
#   (A) one test start (SLIP with feedback, full state, 60 training steps): height deviation from the mean step
#   (B–D) forecast skill against the mean cycle over the horizon: ceiling (noise-free continuation of the true state),
#         Hankel-DMD, ESN (median of 4 seeds), persistence; solid 60 training steps, dashed 300

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
R = deserialize(joinpath(ROOT, "cache", "predictable", "slip_results.jls"))
get(κ, n, obs) = only(filter(o -> o.κ == κ && o.NTR == n && o.obs == obs, R))
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
C_DMD = CAT[1]; C_ESN = CAT[2]
const SPS = 100                                     # samples per stride
binned(v; w = 10) = [mean(v[i:i+w-1]) for i in 1:w:length(v)-w+1]
hx(v; w = 10) = ((0:length(binned(v; w))-1) .+ 0.5) .* w ./ SPS
esn_med(o; y = false) = vec(median(reduce(hcat, [(y ? o.skilly : o.skill)["ESN $sd"] for sd in 1:4]); dims = 2))

fig = Figure(size = (COL2, 112MM))
gA = fig[1, 1] = GridLayout(); gB = fig[1, 2] = GridLayout()
gC = fig[2, 1] = GridLayout(); gD = fig[2, 2] = GridLayout()
colgap!(fig.layout, 16); rowgap!(fig.layout, 10)

# ── A: one forecast ─────────────────────────────────────────────────────────
o = get(0.1, 60, :full); ex = o.examples[2]
th = ((1:size(ex.hist, 1)) .- size(ex.hist, 1)) ./ SPS; tf = (1:size(ex.fut, 1)) ./ SPS
mfut = ex.fc["mean cycle"]
dev(F, M) = F[:, 1] .- M[:, 1]                       # height deviation from the mean step (z units)
axA = Axis(gA[1, 1]; xlabel = "Strides from the forecast start", ylabel = "Height deviation\nfrom the mean step (SD)",
           title = "One forecast: SLIP with feedback")
refline!(axA, 0.0; linestyle = :solid)
vlo = [h for h in 1:size(ex.fut, 1) if mod1(ex.j + h, 50) == 1] ./ SPS
for v in vlo; vlines!(axA, v; color = (C_REF, 0.35), linewidth = 0.6, linestyle = :dot); end
lines!(axA, th, dev(ex.hist, ex.mstep_hist); color = (C_REF, 0.9), linewidth = 0.9, label = "observed (noisy)")
lines!(axA, tf, dev(ex.fut, mfut); color = C_TRUTH, linewidth = 1.4, label = "what happened")
lines!(axA, tf, dev(Float64.(ex.fc["ceiling"]), mfut); color = C_TRUTH, linestyle = :dash, linewidth = 1.1,
       label = "predictable part (ceiling)")
for sd in 1:4
    lines!(axA, tf, dev(Float64.(ex.fc["ESN $sd"]), mfut); color = (C_ESN, 0.6), linewidth = 0.8,
           label = sd == 1 ? "ESN, 4 seeds" : nothing)
end
lines!(axA, tf, dev(Float64.(ex.fc["DMD 0"]), mfut); color = C_DMD, linewidth = 1.2, label = "Hankel-DMD")
axislegend(axA; position = (:right, :top), labelsize = 6.5, patchsize = (12, 5), unique = true)
yl = maximum(abs, dev(ex.hist, ex.mstep_hist))
text!(axA, vlo[1], -yl; text = "dotted: next disturbances", fontsize = 6, color = C_REF,
      align = (:left, :bottom), offset = (2, 0))
xlims!(axA, -2, 3); ylims!(axA, -1.1yl, 1.75yl)
plabel!(gA[1, 1], "A")

# ── B–D: skill over the horizon ─────────────────────────────────────────────
function skillpanel!(g, κ, obs, title; y = false, label = "")
    ax = Axis(g[1, 1]; xlabel = "Forecast horizon (strides)", ylabel = "Skill against the mean cycle",
              title, limits = (0, 3, -0.6, 1.02), yticks = [-0.5, 0, 0.5, 1])
    vspan!(ax, 0, 1; color = C_SHADE)
    refline!(ax, 0.0; linestyle = :solid)
    for (n, ls) in ((60, :solid), (300, :dash))
        o = get(κ, n, obs); S = y ? o.skilly : o.skill
        n == 60 && lines!(ax, hx(S["persistence"]), binned(S["persistence"]); color = C_REF, linestyle = :dot,
                          linewidth = 1.0, label = "persistence (last step)")
        n == 60 && lines!(ax, hx(S["ceiling"]), binned(S["ceiling"]); color = C_TRUTH, linewidth = 1.6,
                          label = "ceiling: no future disturbances")
        lines!(ax, hx(S["DMD 0"]), binned(S["DMD 0"]); color = C_DMD, linestyle = ls, linewidth = 1.3,
               label = n == 60 ? "Hankel-DMD" : nothing)
        lines!(ax, hx(S["ESN 1"]), binned(esn_med(o; y)); color = C_ESN, linestyle = ls, linewidth = 1.3,
               label = n == 60 ? "ESN (median of 4 seeds)" : nothing)
    end
    label != "" && text!(ax, 2.95, 0.97; text = label, fontsize = 6.5, color = C_REF, align = (:right, :top))
    ax
end
axB = skillpanel!(gB, 0.1, :full, "SLIP with feedback, full CoM state"; label = "solid 60, dashed 300 training steps")
axislegend(axB; position = (:left, :bottom), labelsize = 6.5, patchsize = (12, 5))
plabel!(gB[1, 1], "B")
skillpanel!(gC, 0.0, :full, "SLIP open loop, full CoM state"); plabel!(gC[1, 1], "C")
skillpanel!(gD, 0.1, :height, "SLIP with feedback, height only"); plabel!(gD[1, 1], "D")

save_paper(fig, "Figure_06")

bin(v, a, b) = mean(v[round(Int, a * SPS)+1:round(Int, b * SPS)])
for o in R
    S = o.skill
    @printf("κ %.1f %3d %-6s | ceiling %.2f %.2f | DMD %.2f %.2f (%.0f%%, %.0f%% of ceiling) | ESN %.2f %.2f (%.0f%%, %.0f%%)\n",
            o.κ, o.NTR, o.obs, bin(S["ceiling"], 0, 1), bin(S["ceiling"], 1, 3), bin(S["DMD 0"], 0, 1), bin(S["DMD 0"], 1, 3),
            100bin(S["DMD 0"], 0, 1) / bin(S["ceiling"], 0, 1), 100bin(S["DMD 0"], 1, 3) / bin(S["ceiling"], 1, 3),
            bin(esn_med(o), 0, 1), bin(esn_med(o), 1, 3), 100bin(esn_med(o), 0, 1) / bin(S["ceiling"], 0, 1),
            100bin(esn_med(o), 1, 3) / bin(S["ceiling"], 1, 3))
end
