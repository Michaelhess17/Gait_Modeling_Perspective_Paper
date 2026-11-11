#!/usr/bin/env julia
# Figure_04.jl — Gait recordings are mostly the mean cycle (manuscript: @fig-gait-mean). Run: julia --project=. scripts/figures/Figure_04.jl
# Figure 5 — Gait recordings are mostly the mean cycle; the small deviations still change what a model reports (OUTLINE.md §5).
#
# Data and computations: scripts/fig5_gait_data.jl (cache/fig5/gait.jls); 299 treadmill trials of 15 s, 42 people
# (able-bodied AB; higher-/lower-functioning stroke survivors HF/LF), six sagittal joint angles.
#
#   (A) One able-bodied trial: limb-1 knee angle of every stride (thin) and the mean cycle (thick); below, the
#       deviations of each stride from the mean cycle.
#   (B) Share of each trial's variance in the deviations against the number of strides in the trial.
#   (C) Transient growth of the companion paper's operator (delay DMD, rank 70, ridge 1e-4, stride-rescaled
#       time, TG = max over one stride of ‖Ãᵏ‖) against the delay window: recorded trials (solid) and the same
#       trials' mean cycle repeated with no deviations at all (dashed); medians. Dotted: the companion paper's
#       window (32 samples ≈ ⅓ stride). A periodic signal has no disturbances to amplify, yet over a window
#       shorter than the stride its harmonics are non-orthogonal and TG > 1; at a full-stride window TG = 1.
#   (D) What the deviations add: per-trial Δ log₁₀ TG = recorded − mean cycle only; median and IQR.
# Individual identification (formerly panel D) is in FigS5_identification.jl.
# Caveat: the slowest trials (24–43 cm/s, 4–9 strides) are irregular, with deviations at 15–45% of their variance.

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
G = deserialize(joinpath(ROOT, "cache", "fig5", "gait.jls"))
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
q(v, p) = quantile(filter(isfinite, v), p)

fig = Figure(size = (COL2, 118MM)); g = fig[1, 1] = GridLayout()

# ── A: one trial ─────────────────────────────────────────────────────────────
ex = G.example; S = ex.S; ph = range(0, 100; length = size(S, 2) + 1)[1:end-1]
m = dropdims(mean(S, dims = 1), dims = 1); D = S .- reshape(m, 1, size(m)...)
gA = g[1, 1] = GridLayout()
axA1 = Axis(gA[1, 1]; ylabel = "Knee (°)", title = @sprintf("One trial: %d strides at %.0f cm/s", size(S, 1), ex.speed),
            xticklabelsvisible = false, limits = (0, 100, nothing, nothing))
for s in axes(S, 1); lines!(axA1, ph, S[s, :, 2]; color = (C_REF, 0.6), linewidth = 0.5); end
lines!(axA1, ph, m[:, 2]; color = C_TRUTH, linewidth = 1.6)
axA2 = Axis(gA[2, 1]; xlabel = "Stride phase (%)", ylabel = "Deviation (°)", limits = (0, 100, nothing, nothing))
for s in axes(S, 1); lines!(axA2, ph, D[s, :, 2]; color = (CAT[1], 0.55), linewidth = 0.5); end
hlines!(axA2, 0; color = C_TRUTH, linewidth = 0.6)
rowsize!(gA, 2, Relative(0.4)); rowgap!(gA, 3)
sh = sum(abs2, D) / sum(abs2, S .- mean(S, dims = (1, 2)))
text!(axA1, 1, maximum(S[:, :, 2]); text = @sprintf("deviations:\n%.1f%% of variance\n(all six joints)", 100sh),
      fontsize = 6.5, align = (:left, :top))
plabel!(gA, "A")

# ── B: deviation share against strides per trial ─────────────────────────────
axB = Axis(g[1, 2]; yscale = log10, xlabel = "Strides in the trial", ylabel = "Deviations (% of variance)",
           title = "Short records, mostly mean cycle", yticks = ([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"]),
           limits = (2, 20.5, 0.4, 70))
for gr in GROUP_ORDER_FIG
    k = G.grp .== gr
    scatter!(axB, G.nstr[k] .+ 0.25 .* (rand(count(k)) .- 0.5), 100 .* G.devshare[k]; color = (GROUP_COL[gr], 0.65),
             markersize = 4, label = @sprintf("%s (median %.1f%%)", gr, 100median(G.devshare[k])))
end
axislegend(axB; position = (:right, :top), framevisible = false, labelsize = 6.5, rowgap = 0)
plabel!(g[1, 2], "B")

# ── C: transient growth, recorded trials vs their mean cycle alone ────────────
W = G.windows
axC = Axis(g[2, 1]; xlabel = "Delay window (fraction of a stride)", ylabel = "Transient growth (log₁₀ max ‖Ãᵏ‖)",
           title = "The mean cycle alone sets a floor", xticks = (W ./ 100, string.(W ./ 100)),
           limits = (0.1, 1.12, -0.03, 0.95))
vlines!(axC, 0.32; color = C_REF, linestyle = :dot, linewidth = 0.8)
text!(axC, 0.335, 0.93; text = "companion\npaper", fontsize = 6.5, color = C_REF, align = (:left, :top))
for gr in GROUP_ORDER_FIG
    k = G.grp .== gr; col = GROUP_COL[gr]
    re = [G.TG[(:real, d)][k] for d in W]; mc = [G.TG[(:mean, d)][k] for d in W]
    lines!(axC, W ./ 100, median.(re); color = col, linewidth = 1.4)
    lines!(axC, W ./ 100, median.(mc); color = col, linewidth = 1.2, linestyle = :dash)
    text!(axC, 1.02, median(re[end]); text = gr, color = col, fontsize = 7, align = (:left, :center))
end
axislegend(axC, [LineElement(color = C_REF), LineElement(color = C_REF, linestyle = :dash)],
           ["recorded trials", "mean cycle only"]; position = (:right, :top), framevisible = false, labelsize = 6.5,
           rowgap = 0, margin = (0, 14, 0, 0))
plabel!(g[2, 1], "C")

# ── D: what the deviations add ───────────────────────────────────────────────
axD = Axis(g[2, 2]; xlabel = "Delay window (fraction of a stride)", ylabel = "Added by deviations (Δ log₁₀ TG)",
           title = "Small deviations add a graded increase", xticks = (W ./ 100, string.(W ./ 100)),
           limits = (0.1, 1.12, 0, 0.33))
vlines!(axD, 0.32; color = C_REF, linestyle = :dot, linewidth = 0.8)
for gr in GROUP_ORDER_FIG
    k = G.grp .== gr; col = GROUP_COL[gr]
    Δ = [G.TG[(:real, d)][k] .- G.TG[(:mean, d)][k] for d in W]
    band!(axD, W ./ 100, q.(Δ, 0.25), q.(Δ, 0.75); color = (col, 0.15))
    scatterlines!(axD, W ./ 100, median.(Δ); color = col, linewidth = 1.4, markersize = 4)
    text!(axD, 1.02, median(Δ[end]); text = gr, color = col, fontsize = 7, align = (:left, :center))
end
text!(axD, 0.35, 0.32; text = join([@sprintf("%s: %.1f%%", gr, 100median(G.devshare[G.grp .== gr])) for gr in GROUP_ORDER_FIG],
      ", ") * "\nof the variance is deviations", fontsize = 6.5, color = C_REF, align = (:left, :top))
plabel!(g[2, 2], "D")

colsize!(g, 1, Relative(0.5)); colgap!(g, 14); rowgap!(g, 12)
save_paper(fig, "Figure_04")

@printf("example %s trial %d: %d strides, deviation share %.2f%%\n", ex.subject, ex.i, size(S, 1), 100sh)
for d in W, gr in GROUP_ORDER_FIG
    k = G.grp .== gr
    @printf("d=%3d %s  real %.3f  mean-only %.3f  (mean-only share of log TG %.0f%%)\n", d, gr, median(G.TG[(:real, d)][k]),
            median(G.TG[(:mean, d)][k]), 100median(G.TG[(:mean, d)][k]) / median(G.TG[(:real, d)][k]))
end
