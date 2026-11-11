#!/usr/bin/env julia
# Estimation bias and correction — manuscript "What does linear structure buy us?".
# Historical filename retained. Main/supplement panel split is specified in FIGURE_PLAN.md.
#
# Kicked SLIP walks (stride-level kicks at VLO; open loop, truth 0.628 ± 0.713i, ρ = 0.95), 20 walks of 2000
# steps; estimators and noise model in scripts/fig3_slip_maps.jl (cache/fig3/slip_maps.jls). Measurement
# noise is expressed relative to the stride-to-stride SD of each observed variable.
#
#   (A) Eigenvalues of the section (VLO) return map from 20 walks: 30 steps with noise 0.25×, and 2000
#       noise-free steps. Star: truth; arc: unit circle.
#   (B) Spectral radius against record length, noise-free (solid) and with noise 0.25× (dashed): more
#       strides reduce scatter, not the noise bias. Median and interquartile range over walks. Magenta: the
#       section map by instrumental variables (previous step as the instrument), noise 0.25×.
#   (C) Against measurement noise, 250 steps: section map (least squares and instrumental variables),
#       phase-indexed maps (Revzen & Guckenheimer) and 8-sector piecewise-linear maps.
#   (D) Piecewise-linear maps against the number of sectors composed over one step, 250 steps.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
D = deserialize(joinpath(ROOT, "cache", "fig3", "slip_maps.jls")); rows = D.rows
truth_ev = D.truths[0.0]; ρtrue = maximum(abs, truth_ev)
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
sel(; kappa = 0.0, method, K = 0, N = nothing, noise = nothing) =
    [x for x in rows if x.kappa == kappa && x.method == method && x.K == K &&
                        (N === nothing || x.N == N) && (noise === nothing || x.noise == noise)]
q(v, p) = isempty(v) ? NaN : quantile(v, p)
C_SEC, C_PH, C_PW, C_IV = CAT[1], CAT[3], CAT[2], CAT[5]

fig = Figure(size = (COL2, 122MM)); g = fig[1, 1] = GridLayout()

# ── A: eigenvalues in the complex plane ──────────────────────────────────────
axA = Axis(g[1, 1]; xlabel = "Re λ", ylabel = "Im λ", aspect = DataAspect(), title = "Return-map eigenvalues",
           limits = (-0.1, 1.1, -0.02, 1.05))
θs = range(0, π / 2; length = 100)
lines!(axA, cos.(θs), sin.(θs); color = C_REF, linestyle = :dot, linewidth = 0.8)
scatter!(axA, [real(truth_ev[argmax(imag.(truth_ev))])], [maximum(imag.(truth_ev))]; marker = :star5,
         color = CAT[4], markersize = 13, strokecolor = C_TRUTH, strokewidth = 0.5, label = "truth")
for (N, r, col, ms, lab) in ((30, 0.25, (C_SEC, 0.75), 4.5, "30 steps, noise 0.25×"), (2000, 0.0, C_TRUTH, 2.5, "2000 steps, no noise"))
    ev = reduce(vcat, [x.ev for x in sel(; method = "section", N, noise = r)])
    ev = ev[imag.(ev) .>= 0]
    scatter!(axA, real.(ev), imag.(ev); color = col, markersize = ms, label = lab)
end
axislegend(axA; position = (:left, :bottom), framevisible = false, labelsize = 6.5, rowgap = 0)
plabel!(g[1, 1], "A")

# ── B: against record length ─────────────────────────────────────────────────
NS = sort(unique([x.N for x in rows]))
axB = Axis(g[1, 2]; xscale = log10, xticks = (NS[[1, 3, 5, 7, 9]], string.(NS[[1, 3, 5, 7, 9]])),
           xlabel = "Steps of data", ylabel = "Estimated spectral radius ρ̂", title = "More strides do not remove noise bias; IV does",
           limits = (8, 2500, 0.0, 1.05))
hlines!(axB, ρtrue; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
for (m, K, col, lab) in (("section", 0, C_SEC, "section map"), ("piecewise", 8, C_PW, "piecewise, 8 sectors")),
    (r, ls) in ((0.0, :solid), (0.25, :dash))
    vs = [[x.ρ for x in sel(; method = m, K, N, noise = r)] for N in NS]
    r == 0.25 && band!(axB, NS, q.(vs, 0.25), q.(vs, 0.75); color = (col, 0.15))
    lines!(axB, NS, median.(vs); color = col, linestyle = ls, linewidth = 1.4)
end
vs = [[x.ρ for x in sel(; method = "section_iv", N, noise = 0.25)] for N in NS]
band!(axB, NS, q.(vs, 0.25), q.(vs, 0.75); color = (C_IV, 0.15))
lines!(axB, NS, median.(vs); color = C_IV, linestyle = :dash, linewidth = 1.4)
text!(axB, 2200, ρtrue + 0.015; text = "truth", fontsize = 7, align = (:right, :bottom))
Legend(g[1, 3], [LineElement(color = C_SEC), LineElement(color = C_IV), LineElement(color = C_PW), LineElement(color = C_REF),
                 LineElement(color = C_REF, linestyle = :dash)],
       ["section map", "section map, IV", "piecewise, 8 sectors", "no noise", "noise 0.25×"]; framevisible = false, labelsize = 6.5,
       tellheight = false, valign = :center)
plabel!(g[1, 2], "B")

# ── C: against measurement noise ─────────────────────────────────────────────
NOISE = sort(unique([x.noise for x in rows]))
axC = Axis(g[2, 1]; xlabel = "Measurement noise (× stride-to-stride SD)", ylabel = "Estimated spectral radius ρ̂",
           title = "250 steps", limits = (-0.03, 1.03, 0.0, 1.05))
hlines!(axC, ρtrue; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
for (m, K, col, mk, lab) in (("section", 0, C_SEC, :circle, "section map"), ("section_iv", 0, C_IV, :utriangle, "section map, IV"),
                             ("phase", 0, C_PH, :circle, "phase-indexed maps"), ("piecewise", 8, C_PW, :circle, "piecewise, 8 sectors"))
    vs = [[x.ρ for x in sel(; method = m, K, N = 250, noise = r)] for r in NOISE]
    band!(axC, NOISE, q.(vs, 0.25), q.(vs, 0.75); color = (col, 0.15))
    scatterlines!(axC, NOISE, median.(vs); color = col, marker = mk, markersize = mk == :circle ? 4 : 5.5, linewidth = 1.4, label = lab)
end
axislegend(axC; position = (1.0, 0.1), framevisible = false, labelsize = 6.5, rowgap = 0)
plabel!(g[2, 1], "C")

# ── D: piecewise against number of sectors ───────────────────────────────────
KS = sort(unique([x.K for x in rows if x.method == "piecewise"]))
axD = Axis(g[2, 2]; xscale = log2, xticks = (KS, string.(KS)), xlabel = "Sectors per step",
           ylabel = "Estimated spectral radius ρ̂", title = "Piecewise-linear maps compound the bias",
           limits = (1.7, 19, 0.0, 1.05))
hlines!(axD, ρtrue; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
for (r, col) in zip((0.0, 0.1, 0.25), (C_TRUTH, CAT[4], C_PW))
    vs = [[x.ρ for x in sel(; method = "piecewise", K, N = 250, noise = r)] for K in KS]
    band!(axD, KS, q.(vs, 0.25), q.(vs, 0.75); color = (col, 0.15))
    scatterlines!(axD, KS, median.(vs); color = col, markersize = 4, linewidth = 1.4, label = "noise $(r)×")
end
axislegend(axD; position = (:left, :bottom), framevisible = false, labelsize = 6.5, rowgap = 0)
plabel!(g[2, 2], "D")

colsize!(g, 1, Auto(1.0)); colgap!(g, 14); rowgap!(g, 12)
save_paper(fig, "Fig3_floquet_maps")

for κ in (0.0, 0.1), (m, K) in (("section", 0), ("section_iv", 0), ("phase", 0), ("piecewise", 8))
    tr = [x.truth for x in rows if x.kappa == κ][1]
    @printf("κ=%.1f %-9s K=%d truth %.3f | N=30 r=0.25: %.3f [%.3f–%.3f] | N=250 r=0.5: %.3f\n", κ, m, K, tr,
            median([x.ρ for x in sel(; kappa = κ, method = m, K, N = 30, noise = 0.25)]),
            q([x.ρ for x in sel(; kappa = κ, method = m, K, N = 30, noise = 0.25)], 0.25),
            q([x.ρ for x in sel(; kappa = κ, method = m, K, N = 30, noise = 0.25)], 0.75),
            median([x.ρ for x in sel(; kappa = κ, method = m, K, N = 250, noise = 0.5)]))
end
