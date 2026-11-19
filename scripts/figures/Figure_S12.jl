#!/usr/bin/env julia
# Figure_S12.jl — Return-map multipliers of SLIP walking in detail (manuscript: @suppfig-slip-maps). Run: julia --project=. scripts/figures/Figure_S12.jl
# Companion to Figure 7 E–F (the record-length and noise dependence of the estimated multiplier).
#
# Kicked SLIP walks (stride-level kicks at VLO; open loop, truth 0.628 ± 0.713i, ρ = 0.95), 20 walks of 2000
# steps; estimators and noise model in scripts/fig3_slip_maps.jl (cache/fig3/slip_maps.jls). Measurement
# noise is expressed relative to the stride-to-stride SD of each observed variable.
#
#   (A) Eigenvalues of the section (VLO) return map from 20 walks: 30 steps with noise 0.25×, and 2000
#       noise-free steps. Star: truth; arc: unit circle.
#   (B) Piecewise-linear maps against the number of sectors composed over one step, 250 steps.

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
D = deserialize(joinpath(ROOT, "cache", "fig3", "slip_maps.jls")); rows = D.rows
truth_ev = D.truths[0.0]; ρtrue = maximum(abs, truth_ev)
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
sel(; kappa = 0.0, method, K = 0, N = nothing, noise = nothing) =
    [x for x in rows if x.kappa == kappa && x.method == method && x.K == K &&
                        (N === nothing || x.N == N) && (noise === nothing || x.noise == noise)]
q(v, p) = isempty(v) ? NaN : quantile(v, p)
C_SEC = colorant"#6a4c9c"          # the section map, as in Figure 7

fig = Figure(size = (COL2, 62MM)); g = fig[1, 1] = GridLayout()

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

# ── B: piecewise against number of sectors ───────────────────────────────────
KS = sort(unique([x.K for x in rows if x.method == "piecewise"]))
axD = Axis(g[1, 2]; xscale = log2, xticks = (KS, string.(KS)), xlabel = "Sectors per step",
           ylabel = "Estimated spectral radius ρ̂", title = "Piecewise-linear maps compound the bias",
           limits = (1.7, 19, 0.0, 1.05))
hlines!(axD, ρtrue; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
for (r, col) in zip((0.0, 0.1, 0.25), (colorant"#2b2b2b", colorant"#77766f", colorant"#b3b2ad"))
    vs = [[x.ρ for x in sel(; method = "piecewise", K, N = 250, noise = r)] for K in KS]
    band!(axD, KS, q.(vs, 0.25), q.(vs, 0.75); color = (col, 0.15))
    scatterlines!(axD, KS, median.(vs); color = col, markersize = 4, linewidth = 1.4, label = "noise $(r)×")
end
axislegend(axD; position = (:left, :bottom), framevisible = false, labelsize = 6.5, rowgap = 0)
plabel!(g[1, 2], "B")

colgap!(g, 18)
save_paper(fig, "Figure_S12")
