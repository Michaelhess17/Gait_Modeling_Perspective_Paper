#!/usr/bin/env julia
# Figure 6 — What a linear model makes possible: one answer per setting, and an estimator bias that can be
# analysed and removed (OUTLINE.md §6 ¶2, ¶4).
#
#   (A) Coupled FHN: off-cycle relaxation rate per cycle of decay-trained models (scripts/offcycle_v2.jl;
#       10 decays × 3 cycles, same data for every model): large markers = each fit (median over 4 start phases),
#       small markers = each start phase. Hankel-DMD is a single least-squares fit (no seed); ESN (4 seeds) and
#       neural ODE (3 seeds) at fixed settings. Line: truth (band: its range over the start phases).
#       (Stuart–Landau, dropped as too simple: DMD 0.88 vs truth 0.58 at 10 decays, ESN 0.46–0.48, NODE 0.49–0.54.)
#   (B, C) SLIP section-map spectral radius against measurement noise (2000 strides; data and noise model of
#       Fig 3): least squares, forward–backward, total least squares and instrumental variables
#       (scripts/fig6_bias_correction.jl). (B) open loop; (C) with speed feedback. Median and IQR over 20 walks.
#   (D) Against record length at noise 0.25×, open loop.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
const FDIR = joinpath(ROOT, "cache", "floquet")
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
rate(v) = all(isfinite, v) && v[1] > 0 ? (v[end] / v[1])^(1 / (length(v) - 1)) : NaN
q(v, p) = (v = filter(isfinite, v); isempty(v) ? NaN : quantile(v, p))
rows = deserialize(joinpath(ROOT, "cache", "fig6", "bias.jls"))
sel(; kappa, method, N = nothing, noise = nothing) =
    [x.ρ for x in rows if x.kappa == kappa && x.method == method && (N === nothing || x.N == N) && (noise === nothing || x.noise == noise)]
METH = [("LS", "least squares", CAT[1], :circle), ("FB", "forward–backward", CAT[2], :rect),
        ("TLS", "total least squares", CAT[4], :diamond), ("IV", "instrumental variables", CAT[3], :utriangle)]

fig = Figure(size = (COL2, 120MM)); g = fig[1, 1] = GridLayout()

# ── A: seeds and start phases (coupled FHN) ──────────────────────────────────
MODELS = [("dmd", "Hankel-DMD", CAT[1]), ("esn", "ESN", CAT[2]), ("node", "neural ODE", CAT[3])]
axA = Axis(g[1, 1]; ylabel = "Relaxation rate per cycle", title = "Coupled FHN, same data: one fit, or one per seed",
           xticks = (1:3, [m[2] for m in MODELS]), limits = (0.4, 3.6, 0.0, 1.25))
tr = nothing
for (k, (m, _, col)) in enumerate(MODELS)
    fs = sort(filter(f -> startswith(f, "offv2_cfhn_$(m)_"), readdir(FDIR)))
    for (j, f) in enumerate(fs)
        o = deserialize(joinpath(FDIR, f)); global tr = o
        dx = length(fs) == 1 ? 0.0 : (j - (length(fs) + 1) / 2) * 0.16
        ps = rate.(o.all_own)
        scatter!(axA, fill(k + dx, length(ps)), min.(ps, 1.22); color = (col, 0.45), markersize = 3.5)
        scatter!(axA, [k + dx], [rate(o.own)]; color = col, markersize = 7, strokecolor = :white, strokewidth = 0.5)
    end
end
tp = rate.(tr.all_truth)
band!(axA, [0.4, 3.6], fill(minimum(tp), 2), fill(maximum(tp), 2); color = (C_TRUTH, 0.08))
hlines!(axA, rate(tr.truth); color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
text!(axA, 0.45, minimum(tp) - 0.02; text = @sprintf("truth %.2f\n(band: start phases)", rate(tr.truth)), fontsize = 6.5,
      align = (:left, :top))
text!(axA, 3.58, 1.24; text = "large: each fit · small: each start phase", fontsize = 6.5, color = C_REF, align = (:right, :top))
plabel!(g[1, 1], "A")

# ── B, C: against measurement noise ──────────────────────────────────────────
function noise_panel!(pos, κ, title; legend = false)
    NOISE = sort(unique([x.noise for x in rows]))
    truth = [x.truth for x in rows if x.kappa == κ][1]
    ax = Axis(pos; xlabel = "Measurement noise (× stride-to-stride SD)", ylabel = "Estimated spectral radius ρ̂",
              title, limits = (-0.03, 1.03, 0.35, 1.06))
    hlines!(ax, truth; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
    for (m, lab, col, mk) in METH
        vs = [sel(; kappa = κ, method = m, N = 2000, noise = r) for r in NOISE]
        band!(ax, NOISE, q.(vs, 0.25), q.(vs, 0.75); color = (col, 0.18))
        scatterlines!(ax, NOISE, q.(vs, 0.5); color = col, marker = mk, markersize = 4.5, linewidth = 1.3, label = lab)
    end
    text!(ax, 0.62, truth + 0.012; text = @sprintf("truth %.2f", truth), fontsize = 6.5, align = (:left, :bottom))
    legend && axislegend(ax; position = (:left, :bottom), framevisible = false, labelsize = 6.5, rowgap = 0)
    ax
end
noise_panel!(g[1, 2], 0.0, "SLIP, open loop: 2000 strides"; legend = true)
plabel!(g[1, 2], "B")
noise_panel!(g[2, 1], 0.1, "SLIP with speed feedback: 2000 strides")
plabel!(g[2, 1], "C")

# ── D: against record length ─────────────────────────────────────────────────
NS = sort(unique([x.N for x in rows]))
truth0 = [x.truth for x in rows if x.kappa == 0.0][1]
axD = Axis(g[2, 2]; xscale = log10, xticks = (NS, string.(NS)), xlabel = "Strides of data",
           ylabel = "Estimated spectral radius ρ̂", title = "Noise 0.25×: more data, same bias", limits = (25, 2400, 0.7, 1.06))
hlines!(axD, truth0; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
for (m, lab, col, mk) in METH
    vs = [sel(; kappa = 0.0, method = m, N, noise = 0.25) for N in NS]
    band!(axD, NS, q.(vs, 0.25), q.(vs, 0.75); color = (col, 0.18))
    scatterlines!(axD, NS, q.(vs, 0.5); color = col, marker = mk, markersize = 4.5, linewidth = 1.3)
end
plabel!(g[2, 2], "D")

colgap!(g, 16); rowgap!(g, 12)
save_paper(fig, "Fig6_linear_affordances")

for (m, _, _, _) in METH
    @printf("%-4s open loop N=2000: noise 0 %.3f .25 %.3f .5 %.3f 1 %.3f | feedback: %.3f %.3f %.3f %.3f\n", m,
            [median(sel(; kappa = 0.0, method = m, N = 2000, noise = r)) for r in (0.0, 0.25, 0.5, 1.0)]...,
            [median(sel(; kappa = 0.1, method = m, N = 2000, noise = r)) for r in (0.0, 0.25, 0.5, 1.0)]...)
end
