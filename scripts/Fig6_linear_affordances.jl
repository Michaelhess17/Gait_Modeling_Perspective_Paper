#!/usr/bin/env julia
# Figure 6 — What a linear model makes possible: one answer per setting, and an estimator bias that can be
# analysed and removed (OUTLINE.md §6 ¶2, ¶4).
#
#   (A) Off-cycle relaxation of decay-trained models (scripts/offcycle_v2.jl; 10 decays × 3 cycles, the same data
#       for every model) from 4 off-cycle starts, ordered by the truth's rate (black ticks; dashed line joins them).
#       Each fit's rates from the same starts are joined by a thin line: Hankel-DMD is one least-squares fit (no
#       seed), ESN 4 seeds and neural ODE 3 seeds at fixed tuned settings. |error| = mean over starts of
#       |model − truth|, range over seeds. Left: coupled FHN (per cycle). Right: SLIP, centre-of-mass (y, ẋ, ẏ)
#       per step (scripts/slip_decays.jl; step-map ρ 0.77). Rate = (distance after 4 cycles / initial)^(1/4).
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

fig = Figure(size = (COL2, 125MM)); g = fig[1, 1] = GridLayout()

# ── A: per start, per fit (coupled FHN, SLIP) ────────────────────────────────
MODELS = [("dmd", "Hankel-DMD\n(one fit)", CAT[1]), ("esn", "ESN\n(4 seeds)", CAT[2]), ("node", "neural ODE\n(3 seeds)", CAT[3])]
SYSTEMS = [("cfhn", "Coupled FHN: one fit, or one answer per seed", "per cycle", (0.0, 1.25)),
           ("slip", "SLIP (centre of mass): same data and protocol", "per step", (0.6, 1.0))]
gA = g[1, 1:3] = GridLayout()
for (c, (sys, title, per, (lo, hi))) in enumerate(SYSTEMS)
    ax = Axis(gA[1, c]; ylabel = "Relaxation rate " * per, title, xticks = (1:3, [m[2] for m in MODELS]),
              limits = (0.45, 3.55, lo, hi))
    for (k, (m, _, col)) in enumerate(MODELS)
        os = [deserialize(joinpath(FDIR, f)) for f in sort(filter(f -> startswith(f, "offv2_$(sys)_$(m)_"), readdir(FDIR)))]
        tp = rate.(os[1].all_truth); ord = sortperm(tp); xs = k .+ ((1:4) .- 2.5) .* 0.2
        lines!(ax, xs, tp[ord]; color = C_TRUTH, linewidth = 0.6, linestyle = :dash)
        linesegments!(ax, vec([xs' .- 0.07; xs' .+ 0.07]), repeat(tp[ord], inner = 2); color = C_TRUTH, linewidth = 1.4)
        for (s, o) in enumerate(os)
            dx = length(os) == 1 ? 0.0 : (s - (length(os) + 1) / 2) * 0.025
            ps = clamp.(rate.(o.all_own)[ord], lo + 0.01, hi - 0.01)
            lines!(ax, xs .+ dx, ps; color = (col, 0.5), linewidth = 0.6)
            scatter!(ax, xs .+ dx, ps; color = col, markersize = 3.8, strokecolor = :white, strokewidth = 0.3)
        end
        e = [mean(abs.(rate.(o.all_own) .- tp)) for o in os]
        er = unique(round.(extrema(e), digits = 2))
        text!(ax, k, lo + 0.015 * (hi - lo); text = "|error| " * join([@sprintf("%.2f", x) for x in er], "–"),
              fontsize = 6.5, color = col, align = (:center, :bottom))
    end
    c == 1 && text!(ax, 0.5, hi - 0.01 * (hi - lo); text = "black: truth at 4 starts\n(fastest → slowest decay)\ndots: each fit, same starts",
                    fontsize = 6.5, color = C_REF, align = (:left, :top))
end
colgap!(gA, 16)
plabel!(gA, "A")

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
noise_panel!(g[2, 1], 0.0, "SLIP map, open loop: 2000 strides"; legend = true)
plabel!(g[2, 1], "B")
noise_panel!(g[2, 2], 0.1, "SLIP map, speed feedback")
plabel!(g[2, 2], "C")

# ── D: against record length ─────────────────────────────────────────────────
NS = sort(unique([x.N for x in rows]))
truth0 = [x.truth for x in rows if x.kappa == 0.0][1]
axD = Axis(g[2, 3]; xscale = log10, xticks = (NS, string.(NS)), xlabel = "Strides of data",
           ylabel = "Estimated spectral radius ρ̂", title = "Noise 0.25×: more data, same bias", limits = (25, 2400, 0.7, 1.06))
hlines!(axD, truth0; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
for (m, lab, col, mk) in METH
    vs = [sel(; kappa = 0.0, method = m, N, noise = 0.25) for N in NS]
    band!(axD, NS, q.(vs, 0.25), q.(vs, 0.75); color = (col, 0.18))
    scatterlines!(axD, NS, q.(vs, 0.5); color = col, marker = mk, markersize = 4.5, linewidth = 1.3)
end
plabel!(g[2, 3], "D")

colgap!(g, 14); rowgap!(g, 12); rowsize!(g, 1, Relative(0.47))
save_paper(fig, "Fig6_linear_affordances")

for (m, _, _, _) in METH
    @printf("%-4s open loop N=2000: noise 0 %.3f .25 %.3f .5 %.3f 1 %.3f | feedback: %.3f %.3f %.3f %.3f\n", m,
            [median(sel(; kappa = 0.0, method = m, N = 2000, noise = r)) for r in (0.0, 0.25, 0.5, 1.0)]...,
            [median(sel(; kappa = 0.1, method = m, N = 2000, noise = r)) for r in (0.0, 0.25, 0.5, 1.0)]...)
end
