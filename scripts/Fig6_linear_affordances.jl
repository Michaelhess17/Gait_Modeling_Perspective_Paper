#!/usr/bin/env julia
# Figure 6 — One convex answer: the linear fit has no seed, and on gait-like systems it is as accurate as the best
# nonlinear seeds (OUTLINE.md §6 ¶2).
#
#   Off-cycle relaxation of decay-trained models (scripts/offcycle_v2.jl; 10 decays × 3 cycles, the same data for
#   every model) from 4 off-cycle starts, ordered by the truth's rate (black ticks; dashed line joins them). Each
#   fit's rates from the same starts are joined by a thin line: Hankel-DMD is one least-squares fit (no seed), ESN
#   4 seeds and neural ODE 3 seeds at fixed tuned settings. |error| = mean over starts of |model − truth|, range
#   over seeds. Rate = (distance after 4 cycles / initial)^(1/4).
#   (A) Coupled FHN, per cycle. (B) SLIP, centre-of-mass (y, ẋ, ẏ), per step (scripts/slip_decays.jl; ρ 0.77).
#   (Stuart–Landau, dropped as too simple: DMD 0.88 vs truth 0.58 at 10 decays, ESN 0.46–0.48, NODE 0.49–0.54.)
# The SLIP section-map bias correction (instrumental variables) is in Fig 3B, C.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
const FDIR = joinpath(ROOT, "cache", "floquet")
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
rate(v) = all(isfinite, v) && v[1] > 0 ? (v[end] / v[1])^(1 / (length(v) - 1)) : NaN

fig = Figure(size = (COL2, 64MM)); g = fig[1, 1] = GridLayout()

# ── per start, per fit (A coupled FHN, B SLIP) ───────────────────────────────
MODELS = [("dmd", "Hankel-DMD\n(one fit)", CAT[1]), ("esn", "ESN\n(4 seeds)", CAT[2]), ("node", "neural ODE\n(3 seeds)", CAT[3])]
SYSTEMS = [("cfhn", "A", "Coupled FHN: DMD closest, seeds disagree", "per cycle", (0.0, 1.25)),
           ("slip", "B", "SLIP (centre of mass): DMD as close as the best seeds", "per step", (0.6, 1.0))]
for (c, (sys, L, title, per, (lo, hi))) in enumerate(SYSTEMS)
    ax = Axis(g[1, c]; ylabel = "Relaxation rate " * per, title, xticks = (1:3, [m[2] for m in MODELS]),
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
    plabel!(g[1, c], L)
end
colgap!(g, 16)
save_paper(fig, "Fig6_linear_affordances")
