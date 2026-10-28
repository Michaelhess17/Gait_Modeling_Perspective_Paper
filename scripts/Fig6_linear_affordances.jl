#!/usr/bin/env julia
# Figure 6 — One convex answer: the linear fit has no seed, and over the first few cycles it is as accurate as the
# best nonlinear seeds; but no model recovers the SLIP step map's rotation (OUTLINE.md §6 ¶2, ¶7).
#
#   (A, B) Off-cycle relaxation of decay-trained models (scripts/offcycle_v2.jl; 10 decays × 3 cycles, the same
#   data for every model) from 4 off-cycle starts, ordered by the truth's rate (black ticks; dashed line joins
#   them). Each fit's rates from the same starts are joined by a thin line: Hankel-DMD is one least-squares fit
#   (no seed), ESN 4 seeds and neural ODE 3 seeds at fixed tuned settings. |error| = mean over starts of
#   |model − truth|, range over seeds. Rate = (distance after 4 cycles / initial)^(1/4).
#   (A) Coupled FHN, per cycle. (B) SLIP, centre-of-mass (y, ẋ, ẏ), per step (scripts/slip_decays.jl; ρ 0.77).
#   (C) SLIP step multiplier read from the same runs (scripts/slip_rotation.jl): section-map eigenvalue of largest
#   modulus, as modulus against rotation per step. Small: each of 4 sections; large: median. Star: exact
#   (0.769 ∠ 13.6°); open circle: true runs read the same way. Every fit has real eigenvalues (0°); fits are
#   dodged horizontally by model. DMD sits at ≈ 1: a linear model's cycle amplitude is neutral.
#   (Stuart–Landau, dropped as too simple: DMD 0.88 vs truth 0.58 at 10 decays, ESN 0.46–0.48, NODE 0.49–0.54.)
# The SLIP section-map bias correction (instrumental variables) is in Fig 3B, C.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
const FDIR = joinpath(ROOT, "cache", "floquet")
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
rate(v) = all(isfinite, v) && v[1] > 0 ? (v[end] / v[1])^(1 / (length(v) - 1)) : NaN

fig = Figure(size = (COL2, 68MM)); g = fig[1, 1] = GridLayout()

# ── A, B: per start, per fit ─────────────────────────────────────────────────
MODELS = [("dmd", "Hankel-DMD\n(one fit)", CAT[1]), ("esn", "ESN\n(4 seeds)", CAT[2]), ("node", "neural ODE\n(3 seeds)", CAT[3])]
SYSTEMS = [("cfhn", "A", "Coupled FHN: DMD closest", "per cycle", (-0.1, 1.42), 0:0.25:1.25),
           ("slip", "B", "SLIP: DMD as close as the best seeds", "per step", (0.56, 1.0), 0.6:0.1:1.0)]
for (c, (sys, L, title, per, (lo, hi), yt)) in enumerate(SYSTEMS)
    ax = Axis(g[1, c]; ylabel = "Relaxation rate " * per * " (first 4)", title, xticks = (1:3, [m[2] for m in MODELS]),
              yticks = yt, limits = (0.45, 3.55, lo, hi))
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
        text!(ax, k, lo + 0.015 * (hi - lo); text = "|error|\n" * join([@sprintf("%.2f", x) for x in er], "–"),
              fontsize = 6.5, color = col, align = (:center, :bottom))
    end
    c == 1 && text!(ax, 0.5, hi - 0.01 * (hi - lo); text = "black: truth at 4 starts, fastest → slowest\ndots: each fit, same starts",
                    fontsize = 6.5, color = C_REF, align = (:left, :top))
    plabel!(g[1, c], L)
end

# ── C: SLIP step multiplier, modulus and rotation ────────────────────────────
R = deserialize(joinpath(ROOT, "cache", "fig6", "slip_rotation.jls"))
axC = Axis(g[1, 3]; xlabel = "Rotation per step (°)", ylabel = "|λ| per step", title = "SLIP: no model rotates",
           limits = (-3.5, 16.5, 0.7, 1.08), xticks = 0:5:15)
hlines!(axC, 1.0; color = C_REF, linestyle = :dot, linewidth = 0.8)
text!(axC, 16.3, 1.005; text = "neutral", fontsize = 6.5, color = C_REF, align = (:right, :bottom))
λx = R.exact[argmax(imag.(R.exact))]
scatter!(axC, [rad2deg(angle(λx))], [abs(λx)]; marker = :star5, color = CAT[4], markersize = 12, strokecolor = C_TRUTH, strokewidth = 0.5)
text!(axC, rad2deg(angle(λx)), abs(λx) - 0.014; text = "exact", fontsize = 6.5, align = (:center, :top))
tr = only([r for r in R.rows if r.model == "truth"])
scatter!(axC, rad2deg.(abs.(angle.(tr.ev))), abs.(tr.ev); color = (C_TRUTH, 0.5), markersize = 3)
scatter!(axC, [tr.ang], [tr.mod]; color = :white, strokecolor = C_TRUTH, strokewidth = 1.0, markersize = 7)
text!(axC, 16.2, tr.mod + 0.016; text = "truth, read\nthe same way", fontsize = 6.5, align = (:right, :bottom))
for (k, (m, _, col)) in enumerate(MODELS)
    rs = [r for r in R.rows if r.model == m]
    for (s, r) in enumerate(rs)
        x = (k - 2) * 1.1 + (length(rs) == 1 ? 0.0 : (s - (length(rs) + 1) / 2) * 0.3)
        scatter!(axC, fill(x, length(r.ev)) .+ rad2deg.(abs.(angle.(r.ev))), abs.(r.ev); color = (col, 0.45), markersize = 3)
        scatter!(axC, [x + r.ang], [r.mod]; color = col, markersize = 6, strokecolor = :white, strokewidth = 0.5)
    end
end
text!(axC, 2.6, 0.975; text = "every fit: real\neigenvalues, 0°\n(dodged by model)", fontsize = 6.5, color = C_REF, align = (:left, :top))
plabel!(g[1, 3], "C")

colsize!(g, 3, Relative(0.24)); colgap!(g, 14)
save_paper(fig, "Fig6_linear_affordances")
