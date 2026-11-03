#!/usr/bin/env julia
# Figure 5 (file name from the earlier numbering) — Off-cycle relaxation and the multipliers of each model's own cycle
# (OUTLINE.md §4 ¶2–¶4). Data: scripts/fig6_relaxation_data.jl (cache/fig6/relaxation.jls).
#
#   (A, B) Relaxation to the true cycle from real perturbed states: decay-trained models (10 decays × 3 cycles, the same
#   data for every model), warmed up on the true perturbed trajectory and run freely, from 4 off-cycle starts ordered by
#   the truth's rate (black ticks). Rate = (deviation over the 5th cycle / over the 1st)^(1/4), deviation = distance to the
#   curve through the true cycle's samples, for model and truth alike. Each fit's rates from the same starts are joined by
#   a thin line: Hankel-DMD is one least-squares fit (no seed), ESN 4 seeds and neural ODE 3 seeds at fixed tuned settings.
#   |error| = mean over starts of |model − truth|, range over seeds. Offset: distance of each model's settled cycle from
#   the true cycle (median), the floor its deviation cannot go below; against the truth's initial deviation (text).
#   (A) Coupled FHN, per cycle. (B) SLIP, centre-of-mass (y, ẋ, ẏ), per step.
#   (C) SLIP: multipliers of each model's own cycle, as rotation per step against modulus per step. DMD: eigenvalues
#   of its operator in the cycle frame (rotation from the nearest harmonic); the one next to the 8th harmonic (square)
#   matches the exact multiplier (star). ESN: eigenvalues of the monodromy of its own settled cycle (closed-loop Jacobians chained
#   over one period; phase eigenvalue removed), the leading pair per seed. Neural ODE: needs a refit to linearise.
#   Diamond: the lattice-constrained linear fit (Fig S6).
# The SLIP section-map bias correction (instrumental variables) is in Fig 2B, C (file Fig3_floquet_maps.jl).

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const D = deserialize(joinpath(ROOT, "cache", "fig6", "relaxation.jls"))

fig = Figure(size = (COL2, 70MM)); g = fig[1, 1] = GridLayout()

# ── A, B: per start, per fit ─────────────────────────────────────────────────
MODELS = [("dmd", "Hankel-DMD\n(one fit)", CAT[1]), ("esn", "ESN\n(4 seeds)", CAT[2]), ("node", "neural ODE\n(3 seeds)", CAT[3])]
SYSTEMS = [("cfhn", "A", "Coupled FHN: runs stall on displaced cycles", "per cycle", (0.3, 1.2), 0.4:0.1:1.0),
           ("slip", "B", "SLIP: all models track the return", "per step", (0.42, 1.06), 0.5:0.1:1.0)]
for (c, (sys, L, title, per, (lo, hi), yt)) in enumerate(SYSTEMS)
    r = D.relax[sys]; ord = sortperm(collect(zip(round.(r.truth, digits = 2), 1:4))); tp = r.truth[ord]
    ax = Axis(g[1, c]; ylabel = "Relaxation rate " * per * " (first 4)", title, xticks = (1:3, [m[2] for m in MODELS]),
              yticks = yt, limits = (0.45, 3.55, lo, hi))
    for (k, (m, _, col)) in enumerate(MODELS)
        rows = [x for x in r.rows if x.model == m]; xs = k .+ ((1:4) .- 2.5) .* 0.2
        linesegments!(ax, vec([xs' .- 0.07; xs' .+ 0.07]), repeat(tp, inner = 2); color = C_TRUTH, linewidth = 1.4)
        for (s, x) in enumerate(rows)
            dx = length(rows) == 1 ? 0.0 : (s - (length(rows) + 1) / 2) * 0.025
            ps = clamp.(x.rates[ord], lo + 0.01, hi - 0.01)
            lines!(ax, xs .+ dx, ps; color = (col, 0.5), linewidth = 0.6)
            scatter!(ax, xs .+ dx, ps; color = col, markersize = 3.8, strokecolor = :white, strokewidth = 0.3)
        end
        e = [mean(filter(isfinite, abs.(x.rates[ord] .- tp))) for x in rows]
        er = unique(round.(extrema(e), digits = 2))
        offs = median(reduce(vcat, [filter(isfinite, x.offset) for x in rows]))
        text!(ax, k, lo + 0.015 * (hi - lo); text = "|error|\n" * join([@sprintf("%.2f", v) for v in er], "–") *
              @sprintf("\noffset %.2f", offs), fontsize = 6.5, color = col, align = (:center, :bottom))
    end
    text!(ax, 3.52, hi - 0.01 * (hi - lo); text = (c == 1 ? "black: truth, 4 starts, fastest → slowest\n" : "") *
          @sprintf("truth's initial deviation %.2f–%.2f", extrema(r.dev0)...), fontsize = 6.5, color = C_REF, align = (:right, :top))
    plabel!(g[1, c], L)
end

# ── C: multipliers of each model's own cycle (SLIP) ─────────────────────────
axC = Axis(g[1, 3]; xlabel = "Rotation per step (°)", ylabel = "|λ| per step", title = "SLIP: each model's own cycle",
           limits = (-1.5, 26, 0.3, 1.04), xticks = 0:5:25)
hlines!(axC, 1.0; color = C_REF, linestyle = :dot, linewidth = 0.8)
cf = [x for x in D.dmd_cycle_frame if x.ρ > 0.3]
scatter!(axC, abs.([x.δ for x in cf]), [x.ρ for x in cf]; color = (CAT[1], 0.55), markersize = 4)
s8 = cf[argmin([abs(abs(x.δ) - 13.6) + 50abs(x.ρ - 0.769) for x in cf])]
text!(axC, abs(s8.δ) + 1.0, s8.ρ + 0.035; text = "DMD, next to\nharmonic $(s8.n)", fontsize = 6.5, color = CAT[1], align = (:left, :bottom))
text!(axC, 4.5, 1.004; text = "DMD harmonics", fontsize = 6.5, color = CAT[1], align = (:left, :bottom))
for e in D.esn
    e === nothing && continue
    z = e.ev[1]; zs = abs(imag(z)) > 1e-6 ? [z] : [z]
    scatter!(axC, rad2deg.(abs.(angle.(zs))), abs.(zs); marker = :utriangle, color = CAT[2], markersize = 6, strokecolor = :white, strokewidth = 0.4)
end
text!(axC, 25.5, 0.36; text = "ESN cycles\n(leading, 4 seeds)", fontsize = 6.5, color = CAT[2], align = (:right, :bottom))
text!(axC, 25.5, 0.55; text = "neural ODE:\nrefit needed", fontsize = 6.5, color = C_REF, align = (:right, :bottom))
λx = D.exact[argmax(imag.(D.exact))]
scatter!(axC, [rad2deg(angle(λx))], [abs(λx)]; marker = :star5, color = CAT[4], markersize = 14, strokecolor = C_TRUTH, strokewidth = 0.5)
scatter!(axC, [abs(s8.δ)], [s8.ρ]; marker = :rect, color = CAT[1], markersize = 6, strokecolor = :white, strokewidth = 0.5)
text!(axC, rad2deg(angle(λx)) - 0.8, abs(λx) - 0.03; text = "exact", fontsize = 6.5, align = (:right, :top))
scatter!(axC, [D.lattice[2]], [D.lattice[1]]; marker = :diamond, color = CAT[5], markersize = 6, strokecolor = :white, strokewidth = 0.4)
text!(axC, D.lattice[2] + 1.2, D.lattice[1] - 0.025; text = "lattice fit\n(Fig S6)", fontsize = 6.5, color = CAT[5], align = (:left, :top))
plabel!(g[1, 3], "C")

colsize!(g, 3, Relative(0.3)); colgap!(g, 14)
save_paper(fig, "Fig6_linear_affordances")
