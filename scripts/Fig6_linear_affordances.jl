#!/usr/bin/env julia
# Figure 5 (file name from the earlier numbering) — Off-cycle relaxation and the multipliers of each model's own cycle
# (OUTLINE.md §4 ¶2–¶4). Data: scripts/fig6_relaxation_data.jl (cache/fig6/relaxation.jls).
#
#   (A, B) Return to the true cycle from real perturbed states: decay-trained models (10 decays × 3 cycles, the same
#   data for every model), warmed up on the true perturbed trajectory and run freely. Deviation = distance to the curve
#   through the true cycle's samples, RMS over each cycle after the warm-up, for model and truth alike; median over 4
#   starts (kicked at 4 phases of the cycle). Black: truth. One line per fit: Hankel-DMD is one least-squares fit (no
#   seed), ESN 4 seeds, neural ODE 3 seeds at fixed tuned settings. Triangles at the right: distance of each fit's settled
#   cycle from the true cycle, the floor its deviation cannot go below. (A) Coupled FHN, per cycle: every model's cycle
#   is displaced by as much as the perturbation, so its runs stall. (B) SLIP, centre-of-mass (y, ẋ, ẏ), per step.
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

# ── A, B: distance to the true cycle after a real perturbation ───────────────
MODELS = [("dmd", "Hankel-DMD", CAT[1]), ("esn", "ESN seeds", CAT[2]), ("node", "neural-ODE seeds", CAT[3])]
SYSTEMS = [("cfhn", "A", "Coupled FHN: the models stall", "Cycles after the perturbation", (0.018, 0.7),
            [0.02, 0.05, 0.1, 0.2, 0.5]),
           ("slip", "B", "SLIP: all models track the return", "Steps after the perturbation", (0.07, 1.2), [0.1, 0.2, 0.5, 1.0])]
medenv(Es) = [median(filter(isfinite, [e[k] for e in Es])) for k in 1:5]
for (c, (sys, L, title, xlab, ylims, yt)) in enumerate(SYSTEMS)
    r = D.relax[sys]
    ax = Axis(g[1, c]; xlabel = xlab, ylabel = c == 1 ? "Distance to the true cycle (SD units)" : "", title, yscale = log10,
              xticks = 0:4, yticks = (yt, string.(yt)), limits = (-0.25, 4.7, ylims...))
    for (m, _, col) in MODELS, x in [x for x in r.rows if x.model == m]
        lines!(ax, 0:4, medenv(x.envs); color = (col, m == "dmd" ? 1.0 : 0.75), linewidth = m == "dmd" ? 1.6 : 0.9)
        scatter!(ax, [4.42], [median(filter(isfinite, x.offset))]; marker = :ltriangle, color = col, markersize = 6)
    end
    te = medenv(r.truth_envs)
    lines!(ax, 0:4, te; color = C_TRUTH, linewidth = 1.8); scatter!(ax, 0:4, te; color = C_TRUTH, markersize = 4)
    offs = reduce(vcat, [filter(isfinite, x.offset) for x in r.rows])
    text!(ax, 4.6, ylims[2] / 1.06; text = @sprintf("perturbation %.2f–%.2f\nmodel cycle offsets %.2f–%.2f", extrema(r.dev0)..., extrema(offs)...),
          fontsize = 6.5, color = C_REF, align = (:right, :top))
    c == 1 && axislegend(ax, vcat([LineElement(color = C_TRUTH, linewidth = 1.8)], [LineElement(color = col, linewidth = 1.2) for (_, _, col) in MODELS],
                                  [MarkerElement(marker = :ltriangle, color = C_REF, markersize = 6)]),
                         vcat(["truth"], [lab for (_, lab, _) in MODELS], ["settled offset"]);
                         position = :lb, framevisible = false, labelsize = 6.5, padding = (2, 2, 2, 2), rowgap = 0)
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
