#!/usr/bin/env julia
# Figure_S01.jl — Ground-truth oscillators (manuscript: @suppfig-oscillators). Run: julia --project=. scripts/figures/Figure_S01.jl
# Figure S8 — the oscillator family behind Fig 1B–C (formerly panels A, B, D of Fig4_nonnormality.jl).
# Ground truth only: scripts/systems_truth.jl (Van der Pol μ sweep, Stuart–Landau, coupled FHN; forward-integrated
# variational equations). Norms in OBSERVED coordinates (each variable scaled by its SD along the cycle), worst case
# over the phase at which the perturbation arrives.
#   (A) Transverse distance from the cycle after a perturbation; dashed, the coupled-FHN multiplier's |μ|^τ.
#   (B) Peak within-cycle transverse amplification against the transverse multiplier; shaded, human walking.
#   (C) Deviation from the UNPERTURBED trajectory after n cycles, including lasting phase shifts (eigenvalue
#       prediction 1: the multiplier along the cycle is 1).

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
st = deserialize(joinpath(ROOT, "cache", "truth", "systems_truth.jls"))
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
C_FHN = CAT[2]; C_VDP = CAT[1]; C_SL = C_REF

fig = Figure(size = (COL2, 62MM)); g = fig[1, 1] = GridLayout()

# ── A ────────────────────────────────────────────────────────────────────────
axA = Axis(g[1, 1]; yscale = log10, xlabel = "Time since perturbation (cycles)",
           ylabel = "Distance from cycle\n(relative to perturbation)", title = "Amplification within the cycle",
           limits = (0, 2, 1e-3, 40))
for (key, lab, col, ls) in (("cfhn", "coupled FHN", C_FHN, :solid), ("vdp_0.2", "Van der Pol μ = 0.2", C_VDP, :solid),
                            ("vdp_1.0", "Van der Pol μ = 1", C_VDP, :dot), ("sl_0.05", "Stuart–Landau", C_SL, :solid))
    r = st[key]
    lines!(axA, r.τ, max.(r.curve, 5e-4); color = col, linestyle = ls, linewidth = 1.3,
           label = @sprintf("%s (|μ| = %s)", lab, r.ρ < 0.01 ? @sprintf("%.0e", r.ρ) : @sprintf("%.2f", r.ρ)))
end
lines!(axA, st["cfhn"].τ, st["cfhn"].ρ .^ st["cfhn"].τ; color = C_FHN, linestyle = :dash, linewidth = 0.9)
refline!(axA, 1.0; linestyle = :dot)
axislegend(axA; position = (:right, :bottom), framevisible = false, labelsize = 6.5, rowgap = 0)
plabel!(g[1, 1], "A")

# ── B ────────────────────────────────────────────────────────────────────────
axB = Axis(g[1, 2]; xscale = log10, yscale = log10, xlabel = "Transverse multiplier |μ| (per cycle)",
           ylabel = "Peak within-cycle amplification", title = "Not predicted by |μ|", limits = (1e-8, 1.5, 0.8, 30))
vspan!(axB, 0.3, 0.8; color = C_SHADE)
text!(axB, 0.27, 4.5; text = "human\nwalking", fontsize = 6.5, align = (:right, :center), color = C_REF)
vk = sort([k for k in keys(st) if startswith(k, "vdp_")]; by = k -> st[k].p)
scatterlines!(axB, [st[k].ρ for k in vk], [maximum(st[k].within) for k in vk]; color = C_VDP, linewidth = 1.2,
              markersize = 5)
text!(axB, st[vk[end]].ρ * 1.5, maximum(st[vk[end]].within) * 1.12; text = "μ = $(st[vk[end]].p)", fontsize = 6.5,
      color = C_VDP, align = (:left, :bottom))
text!(axB, st[vk[1]].ρ * 0.9, maximum(st[vk[1]].within) * 1.35; text = "μ = $(st[vk[1]].p)", fontsize = 6.5,
      color = C_VDP, align = (:right, :bottom))
text!(axB, 1e-5, 2.0; text = "Van der Pol", color = C_VDP, fontsize = 7, align = (:center, :bottom))
slk = [k for k in keys(st) if startswith(k, "sl_")]
scatter!(axB, [st[k].ρ for k in slk], [maximum(st[k].within) for k in slk]; color = C_SL, markersize = 5)
text!(axB, 0.05, 0.93; text = "Stuart–Landau", color = C_SL, fontsize = 7, align = (:right, :center))
scatter!(axB, [st["cfhn"].ρ], [maximum(st["cfhn"].within)]; color = C_FHN, marker = :star5, markersize = 10)
text!(axB, st["cfhn"].ρ * 0.6, maximum(st["cfhn"].within); text = "coupled FHN", color = C_FHN, fontsize = 7,
      align = (:right, :center))
plabel!(g[1, 2], "B")

# ── C ────────────────────────────────────────────────────────────────────────
n = 0:8; fh = st["cfhn"]
axC = Axis(g[1, 3]; yscale = log10, xlabel = "Cycles n", ylabel = "Deviation from unperturbed\ntrajectory (worst case)",
           title = "Lasting phase shifts", limits = (-0.2, 8.2, 0.8, 30))
for (k, ls) in zip(("vdp_0.2", "vdp_0.5", "vdp_1.0", "vdp_2.0"), (:solid, :dash, :dot, :dashdot))
    lines!(axC, n, st[k].full[1:9]; color = C_VDP, linestyle = ls, linewidth = 1.2)
    text!(axC, 7.9, st[k].full[9] * 1.06; text = "Van der Pol μ = $(st[k].p)", color = C_VDP, fontsize = 6.5,
          align = (:right, :bottom))
end
lines!(axC, n, fh.full[1:9]; color = C_FHN, linewidth = 1.4)
text!(axC, 7.9, fh.full[9] * 0.95; text = "coupled FHN", color = C_FHN, fontsize = 6.5, align = (:right, :top))
lines!(axC, n, st["sl_0.05"].full[1:9]; color = C_SL, linewidth = 1.2)
text!(axC, 7.9, 0.95; text = "Stuart–Landau (unit multiplier)", fontsize = 6.5, color = C_SL, align = (:right, :top))
plabel!(g[1, 3], "C")

colgap!(g, 16)
save_paper(fig, "Figure_S01")
