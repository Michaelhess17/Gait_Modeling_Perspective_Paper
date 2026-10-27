#!/usr/bin/env julia
# Figure 4 — Eigenvalues are not the whole story (OUTLINE.md §4 ¶4–6).
#
# Ground truth only (no fitting): scripts/systems_truth.jl (Van der Pol μ sweep, Stuart–Landau, coupled FHN;
# forward-integrated variational equations) and scripts/slip_truth.jl (SLIP single-step VLO map, open loop
# and with speed feedback κ = 0.1). All norms are in OBSERVED coordinates — each variable scaled by its SD
# along the cycle (SLIP: stationary SD of (y, θ) at VLO under the stride kicks of slip_truth.jl). Worst case
# over the phase at which the perturbation arrives.
#
#   (A) Transverse distance from the cycle after a perturbation, against time since the perturbation.
#       Dashed: the multiplier's prediction |μ|^τ.
#   (B) Peak within-cycle transverse amplification against the transverse multiplier. Shaded: multipliers
#       reported for human walking (≈0.3–0.8).
#   (C) Stride-to-stride: worst-case growth of the n-stride transverse map against the eigenvalue
#       prediction ρⁿ (dashed).
#   (D) Deviation from the UNPERTURBED trajectory after n cycles (includes lasting phase shifts). Every
#       oscillator has a unit multiplier, so the eigenvalue prediction is 1.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf, LinearAlgebra

const ROOT = dirname(@__DIR__)
st = deserialize(joinpath(ROOT, "cache", "truth", "systems_truth.jls"))
sl_ = deserialize(joinpath(ROOT, "cache", "truth", "slip_truth.jls"))
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)

# SLIP: stationary covariance of (y, θ) under kicks Q = diag(σ²) → observed-coordinate scaling
const KICK = [0.0005, 0.001]
function slip_scaled(t)
    J = t.J; Q = Diagonal(KICK .^ 2)
    Σ = reshape((I - kron(J, J)) \ vec(Matrix(Q)), 2, 2)
    D = Diagonal(sqrt.(diag(Σ))); Js = D \ J * D
    (; Js, ρ = t.ρ, growth = [opnorm(Js^n) for n in 0:12], κV = cond(eigvecs(Js)))
end
slip0 = slip_scaled(sl_["truth_0.0"]); slip1 = slip_scaled(sl_["truth_0.1"])

C_FHN = CAT[2]; C_VDP = CAT[1]; C_SL = C_REF; C_SLIP = CAT[3]
fig = Figure(size = (COL2, 128MM)); g = fig[1, 1] = GridLayout()

# ── A: within-cycle transverse amplification ────────────────────────────────
axA = Axis(g[1, 1]; yscale = log10, xlabel = "Time since perturbation (cycles)",
           ylabel = "Distance from cycle\n(relative to perturbation)", title = "A perturbation first grows",
           limits = (0, 2, 3e-3, 40))
for (key, lab, col, ls) in (("cfhn", "coupled FHN", C_FHN, :solid), ("vdp_0.2", "Van der Pol μ = 0.2", C_VDP, :solid),
                            ("vdp_1.0", "Van der Pol μ = 1", C_VDP, :dot), ("sl_0.05", "Stuart–Landau", C_SL, :solid))
    r = st[key]
    lines!(axA, r.τ, max.(r.curve, 3e-3); color = col, linestyle = ls, linewidth = 1.4,
           label = @sprintf("%s (|μ| = %s)", lab, r.ρ < 0.01 ? @sprintf("%.0e", r.ρ) : @sprintf("%.2f", r.ρ)))
end
lines!(axA, st["cfhn"].τ, st["cfhn"].ρ .^ st["cfhn"].τ; color = C_FHN, linestyle = :dash, linewidth = 0.9)
refline!(axA, 1.0; linestyle = :dot)
axislegend(axA; position = (:right, :bottom), framevisible = false, labelsize = 6.5, rowgap = 0)
text!(axA, 1.6, st["cfhn"].ρ^1.6 * 0.8; text = "|μ|^τ", color = C_FHN, fontsize = 7, align = (:center, :top))
plabel!(g[1, 1], "A")

# ── B: peak within-cycle amplification vs multiplier ────────────────────────
axB = Axis(g[1, 2]; xscale = log10, yscale = log10, xlabel = "Transverse multiplier |μ| (per cycle)",
           ylabel = "Peak within-cycle amplification", title = "Not predicted by |μ|",
           limits = (1e-8, 1.5, 0.8, 30))
vspan!(axB, 0.3, 0.8; color = C_SHADE)
text!(axB, 0.49, 4.5; text = "human\nwalking", fontsize = 6.5, align = (:center, :center), color = C_REF)
vk = sort([k for k in keys(st) if startswith(k, "vdp_")]; by = k -> st[k].p)
lines!(axB, [st[k].ρ for k in vk], [maximum(st[k].within) for k in vk]; color = C_VDP, linewidth = 1.2)
scatter!(axB, [st[k].ρ for k in vk], [maximum(st[k].within) for k in vk]; color = C_VDP, markersize = 5)
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

# ── C: stride-to-stride transverse map ───────────────────────────────────────
n = 0:8
axC = Axis(g[2, 1]; yscale = log10, xlabel = "Strides (cycles) n", ylabel = "Worst-case growth, n-stride map",
           title = "Stride-to-stride growth beyond ρⁿ", limits = (-0.2, 8.2, 0.05, 6))
fh = st["cfhn"]
lines!(axC, n, fh.trans[1:9]; color = C_FHN, linewidth = 1.4, label = @sprintf("coupled FHN (ρ = %.2f)", fh.ρ))
lines!(axC, n, fh.ρ .^ n; color = C_FHN, linestyle = :dash, linewidth = 0.9)
lines!(axC, n, slip0.growth[1:9]; color = C_SLIP, linewidth = 1.4, label = @sprintf("SLIP, open loop (ρ = %.2f)", slip0.ρ))
lines!(axC, n, slip0.ρ .^ n; color = C_SLIP, linestyle = :dash, linewidth = 0.9)
lines!(axC, n, slip1.growth[1:9]; color = C_SLIP, linestyle = :dot, linewidth = 1.4,
       label = @sprintf("SLIP, speed feedback (ρ = %.2f)", slip1.ρ))
lines!(axC, n, slip1.ρ .^ n; color = C_SLIP, linestyle = :dash, linewidth = 0.7)
refline!(axC, 1.0; linestyle = :dot)
axislegend(axC; position = (:left, :bottom), framevisible = false, labelsize = 6.5, rowgap = 0)
text!(axC, 8, 0.12; text = "dashed: ρⁿ", fontsize = 7, color = C_REF, align = (:right, :bottom))
plabel!(g[2, 1], "C")

# ── D: deviation from the unperturbed trajectory (phase shifts) ──────────────
axD = Axis(g[2, 2]; yscale = log10, xlabel = "Cycles n", ylabel = "Deviation from unperturbed\ntrajectory (worst case)",
           title = "Lasting phase shifts", limits = (-0.2, 8.2, 0.8, 30))
for (k, ls) in zip(("vdp_0.2", "vdp_0.5", "vdp_1.0", "vdp_2.0"), (:solid, :dash, :dot, :dashdot))
    lines!(axD, n, st[k].full[1:9]; color = C_VDP, linestyle = ls, linewidth = 1.2)
    text!(axD, 7.9, st[k].full[9] * 1.06; text = "Van der Pol μ = $(st[k].p)", color = C_VDP, fontsize = 6.5,
          align = (:right, :bottom))
end
lines!(axD, n, fh.full[1:9]; color = C_FHN, linewidth = 1.4)
text!(axD, 7.9, fh.full[9] * 0.95; text = "coupled FHN", color = C_FHN, fontsize = 6.5, align = (:right, :top))
lines!(axD, n, st["sl_0.05"].full[1:9]; color = C_SL, linewidth = 1.2)
text!(axD, 7.9, 0.95; text = "Stuart–Landau = eigenvalue prediction (unit multiplier)", fontsize = 6.5, color = C_SL,
      align = (:right, :top))
plabel!(g[2, 2], "D")

colgap!(g, 16); rowgap!(g, 12)
save_paper(fig, "Fig4_nonnormality")

@printf("SLIP open loop: ρ %.3f κV %.2f max ‖Jⁿ‖/ρⁿ %.2f | feedback: ρ %.3f κV %.2f max %.2f\n", slip0.ρ, slip0.κV,
        maximum(slip0.growth ./ slip0.ρ .^ (0:12)), slip1.ρ, slip1.κV, maximum(slip1.growth ./ slip1.ρ .^ (0:12)))
@printf("coupled FHN: max within %.1f, max trans/ρⁿ %.2f, max full %.2f\n", maximum(fh.within),
        maximum(filter(isfinite, fh.trans_ratio)), maximum(fh.full))
