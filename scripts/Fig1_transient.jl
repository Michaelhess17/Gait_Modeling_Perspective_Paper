#!/usr/bin/env julia
# Fig 1 — Walking lives in the transient, and the transient is not the multiplier. Ground truth only.
# Data: scripts/fig1_transient_data.jl → cache/fig1/transient.jls; cache/truth/{systems,slip}_truth.jls.
#   (A) SLIP with speed feedback, kicked at every step: CoM height, and its deviation at each VLO as the sum of the
#       responses to every past kick (step map linearized at the gait, so the sum is exact).
#   (B) Within a stride: coupled-FHN distance from the cycle after one disturbance, typical (RMS over directions and
#       phases) and worst case, against the multiplier's prediction ρ^τ.
#   (C) Stride to stride: worst-case growth of the n-stride transverse map against ρⁿ (coupled FHN, SLIP).
#   (D) Share of the observed deviation contributed by disturbances younger than a given age.
# The oscillator family (Van der Pol, Stuart–Landau) and lasting phase shifts are in FigS8_oscillators.jl.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf, LinearAlgebra

const ROOT = dirname(@__DIR__)
d = deserialize(joinpath(ROOT, "cache", "fig1", "transient.jls"))
st = deserialize(joinpath(ROOT, "cache", "truth", "systems_truth.jls"))
sl_ = deserialize(joinpath(ROOT, "cache", "truth", "slip_truth.jls"))
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
C_FHN = CAT[2]; C_SLIP = CAT[3]

# SLIP step map in observed coordinates (stationary SD of (y, θ) under the kicks), as in Fig4_nonnormality.jl
const KICK = [0.0005, 0.001]
function slip_scaled(t)
    J = t.J; Q = Diagonal(KICK .^ 2)
    Σ = reshape((I - kron(J, J)) \ vec(Matrix(Q)), 2, 2)
    D = Diagonal(sqrt.(diag(Σ))); Js = D \ J * D
    (; ρ = t.ρ, growth = [opnorm(Js^n) for n in 0:16])
end
slip0 = slip_scaled(sl_["truth_0.0"]); slip1 = slip_scaled(sl_["truth_0.1"])

fig = Figure(size = (COL2, 118MM))
gA = fig[1, 1] = GridLayout(); gB = fig[1, 2] = GridLayout()
gC = fig[2, 1] = GridLayout(); gD = fig[2, 2] = GridLayout()
colgap!(fig.layout, 16); rowgap!(fig.layout, 10)

# ── A: a disturbance at every step ──────────────────────────────────────────
w = d.walk; NW = length(w.dev_obs) - 1
raw = w.raw; tv = raw[w.vlo_idx, 1]; Tstep = mean(diff(tv))
NA = min(NW, length(tv) - 1)                                  # steps with a continuous record
axA0 = Axis(gA[1, 1]; title = "Disturbed at every step, never back on the cycle",
            yticksvisible = false, yticklabelsvisible = false, xticksvisible = false, xticklabelsvisible = false,
            bottomspinevisible = false)
keep = raw[:, 1] .<= tv[NA+1]
lines!(axA0, raw[keep, 1] ./ Tstep, raw[keep, 2]; color = C_TRUTH, linewidth = 0.9)
ylo, yhi = extrema(raw[keep, 2]); pad = 0.25 * (yhi - ylo)
scatter!(axA0, tv[1:NA] ./ Tstep, raw[w.vlo_idx[1:NA], 2] .+ pad; marker = :dtriangle, markersize = 4.5, color = C_SLIP)
text!(axA0, tv[2] / Tstep, yhi + 1.15pad; text = "SLIP walking (CoM height): a disturbance at each step", fontsize = 6.5,
      color = C_TRUTH, align = (:left, :bottom))
ylims!(axA0, ylo - 0.1pad, yhi + 2.6pad)

axA = Axis(gA[2, 1]; xlabel = "Step", ylabel = "Height deviation at\nmid-stance (SD)")
refline!(axA, 0.0; linestyle = :solid)
ks = 0:NA
HL = 3                                                       # the highlighted kick
for j in 1:NA
    j == HL && continue
    lines!(axA, ks[j+1:end], w.resp[j, j+1:NA+1]; color = (C_REF, 0.55), linewidth = 0.6)
end
lines!(axA, ks[HL+1:end], w.resp[HL, HL+1:NA+1]; color = C_SLIP, linewidth = 1.3)
dev_lin = [sum(filter(!isnan, w.resp[1:NA, k]); init = 0.0) for k in 1:NA+1]
scatterlines!(axA, ks, dev_lin; color = C_TRUTH, linewidth = 1.2, markersize = 3.5)
text!(axA, HL + 0.3, w.resp[HL, HL+1]; text = "one disturbance's\nresponse", color = C_SLIP, fontsize = 6.5,
      align = (:left, :center), offset = (3, 0))
text!(axA, 0.1, 1.35; text = "observed:\nsum of all\nresponses", color = C_TRUTH, fontsize = 6.5,
      align = (:left, :top))
ylims!(axA, -2.15, 1.4)
linkxaxes!(axA0, axA); xlims!(axA, -0.3, NA + 0.3)
rowsize!(gA, 1, Relative(0.25)); rowgap!(gA, 2)
plabel!(gA[1, 1], "A")

# ── B: within a stride ──────────────────────────────────────────────────────
c = d.cfhn
axB = Axis(gB[1, 1]; yscale = log10, xlabel = "Time since the disturbance (cycles)",
           ylabel = "Distance from the cycle\n(relative to the disturbance)",
           title = "Within a stride: growth the multiplier misses",
           limits = (0, 4, 0.08, 40), yticks = ([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"]))
vspan!(axB, 0, 1; color = C_SHADE)
keepc = c.age .<= 4
lines!(axB, c.age[keepc], c.ρ .^ c.age[keepc]; color = C_REF, linestyle = :dash, linewidth = 1.0)
refline!(axB, 1.0; linestyle = :dot)
lines!(axB, c.worst_τ, c.worst; color = (C_FHN, 0.55), linewidth = 0.9,
       label = @sprintf("worst phase and direction (peak %.0f×)", maximum(c.worst)))
lines!(axB, c.age[keepc], c.typical[keepc]; color = C_FHN, linewidth = 1.5,
       label = @sprintf("typical (peak %.1f×)", maximum(c.typical)))
lines!(axB, [NaN], [NaN]; color = C_REF, linestyle = :dash, linewidth = 1.0,
       label = @sprintf("multiplier, ×%.2f per cycle", c.ρ))
axislegend(axB, "coupled FHN"; position = (:right, :top), labelsize = 6.5, titlesize = 7, titlefont = :regular,
           titlecolor = C_FHN, titlehalign = :left, patchsize = (12, 5))
plabel!(gB[1, 1], "B")

# ── C: stride to stride ─────────────────────────────────────────────────────
n = 0:8; fh = st["cfhn"]
axC = Axis(gC[1, 1]; yscale = log10, xlabel = "Strides since the disturbance",
           ylabel = "Worst-case growth over\nn strides", title = "Stride to stride: growth beyond ρⁿ",
           limits = (-0.2, 8.2, 0.03, 4), yticks = ([0.03, 0.1, 0.3, 1, 3], ["0.03", "0.1", "0.3", "1", "3"]))
vspan!(axC, 0, 1; color = C_SHADE)
refline!(axC, 1.0; linestyle = :dot)
lines!(axC, n, fh.ρ .^ n; color = C_FHN, linestyle = :dash, linewidth = 0.9)
lines!(axC, n, fh.trans[1:9]; color = C_FHN, linewidth = 1.5, label = @sprintf("coupled FHN, ρ = %.2f", fh.ρ))
ns = 0:2:16                                                  # SLIP: two steps per stride
lines!(axC, n, slip1.ρ .^ ns; color = C_SLIP, linestyle = :dash, linewidth = 0.9)
lines!(axC, n, slip1.growth[ns .+ 1]; color = C_SLIP, linewidth = 1.5,
       label = @sprintf("SLIP, feedback, ρ = %.2f", slip1.ρ^2))
axislegend(axC; position = (:left, :bottom), labelsize = 6.5, patchsize = (12, 5))
text!(axC, 8.1, 1.25; text = "dashed: multiplier, ρⁿ", fontsize = 6.5, color = C_REF, align = (:right, :bottom))
plabel!(gC[1, 1], "C")

# ── D: how old is what we observe? ──────────────────────────────────────────
XMAX = 6
axD = Axis(gD[1, 1]; xlabel = "Age of the disturbance (strides)",
           ylabel = "Share of the observed deviation\nfrom younger disturbances",
           title = "What we observe is mostly young", limits = (0, XMAX, 0, 1.02),
           yticks = ([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"]))
vspan!(axD, 0, 1; color = C_SHADE)
h = d.human
band!(axD, h.age, h.lo, h.hi; color = (C_REF, 0.22), label = "human, ρ 0.3–0.8")
lines!(axD, h.age, h.lo; color = C_REF, linewidth = 0.6); lines!(axD, h.age, h.hi; color = C_REF, linewidth = 0.6)
lines!(axD, vcat(0, c.age), vcat(0, c.share); color = C_FHN, linewidth = 1.5, label = "coupled FHN")
for (s_, ls, lab) in ((d.slip.fb, :solid, "SLIP, feedback"), (d.slip.open, :dot, "SLIP, open loop"))
    lines!(axD, vcat(0, s_.age), vcat(0, s_.share); color = C_SLIP, linestyle = ls, linewidth = 1.4, label = lab)
end
i1 = findfirst(>=(1.0), c.age)
scatter!(axD, [1.0], [c.share[i1]]; color = C_FHN, markersize = 4)
text!(axD, 1.0, c.share[i1]; text = @sprintf("%.0f%%", 100c.share[i1]), color = C_FHN, fontsize = 6.5,
      align = (:right, :bottom), offset = (-2, 1))
axislegend(axD; position = (:right, :bottom), labelsize = 6.5, patchsize = (10, 5))
plabel!(gD[1, 1], "D")

save_paper(fig, "Fig1_transient")
@printf("cFHN: typical peak %.2f, worst within %.1f, worst n-stride growth at n = 1: %.2f (ρ %.2f)\n",
        maximum(c.typical), maximum(c.worst), fh.trans[2], fh.ρ)
