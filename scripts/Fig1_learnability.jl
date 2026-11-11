#!/usr/bin/env julia
# Preliminary coverage diagnostic, retained while the replacement learnability experiment is built.
# Not the planned three-panel main figure: see scripts/FIGURE_PLAN.md.
# Stabilisation rollouts now live in Fig_stabilization.jl.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
cachef(p...) = joinpath(ROOT, "cache", p...)
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)

fig = Figure(size = (COL2, 58MM))
gA = fig[1, 1] = GridLayout()

# ── A: data coverage (coupled FHN) ──────────────────────────────────────────
cov = deserialize(cachef("fig1", "coverage_cfhn.jls"))
se, sd = cov.sec_excited, cov.sec_decays
slope(seqs) = (x = reduce(vcat, [q[1:end-1] for q in seqs]); y = reduce(vcat, [q[2:end] for q in seqs]); (x' * y) / (x' * x))
axA1 = Axis(gA[1, 1]; xlabel = "Cycle", ylabel = "Units out of step\n(w₁ − w₂, once per cycle)",
            title = "Passive data barely leave the cycle")
refline!(axA1, 0.0; linestyle = :solid)
scatterlines!(axA1, 1:length(se), se; color = (C_REF, 0.9), markersize = 3.5, linewidth = 0.8, label = "passive: one long run")
for (k, q) in enumerate(sd)
    scatterlines!(axA1, 3 * (k - 1) .+ (1:length(q)), q; color = CAT[1], markersize = 3.5, linewidth = 0.9,
                  label = k == 1 ? "transients: 10 decays × 3 cycles" : nothing)
end
axislegend(axA1; position = (:right, :bottom), framevisible = false, labelsize = 7, unique = true)
axA2 = Axis(gA[1, 2]; xlabel = "Out of step, cycle k", ylabel = "Out of step, cycle k + 1",
            title = "Return estimated from each", aspect = 1)
lo = minimum(vcat(se, reduce(vcat, sd))) * 1.1; hi = 0.08; xs = [lo, hi]
lines!(axA2, xs, cov.μ_true .* xs; color = C_TRUTH, linewidth = 1.4, label = @sprintf("truth: %.2f per cycle", cov.μ_true))
scatter!(axA2, se[1:end-1], se[2:end]; color = (C_REF, 0.9), markersize = 4)
lines!(axA2, xs, slope([se]) .* xs; color = C_REF, linestyle = :dash, linewidth = 1.2, label = @sprintf("passive: %.2f", slope([se])))
scatter!(axA2, reduce(vcat, [q[1:end-1] for q in sd]), reduce(vcat, [q[2:end] for q in sd]); color = CAT[1], markersize = 4)
lines!(axA2, xs, slope(sd) .* xs; color = CAT[1], linestyle = :dash, linewidth = 1.2, label = @sprintf("transients: %.2f", slope(sd)))
limits!(axA2, lo, hi, lo, hi)
axislegend(axA2; position = (:right, :bottom), framevisible = false, labelsize = 7)
plabel!(gA[1, 1], "A"); colsize!(gA, 2, Auto(0.55))

save_paper(fig, "Fig1_learnability")
