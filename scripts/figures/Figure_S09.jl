#!/usr/bin/env julia
# Figure_S09.jl — Forecast baselines (manuscript: @suppfig-forecast). Run: julia --project=. scripts/figures/Figure_S09.jl
# Forecasting unperturbed gait is mostly forecasting the mean cycle
#
# Every method gets the same information (z-scored joint angles up to sample
# 1000 of each 15-s trial) and is scored on the same held-out targets with the
# same instantaneous RMSE across the six channels (SSM_baselines.py).
#
#   (A) Median error against true lead time (seconds past the last observed
#       sample) for the tuned SSMLearnPy delay map and a rank-70 linear Hankel-DMD (ridge 1e-4, stabilised; companion
#       settings), with two non-dynamical reference levels: a phase-aligned mean training cycle and the last stride
#       repeated. The template has no state and cannot be started off the cycle, so it is a reference, not a model.
#   (B) Per trial, at 2.7 s lead: SSM error against the template's.
#   (C-E) Knee angle in one representative trial per group.
#
# Inputs:  cache/ssm_baselines.npz          (SSM_baselines.py)
#          cache/fig_forecast_examples.npz  (export_forecast_examples.py)

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using NPZ, Statistics

const FS = 100.0
const INFO_END = 1001                   # samples 0..1000 are known (0-based)

B = npzread(joinpath(dirname(dirname(@__DIR__)), "cache/ssm_baselines.npz"),
            ["err_ssm", "err_mean", "err_repeat_stride", "err_phase_template",
             "err_hankel_dmd", "ssm_finite", "target_sample_index"])
E = npzread(joinpath(dirname(dirname(@__DIR__)), "cache/fig_forecast_examples.npz"))

targets = B["target_sample_index"]
lead    = (targets .- (INFO_END - 1)) ./ FS      # s past the last observed sample
fwd     = findall(>=(0), lead)                   # only genuine forecasts
ok      = B["ssm_finite"] .== 1

methods = [
    ("phase template",     B["err_phase_template"], CAT[1], :solid),
    ("repeat last stride", B["err_repeat_stride"],  CAT[3], :solid),
    ("Hankel-DMD (r = 70)", B["err_hankel_dmd"],    CAT[4], :dash),
    ("SSMLearnPy",         B["err_ssm"],            CAT[2], :solid),
]

colmedian(M) = [median(filter(isfinite, M[ok, j])) for j in axes(M, 2)]
colq(M, q)   = [quantile(filter(isfinite, M[ok, j]), q) for j in axes(M, 2)]

fig = Figure(size = (COL2, 100MM))
gA = fig[1, 1:2] = GridLayout()
gB = fig[1, 3]   = GridLayout()
gC = fig[2, 1:3] = GridLayout()

# ── (A) median error vs lead ─────────────────────────────────────────────────
axA = Axis(gA[1, 1]; xlabel = "Lead time past last observed sample (s)",
           ylabel = "Forecast error (SD units)",
           title = "Median across trials (IQR shaded for template and SSM)")
refline!(axA, 1.0)
text!(axA, lead[fwd][end], 1.0; text = "predict the mean", color = C_REF,
      fontsize = 7, align = (:right, :bottom))
for (name, M, c, ls) in methods
    med = colmedian(M)[fwd]
    if name in ("phase template", "SSMLearnPy")
        band!(axA, lead[fwd], colq(M, 0.25)[fwd], colq(M, 0.75)[fwd];
              color = (c, 0.15))
    end
    lines!(axA, lead[fwd], med; color = c, linestyle = ls, label = name,
           linewidth = 1.4)
end
xlims!(axA, 0, lead[fwd][end])
ylims!(axA, 0, 1.15)
axislegend(axA; position = :lt, margin = (6, 4, 4, 22))
panel_label!(gA, "A")

# ── (B) paired per-trial comparison ──────────────────────────────────────────
jB = argmin(abs.(lead .- 2.69))
s_ssm = B["err_ssm"][ok, jB]; s_tpl = B["err_phase_template"][ok, jB]
axB = Axis(gB[1, 1]; xlabel = "Phase template error", ylabel = "SSMLearnPy error",
           title = @sprintf("Per trial, %.1f s lead", lead[jB]), aspect = 1)
lim = 1.5
n_off = count((s_ssm .> lim) .| (s_tpl .> lim))
lines!(axB, [0, lim], [0, lim]; color = C_REF, linewidth = 0.8, linestyle = :dash)
scatter!(axB, s_tpl, s_ssm; color = (C_TRUTH, 0.45), markersize = 3.5)
frac_worse = mean(s_ssm .> s_tpl)
text!(axB, 0.04lim, 0.96lim; align = (:left, :top), fontsize = 7,
      text = @sprintf("%d trials\n(%d off-scale)", length(s_ssm), n_off))
limits!(axB, 0, lim, 0, lim)
panel_label!(gB, "B")

# ── (C–E) representative knee traces ─────────────────────────────────────────
KNEE = 2
for (k, g) in enumerate(GROUP_ORDER_FIG)
    i = k - 1
    axk = Axis(gC[1, k]; xlabel = "Lead time (s)",
               ylabel = k == 1 ? "Knee angle (z)" : "",
               title = @sprintf("%s trial %d", g, E["example_indices"][k]))
    tru = E["ex$(i)_true"][fwd, KNEE]
    lines!(axk, lead[fwd], tru; color = C_TRUTH, linewidth = 1.6, label = "observed")
    lines!(axk, lead[fwd], E["ex$(i)_template"][fwd, KNEE]; color = CAT[1],
           label = "phase template")
    lines!(axk, lead[fwd], E["ex$(i)_ssm"][fwd, KNEE]; color = CAT[2],
           label = "SSMLearnPy")
    xlims!(axk, 0, lead[fwd][end])
end
for (k, letter) in enumerate(("C", "D", "E"))
    Label(gC[1, k, TopLeft()], letter; fontsize = 10, font = :bold,
          padding = (0, 6, 2, 0), halign = :right)
end
Legend(gC[0, 1:3], [LineElement(color = C_TRUTH, linewidth = 1.6),
                    LineElement(color = CAT[1]), LineElement(color = CAT[2])],
       ["observed", "phase template", "SSMLearnPy"]; orientation = :horizontal,
       tellheight = true, halign = :right)
rowgap!(gC, 2)
colgap!(gC, 10)
rowgap!(fig.layout, 8)

save_paper(fig, "Figure_S09")

# numbers quoted in the text
for (name, M, _, _) in methods
    med = colmedian(M)
    @printf("%-20s median error at lead 0.69/2.69/4.69 s: %.2f %.2f %.2f\n", name,
            med[argmin(abs.(lead .- 0.69))], med[argmin(abs.(lead .- 2.69))],
            med[argmin(abs.(lead .- 4.69))])
end
@printf("SSM worse than template in %.0f%% of %d finite-SSM trials at %.2f s lead\n",
        100frac_worse, length(s_ssm), lead[jB])
