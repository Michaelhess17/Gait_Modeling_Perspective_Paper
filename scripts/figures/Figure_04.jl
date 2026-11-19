#!/usr/bin/env julia
# Figure_04.jl — Modelling the deviation from the cycle on long running recordings (manuscript: @fig-running). Run: julia --project=. scripts/figures/Figure_04.jl
# What does modelling the whole system cost, and how many strides, and how clean, does each model class need to
# forecast the deviations of human running? (maus_ladder.jl, maus_learning_all.jl, maus_node.jl, maus_ssm.py; 22 long
# treadmill running trials, Maus et al. 2015). Deviation: relative remaining variance, MSE(forecast) / MSE(average stride)
# = 1 − R² of the deviation from the 61-stride moving-average stride (1: no better than the average stride, 0: exact);
# medians and interquartile ranges over trials. Models: linear (solid; DMD's blue family) and nonlinear (dashed; ESN, LSTM, neural
# ODE, SSMLearnPy in the colours of the other figures).
#   A, B  prediction horizons of the whole signal, 150 training strides, each scored in its own time frame: NRMSE of the
#         joint angles (RMSE / SD of the signal), against a mean-cycle template (A: real time, the mean training stride
#         with its phase taken from the history and advanced at the previous stride's rate; B: stride-normalised time,
#         the mean training stride)
#   C     the deviation (the moving-average stride and the stride timing given) against the forecast horizon, 150 training strides
#   (training strides and added noise: Figure_07.jl, R4)
# Usage: julia --project=. scripts/figures/Figure_04.jl   → figures/Figure_04.{pdf,svg,png}

include(joinpath(@__DIR__, "running_common.jl"))

LAD = deserialize(joinpath(ROOT, "cache", "maus", "ladder.jls"))
REF = deserialize(joinpath(ROOT, "cache", "maus", "ladder_reference.jls"))
NODEW = (f = joinpath(ROOT, "cache", "maus", "node_whole.jls"); isfile(f) ? deserialize(f).out : nothing)
"Skill of the whole-signal model m on trial i (the same 22 trials, in the same order, in every cache)."
function wskill(rung, m, i, h)
    m == :template && return REF.R[i][rung].ref_causal[h]
    m == :node && return NODEW[rung].rows[i].r2[h]
    m == :ssm && return SSM["whole"][i, rung == :rt ? 1 : 2, h]
    LAD.R[i][(rung, m, 150)].r2[h]
end
"NRMSE of the predicted joint angles (RMSE / SD of the signal) over trials, in representation rung (:rt or :sn)."
nrmse(rung, m, h) = filter(isfinite, [sqrt(max((1 - wskill(rung, m, i, h)) * REF.R[i][rung].devfrac[h], 0))
                                      for i in eachindex(LAD.R) if LAD.R[i] !== nothing])
WHOLE = [(:ltp, C_LTP, :diamond, :solid), (:dmd, CAT[1], :circle, :solid), (:esn, CAT[2], :utriangle, :dash)]
NODEW === nothing || push!(WHOLE, (:node, CAT[3], :hexagon, :dash))
SSM === nothing || push!(WHOLE, (:ssm, CAT[4], :star5, :dash))
fig = Figure(size = (COL2, 70MM))
# A, B: prediction horizons of the whole signal, each scored in its own time frame against its own mean-cycle template
for (j, (rung, ttl, tl)) in enumerate(((:rt, "whole signal, real time", "mean stride,\nphase from the history"),
                                       (:sn, "whole signal, stride time", "mean stride")))
    ax = Axis(fig[1, j]; xlabel = "forecast horizon (stride)", ylabel = j == 1 ? "error of the joint angles (NRMSE)" : "",
              title = ttl, xticks = ([0.25, 0.5, 0.75, 1.0], ["¼", "½", "¾", "1"]))
    for (m, col, mk, ls) in WHOLE
        band!(ax, HZ ./ 100, [q(nrmse(rung, m, h), 0.25) for h in eachindex(HZ)], [q(nrmse(rung, m, h), 0.75) for h in eachindex(HZ)]; color = (col, 0.12))
        scatterlines!(ax, HZ ./ 100, [median(nrmse(rung, m, h)) for h in eachindex(HZ)]; color = col, marker = mk, markersize = 5, linestyle = ls)
    end
    lines!(ax, HZ ./ 100, [median(nrmse(rung, :template, h)) for h in eachindex(HZ)]; color = C_TRUTH, linestyle = :dash, linewidth = 1.2)
    text!(ax, 1.0, 0.24; text = "dashed: " * tl, fontsize = 5.5, align = (:right, :top))
    xlims!(ax, 0, 1.05); ylims!(ax, 0, 0.25)
    panel_label!(fig[1, j], ("A", "B")[j])
end
# C: the deviation, against the horizon
axC = Axis(fig[1, 3]; xlabel = "forecast horizon (stride)", ylabel = "remaining variance (rel. to average stride)", title = "deviation (cycle and timing given)",
           xticks = ([0.25, 0.5, 0.75, 1.0], ["¼", "½", "¾", "1"]))
refline!(axC, 1)
for (m, nm, col, mk, ls) in MODELS
    hs = [h for h in eachindex(HZ) if !isempty(vals(m, 150, 0.0, h))]
    isempty(hs) && continue
    band!(axC, HZ[hs] ./ 100, [q(vals(m, 150, 0.0, h), 0.25) for h in hs], [q(vals(m, 150, 0.0, h), 0.75) for h in hs]; color = (col, 0.12))
    scatterlines!(axC, HZ[hs] ./ 100, [median(vals(m, 150, 0.0, h)) for h in hs]; color = col, marker = mk, markersize = 5, linestyle = ls, label = nm)
end
xlims!(axC, 0, 1.05); ylims!(axC, 0, 1.3)
panel_label!(fig[1, 3], "C")
leg_el = vcat([[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 6)] for (_, _, col, mk, ls) in MODELS],
              [[LineElement(color = C_TRUTH, linestyle = :dash)]])
leg_lab = vcat([m in PHASE_INPUT ? nm * " (phase input with the deviation)" : nm for (m, nm, _, _, _) in MODELS], ["mean-cycle template (A, B)"])
Legend(fig[2, 1:3], leg_el, leg_lab; framevisible = false, labelsize = 6.5, orientation = :horizontal, nbanks = 3, colgap = 10, rowgap = 0)
rowgap!(fig.layout, 6)
save_paper(fig, "Figure_04")

for (m, nm, _, _, _) in MODELS, n in (16, 32, 150)
    v = vals(m, n, 0.0, 2); isempty(v) && continue
    @printf("%-8s %3d strides: remaining variance 1/8 stride %.2f [%.2f–%.2f], 1/4 %.2f (%d trials)\n", m, n, median(v), q(v, 0.25), q(v, 0.75), median(vals(m, n, 0.0, 4)), length(v))
end
KEY = Dict(:ltp => "FIP", :lpv => "LPV", :dmd => "DMD", :esn => "ESN", :lstm => "LSTM", :node => "NODE", :ssm => "SSM")
scorecard!("R1run", Dict(KEY[m] => @sprintf("%.2f", median(vals(m, 150, 0.0, 2))) for (m, _, _, _, _) in MODELS if haskey(KEY, m) && !isempty(vals(m, 150, 0.0, 2))))
