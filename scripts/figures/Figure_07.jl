#!/usr/bin/env julia
# Figure_07.jl — Modelling the deviation from the cycle on long running recordings (manuscript: @fig-running). Run: julia --project=. scripts/figures/Figure_07.jl
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
#   D     the deviation, 1/8 stride ahead, against the number of training strides. Shaded: strides in a 15-s walking trial
#   E     the deviation, 150 training strides, against added measurement noise (× each channel's deviation SD); the LSTM was run without noise only
# Usage: julia --project=. scripts/figures/Figure_07.jl   → figures/Figure_07.{pdf,svg,png}

include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf, NPZ

const ROOT = dirname(dirname(@__DIR__))
L = Dict(η => deserialize(joinpath(ROOT, "cache", "maus", "learning_all_cpu_$(η).jls")) for η in (0.0, 0.5, 1.0))
LSTMF = Dict(η => joinpath(ROOT, "cache", "maus", "learning_all_lstm_$(η).jls") for η in (0.0, 0.5, 1.0))
const SZ = L[0.0].SIZES; const HZ = L[0.0].HZ
# model, label, colour, marker, line style. Colours by model as in the other figures (DMD blue, ESN orange, neural ODE
# aqua, SSMLearnPy yellow); the phase-varying and phase-binned linear models in DMD's blue family; nonlinear models dashed
const C_LTP = colorant"#1a4f99"; const C_LPV = colorant"#78b0f5"
MODELS = [(:ltp, "phase-varying linear (Fourier in phase)", C_LTP, :diamond, :solid),
          (:lpv, "phase-binned linear (LPV)", C_LPV, :rect, :solid),
          (:dmd, "time-invariant linear (DMD)", CAT[1], :circle, :solid),
          (:lattice, "lattice (imposed cycle + one mode)", C_REF, :xcross, :solid),
          (:esn, "ESN", CAT[2], :utriangle, :dash)]
haslstm = isfile(LSTMF[0.0])
haslstm && push!(MODELS, (:lstm, "LSTM", CAT[5], :dtriangle, :dash))
NODEF = Dict(η => joinpath(ROOT, "cache", "maus", "node_dev_$(η).jls") for η in (0.0, 0.5, 1.0))
NODE = Dict(η => deserialize(f).out for (η, f) in NODEF if isfile(f))
haskey(NODE, 0.0) && push!(MODELS, (:node, "neural ODE", CAT[3], :hexagon, :dash))
SSMF = joinpath(ROOT, "cache", "maus", "ssm.npz"); SSM = isfile(SSMF) ? npzread(SSMF) : nothing
SSM === nothing || push!(MODELS, (:ssm, "SSM (SSMLearnPy)", CAT[4], :star5, :dash))
const PHASE_INPUT = (:esn, :lstm, :node)          # given the stride phase when modelling the deviation

"R² values over trials for model m, training size n, noise η, horizon index h."
function skillvals(m, n, η, h)
    if m == :lstm
        isfile(LSTMF[η]) || return Float64[]
        O = deserialize(LSTMF[η]).out; haskey(O, n) || return Float64[]
        return [r.r2[h] for r in O[n].rows]
    elseif m == :node
        (haskey(NODE, η) && haskey(NODE[η], n)) || return Float64[]
        return [r.r2[h] for r in NODE[η][n].rows]
    elseif m == :ssm
        SSM === nothing && return Float64[]
        e = findfirst(==(η), SSM["etas"]); k = findfirst(==(n), SSM["sizes"])
        (e === nothing || k === nothing) && return Float64[]
        return filter(isfinite, SSM["dev"][:, e, k, h])
    end
    [r[(m, n)].r2[h] for r in L[η].R if r !== nothing && haskey(r, (m, n))]
end
q(v, p) = isempty(v) ? NaN : quantile(v, p)
"Relative remaining variance, MSE / MSE(average stride) = 1 − R² of the deviation, over trials."
vals(m, n, η, h) = 1 .- skillvals(m, n, η, h)

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
fig = Figure(size = (COL2, 118MM))
walk = (9.4, 14.9)                       # interquartile strides in a 15-s walking trial (gait_group_fits.jl periods)
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
# D: learning curves of the deviation
axD = Axis(fig[2, 1]; xscale = log2, xticks = SZ, xlabel = "training strides", ylabel = "remaining variance, ⅛ stride ahead",
           title = "deviation: training strides")
vspan!(axD, walk...; color = C_SHADE)
text!(axD, sqrt(walk[1] * walk[2]), 0.18; text = "15-s walking\ntrial", fontsize = 6, align = (:center, :bottom), color = C_REF)
refline!(axD, 1)
for (m, nm, col, mk, ls) in MODELS
    ns = [n for n in SZ if !isempty(vals(m, n, 0.0, 2))]
    isempty(ns) && continue
    band!(axD, ns, [q(vals(m, n, 0.0, 2), 0.25) for n in ns], [q(vals(m, n, 0.0, 2), 0.75) for n in ns]; color = (col, 0.12))
    scatterlines!(axD, ns, [median(vals(m, n, 0.0, 2)) for n in ns]; color = col, marker = mk, markersize = 6, linestyle = ls)
end
ylims!(axD, 0.15, 1.6)
panel_label!(fig[2, 1], "D")
# E: noise
axE = Axis(fig[2, 2]; xlabel = "added measurement noise (× deviation SD)", ylabel = "remaining variance, ⅛ stride ahead",
           title = "deviation: added noise", xticks = [0, 0.5, 1])
MODELS_E = [x for x in MODELS if x[1] ∉ (:lattice, :lstm)]   # lattice: not fitted with noise; LSTM: noise-free runs only
for (k, (m, nm, col, mk, ls)) in enumerate(MODELS_E)
    ηs = [η for η in (0.0, 0.5, 1.0) if !isempty(vals(m, 150, η, 2))]
    isempty(ηs) && continue
    off = (k - (length(MODELS_E) + 1) / 2) * 0.022
    rangebars!(axE, ηs .+ off, [q(vals(m, 150, η, 2), 0.25) for η in ηs], [q(vals(m, 150, η, 2), 0.75) for η in ηs]; color = col, whiskerwidth = 0)
    scatterlines!(axE, ηs .+ off, [median(vals(m, 150, η, 2)) for η in ηs]; color = col, marker = mk, markersize = 6, linestyle = ls)
end
ylims!(axE, 0.15, 1.0)
panel_label!(fig[2, 2], "E")
leg_el = vcat([[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 6)] for (_, _, col, mk, ls) in MODELS],
              [[LineElement(color = C_TRUTH, linestyle = :dash)]])
leg_lab = vcat([m in PHASE_INPUT ? nm * " (phase input with the deviation)" : nm for (m, nm, _, _, _) in MODELS], ["mean-cycle template (A, B)"])
Legend(fig[2, 3], leg_el, leg_lab; framevisible = false, labelsize = 6.5, rowgap = 1, tellheight = false)
save_paper(fig, "Figure_07")

for (m, nm, _, _, _) in MODELS, n in (16, 32, 150)
    v = vals(m, n, 0.0, 2); isempty(v) && continue
    @printf("%-8s %3d strides: remaining variance 1/8 stride %.2f [%.2f–%.2f], 1/4 %.2f (%d trials)\n", m, n, median(v), q(v, 0.25), q(v, 0.75), median(vals(m, n, 0.0, 4)), length(v))
end
