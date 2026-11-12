#!/usr/bin/env julia
# Figure_07.jl — Modelling the deviation from the cycle on long running recordings (manuscript: @fig-running). Run: julia --project=. scripts/figures/Figure_07.jl
# What does modelling the whole system cost, and how many strides, and how clean, does each model class need to
# forecast the deviations of human running? (maus_ladder.jl, maus_learning_all.jl; 22 long treadmill running trials,
# Maus et al. 2015). Skill against the average stride, 1 − MSE(forecast) / MSE(average stride) = R² of the deviation from
# the 61-stride moving-average stride (0: no better than the average stride); medians and interquartile ranges over trials.
#   A, B  prediction horizons of the whole signal, 150 training strides, each scored in its own time frame: NRMSE of the
#         joint angles (RMSE / SD of the signal), against a mean-cycle template (A: real time, the mean training stride
#         with its phase taken from the history and advanced at the previous stride's rate; B: stride-normalised time,
#         the mean training stride)
#   C     the deviation (cycle supplied) against the forecast horizon, 150 training strides
#   D     the deviation, 1/8 stride ahead, against the number of training strides. Shaded: strides in a 15-s walking trial
#   E     the deviation, 150 training strides, against added measurement noise (× each channel's deviation SD); the LSTM was run without noise only
# Usage: julia --project=. scripts/figures/Figure_07.jl   → figures/Figure_07.{pdf,svg,png}

include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
L = Dict(η => deserialize(joinpath(ROOT, "cache", "maus", "learning_all_cpu_$(η).jls")) for η in (0.0, 0.5, 1.0))
LSTMF = Dict(η => joinpath(ROOT, "cache", "maus", "learning_all_lstm_$(η).jls") for η in (0.0, 0.5, 1.0))
const SZ = L[0.0].SIZES; const HZ = L[0.0].HZ
# model, label, colour, marker (colour by model, as in the other figures: DMD blue, ESN orange)
MODELS = [(:ltp, "phase-varying linear (Fourier in phase)", CAT[3], :diamond),
          (:lpv, "phase-binned linear (LPV)", CAT[4], :rect),
          (:dmd, "time-invariant linear (DMD)", CAT[1], :circle),
          (:lattice, "lattice (imposed cycle + one mode)", C_REF, :xcross),
          (:esn, "ESN (phase input)", CAT[2], :utriangle)]
haslstm = isfile(LSTMF[0.0])
haslstm && push!(MODELS, (:lstm, "LSTM (phase input)", CAT[5], :dtriangle))

"R² values over trials for model m, training size n, noise η, horizon index h."
function vals(m, n, η, h)
    if m == :lstm
        isfile(LSTMF[η]) || return Float64[]
        O = deserialize(LSTMF[η]).out; haskey(O, n) || return Float64[]
        return [r.r2[h] for r in O[n].rows]
    end
    [r[(m, n)].r2[h] for r in L[η].R if r !== nothing && haskey(r, (m, n))]
end
q(v, p) = isempty(v) ? NaN : quantile(v, p)

LAD = deserialize(joinpath(ROOT, "cache", "maus", "ladder.jls"))
REF = deserialize(joinpath(ROOT, "cache", "maus", "ladder_reference.jls"))
"NRMSE of the predicted joint angles (RMSE / SD of the signal) over trials, in representation rung (:rt or :sn)."
nrmse(rung, m, h) = [sqrt(max((1 - (m == :template ? REF.R[i][rung].ref_causal[h] : LAD.R[i][(rung, m, 150)].r2[h])) * REF.R[i][rung].devfrac[h], 0))
                     for i in eachindex(LAD.R) if LAD.R[i] !== nothing]
WHOLE = [(:ltp, CAT[3], :diamond), (:dmd, CAT[1], :circle), (:esn, CAT[2], :utriangle)]
fig = Figure(size = (COL2, 118MM))
walk = (9.4, 14.9)                       # interquartile strides in a 15-s walking trial (gait_group_fits.jl periods)
# A, B: prediction horizons of the whole signal, each scored in its own time frame against its own mean-cycle template
for (j, (rung, ttl, tl)) in enumerate(((:rt, "whole signal, real time", "mean stride,\nphase from the history"),
                                       (:sn, "whole signal, stride time", "mean stride")))
    ax = Axis(fig[1, j]; xlabel = "forecast horizon (stride)", ylabel = j == 1 ? "error of the joint angles (NRMSE)" : "",
              title = ttl, xticks = ([0.25, 0.5, 0.75, 1.0], ["¼", "½", "¾", "1"]))
    for (m, col, mk) in WHOLE
        band!(ax, HZ ./ 100, [q(nrmse(rung, m, h), 0.25) for h in eachindex(HZ)], [q(nrmse(rung, m, h), 0.75) for h in eachindex(HZ)]; color = (col, 0.15))
        scatterlines!(ax, HZ ./ 100, [median(nrmse(rung, m, h)) for h in eachindex(HZ)]; color = col, marker = mk, markersize = 5)
    end
    lines!(ax, HZ ./ 100, [median(nrmse(rung, :template, h)) for h in eachindex(HZ)]; color = C_TRUTH, linestyle = :dash, linewidth = 1.2)
    text!(ax, 1.0, 0.24; text = "dashed: " * tl, fontsize = 5.5, align = (:right, :top))
    xlims!(ax, 0, 1.05); ylims!(ax, 0, 0.25)
    panel_label!(fig[1, j], ("A", "B")[j])
end
# C: the deviation, against the horizon
axC = Axis(fig[1, 3]; xlabel = "forecast horizon (stride)", ylabel = "skill against the average stride", title = "deviation (cycle supplied)",
           xticks = ([0.25, 0.5, 0.75, 1.0], ["¼", "½", "¾", "1"]))
refline!(axC, 0)
for (m, nm, col, mk) in MODELS
    hs = [h for h in eachindex(HZ) if !isempty(vals(m, 150, 0.0, h))]
    isempty(hs) && continue
    band!(axC, HZ[hs] ./ 100, [q(vals(m, 150, 0.0, h), 0.25) for h in hs], [q(vals(m, 150, 0.0, h), 0.75) for h in hs]; color = (col, 0.15))
    scatterlines!(axC, HZ[hs] ./ 100, [median(vals(m, 150, 0.0, h)) for h in hs]; color = col, marker = mk, markersize = 5, label = nm)
end
xlims!(axC, 0, 1.05); ylims!(axC, -0.3, 1.0)
panel_label!(fig[1, 3], "C")
# D: learning curves of the deviation
axD = Axis(fig[2, 1]; xscale = log2, xticks = SZ, xlabel = "training strides", ylabel = "skill, ⅛ stride ahead",
           title = "deviation: training strides")
vspan!(axD, walk...; color = C_SHADE)
text!(axD, sqrt(walk[1] * walk[2]), -0.55; text = "15-s walking\ntrial", fontsize = 6, align = (:center, :bottom), color = C_REF)
refline!(axD, 0)
for (m, nm, col, mk) in MODELS
    ns = [n for n in SZ if !isempty(vals(m, n, 0.0, 2))]
    isempty(ns) && continue
    band!(axD, ns, [q(vals(m, n, 0.0, 2), 0.25) for n in ns], [q(vals(m, n, 0.0, 2), 0.75) for n in ns]; color = (col, 0.15))
    scatterlines!(axD, ns, [median(vals(m, n, 0.0, 2)) for n in ns]; color = col, marker = mk, markersize = 6)
end
ylims!(axD, -0.6, 0.85)
panel_label!(fig[2, 1], "D")
# E: noise
axE = Axis(fig[2, 2]; xlabel = "added measurement noise (× deviation SD)", ylabel = "skill, ⅛ stride ahead",
           title = "deviation: added noise", xticks = [0, 0.5, 1])
for (k, (m, nm, col, mk)) in enumerate(MODELS)
    m in (:lattice, :lstm) && continue                    # LSTM: noise-free runs only
    ηs = [η for η in (0.0, 0.5, 1.0) if !isempty(vals(m, 150, η, 2))]
    isempty(ηs) && continue
    off = (k - 3) * 0.025
    rangebars!(axE, ηs .+ off, [q(vals(m, 150, η, 2), 0.25) for η in ηs], [q(vals(m, 150, η, 2), 0.75) for η in ηs]; color = col, whiskerwidth = 0)
    scatterlines!(axE, ηs .+ off, [median(vals(m, 150, η, 2)) for η in ηs]; color = col, marker = mk, markersize = 6)
end
ylims!(axE, 0, 0.85)
panel_label!(fig[2, 2], "E")
leg_el = vcat([[LineElement(color = col), MarkerElement(color = col, marker = mk, markersize = 6)] for (_, _, col, mk) in MODELS],
              [[LineElement(color = C_TRUTH, linestyle = :dash)]])
leg_lab = vcat([nm == "ESN (phase input)" ? "ESN (phase input with the deviation)" : nm for (_, nm, _, _) in MODELS], ["mean-cycle template (A, B)"])
Legend(fig[2, 3], leg_el, leg_lab; framevisible = false, labelsize = 6.5, rowgap = 1, tellheight = false)
save_paper(fig, "Figure_07")

for (m, nm, _, _) in MODELS, n in (16, 32, 150)
    v = vals(m, n, 0.0, 2); isempty(v) && continue
    @printf("%-8s %3d strides: R² 1/8 stride %.2f [%.2f–%.2f], 1/4 %.2f (%d trials)\n", m, n, median(v), q(v, 0.25), q(v, 0.75), median(vals(m, n, 0.0, 4)), length(v))
end
