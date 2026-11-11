#!/usr/bin/env julia
# Figure_07.jl — Modelling the deviation from the cycle on long running recordings (manuscript: @fig-running). Run: julia --project=. scripts/figures/Figure_07.jl
# What does modelling the whole system cost, and how many strides, and how clean, does each model class need to
# forecast the deviations of human running? (maus_ladder.jl, maus_learning_all.jl; 22 long treadmill running trials,
# Maus et al. 2015). R² of the deviation from the 61-stride moving-average stride; medians and interquartile ranges over
# trials.
#   A  the same forecasting task in three representations, 150 training strides, 1/8 stride ahead: the whole signal in
#      real time, the whole signal in stride-normalised time (timing supplied), the deviation (cycle supplied)
#   B  deviation, 1/8 stride ahead, against the number of training strides. Shaded band: strides in a 15-s walking trial
#   C  deviation, 150 training strides, against added measurement noise (× each channel's deviation SD)
#   D  deviation, 150 training strides, against the forecast horizon
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

LADF = joinpath(ROOT, "cache", "maus", "ladder.jls")
LAD = isfile(LADF) ? deserialize(LADF) : nothing
"R² over trials in a whole-signal representation (rung :rt or :sn) of maus_ladder.jl."
lvals(rung, m, n, h) = LAD === nothing ? Float64[] : [r[(rung, m, n)].r2[h] for r in LAD.R if r !== nothing && haskey(r, (rung, m, n))]
const YMIN = -1.0
fig = Figure(size = (COL2, 118MM))
walk = (9.4, 14.9)                       # interquartile strides in a 15-s walking trial (gait_group_fits.jl periods)
# A: three representations of the same task
axA = Axis(fig[1, 1]; ylabel = "R², ⅛ stride ahead", title = "what modelling the whole system costs",
           xticks = (1:3, ["whole signal,\nreal time", "whole signal,\nstride time", "deviation"]))
refline!(axA, 0)
for (k, (m, nm, col, mk)) in enumerate(MODELS)
    m in (:dmd, :esn, :ltp, :lstm) || continue
    xs = Float64[]; md = Float64[]; lo = Float64[]; hi = Float64[]
    for (x, getv) in ((1, () -> lvals(:rt, m, 150, 2)), (2, () -> lvals(:sn, m, 150, 2)), (3, () -> vals(m, 150, 0.0, 2)))
        v = getv(); isempty(v) && continue
        push!(xs, x + (k - 3.5) * 0.09); push!(md, median(v)); push!(lo, q(v, 0.25)); push!(hi, q(v, 0.75))
    end
    isempty(xs) && continue
    lines!(axA, xs, max.(md, YMIN); color = (col, 0.5), linewidth = 0.8)
    rangebars!(axA, xs, max.(lo, YMIN), max.(hi, YMIN); color = col, whiskerwidth = 0)
    scatter!(axA, xs, max.(md, YMIN); color = col, marker = mk, markersize = 6)
    for (x, v) in zip(xs, md); v < YMIN && text!(axA, x, YMIN; text = @sprintf("%.0f", v), fontsize = 5, align = (:center, :bottom), offset = (0, 3)); end
end
# the reference a forecaster could use: the training-average stride at the extrapolated phase (maus_ladder_reference.jl)
REFF = joinpath(ROOT, "cache", "maus", "ladder_reference.jls")
if isfile(REFF)
    RF = deserialize(REFF).R
    refv = Dict(rung => median([r[rung].ref_causal[2] for r in RF if r !== nothing]) for rung in (:rt, :sn))
    for (x, rung) in ((1, :rt), (2, :sn))
        lines!(axA, [x - 0.3, x + 0.3], fill(max(refv[rung], YMIN), 2); color = C_TRUTH, linewidth = 1.2, linestyle = :dash)
    end
    text!(axA, 3.35, YMIN + 0.02; text = @sprintf("dashed: average stride with\nextrapolated timing\n(real time %.1f, off scale)", refv[:rt]),
          fontsize = 5.5, align = (:right, :bottom), color = C_TRUTH)
end
ylims!(axA, YMIN - 0.05, 0.85)
panel_label!(fig[1, 1], "A")
# B: learning curves of the deviation
axB = Axis(fig[1, 2]; xscale = log2, xticks = SZ, xlabel = "training strides", ylabel = "R², ⅛ stride ahead",
           title = "deviation: training strides")
vspan!(axB, walk...; color = C_SHADE)
text!(axB, sqrt(walk[1] * walk[2]), -0.55; text = "15-s walking\ntrial", fontsize = 6, align = (:center, :bottom), color = C_REF)
refline!(axB, 0)
for (m, nm, col, mk) in MODELS
    ns = [n for n in SZ if !isempty(vals(m, n, 0.0, 2))]
    isempty(ns) && continue
    band!(axB, ns, [q(vals(m, n, 0.0, 2), 0.25) for n in ns], [q(vals(m, n, 0.0, 2), 0.75) for n in ns]; color = (col, 0.15))
    scatterlines!(axB, ns, [median(vals(m, n, 0.0, 2)) for n in ns]; color = col, marker = mk, markersize = 6, label = nm)
end
ylims!(axB, -0.6, 0.85)
panel_label!(fig[1, 2], "B")
Legend(fig[1:2, 3], axB; framevisible = false, labelsize = 7, rowgap = 2)
# C: noise
axC = Axis(fig[2, 1]; xlabel = "added measurement noise (× deviation SD)", ylabel = "R², ⅛ stride ahead",
           title = "deviation: added noise", xticks = [0, 0.5, 1])
for (k, (m, nm, col, mk)) in enumerate(MODELS)
    m == :lattice && continue
    ηs = [η for η in (0.0, 0.5, 1.0) if !isempty(vals(m, 150, η, 2))]
    isempty(ηs) && continue
    off = (k - 3) * 0.025
    rangebars!(axC, ηs .+ off, [q(vals(m, 150, η, 2), 0.25) for η in ηs], [q(vals(m, 150, η, 2), 0.75) for η in ηs]; color = col, whiskerwidth = 0)
    scatterlines!(axC, ηs .+ off, [median(vals(m, 150, η, 2)) for η in ηs]; color = col, marker = mk, markersize = 6)
end
ylims!(axC, 0, 0.85)
panel_label!(fig[2, 1], "C")
# D: horizon profile
axD = Axis(fig[2, 2]; xlabel = "forecast horizon (stride)", ylabel = "R² of the deviation", title = "deviation: horizon",
           xticks = ([0.125, 0.25, 0.5, 0.75, 1.0], ["⅛", "¼", "½", "¾", "1"]))
refline!(axD, 0)
for (m, nm, col, mk) in MODELS
    hs = [h for h in eachindex(HZ) if !isempty(vals(m, 150, 0.0, h))]
    isempty(hs) && continue
    band!(axD, HZ[hs] ./ 100, [q(vals(m, 150, 0.0, h), 0.25) for h in hs], [q(vals(m, 150, 0.0, h), 0.75) for h in hs]; color = (col, 0.15))
    scatterlines!(axD, HZ[hs] ./ 100, [median(vals(m, 150, 0.0, h)) for h in hs]; color = col, marker = mk, markersize = 5)
end
ylims!(axD, -0.3, 1.0)
panel_label!(fig[2, 2], "D")
colsize!(fig.layout, 3, Auto(0.55))
save_paper(fig, "Figure_07")

for rung in (:rt, :sn), m in (:dmd, :esn, :ltp)
    v = lvals(rung, m, 150, 2); isempty(v) && continue
    dv = LAD === nothing ? NaN : median([r[(rung, m, 150)].diverged for r in LAD.R if r !== nothing && haskey(r, (rung, m, 150))])
    @printf("%s %-4s 150 strides: R² 1/8 stride %.2f [%.2f–%.2f], 1/4 %.2f | diverged starts %.0f%%\n", rung, m, median(v), q(v, 0.25), q(v, 0.75), median(lvals(rung, m, 150, 4)), 100dv)
end
for (m, nm, _, _) in MODELS, n in (16, 32, 150)
    v = vals(m, n, 0.0, 2); isempty(v) && continue
    @printf("%-8s %3d strides: R² 1/8 stride %.2f [%.2f–%.2f], 1/4 %.2f (%d trials)\n", m, n, median(v), q(v, 0.25), q(v, 0.75), median(vals(m, n, 0.0, 4)), length(v))
end
