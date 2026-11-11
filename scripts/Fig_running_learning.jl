#!/usr/bin/env julia
# How many strides, and how clean, does each model class need to forecast the deviations of human running?
# (maus_learning_all.jl; 22 long treadmill running trials, Maus et al. 2015)
#   A, B  median R² of the deviation 1/8 and 1/4 stride ahead against the number of training strides, interquartile
#         range over trials; noise-free recordings as measured. Shaded band: strides in a 15-s walking trial
#   C     R² 1/8 stride ahead with 150 training strides against added measurement noise (× each channel's deviation SD)
#   D     R² against forecast horizon with 150 training strides, as measured
# Usage: julia --project=. scripts/Fig_running_learning.jl   → figures/Fig_running_learning.{pdf,svg,png}

include("paper_style.jl")
using Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
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

fig = Figure(size = (COL2, 118MM))
walk = (9.4, 14.9)                       # interquartile strides in a 15-s walking trial (gait_group_fits.jl periods)
for (j, (h, lab)) in enumerate(((2, "1/8 stride ahead"), (4, "1/4 stride ahead")))
    ax = Axis(fig[1, j]; xscale = log2, xticks = SZ, xlabel = "training strides", ylabel = j == 1 ? "R² of the deviation" : "",
              title = "forecast $lab")
    vspan!(ax, walk...; color = C_SHADE)
    j == 1 && text!(ax, sqrt(walk[1] * walk[2]), -0.55; text = "15-s walking\ntrial", fontsize = 6, align = (:center, :bottom), color = C_REF)
    refline!(ax, 0)
    for (m, nm, col, mk) in MODELS
        ns = [n for n in SZ if !isempty(vals(m, n, 0.0, h))]
        isempty(ns) && continue
        band!(ax, ns, [q(vals(m, n, 0.0, h), 0.25) for n in ns], [q(vals(m, n, 0.0, h), 0.75) for n in ns]; color = (col, 0.15))
        scatterlines!(ax, ns, [median(vals(m, n, 0.0, h)) for n in ns]; color = col, marker = mk, markersize = 6, label = nm)
    end
    ylims!(ax, -0.6, 0.85)
    panel_label!(fig[1, j], ("A", "B")[j])
end
Legend(fig[1, 3], fig.content[1]; framevisible = false, labelsize = 7, rowgap = 2)
# C: noise
axC = Axis(fig[2, 1]; xlabel = "added measurement noise (× deviation SD)", ylabel = "R², 1/8 stride ahead",
           title = "150 training strides: noise", xticks = [0, 0.5, 1])
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
axD = Axis(fig[2, 2]; xlabel = "forecast horizon (stride)", ylabel = "R² of the deviation", title = "150 training strides: horizon",
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
save_paper(fig, "Fig_running_learning")

for (m, nm, _, _) in MODELS, n in (16, 32, 150)
    v = vals(m, n, 0.0, 2); isempty(v) && continue
    @printf("%-8s %3d strides: R² 1/8 stride %.2f [%.2f–%.2f], 1/4 %.2f (%d trials)\n", m, n, median(v), q(v, 0.25), q(v, 0.75), median(vals(m, n, 0.0, 4)), length(v))
end
