#!/usr/bin/env julia
# Figure_03.jl — Forecasting the predictable short-time response (SLIP, with ceiling) (manuscript: @fig-predictable). Run: julia --project=. scripts/figures/Figure_03.jl
# Figure: how much of the predictable short-time response do fitted models forecast? (R1)
# Data: scripts/slip_predictable_truth.jl (SLIP.jl env) → scripts/predictable_response.jl → cache/predictable/slip_results.jls
#   (A) one test start (SLIP with feedback, full state, 60 training steps): height deviation from the mean step
#   (B–D) remaining variance relative to the mean-cycle forecast (MSE / MSE of the mean cycle) over the horizon, 60
#         training steps: floor (noise-free continuation of the true state), every model class (median over seeds:
#         ESN 4, LSTM 2, neural ODE 2), persistence; linear models solid, nonlinear dashed (predictable_more.jl adds
#         Fourier in phase, neural ODE, LSTM and SSMLearnPy → cache/predictable/more_*.jls)
#   (E)   share of the attainable variance reduction, (1 − model) / (1 − floor), over the first stride and strides 1–3,
#         for every condition; filled 60 training steps, open 300

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
R = deserialize(joinpath(ROOT, "cache", "predictable", "slip_results.jls"))
get(κ, n, obs) = only(filter(o -> o.κ == κ && o.NTR == n && o.obs == obs, R))
MORE = [deserialize(f) for f in (joinpath(ROOT, "cache", "predictable", "more_$(m).jls") for m in ("fip", "node", "lstm", "ssm")) if isfile(f)]
getm(κ, n, obs) = [only(filter(o -> o.κ == κ && o.NTR == n && o.obs == obs, M)) for M in MORE]
const C_LTP = colorant"#1a4f99"
# model, label, colour, line style (linear solid, nonlinear dashed; colours as Figures 5–6)
SLIPMODELS = [("FIP", "Fourier in phase", C_LTP, :solid, :diamond), ("DMD", "Hankel-DMD", CAT[1], :solid, :circle),
              ("ESN", "ESN", CAT[2], :dash, :utriangle), ("LSTM", "LSTM", CAT[5], :dash, :dtriangle),
              ("NODE", "neural ODE", CAT[3], :dash, :hexagon), ("SSM", "SSMLearnPy", CAT[4], :dash, :star5)]
"Remaining variance over the horizon for model `base` (median over its seeds); nothing if not run."
function rrv(κ, n, obs, base)
    objs = vcat([get(κ, n, obs)], getm(κ, n, obs))
    v = [1 .- o.skill[nm] for o in objs for nm in o.names if startswith(nm, base * " ")]
    isempty(v) ? nothing : vec(median(reduce(hcat, v); dims = 2))
end
floorv(κ, n, obs) = 1 .- get(κ, n, obs).skill["ceiling"]
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
C_DMD = CAT[1]; C_ESN = CAT[2]
const SPS = 100                                     # samples per stride
binned(v; w = 10) = [mean(v[i:i+w-1]) for i in 1:w:length(v)-w+1]
hx(v; w = 10) = ((0:length(binned(v; w))-1) .+ 0.5) .* w ./ SPS
esn_med(o; y = false) = vec(median(reduce(hcat, [(y ? o.skilly : o.skill)["ESN $sd"] for sd in 1:4]); dims = 2))

fig = Figure(size = (COL2, 150MM))
gA = fig[1, 1] = GridLayout(); gB = fig[1, 2] = GridLayout()
gC = fig[2, 1] = GridLayout(); gD = fig[2, 2] = GridLayout(); gE = fig[3, 1:2] = GridLayout()
colgap!(fig.layout, 16); rowgap!(fig.layout, 10)

# ── A: one forecast ─────────────────────────────────────────────────────────
o = get(0.1, 60, :full); ex = o.examples[2]
th = ((1:size(ex.hist, 1)) .- size(ex.hist, 1)) ./ SPS; tf = (1:size(ex.fut, 1)) ./ SPS
mfut = ex.fc["mean cycle"]
dev(F, M) = F[:, 1] .- M[:, 1]                       # height deviation from the mean step (z units)
axA = Axis(gA[1, 1]; xlabel = "Strides from the forecast start", ylabel = "Height deviation\nfrom the mean step (SD)",
           title = "One forecast: SLIP with feedback")
refline!(axA, 0.0; linestyle = :solid)
vlo = [h for h in 1:size(ex.fut, 1) if mod1(ex.j + h, 50) == 1] ./ SPS
for v in vlo; vlines!(axA, v; color = (C_REF, 0.35), linewidth = 0.6, linestyle = :dot); end
lines!(axA, th, dev(ex.hist, ex.mstep_hist); color = (C_REF, 0.9), linewidth = 0.9, label = "observed (noisy)")
lines!(axA, tf, dev(ex.fut, mfut); color = C_TRUTH, linewidth = 1.4, label = "what happened")
lines!(axA, tf, dev(Float64.(ex.fc["ceiling"]), mfut); color = C_TRUTH, linestyle = :dash, linewidth = 1.1,
       label = "predictable part (floor)")
for sd in 1:4
    lines!(axA, tf, dev(Float64.(ex.fc["ESN $sd"]), mfut); color = (C_ESN, 0.6), linewidth = 0.8,
           label = sd == 1 ? "ESN, 4 seeds" : nothing)
end
lines!(axA, tf, dev(Float64.(ex.fc["DMD 0"]), mfut); color = C_DMD, linewidth = 1.2, label = "Hankel-DMD")
axislegend(axA; position = (:right, :top), labelsize = 6.5, patchsize = (12, 5), unique = true)
yl = maximum(abs, dev(ex.hist, ex.mstep_hist))
text!(axA, vlo[1], -yl; text = "dotted: next disturbances", fontsize = 6, color = C_REF,
      align = (:left, :bottom), offset = (2, 0))
xlims!(axA, -2, 3); ylims!(axA, -1.1yl, 1.75yl)
plabel!(gA[1, 1], "A")

# ── B–D: relative remaining variance over the horizon (1 − skill; skill = 1 − MSE / MSE of the mean cycle) ──
function skillpanel!(g, κ, obs, title)
    ax = Axis(g[1, 1]; xlabel = "Forecast horizon (strides)", ylabel = "Remaining variance\n(relative to the mean cycle)",
              title, limits = (0, 3, -0.02, 1.6), yticks = [0, 0.5, 1, 1.5])
    vspan!(ax, 0, 1; color = C_SHADE)
    refline!(ax, 1.0; linestyle = :solid)
    o = get(κ, 60, obs); S = Dict(k => 1 .- v for (k, v) in o.skill)
    lines!(ax, hx(S["persistence"]), binned(S["persistence"]); color = C_REF, linestyle = :dot, linewidth = 1.0, label = "persistence (last step)")
    lines!(ax, hx(S["ceiling"]), binned(S["ceiling"]); color = C_TRUTH, linewidth = 1.8, label = "floor: no future disturbances")
    for (base, lab, col, ls, _) in SLIPMODELS
        v = rrv(κ, 60, obs, base); v === nothing && continue
        lines!(ax, hx(v), binned(v); color = col, linestyle = ls, linewidth = 1.2, label = lab)
    end
    ax
end
axB = skillpanel!(gB, 0.1, :full, "SLIP with feedback, full CoM state")
plabel!(gB[1, 1], "B")
skillpanel!(gC, 0.0, :full, "SLIP open loop, full CoM state"); plabel!(gC[1, 1], "C")
skillpanel!(gD, 0.1, :height, "SLIP with feedback, height only"); plabel!(gD[1, 1], "D")
# ── E: share of the attainable variance reduction ───────────────────────────
# shares below −60% (more variance than the mean cycle by far) are drawn in a grey band at the bottom
const OFF_LO, OFF_HI, OFF_Y = -0.9, -0.62, -0.76
offscale(v) = v < OFF_HI ? OFF_Y : v
offband!(ax) = hspan!(ax, OFF_LO, OFF_HI; color = (C_REF, 0.12))
bin(v, a, b) = mean(v[round(Int, a * SPS)+1:round(Int, b * SPS)])
share(κ, n, obs, base, a, b) = (v = rrv(κ, n, obs, base); v === nothing ? NaN : (1 - bin(v, a, b)) / (1 - bin(floorv(κ, n, obs), a, b)))
CONDS = [(0.1, :full, "feedback,\nfull state"), (0.0, :full, "open loop,\nfull state"), (0.1, :height, "feedback,\nheight only")]
WINS = [(0, 1, "first stride"), (1, 3, "strides 1–3")]
axE = Axis(gE[1, 1]; ylabel = "share of the attainable\nvariance reduction", title = "E   How much of the predictable response each model forecasts",
           titlefont = :bold, xticks = (1:6, ["$(c[3])\n$(w[3])" for w in WINS for c in CONDS]), xticklabelsize = 6.5,
           limits = (0.4, 6.6, OFF_LO, 1.08), yticks = ([OFF_Y, -0.5, 0, 0.5, 1], ["below\n−60%", "−50%", "0", "50%", "100%"]))
offband!(axE)
hlines!(axE, 1; color = C_TRUTH, linewidth = 0.8); hlines!(axE, 0; color = C_REF, linewidth = 0.6)
for (g, (w, c)) in enumerate([(w, c) for w in WINS for c in CONDS])
    for (k, (base, _, col, _, mk)) in enumerate(SLIPMODELS)
        x = g + (k - 3.5) * 0.12
        for (n, filled) in ((60, true), (300, false))
            v = share(c[1], n, c[2], base, w[1], w[2]); isfinite(v) || continue
            scatter!(axE, [x], [offscale(v)]; color = filled ? col : :white, strokecolor = col, strokewidth = filled ? 0 : 0.9,
                     marker = mk, markersize = 6)
        end
    end
end
gL = gE[1, 2] = GridLayout()
Legend(gL[1, 1], vcat([LineElement(color = C_TRUTH, linewidth = 1.8), LineElement(color = C_REF, linestyle = :dot)],
                      [[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 5)] for (_, _, col, ls, mk) in SLIPMODELS]),
       vcat(["floor: no future disturbances", "persistence (last step)"], [lab for (_, lab, _, _, _) in SLIPMODELS]);
       framevisible = false, labelsize = 6.5, patchsize = (14, 5), rowgap = 0, tellheight = false)
Label(gL[2, 1], "B–D: 60 training steps\nE: filled 60, open 300"; fontsize = 6.5, color = C_REF, tellwidth = false)
rowgap!(gL, 2)
colsize!(gE, 2, Fixed(120))
rowsize!(fig.layout, 3, Relative(0.3))
save_paper(fig, "Figure_03")

for (κ, obs, _) in CONDS, n in (60, 300)
    @printf("κ %.1f %3d %-6s | floor %.2f / %.2f |", κ, n, obs, bin(floorv(κ, n, obs), 0, 1), bin(floorv(κ, n, obs), 1, 3))
    for (base, _, _, _, _) in SLIPMODELS
        v = rrv(κ, n, obs, base); v === nothing && continue
        @printf(" %s %.2f / %.2f (%.0f%%, %.0f%%) |", base, bin(v, 0, 1), bin(v, 1, 3), 100share(κ, n, obs, base, 0, 1), 100share(κ, n, obs, base, 1, 3))
    end
    println()
end
pct(v) = v < OFF_HI ? "below −60%" : replace(@sprintf("%.0f%%", 100v), "-" => "−")
scorecard!("R1slip", Dict(base => pct(share(0.1, 60, :full, base, 1, 3))
                          for (base, _, _, _, _) in SLIPMODELS if rrv(0.1, 60, :full, base) !== nothing))
