#!/usr/bin/env julia
# Figure_09.jl — Working with realistic recordings (R4) (manuscript: @fig-realistic). Run: julia --project=. scripts/figures/Figure_09.jl
#   A  record length: running deviations (as Figure 6C), remaining variance ⅛ stride ahead against training strides;
#      shaded, the strides in a 15-s walking trial
#   B  measurement noise: the same with added white noise (× each channel's deviation SD), 150 training strides; the LSTM
#      and the lattice were run without noise only
#   C  partial observation: SLIP with speed feedback (Figure 5; 60 training steps), the share of the attainable variance
#      reduction over strides 1–3 with the full centre-of-mass state and with height only (median over seeds)
#   D  cost: time to choose the settings and fit one running trial (150 strides); CPU models on one core (median over
#      trials; SSMLearnPy timed on one trial); GPU models (RTX 4080) trained 22 trials at once, time divided by 22
include(joinpath(@__DIR__, "running_common.jl"))

walk = (9.4, 14.9)                       # interquartile strides in a 15-s walking trial (gait_group_fits.jl periods)
fig = Figure(size = (COL2, 116MM))
# A: training strides
axA = Axis(fig[1, 1]; xscale = log2, xticks = SZ, xlabel = "training strides", ylabel = "remaining variance, ⅛ stride ahead",
           title = "record length (running)")
vspan!(axA, walk...; color = C_SHADE)
text!(axA, sqrt(walk[1] * walk[2]), 0.18; text = "15-s walking\ntrial", fontsize = 6, align = (:center, :bottom), color = C_REF)
refline!(axA, 1)
for (m, nm, col, mk, ls) in MODELS
    ns = [n for n in SZ if !isempty(vals(m, n, 0.0, 2))]
    isempty(ns) && continue
    band!(axA, ns, [q(vals(m, n, 0.0, 2), 0.25) for n in ns], [q(vals(m, n, 0.0, 2), 0.75) for n in ns]; color = (col, 0.12))
    scatterlines!(axA, ns, [median(vals(m, n, 0.0, 2)) for n in ns]; color = col, marker = mk, markersize = 6, linestyle = ls)
end
ylims!(axA, 0.15, 1.6)
panel_label!(fig[1, 1], "A")
# B: noise
axB = Axis(fig[1, 2]; xlabel = "added measurement noise (× deviation SD)", ylabel = "remaining variance, ⅛ stride ahead",
           title = "measurement noise (running)", xticks = [0, 0.5, 1])
MODELS_E = [x for x in MODELS if x[1] ∉ (:lattice, :lstm)]
for (k, (m, nm, col, mk, ls)) in enumerate(MODELS_E)
    ηs = [η for η in (0.0, 0.5, 1.0) if !isempty(vals(m, 150, η, 2))]
    isempty(ηs) && continue
    off = (k - (length(MODELS_E) + 1) / 2) * 0.022
    rangebars!(axB, ηs .+ off, [q(vals(m, 150, η, 2), 0.25) for η in ηs], [q(vals(m, 150, η, 2), 0.75) for η in ηs]; color = col, whiskerwidth = 0)
    scatterlines!(axB, ηs .+ off, [median(vals(m, 150, η, 2)) for η in ηs]; color = col, marker = mk, markersize = 6, linestyle = ls)
end
ylims!(axB, 0.15, 1.0)
panel_label!(fig[1, 2], "B")
# C: partial observation (SLIP, Figure 5's data)
PRED = joinpath(ROOT, "cache", "predictable")
PR = deserialize(joinpath(PRED, "slip_results.jls"))
PMORE = [deserialize(f) for f in (joinpath(PRED, "more_$(m).jls") for m in ("fip", "node", "lstm", "ssm")) if isfile(f)]
pick(V, κ, n, obs) = only(filter(o -> o.κ == κ && o.NTR == n && o.obs == obs, V))
function srrv(κ, n, obs, base)
    objs = vcat([pick(PR, κ, n, obs)], [pick(M, κ, n, obs) for M in PMORE])
    v = [1 .- o.skill[nm] for o in objs for nm in o.names if startswith(nm, base * " ")]
    isempty(v) ? nothing : vec(median(reduce(hcat, v); dims = 2))
end
sbin(v, a, b) = mean(v[round(Int, a * 100)+1:round(Int, b * 100)])
sfloor(κ, n, obs) = 1 .- pick(PR, κ, n, obs).skill["ceiling"]
sshare(κ, n, obs, base) = (v = srrv(κ, n, obs, base); v === nothing ? NaN : (1 - sbin(v, 1, 3)) / (1 - sbin(sfloor(κ, n, obs), 1, 3)))
SLIPM = [("FIP", "Fourier in phase", C_LTP, :diamond), ("DMD", "Hankel-DMD", CAT[1], :circle), ("ESN", "ESN", CAT[2], :utriangle),
         ("LSTM", "LSTM", CAT[5], :dtriangle), ("NODE", "neural ODE", CAT[3], :hexagon), ("SSM", "SSMLearnPy", CAT[4], :star5)]
axC = Axis(fig[2, 1]; xticks = (1:length(SLIPM), [x[2] for x in SLIPM]), xticklabelrotation = π / 5, xticklabelsize = 6.5,
           ylabel = "share of the attainable\nvariance reduction, strides 1–3", title = "partial observation (SLIP with feedback)",
           limits = (0.4, length(SLIPM) + 0.6, -0.6, 1.1), yticks = ([-0.5, 0, 0.5, 1], ["−50%", "0", "50%", "100%"]))
hlines!(axC, 1; color = C_TRUTH, linewidth = 0.8); hlines!(axC, 0; color = C_REF, linewidth = 0.6)
for (k, (base, _, col, mk)) in enumerate(SLIPM)
    f, h = sshare(0.1, 60, :full, base), sshare(0.1, 60, :height, base)
    (isfinite(f) && isfinite(h)) || continue
    lines!(axC, [k - 0.12, k + 0.12], [f, max(h, -0.58)]; color = (col, 0.6), linewidth = 0.9)
    scatter!(axC, [k - 0.12], [f]; color = col, marker = mk, markersize = 6)
    scatter!(axC, [k + 0.12], [max(h, -0.58)]; color = :white, strokecolor = col, strokewidth = 0.9, marker = mk, markersize = 6)
end
text!(axC, length(SLIPM) + 0.55, -0.55; text = "filled: full state; open: height only", fontsize = 6, color = C_REF, align = (:right, :bottom))
panel_label!(fig[2, 1], "C")
# D: cost
Lc = L[0.0].R
cpu(m) = median(r[(m, 150)].fit_secs + r[(m, 150)].select_secs for r in Lc if r !== nothing && haskey(r, (m, 150)))
NODE150 = haskey(NODE, 0.0) ? NODE[0.0][150].fit_secs / length(NODE[0.0][150].rows) : NaN
LSTM150 = isfile(LSTMF[0.0]) ? (O = deserialize(LSTMF[0.0]).out[150]; O.fit_secs / O.nmodels) : NaN
COST = [("phase-binned", cpu(:lpv), C_LPV, false), ("time-invariant (DMD)", cpu(:dmd), CAT[1], false), ("Fourier in phase", cpu(:ltp), C_LTP, false),
        ("ESN", cpu(:esn), CAT[2], false), ("SSMLearnPy", 13.9, CAT[4], false), ("neural ODE", NODE150, CAT[3], true), ("LSTM", LSTM150, CAT[5], true)]
axD = Axis(fig[2, 2]; xscale = log10, yticks = (1:length(COST), [c[1] for c in COST]), yticklabelsize = 6.5, xlabel = "seconds per trial (150 strides)",
           title = "cost of choosing settings and fitting", xticks = ([1, 10, 100, 1000], ["1", "10", "100", "1000"]),
           limits = (0.5, 3000, 0.4, length(COST) + 0.6), yreversed = true)
for (k, (nm, t, col, gpu)) in enumerate(COST)
    isfinite(t) || continue
    barplot!(axD, [k], [t]; direction = :x, fillto = 0.5, color = gpu ? (col, 0.45) : col, strokecolor = col, strokewidth = 0.8, width = 0.6)
    text!(axD, t, k; text = (t < 10 ? @sprintf(" %.1f s", t) : @sprintf(" %.0f s", t)) * (gpu ? " (GPU, batched)" : ""), fontsize = 6, align = (:left, :center))
end
panel_label!(fig[2, 2], "D")
Legend(fig[3, 1:2], [[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 6)] for (_, _, col, mk, ls) in MODELS],
       [nm for (_, nm, _, _, _) in MODELS]; framevisible = false, labelsize = 6.5, orientation = :horizontal, nbanks = 2, colgap = 10, rowgap = 0)
colgap!(fig.layout, 18); rowgap!(fig.layout, 10); rowgap!(fig.layout, 2, 4)
save_paper(fig, "Figure_09")

for (base, nm, _, _) in SLIPM
    @printf("%-18s strides 1–3 share: full %.0f%%, height %.0f%% | first stride: full %.0f%%, height %.0f%%\n", nm, 100sshare(0.1, 60, :full, base),
            100sshare(0.1, 60, :height, base),
            100 * (v = srrv(0.1, 60, :full, base); v === nothing ? NaN : (1 - sbin(v, 0, 1)) / (1 - sbin(sfloor(0.1, 60, :full), 0, 1))),
            100 * (v = srrv(0.1, 60, :height, base); v === nothing ? NaN : (1 - sbin(v, 0, 1)) / (1 - sbin(sfloor(0.1, 60, :height), 0, 1))))
end
for (nm, t, _, gpu) in COST; @printf("cost %-22s %.1f s%s\n", nm, t, gpu ? " (GPU, per trial of a 22-trial batch)" : ""); end
KEY = Dict(:ltp => "FIP", :lpv => "LPV", :dmd => "DMD", :esn => "ESN", :lstm => "LSTM", :node => "NODE", :ssm => "SSM")
r16(m) = (v = vals(m, 16, 0.0, 2); isempty(v) ? "—" : @sprintf("%.2f", median(v)))
hgt(b) = (h = sshare(0.1, 60, :height, b); isfinite(h) ? replace(@sprintf("%.0f%%", 100h), "-" => "−") : "—")
COSTK = Dict("phase-binned" => "LPV", "time-invariant (DMD)" => "DMD", "Fourier in phase" => "FIP", "ESN" => "ESN", "SSMLearnPy" => "SSM",
             "neural ODE" => "NODE", "LSTM" => "LSTM")
costs = Dict(COSTK[nm] => (isfinite(t) ? (t < 10 ? @sprintf("%.1f s", t) : @sprintf("%.0f s", t)) * (gpu ? " (GPU)" : "") : "—") for (nm, t, _, gpu) in COST)
scorecard!("R4", Dict(k => join([r16(m), hgt(k), get(costs, k, "—")], " · ") for (m, k) in KEY))
