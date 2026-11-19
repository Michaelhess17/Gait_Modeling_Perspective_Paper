#!/usr/bin/env julia
# Figure_07.jl — Working with realistic recordings (R4) (manuscript: @fig-realistic). Run: julia --project=. scripts/figures/Figure_07.jl
#   A  what a gait recording contains: one able-bodied 15-s trial (fig5_gait_data.jl, cache/fig5/gait.jls), the knee angle
#      of every stride and the mean cycle; below, each stride's deviation from the mean cycle
#   B  share of each trial's variance in the deviations against the strides in the trial; 299 trials, 42 people
#      (able-bodied AB; higher- and lower-functioning stroke survivors HF and LF)
#   C  record length: running deviations (as Figure 4 C), remaining variance ⅛ stride ahead against training strides;
#      shaded, the strides in a 15-s walking trial
#   D  measurement noise: the same with added white noise (× each channel's deviation SD), 150 training strides; the LSTM
#      and the lattice were run without noise only
#   E  the multiplier itself: kicked SLIP walks (open loop, ρ = 0.95; fig3_slip_maps.jl, cache/fig3/slip_maps.jls), the
#      spectral radius of fitted step maps against record length, without noise (solid) and with noise 0.25× the
#      stride-to-stride SD (dashed); median and interquartile range over 20 walks. Section map by least squares and by
#      instrumental variables (IV, previous step as the instrument), and 8-sector piecewise-linear maps
#   F  the same against measurement noise, 250 steps, with phase-indexed maps (Revzen & Guckenheimer) added
#   G  partial observation: SLIP with speed feedback (Figure 3; 60 training steps), the share of the attainable variance
#      reduction over strides 1–3 with the full centre-of-mass state and with height only (median over seeds)
#   H  cost: time to choose the settings and fit one running trial (150 strides); CPU models on one core (median over
#      trials; SSMLearnPy timed on one trial); GPU models (RTX 4080) trained 22 trials at once, time divided by 22
# Groups are drawn in greys and the step-map estimators in their own colours, so that colour means model class only in
# C, D, G and H. The eigenvalue scatter and the piecewise maps' dependence on the number of sectors are in Figure S12.
include(joinpath(@__DIR__, "running_common.jl"))

const GREY_GROUP = Dict("AB" => colorant"#2b2b2b", "HF" => colorant"#77766f", "LF" => colorant"#b3b2ad")
const MK_GROUP = Dict("AB" => :circle, "HF" => :rect, "LF" => :utriangle)
#                 estimator        label                  colour               marker
const EST = [("section", 0, "section map", colorant"#6a4c9c", :circle), ("section_iv", 0, "section, IV", colorant"#b23a48", :utriangle),
             ("phase", 0, "phase-indexed", colorant"#6b8e23", :rect), ("piecewise", 8, "piecewise (8)", colorant"#7a7975", :xcross)]
walk = (9.4, 14.9)                       # interquartile strides in a 15-s walking trial (gait_group_fits.jl periods)
fig = Figure(size = (COL2, 150MM))
TS = (; titlesize = 7.5, xlabelsize = 7, ylabelsize = 7)

# A: one gait trial
G = deserialize(joinpath(ROOT, "cache", "fig5", "gait.jls"))
ex = G.example; S = ex.S; ph = range(0, 100; length = size(S, 2) + 1)[1:end-1]
mcyc = dropdims(mean(S, dims = 1), dims = 1); Dv = S .- reshape(mcyc, 1, size(mcyc)...)
gA = fig[1, 1] = GridLayout()
axA1 = Axis(gA[1, 1]; ylabel = "knee (°)", title = @sprintf("one walking trial: %d strides", size(S, 1)), xticklabelsvisible = false,
            limits = (0, 100, nothing, nothing), TS...)
for s in axes(S, 1); lines!(axA1, ph, S[s, :, 2]; color = (C_REF, 0.6), linewidth = 0.5); end
lines!(axA1, ph, mcyc[:, 2]; color = C_TRUTH, linewidth = 1.4)
axA2 = Axis(gA[2, 1]; xlabel = "stride phase (%)", ylabel = "deviation (°)", limits = (0, 100, nothing, nothing), TS...)
for s in axes(S, 1); lines!(axA2, ph, Dv[s, :, 2]; color = (C_TRUTH, 0.45), linewidth = 0.5); end
hlines!(axA2, 0; color = C_TRUTH, linewidth = 0.6)
rowsize!(gA, 2, Relative(0.38)); rowgap!(gA, 3)
devsh = sum(abs2, Dv) / sum(abs2, S .- mean(S, dims = (1, 2)))
text!(axA1, 2, maximum(S[:, :, 2]); text = @sprintf("deviations: %.1f%%\nof the variance", 100devsh), fontsize = 6, align = (:left, :top))
panel_label!(gA, "A")

# B: deviation share against strides per trial
axB = Axis(fig[1, 2]; yscale = log10, xlabel = "strides in the trial", ylabel = "deviations (% of variance)",
           title = "15-s walking trials", yticks = ([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"]),
           limits = (2, 20.5, 0.4, 70), TS...)
for gr in GROUP_ORDER_FIG
    k = G.grp .== gr
    scatter!(axB, G.nstr[k] .+ 0.25 .* (rand(count(k)) .- 0.5), 100 .* G.devshare[k]; color = (GREY_GROUP[gr], 0.75), marker = MK_GROUP[gr],
             markersize = 4, label = @sprintf("%s (median %.1f%%)", gr, 100median(G.devshare[k])))
end
axislegend(axB; position = (:right, :top), framevisible = false, labelsize = 6, rowgap = 0, patchsize = (6, 6))
panel_label!(fig[1, 2], "B")

# C: record length (running)
axC = Axis(fig[1, 3]; xscale = log2, xticks = SZ, xlabel = "training strides", ylabel = "remaining variance, ⅛ stride ahead",
           title = "record length (running)", TS...)
vspan!(axC, walk...; color = C_SHADE)
text!(axC, sqrt(walk[1] * walk[2]), 0.18; text = "15-s\nwalking\ntrial", fontsize = 5.5, align = (:center, :bottom), color = C_REF)
refline!(axC, 1)
for (m, nm, col, mk, ls) in MODELS
    ns = [n for n in SZ if !isempty(vals(m, n, 0.0, 2))]
    isempty(ns) && continue
    band!(axC, ns, [q(vals(m, n, 0.0, 2), 0.25) for n in ns], [q(vals(m, n, 0.0, 2), 0.75) for n in ns]; color = (col, 0.12))
    scatterlines!(axC, ns, [median(vals(m, n, 0.0, 2)) for n in ns]; color = col, marker = mk, markersize = 5, linestyle = ls)
end
ylims!(axC, 0.15, 1.6)
panel_label!(fig[1, 3], "C")

# D: measurement noise (running)
axD = Axis(fig[2, 1]; xlabel = "added noise (× deviation SD)", ylabel = "remaining variance, ⅛ stride ahead",
           title = "measurement noise (running)", xticks = [0, 0.5, 1], TS...)
MODELS_E = [x for x in MODELS if x[1] ∉ (:lattice, :lstm)]
for (k, (m, nm, col, mk, ls)) in enumerate(MODELS_E)
    ηs = [η for η in (0.0, 0.5, 1.0) if !isempty(vals(m, 150, η, 2))]
    isempty(ηs) && continue
    off = (k - (length(MODELS_E) + 1) / 2) * 0.022
    rangebars!(axD, ηs .+ off, [q(vals(m, 150, η, 2), 0.25) for η in ηs], [q(vals(m, 150, η, 2), 0.75) for η in ηs]; color = col, whiskerwidth = 0)
    scatterlines!(axD, ηs .+ off, [median(vals(m, 150, η, 2)) for η in ηs]; color = col, marker = mk, markersize = 5, linestyle = ls)
end
ylims!(axD, 0.15, 1.0)
panel_label!(fig[2, 1], "D")

# E, F: the multiplier estimated from step maps (SLIP)
SM = deserialize(joinpath(ROOT, "cache", "fig3", "slip_maps.jls")); rows = SM.rows
ρtrue = maximum(abs, SM.truths[0.0])
sel(; kappa = 0.0, method, K = 0, N = nothing, noise = nothing) =
    [x for x in rows if x.kappa == kappa && x.method == method && x.K == K && (N === nothing || x.N == N) && (noise === nothing || x.noise == noise)]
qq(v, p) = isempty(v) ? NaN : quantile(v, p)
NS = sort(unique([x.N for x in rows]))
axE = Axis(fig[2, 2]; xscale = log10, xticks = (NS[[1, 3, 5, 7, 9]], string.(NS[[1, 3, 5, 7, 9]])), xlabel = "steps of data",
           ylabel = "estimated multiplier, ρ̂", title = "the multiplier: record length (SLIP)", limits = (8, 2500, 0.0, 1.05), TS...)
hlines!(axE, ρtrue; color = C_TRUTH, linestyle = :dash, linewidth = 0.9)
for (m, K, _, col, _) in EST[[1, 4]], (r, ls) in ((0.0, :solid), (0.25, :dash))
    vs = [[x.ρ for x in sel(; method = m, K, N, noise = r)] for N in NS]
    r == 0.25 && band!(axE, NS, qq.(vs, 0.25), qq.(vs, 0.75); color = (col, 0.15))
    lines!(axE, NS, median.(vs); color = col, linestyle = ls, linewidth = 1.3)
end
vs = [[x.ρ for x in sel(; method = "section_iv", N, noise = 0.25)] for N in NS]
band!(axE, NS, qq.(vs, 0.25), qq.(vs, 0.75); color = (EST[2][4], 0.15))
lines!(axE, NS, median.(vs); color = EST[2][4], linestyle = :dash, linewidth = 1.3)
text!(axE, 2200, ρtrue + 0.015; text = "truth", fontsize = 6, align = (:right, :bottom))
axislegend(axE, vcat([[LineElement(color = col), MarkerElement(color = col, marker = mk, markersize = 4.5)] for (_, _, _, col, mk) in EST],
                     [LineElement(color = C_REF), LineElement(color = C_REF, linestyle = :dash)]),
           vcat([lab for (_, _, lab, _, _) in EST], ["no noise", "noise 0.25×"]); position = (0.55, 0.62), nbanks = 3,
           orientation = :horizontal, framevisible = false, labelsize = 6, rowgap = 0, colgap = 6, patchsize = (12, 5))
panel_label!(fig[2, 2], "E")
NOISE = sort(unique([x.noise for x in rows]))
axF = Axis(fig[2, 3]; xlabel = "noise (× stride-to-stride SD)", ylabel = "estimated multiplier, ρ̂",
           title = "the multiplier: noise (SLIP)", limits = (-0.03, 1.03, 0.0, 1.05), TS...)
hlines!(axF, ρtrue; color = C_TRUTH, linestyle = :dash, linewidth = 0.9)
for (m, K, lab, col, mk) in EST
    vs = [[x.ρ for x in sel(; method = m, K, N = 250, noise = r)] for r in NOISE]
    band!(axF, NOISE, qq.(vs, 0.25), qq.(vs, 0.75); color = (col, 0.15))
    scatterlines!(axF, NOISE, median.(vs); color = col, marker = mk, markersize = 4.5, linewidth = 1.3)
end
panel_label!(fig[2, 3], "F")

# G: partial observation (SLIP, Figure 3's data)
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
const OFF_LO, OFF_HI, OFF_Y = -0.9, -0.62, -0.76
offscale(v) = v < OFF_HI ? OFF_Y : v
axG = Axis(fig[3, 1]; xticks = (1:length(SLIPM), [x[2] for x in SLIPM]), xticklabelrotation = π / 5, xticklabelsize = 6,
           ylabel = "share of the attainable reduction,\nstrides 1–3", title = "partial observation (SLIP)",
           limits = (0.4, length(SLIPM) + 0.6, OFF_LO, 1.1), yticks = ([OFF_Y, -0.5, 0, 0.5, 1], ["below\n−60%", "−50%", "0", "50%", "100%"]), TS...)
hspan!(axG, OFF_LO, OFF_HI; color = (C_REF, 0.12))                 # shares below −60% are drawn in this band
hlines!(axG, 1; color = C_TRUTH, linewidth = 0.8); hlines!(axG, 0; color = C_REF, linewidth = 0.6)
for (k, (base, _, col, mk)) in enumerate(SLIPM)
    f, h = sshare(0.1, 60, :full, base), sshare(0.1, 60, :height, base)
    (isfinite(f) && isfinite(h)) || continue
    lines!(axG, [k - 0.12, k + 0.12], offscale.([f, h]); color = (col, 0.6), linewidth = 0.9)
    scatter!(axG, [k - 0.12], [offscale(f)]; color = col, marker = mk, markersize = 5)
    scatter!(axG, [k + 0.12], [offscale(h)]; color = :white, strokecolor = col, strokewidth = 0.9, marker = mk, markersize = 5)
end
panel_label!(fig[3, 1], "G")

# H: cost
Lc = L[0.0].R
cpu(m) = median(r[(m, 150)].fit_secs + r[(m, 150)].select_secs for r in Lc if r !== nothing && haskey(r, (m, 150)))
NODE150 = haskey(NODE, 0.0) ? NODE[0.0][150].fit_secs / length(NODE[0.0][150].rows) : NaN
LSTM150 = isfile(LSTMF[0.0]) ? (O = deserialize(LSTMF[0.0]).out[150]; O.fit_secs / O.nmodels) : NaN
COST = [("phase-binned", cpu(:lpv), C_LPV, false), ("time-invariant (DMD)", cpu(:dmd), CAT[1], false), ("Fourier in phase", cpu(:ltp), C_LTP, false),
        ("ESN", cpu(:esn), CAT[2], false), ("SSMLearnPy", 13.9, CAT[4], false), ("neural ODE", NODE150, CAT[3], true), ("LSTM", LSTM150, CAT[5], true)]
axH = Axis(fig[3, 2:3]; xscale = log10, yticks = (1:length(COST), [c[1] for c in COST]), yticklabelsize = 6.5, xlabel = "seconds per trial (150 strides)",
           title = "cost of choosing settings and fitting (running)", xticks = ([1, 10, 100, 1000], ["1", "10", "100", "1000"]),
           limits = (0.5, 3000, 0.4, length(COST) + 0.6), yreversed = true, TS...)
for (k, (nm, t, col, gpu)) in enumerate(COST)
    isfinite(t) || continue
    barplot!(axH, [k], [t]; direction = :x, fillto = 0.5, color = gpu ? (col, 0.45) : col, strokecolor = col, strokewidth = 0.8, width = 0.6)
    text!(axH, t, k; text = (t < 10 ? @sprintf(" %.1f s", t) : @sprintf(" %.0f s", t)) * (gpu ? " (GPU, batched)" : ""), fontsize = 6, align = (:left, :center))
end
panel_label!(fig[3, 2:3], "H")
Legend(fig[4, 1:3], [[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 5)] for (_, _, col, mk, ls) in MODELS],
       [nm for (_, nm, _, _, _) in MODELS]; framevisible = false, labelsize = 6.5, orientation = :horizontal, nbanks = 2, colgap = 10, rowgap = 0,
       patchsize = (14, 5))
Label(fig[5, 1:3], "C, D, G, H: model classes; E, F: estimators of the step map's multiplier; B: groups (greys)"; fontsize = 6, color = C_REF)
colgap!(fig.layout, 16); rowgap!(fig.layout, 10); rowgap!(fig.layout, 3, 6); rowgap!(fig.layout, 4, 0)
save_paper(fig, "Figure_07")

for (base, nm, _, _) in SLIPM
    @printf("%-18s strides 1–3 share: full %.0f%%, height %.0f%% | first stride: full %.0f%%, height %.0f%%\n", nm, 100sshare(0.1, 60, :full, base),
            100sshare(0.1, 60, :height, base),
            100 * (v = srrv(0.1, 60, :full, base); v === nothing ? NaN : (1 - sbin(v, 0, 1)) / (1 - sbin(sfloor(0.1, 60, :full), 0, 1))),
            100 * (v = srrv(0.1, 60, :height, base); v === nothing ? NaN : (1 - sbin(v, 0, 1)) / (1 - sbin(sfloor(0.1, 60, :height), 0, 1))))
end
for (nm, t, _, gpu) in COST; @printf("cost %-22s %.1f s%s\n", nm, t, gpu ? " (GPU, per trial of a 22-trial batch)" : ""); end
@printf("example %s trial %d: %d strides, deviation share %.2f%%\n", ex.subject, ex.i, size(S, 1), 100devsh)
for (m, K, lab, _, _) in EST
    @printf("%-22s ρ̂ (truth %.3f) | 30 steps, noise 0.25: %.3f | 2000 steps, noise 0.25: %.3f | 250 steps, noise 0.5: %.3f\n", lab, ρtrue,
            median([x.ρ for x in sel(; method = m, K, N = 30, noise = 0.25)]), median([x.ρ for x in sel(; method = m, K, N = 2000, noise = 0.25)]),
            median([x.ρ for x in sel(; method = m, K, N = 250, noise = 0.5)]))
end
KEY = Dict(:ltp => "FIP", :lpv => "LPV", :dmd => "DMD", :esn => "ESN", :lstm => "LSTM", :node => "NODE", :ssm => "SSM")
r16(m) = (v = vals(m, 16, 0.0, 2); isempty(v) ? "—" : @sprintf("%.2f", median(v)))
hgt(b) = (h = sshare(0.1, 60, :height, b); !isfinite(h) ? "—" : h < OFF_HI ? "below −60%" : replace(@sprintf("%.0f%%", 100h), "-" => "−"))
COSTK = Dict("phase-binned" => "LPV", "time-invariant (DMD)" => "DMD", "Fourier in phase" => "FIP", "ESN" => "ESN", "SSMLearnPy" => "SSM",
             "neural ODE" => "NODE", "LSTM" => "LSTM")
costs = Dict(COSTK[nm] => (isfinite(t) ? (t < 10 ? @sprintf("%.1f s", t) : @sprintf("%.0f s", t)) * (gpu ? " (GPU)" : "") : "—") for (nm, t, _, gpu) in COST)
scorecard!("R4", Dict(k => join([r16(m), hgt(k), get(costs, k, "—")], " · ") for (m, k) in KEY))
