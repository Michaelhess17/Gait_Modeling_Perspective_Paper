#!/usr/bin/env julia
# Figure_07.jl — R2: models run on their own; who keeps the cycle (manuscript: @fig-selfdriving).
# Run: julia --project=. scripts/figures/Figure_07.jl
#
# (A) Free runs from the end of a 12-cycle training window (realistic regime: 1 % measurement noise, process noise on
#     the simulated systems). Grey: held-out data; colour: free runs, every seed (ESN 8, LSTM 2, neural ODE 4) or
#     polynomial degree (SSMLearnPy 2) overlaid. Axes framed on the data; runs that leave the frame are cut. The label
#     gives the outcome counts after 30 cycles.
# (B) Outcome after 30 free-running cycles in three settings, every run pooled:
#       benchmark  13 cells: 4 simulated systems × 2 regimes (30 training cycles without noise; 12 with noise) and 5
#                  human gait recordings (self_driving_benchmark.jl; Fourier in phase and LSTM from benchmark_more.jl)
#       SLIP       the walker of Figure 5: 8 conditions × 10 held-out walks, 30 strides from the first test start of
#                  each walk (predictable_more.jl; predictable_ssm.py)
#       running    the whole signal of Figure 6: 22 trials, 150 training strides, real and stride-normalised time
#                  (maus_whole_r2.jl); the neural ODE and LSTM were not run freely on the whole running signal
#     Outcomes (self_driving_benchmark.jl): ok (amplitude 0.75–1.33 of the data's, period within 10 %), decays (right
#     period, 0.25–0.75), wrong rhythm (period off, or amplitude > 1.33), collapse (< 0.25), blow-up (non-finite or > 4).
# (C) Benchmark: error over the first free-running cycle, per system, against the phase-aligned mean-cycle template.
# (D) Benchmark: time each free run stays within ε of the true cycle (r2_tube.jl).
# (E) Running: free runs of the deviation models (maus_requirements.jl), size of the deviation relative to its SD.
#
# Settings are chosen on the training data only (DMD and ESN: dmd_tuning.jl, esn_tuning.jl; Fourier in phase:
# benchmark_more.jl, by the same criterion); the neural ODE uses one fixed recipe, SSMLearnPy the configuration tuned on
# gait, the LSTM the published recipe.

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, NPZ, Random, Printf

const ROOTF = dirname(dirname(@__DIR__))
const DIR = joinpath(ROOTF, "cache", "selfdrive")
const C_LTP = colorant"#1a4f99"
const SYSTEMS = [("fhn", "FitzHugh–Nagumo", (1, 2), ("v", "w")),
                 ("cfhn", "coupled FHN", (1, 2), ("v₁", "w₁")),
                 ("vdp2", "coupled Van der Pol", (1, 2), ("x_A", "ẋ_A")),
                 ("slip", "SLIP walking", (1, 3), ("CoM height", "vertical vel.")),
                 ("gait91", "human gait", (1, 2), ("hip", "knee"))]
#            key     label               colour  marker      linestyle
const MODELS = [("FIP", "Fourier in phase", C_LTP, :diamond, :solid), ("DMD", "Hankel-DMD", CAT[1], :circle, :solid),
                ("ESN", "ESN", CAT[2], :utriangle, :dash), ("LSTM", "LSTM", CAT[5], :dtriangle, :dash),
                ("NODE", "neural ODE", CAT[3], :hexagon, :dash), ("SSM", "SSMLearnPy", CAT[4], :star5, :dash)]
const FATES = ["ok", "decays", "wrong", "collapse", "blow-up"]
const FATE_COL = Dict("ok" => colorant"#3d7a5c", "decays" => colorant"#9cc7ae", "wrong" => colorant"#d9b45a",
                      "collapse" => colorant"#a9a8a4", "blow-up" => colorant"#b5523b")

"All free runs of one benchmark cell: template, tuned DMD and ESN, neural ODE, SSMLearnPy, Fourier in phase, LSTM."
function load_cell(sys, regime)
    rows = NamedTuple[]; Yte = nothing; Ytr = nothing; P = NaN
    for tag in ("dmd-esn-template", "node")
        f = joinpath(DIR, "$(sys)_$(regime)_$(tag).jls"); isfile(f) || continue
        o = deserialize(f)
        append!(rows, [(; r.model, r.variant, r.e1, r.fate, Fc = Float64.(r.Fc)) for r in o.rows if !(r.model in ("DMD", "ESN"))])
        Yte = o.Yte; Ytr = o.Ytr; P = o.P
    end
    f = joinpath(DIR, "dmdtuned_$(sys)_$(regime).jls")          # DMD and ESN with settings tuned on the training data
    if isfile(f)
        o = deserialize(f); c = o.chosen
        push!(rows, (; model = "DMD", variant = "τ=$(c.τ) r=$(c.r) $(c.variant)", c.e1, c.fate, Fc = Float64.(o.Fc)))
    end
    f = joinpath(DIR, "esntuned_$(sys)_$(regime).jls")
    isfile(f) && append!(rows, [(; model = "ESN", variant = "seed $(r.seed)", r.e1, r.fate, Fc = Float64.(r.Fc)) for r in deserialize(f).rows])
    f = joinpath(DIR, "ssm_$(sys)_$(regime).npz")
    if isfile(f)
        S = npzread(f)
        append!(rows, [(; model = "SSM", variant = "deg $(S["degrees"][k])", e1 = S["e1"][k], fate = FATES[S["fate_code"][k] + 1],
                          Fc = S["fc"][k, :, :]) for k in eachindex(S["degrees"])])
    end
    for tag in ("fip", "lstm")
        f = joinpath(DIR, "$(tag)_$(sys)_$(regime).jls")
        isfile(f) && append!(rows, [(; r.model, r.variant, r.e1, r.fate, Fc = Float64.(r.Fc)) for r in deserialize(f).rows])
    end
    return rows, Ytr, Yte, P
end

CELLS = [(m[1], m[2]) for f in readdir(DIR) for m in (match(r"^(.+)_(realistic|long)_dmd-esn-template\.jls$", f),) if m !== nothing]
allrows = [(; x.model, x.fate, x.e1, sys, reg) for (sys, reg) in CELLS for x in load_cell(sys, reg)[1]]
fsb = joinpath(DIR, "stabilization_benchmark.jls")              # Hankel-DMD retuned without its spectral constraint
isfile(fsb) && append!(allrows, [(; model = "DMD0", o.retuned.fate, o.retuned.e1, sys = o.system, reg = o.regime) for o in deserialize(fsb)])

fig = Figure(size = (COL2, 240MM))
gA = fig[1, 1] = GridLayout()
gB = fig[2, 1] = GridLayout()
gC = fig[3, 1] = GridLayout()

# ── (A) free runs, realistic regime ──────────────────────────────────────────
for (c, (sys, title, (i, j), (li, lj))) in enumerate(SYSTEMS)
    rows, Ytr, Yte, P = load_cell(sys, "realistic")
    Yte === nothing && continue
    n8 = min(size(Yte, 1), round(Int, 8P))
    lims = framed_limits(vcat(Ytr[:, i], Yte[1:n8, i]), vcat(Ytr[:, j], Yte[1:n8, j]); pad = 0.25)
    for (r, (m, nm, col, _, _)) in enumerate(MODELS)
        last = r == length(MODELS)
        ax = Axis(gA[r, c]; xticklabelsvisible = false, yticklabelsvisible = false, xlabel = last ? li : "",
                  ylabel = last ? lj : "", title = r == 1 ? title : "", titlealign = :center, xlabelsize = 7, ylabelsize = 7,
                  xticks = LinearTicks(3), yticks = LinearTicks(3))
        lines!(ax, Yte[1:n8, i], Yte[1:n8, j]; color = (C_REF, 0.5), linewidth = 1.0)
        mr = [x for x in rows if x.model == m]
        for x in mr
            xs, ys, _ = clip_to_view(x.Fc[:, i], x.Fc[:, j], lims)
            lines!(ax, xs, ys; color = (col, 0.75), linewidth = 0.8)
        end
        limits!(ax, lims...)
        if !isempty(mr)
            parts = ["$(n) $(f)" for (n, f) in ((count(x -> x.fate == f, mr), f) for f in FATES) if n > 0]
            txt = length(parts) > 2 ? join(parts[1:2], ", ") * ",\n" * join(parts[3:end], ", ") : join(parts, ", ")
            text!(ax, 0.03, 0.97; space = :relative, align = (:left, :top), fontsize = 5.5, text = txt, color = C_TRUTH)
        end
        c == length(SYSTEMS) && Label(gA[r, c+1], nm; rotation = -π / 2, fontsize = 7, tellheight = false, color = col)
    end
end
colgap!(gA, 4); rowgap!(gA, 3)
Label(gA[0, 1:length(SYSTEMS)], "Benchmark: free runs from the end of a 12-cycle training window (grey: held-out data; z-scored units)";
      fontsize = 8, halign = :left)
Label(gA[1, 1, TopLeft()], "A"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (B) outcome after 30 free-running cycles, three settings ─────────────────
PRED = joinpath(ROOTF, "cache", "predictable")
slipf = Dict{String, Vector{String}}()                           # SLIP walker: model => fates (all conditions and seeds)
for tag in ("fip", "dmdesn", "lstm", "node", "ssm")
    f = joinpath(PRED, "more_$(tag).jls"); isfile(f) || continue
    for o in deserialize(f)
        haskey(o, :fates) || continue
        for (k, v) in o.fates; append!(get!(slipf, String(split(k)[1]), String[]), v); end
    end
end
runf = Dict{String, Vector{String}}()                            # running, whole signal: model => fates (both time frames)
WR = joinpath(ROOTF, "cache", "maus", "whole_r2.jls")
if isfile(WR)
    MAP = Dict(:ltp => "FIP", :dmd => "DMD", :esn => "ESN", :ssm => "SSM", :lstm => "LSTM", :node => "NODE")
    for r in deserialize(WR), (k, v) in (r === nothing ? () : pairs(r))
        append!(get!(runf, MAP[k[2]], String[]), [v.fate])
    end
end
ROWS_B = vcat([(m, nm) for (m, nm, _, _, _) in MODELS[1:2]], [("DMD0", "Hankel-DMD, unconstrained")],
              [(m, nm) for (m, nm, _, _, _) in MODELS[3:end]])
SETS = [("benchmark (13 cells)", m -> [x.fate for x in allrows if x.model == m]),
        ("SLIP walker (Figure 5)", m -> get(slipf, m, String[])),
        ("running, whole signal (Figure 6)", m -> get(runf, m, String[]))]
for (s, (stitle, getf)) in enumerate(SETS)
    ax = Axis(gB[1, s]; title = stitle, xlabel = "share of free runs", yreversed = true, limits = (0, 1.3, 0.4, length(ROWS_B) + 0.6),
              yticks = (1:length(ROWS_B), [nm for (_, nm) in ROWS_B]), yticklabelsvisible = s == 1, xticks = [0, 0.5, 1],
              yticklabelsize = 6.5)
    hidespines!(ax, :t, :r)
    for (r, (m, _)) in enumerate(ROWS_B)
        v = getf(m)
        if isempty(v)
            text!(ax, 0.02, r; text = "not run", fontsize = 6, color = C_REF, align = (:left, :center)); continue
        end
        left = 0.0
        for f in FATES
            sh = count(==(f), v) / length(v); sh > 0 || continue
            barplot!(ax, [r], [sh]; offset = left, direction = :x, color = FATE_COL[f], width = 0.7, gap = 0, strokecolor = :white, strokewidth = 0.8)
            sh > 0.12 && text!(ax, left + sh / 2, r; text = "$(round(Int, 100sh))%", align = (:center, :center), fontsize = 5.5, color = :white)
            left += sh
        end
        text!(ax, 1.02, r; text = "n = $(length(v))", align = (:left, :center), fontsize = 5.5, color = C_REF)
    end
end
Legend(gB[2, 1:3], [PolyElement(color = FATE_COL[f]) for f in FATES], ["ok", "decays (right rhythm)", "wrong rhythm", "collapse", "blow-up"];
       orientation = :horizontal, tellwidth = false, framevisible = false, labelsize = 6.5)
Label(gB[0, 1:3], "Outcome after 30 free-running cycles"; fontsize = 8, halign = :left)
colgap!(gB, 10); rowgap!(gB, 3)
Label(gB[1, 1, TopLeft()], "B"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (C) first-cycle error per benchmark system ───────────────────────────────
sysorder = ["fhn", "cfhn", "vdp2", "slip", "gait91", "gait243", "gait231", "gait63", "gait267"]
axC = Axis(gC[1, 1]; title = "First free cycle: error", ylabel = "error (cycle SD units)", yscale = log10,
           xticks = (1:length(sysorder), ["FHN", "cFHN", "VdP", "SLIP", "gait 1", "2", "3", "4", "5"]), xticklabelrotation = π / 3,
           xticklabelsize = 6.5, yticks = ([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"]))
for (k, s) in enumerate(sysorder)
    tpl = [x.e1 for x in allrows if x.sys == s && x.model == "template"]
    isempty(tpl) || lines!(axC, [k - 0.45, k + 0.45], fill(median(tpl), 2); color = C_TRUTH, linewidth = 1.0)
    for (q, (m, _, col, mk, _)) in enumerate(MODELS)
        v = [x.e1 for x in allrows if x.sys == s && x.model == m]
        isempty(v) && continue
        scatter!(axC, fill(k - 0.375 + 0.15(q - 1), length(v)), clamp.(replace(v, NaN => 10.0), 1e-2, 10.0); color = (col, 0.8),
                 marker = mk, markersize = 3.5)
    end
end
ylims!(axC, 7e-3, 14)
Label(gC[1, 1, TopLeft()], "C"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (D) time within ε of the true cycle ──────────────────────────────────────
ftube = joinpath(DIR, "r2_tube.jls")
if isfile(ftube)
    tube = deserialize(ftube); EPS_D = 0.5; tg = 0:0.1:8
    surv(recs) = [count(r -> r.exits[EPS_D] > t, recs) / length(recs) for t in tg]
    axD = Axis(gC[1, 2]; title = "Time within ε of the true cycle", xlabel = "free-running cycles",
               ylabel = "share of runs within ε", limits = (0, 8, 0, 1.03))
    for (m, col, ls) in vcat([(m, col, :solid) for (m, _, col, _, _) in MODELS], [("DMD0", CAT[1], :dot)])
        recs = [r for o in tube for r in o.recs if r.model == m]
        isempty(recs) && continue
        lines!(axD, collect(tg), surv(recs); color = col, linestyle = ls, linewidth = 1.3)
    end
    text!(axD, 0.2, 0.04; text = "ε = $(EPS_D) cycle SD; held-out data stay\nwithin $(round(maximum(o.data_max for o in tube), digits = 2)); dotted: DMD unconstrained",
          fontsize = 5.5, color = C_REF, align = (:left, :bottom))
    Label(gC[1, 2, TopLeft()], "D"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))
end

# ── (E) running: free runs of the deviation models ───────────────────────────
RQ = joinpath(ROOTF, "cache", "maus", "req")
function runlong(m)
    if m == "SSM"
        fs = [joinpath(RQ, @sprintf("ssm_trial_%02d.npz", i)) for i in 1:22]
        return all(isfile, fs) ? reduce(vcat, [npzread(f)["long"] for f in fs]) : nothing
    end
    key = Dict("FIP" => "ltp", "DMD" => "dmd", "ESN" => "esn", "LSTM" => "lstm", "NODE" => "node")[m]
    f = joinpath(RQ, m in ("ESN", "LSTM", "NODE") ? "$(key)_seed1.jls" : "$(key).jls"); isfile(f) || return nothing
    R = deserialize(f); (length(R) == 22 && all(!isnothing, R)) || return nothing
    reduce(vcat, [Matrix(Float64.(r.long)) for r in R])
end
axE = Axis(gC[1, 3]; title = "Running: deviation models", xlabel = "free-running strides", yscale = log10,
           ylabel = "deviation size (× its SD)", limits = (0.5, 30.5, 3e-3, 30), yticks = ([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"]))
hlines!(axE, 1; color = C_REF, linewidth = 0.6, linestyle = :dash)
for (m, _, col, _, ls) in MODELS
    Lm = runlong(m); Lm === nothing && continue
    Lm = clamp.(replace(Lm, Inf => 1e3, NaN => 1e3), 1e-6, 1e3)
    band!(axE, 1:30, [quantile(c, 0.25) for c in eachcol(Lm)], [quantile(c, 0.75) for c in eachcol(Lm)]; color = (col, 0.12))
    lines!(axE, 1:30, [median(c) for c in eachcol(Lm)]; color = col, linestyle = ls, linewidth = 1.3)
end
text!(axE, 30, 1.1; text = "the deviation's own size", fontsize = 5.5, color = C_REF, align = (:right, :bottom))
Label(gC[1, 3, TopLeft()], "E"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))
Legend(gC[2, 1:3], [[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 5)] for (_, _, col, mk, ls) in MODELS],
       [nm for (_, nm, _, _, _) in MODELS]; orientation = :horizontal, tellwidth = false, framevisible = false, labelsize = 6.5, colgap = 10)
colgap!(gC, 16); rowgap!(gC, 2)

rowsize!(fig.layout, 1, Relative(0.53)); rowsize!(fig.layout, 2, Relative(0.2)); rowgap!(fig.layout, 8)
save_paper(fig, "Figure_07")

# ── numbers for the text ─────────────────────────────────────────────────────
println("\nOutcome after 30 free-running cycles (counts):")
for (stitle, getf) in SETS
    println("  ", stitle)
    for (m, nm) in vcat(ROWS_B, [("template", "template")])
        v = getf(m); isempty(v) && continue
        @printf("    %-26s n=%3d  %s\n", nm, length(v), join(["$(f) $(count(==(f), v))" for f in FATES if count(==(f), v) > 0], ", "))
    end
end
fin(v) = filter(isfinite, v)
println("\nBenchmark, first-cycle relative remaining variance (model error / template error)², median over seeds, range over cells:")
for reg in ("long", "realistic"), (m, nm, _, _, _) in MODELS
    r = Float64[]
    for s in unique(x.sys for x in allrows if x.reg == reg)
        t = fin([x.e1 for x in allrows if x.sys == s && x.reg == reg && x.model == "template"])
        v = fin([x.e1 for x in allrows if x.sys == s && x.reg == reg && x.model == m])
        (isempty(t) || isempty(v)) || push!(r, (median(v) / median(t))^2)
    end
    isempty(r) || @printf("  %-9s %-17s %.3g to %.3g (median %.3g, %d cells)\n", reg, nm, extrema(r)..., median(r), length(r))
end
if isfile(ftube)
    println("\nBenchmark, share of runs within ε = 0.5 for all 8 cycles:")
    for (m, nm) in ROWS_B
        recs = [r for o in tube for r in o.recs if r.model == m]
        isempty(recs) || @printf("  %-26s %3.0f%% (n = %d)\n", nm, 100 * count(r -> isinf(r.exits[0.5]), recs) / length(recs), length(recs))
    end
end
pct(v) = isempty(v) ? "—" : @sprintf("%.0f%%", 100 * count(==("ok"), v) / length(v))
sc = Dict(m => join([pct(getf(m)) for (_, getf) in SETS], " · ") for (m, _, _, _, _) in MODELS)
sc["LPV"] = "cannot run freely"
scorecard!("R2", sc)
