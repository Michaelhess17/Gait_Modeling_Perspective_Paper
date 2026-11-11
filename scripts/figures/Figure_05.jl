#!/usr/bin/env julia
# Figure_05.jl — Whole-system models run freely: who keeps the cycle (R1, R2) (manuscript: @fig-selfdriving). Run: julia --project=. scripts/figures/Figure_05.jl
# Self-driving mode: what each model does when it has to run on its own
#
# (A) Phase portraits in the realistic regime (12 training cycles, 1 %
#     measurement noise, process noise on the simulated systems). Grey: the
#     held-out data; colour: free runs from the end of the training window,
#     every seed (ESN, NODE), polynomial degree (SSMLearnPy) or rank (DMD)
#     overlaid. Axes are framed on the data; runs that leave the frame are cut.
#     The label gives the 30-cycle outcome counts.
# (B) Outcome after 30 free-running cycles, pooled over all systems and both
#     regimes: ok / decays with the right rhythm / collapse / blow-up / wrong
#     rhythm (definitions in self_driving_benchmark.jl).
# (C) Free-run error over the first cycle, per system, against the
#     phase-aligned mean-cycle template.
#
# DMD and ESN use hyperparameters chosen on the training data only
# (dmd_tuning.jl, esn_tuning.jl: fit on all but the last two training cycles,
# pick by free-run error on those cycles). The neural ODE uses one fixed recipe
# (too costly to tune per cell); SSMLearnPy uses the configuration tuned on gait.
#
# Inputs: cache/selfdrive/*.jls (self_driving_benchmark.jl), dmdtuned_*.jls,
#         esntuned_*.jls, ssm_*.npz (ssm_self_driving.py)

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, NPZ, Random

const DIR = joinpath(dirname(dirname(@__DIR__)), "cache", "selfdrive")
const SYSTEMS = [("fhn", "FitzHugh–Nagumo", (1, 2), ("v", "w")),
                 ("cfhn", "coupled FHN", (1, 2), ("v₁", "w₁")),
                 ("vdp2", "coupled Van der Pol", (1, 2), ("x_A", "ẋ_A")),
                 ("slip", "SLIP walking", (1, 3), ("CoM height", "vertical vel.")),
                 ("gait91", "human gait (AB)", (1, 2), ("hip", "knee"))]
const MODELS = [("ESN", CAT[2]), ("NODE", CAT[3]), ("SSM", CAT[4]), ("DMD", CAT[1])]
const MODEL_LABEL = Dict("ESN" => "echo state network", "NODE" => "neural ODE",
                         "SSM" => "SSMLearnPy", "DMD" => "Hankel-DMD (linear)")
const FATES = ["ok", "decays", "wrong", "collapse", "blow-up"]
const FATE_COL = Dict("ok" => colorant"#3d7a5c", "decays" => colorant"#9cc7ae",
                      "wrong" => colorant"#d9b45a", "collapse" => colorant"#a9a8a4",
                      "blow-up" => colorant"#b5523b")

"Tuned DMD (one deterministic run) and tuned ESN (8 seeds) rows, if available."
function tuned_rows(sys, regime)
    rows = NamedTuple[]
    f = joinpath(DIR, "dmdtuned_$(sys)_$(regime).jls")
    if isfile(f)
        o = deserialize(f); c = o.chosen
        push!(rows, (; model = "DMD", variant = "τ=$(c.τ) r=$(c.r) $(c.variant)", c.e1, c.fate,
                       Fc = Float64.(o.Fc)))
    end
    f = joinpath(DIR, "esntuned_$(sys)_$(regime).jls")
    if isfile(f)
        o = deserialize(f)
        append!(rows, [(; model = "ESN", variant = "seed $(r.seed)", r.e1, r.fate,
                          Fc = Float64.(r.Fc)) for r in o.rows])
    end
    return rows
end

"Rows for one (system, regime): DMD/ESN/template + NODE from .jls, SSM from .npz."
function load_cell(sys, regime)
    rows = NamedTuple[]; Yte = nothing; Ytr = nothing; P = NaN
    for tag in ("dmd-esn-template", "node")
        f = joinpath(DIR, "$(sys)_$(regime)_$(tag).jls")
        isfile(f) || continue
        o = deserialize(f)
        append!(rows, [(; r.model, r.variant, r.e1, r.fate, Fc = Float64.(r.Fc)) for r in o.rows])
        Yte = o.Yte; Ytr = o.Ytr; P = o.P
    end
    tr = tuned_rows(sys, regime)            # tuned DMD / ESN replace the defaults
    for m in unique([x.model for x in tr])
        filter!(x -> x.model != m, rows)
    end
    append!(rows, tr)
    f = joinpath(DIR, "ssm_$(sys)_$(regime).npz")
    if isfile(f)
        S = npzread(f)
        for k in eachindex(S["degrees"])
            push!(rows, (; model = "SSM", variant = "deg $(S["degrees"][k])", e1 = S["e1"][k],
                           fate = FATES[S["fate_code"][k] + 1], Fc = S["fc"][k, :, :]))
        end
    end
    return rows, Ytr, Yte, P
end

fig = Figure(size = (COL2, 205MM))
gA = fig[1, 1] = GridLayout()
gB = fig[2, 1] = GridLayout()

# ── (A) phase-portrait grid ──────────────────────────────────────────────────
for (c, (sys, title, (i, j), (li, lj))) in enumerate(SYSTEMS)
    rows, Ytr, Yte, P = load_cell(sys, "realistic")
    Yte === nothing && continue
    n8 = min(size(Yte, 1), round(Int, 8P))
    lims = framed_limits(vcat(Ytr[:, i], Yte[1:n8, i]), vcat(Ytr[:, j], Yte[1:n8, j]); pad = 0.25)
    for (r, (m, col)) in enumerate(MODELS)
        ax = Axis(gA[r, c]; xticklabelsvisible = r == length(MODELS),
                  yticklabelsvisible = c == 1, xlabel = r == length(MODELS) ? li : "",
                  ylabel = c == 1 ? lj : "", title = r == 1 ? title : "",
                  titlealign = :center, xticks = LinearTicks(3), yticks = LinearTicks(3))
        lines!(ax, Yte[1:n8, i], Yte[1:n8, j]; color = (C_REF, 0.5), linewidth = 1.0)
        mr = [x for x in rows if x.model == m]
        for x in mr
            xs, ys, _ = clip_to_view(x.Fc[:, i], x.Fc[:, j], lims)
            lines!(ax, xs, ys; color = (col, 0.75), linewidth = 0.8)
        end
        limits!(ax, lims...)
        if !isempty(mr)
            counts = [count(x -> x.fate == f, mr) for f in FATES]
            parts = ["$(n) $(f)" for (n, f) in zip(counts, FATES) if n > 0]
            txt = length(parts) > 2 ? join(parts[1:2], ", ") * ",\n" * join(parts[3:end], ", ") : join(parts, ", ")
            text!(ax, 0.03, 0.97; space = :relative, align = (:left, :top), fontsize = 6,
                  text = txt, color = C_TRUTH)
        end
        c == length(SYSTEMS) && Label(gA[r, c + 1], MODEL_LABEL[m]; rotation = -π / 2,
                                      fontsize = 7, tellheight = false)
    end
end
colgap!(gA, 4); rowgap!(gA, 4)
Label(gA[0, 1:length(SYSTEMS)], "Free runs from the end of a 12-cycle training window " *
      "(grey: held-out data)"; fontsize = 8, halign = :left)
Label(gA[1, 1, TopLeft()], "A"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (B) pooled outcomes ──────────────────────────────────────────────────────
allrows = NamedTuple[]
for f in readdir(DIR)
    m = match(r"^(.+)_(realistic|long)_(dmd-esn-template|node)\.jls$", f)
    m === nothing && continue
    o = deserialize(joinpath(DIR, f))
    tr = tuned_rows(m[1], m[2]); tuned = unique([x.model for x in tr])
    append!(allrows, [(; r.model, r.fate, sys = m[1], reg = m[2], r.e1) for r in o.rows if !(r.model in tuned)])
    m[3] == "dmd-esn-template" &&
        append!(allrows, [(; x.model, x.fate, sys = m[1], reg = m[2], x.e1) for x in tr])
end
for f in readdir(DIR)
    m = match(r"^ssm_(.+)_(realistic|long)\.npz$", f)
    m === nothing && continue
    S = npzread(joinpath(DIR, f))
    append!(allrows, [(; model = "SSM", fate = FATES[S["fate_code"][k] + 1], sys = m[1], reg = m[2],
                         e1 = S["e1"][k]) for k in eachindex(S["fate_code"])])
end

# unconstrained Hankel-DMD, tuned on the training data by the same criterion (stabilization_benchmark.jl)
fsb = joinpath(DIR, "stabilization_benchmark.jls")
if isfile(fsb)
    append!(allrows, [(; model = "DMD0", fate = o.retuned.fate, sys = o.system, reg = o.regime, e1 = o.retuned.e1)
                      for o in deserialize(fsb)])
end
MODELS_B = vcat(MODELS, isfile(fsb) ? [("DMD0", CAT[1])] : Tuple{String, Any}[])
LABEL_B = Dict("ESN" => "ESN", "NODE" => "neural ODE", "SSM" => "SSMLearnPy", "DMD" => "DMD, as tuned", "DMD0" => "DMD, no constraint")
axB = Axis(gB[1, 1]; title = "Outcome after 30 free-running cycles",
           xlabel = "Share of runs", yticks = (1:length(MODELS_B), [LABEL_B[m] for (m, _) in MODELS_B]),
           yreversed = true, limits = (0, 1, 0.4, length(MODELS_B) + 0.6))
for (r, (m, _)) in enumerate(MODELS_B)
    mr = [x for x in allrows if x.model == m]
    isempty(mr) && continue
    left = 0.0
    for f in FATES
        share = count(x -> x.fate == f, mr) / length(mr)
        share > 0 || continue
        barplot!(axB, [r], [share]; offset = left, direction = :x, color = FATE_COL[f],
                 width = 0.7, gap = 0, strokecolor = :white, strokewidth = 1)
        share > 0.08 && text!(axB, left + share / 2, r; text = "$(round(Int, 100share))%",
                              align = (:center, :center), fontsize = 6, color = :white)
        left += share
    end
    text!(axB, 1.01, r; text = "n = $(length(mr))", align = (:left, :center), fontsize = 6)
end
Legend(gB[2, 1], [PolyElement(color = FATE_COL[f]) for f in FATES],
       ["ok", "decays (right rhythm)", "wrong rhythm", "collapse", "blow-up"];
       orientation = :horizontal, tellwidth = false, nbanks = 3)
Label(gB[1, 1, TopLeft()], "B"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (C) first-cycle error per system ─────────────────────────────────────────
sysorder = ["fhn", "cfhn", "vdp2", "slip", "gait91", "gait243", "gait231", "gait63", "gait267"]
axC = Axis(gB[1, 2]; title = "Error over the first free-running cycle",
           xticks = (1:length(sysorder), ["FHN", "cFHN", "VdP", "SLIP", "gait 1", "2", "3", "4", "5"]),
           ylabel = "Error (SD units)", yscale = log10, xticklabelrotation = π / 3, xticklabelsize = 6.5)
for (k, s) in enumerate(sysorder)
    tpl = [x.e1 for x in allrows if x.sys == s && x.model == "template"]
    isempty(tpl) || lines!(axC, [k - 0.4, k + 0.4], fill(median(tpl), 2); color = C_TRUTH,
                           linewidth = 1.2)
    for (q, (m, col)) in enumerate(MODELS)
        v = [x.e1 for x in allrows if x.sys == s && x.model == m]
        isempty(v) && continue
        v = clamp.(v, 1e-2, 10.0)
        scatter!(axC, fill(k - 0.3 + 0.2(q - 1), length(v)), v; color = (col, 0.8), markersize = 3.5)
    end
end
refline!(axC, 1.0)
ylims!(axC, 1e-2, 12)
text!(axC, 0.6, 9; text = "points at 10 = diverged", fontsize = 6, color = C_REF, align = (:left, :top))
Legend(gB[2, 2], vcat([MarkerElement(color = c, marker = :circle, markersize = 5) for (_, c) in MODELS],
                      [LineElement(color = C_TRUTH, linewidth = 1.2)]),
       vcat([m == "DMD" ? "DMD" : m for (m, _) in MODELS], ["mean-cycle template"]);
       orientation = :horizontal, tellwidth = false, nbanks = 3)
Label(gB[1, 2, TopLeft()], "C"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))
# ── (D) R2: time within ε of the true cycle ───────────────────────────────────
ftube = joinpath(DIR, "r2_tube.jls")
if isfile(ftube)
    tube = deserialize(ftube); EPS_D = 0.5; tg = 0:0.1:8
    surv(recs) = [count(r -> r.exits[EPS_D] > t, recs) / length(recs) for t in tg]
    axD = Axis(gB[1, 3]; title = "Time within ε of the true cycle", xlabel = "Free-running cycles",
               ylabel = "Share of runs within ε", limits = (0, 8, 0, 1.03))
    rng = MersenneTwister(3)
    for (m, col, ls) in (("ESN", CAT[2], :solid), ("NODE", CAT[3], :solid), ("SSM", CAT[4], :solid),
                         ("DMD0", CAT[1], :dash), ("DMD", CAT[1], :solid))
        cellsm = [[r for r in o.recs if r.model == m] for o in tube]; cellsm = filter(!isempty, cellsm)
        boot = reduce(hcat, [surv(reduce(vcat, cellsm[rand(rng, 1:length(cellsm), length(cellsm))])) for _ in 1:1000])
        lo = [quantile(boot[i, :], 0.025) for i in eachindex(tg)]; hi = [quantile(boot[i, :], 0.975) for i in eachindex(tg)]
        ls == :solid && band!(axD, collect(tg), lo, hi; color = (col, 0.15))
        lines!(axD, collect(tg), surv(reduce(vcat, cellsm)); color = col, linestyle = ls, linewidth = 1.4)
    end
    text!(axD, 0.2, 0.05; text = "ε = $(EPS_D) cycle SD; held-out data\nstay within $(round(maximum(o.data_max for o in tube), digits = 2))",
          fontsize = 6, color = C_REF, align = (:left, :bottom))
    Legend(gB[2, 3], [LineElement(color = CAT[1], linestyle = :solid), LineElement(color = CAT[1], linestyle = :dash)],
           ["DMD as tuned", "DMD, no constraint"]; orientation = :horizontal, tellwidth = false, nbanks = 2)
    Label(gB[1, 3, TopLeft()], "D"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))
end
colsize!(gB, 1, Relative(0.34)); colsize!(gB, 2, Relative(0.35)); colgap!(gB, 18)

rowsize!(fig.layout, 1, Relative(0.70))
save_paper(fig, "Figure_05")

println("\nOutcome counts (all systems, both regimes):")
for (m, _) in vcat(MODELS_B, [("template", nothing)])
    mr = [x for x in allrows if x.model == m]
    isempty(mr) && continue
    println(rpad(m, 9), " n=", lpad(length(mr), 3), "  ",
            join(["$(f) $(count(x -> x.fate == f, mr))" for f in FATES], "  "))
end

# Panel C as numbers: median first-cycle free-run error per cell, and skill against the mean-cycle reference forecast
# (1 − model / template; 0 = no better than the mean cycle, 1 = exact), as weather forecasts are scored against
# climatology.
fin(v) = filter(isfinite, v)
for reg in ("long", "realistic")
    println("\nFirst-cycle forecast error, $reg regime (median over seeds/settings; skill against the mean cycle):")
    for s in sort(unique(x.sys for x in allrows if x.reg == reg))
        t = median(fin([x.e1 for x in allrows if x.sys == s && x.reg == reg && x.model == "template"]))
        cells = map(("DMD", "ESN", "NODE", "SSM")) do m
            v = fin([x.e1 for x in allrows if x.sys == s && x.reg == reg && x.model == m])
            isempty(v) ? "$(m) —" : @sprintf("%s %.3f (%+.2f)", m, median(v), 1 - median(v) / t)
        end
        println(rpad(s, 8), @sprintf(" template %.3f | ", t), join(cells, "  "))
    end
end
