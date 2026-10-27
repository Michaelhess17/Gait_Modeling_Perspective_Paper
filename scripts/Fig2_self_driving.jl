#!/usr/bin/env julia
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

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, NPZ

const DIR = joinpath(dirname(@__DIR__), "cache", "selfdrive")
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
    append!(allrows, [(; r.model, r.fate, sys = m[1], r.e1) for r in o.rows if !(r.model in tuned)])
    m[3] == "dmd-esn-template" &&
        append!(allrows, [(; x.model, x.fate, sys = m[1], x.e1) for x in tr])
end
for f in readdir(DIR)
    m = match(r"^ssm_(.+)_(realistic|long)\.npz$", f)
    m === nothing && continue
    S = npzread(joinpath(DIR, f))
    append!(allrows, [(; model = "SSM", fate = FATES[S["fate_code"][k] + 1], sys = m[1],
                         e1 = S["e1"][k]) for k in eachindex(S["fate_code"])])
end

axB = Axis(gB[1, 1]; title = "Outcome after 30 free-running cycles",
           xlabel = "Share of runs", yticks = (1:length(MODELS), [MODEL_LABEL[m] for (m, _) in MODELS]),
           yreversed = true, limits = (0, 1, 0.4, length(MODELS) + 0.6))
for (r, (m, _)) in enumerate(MODELS)
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
       orientation = :horizontal, tellwidth = false, nbanks = 2)
Label(gB[1, 1, TopLeft()], "B"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (C) first-cycle error per system ─────────────────────────────────────────
sysorder = ["fhn", "cfhn", "vdp2", "slip", "gait91", "gait243", "gait231", "gait63", "gait267"]
axC = Axis(gB[1, 2]; title = "Free-run error, first cycle (realistic + long)",
           xticks = (1:length(sysorder), ["FHN", "cFHN", "VdP", "SLIP", "gait 1", "2", "3", "4", "5"]),
           ylabel = "Error (SD units)", yscale = log10, xticklabelrotation = 0)
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
       orientation = :horizontal, tellwidth = false, nbanks = 2)
Label(gB[1, 2, TopLeft()], "C"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))
colsize!(gB, 1, Relative(0.48)); colgap!(gB, 16)

rowsize!(fig.layout, 1, Relative(0.70))
save_paper(fig, "Fig2_self_driving")

println("\nOutcome counts (all systems, both regimes):")
for (m, _) in vcat(MODELS, [("template", nothing)])
    mr = [x for x in allrows if x.model == m]
    isempty(mr) && continue
    println(rpad(m, 9), " n=", lpad(length(mr), 3), "  ",
            join(["$(f) $(count(x -> x.fate == f, mr))" for f in FATES], "  "))
end
