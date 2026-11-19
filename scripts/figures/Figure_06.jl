#!/usr/bin/env julia
# Figure_06.jl — R2: who keeps the cycle when run on its own (manuscript: @fig-selfdriving).
# Run: julia --project=. scripts/figures/Figure_06.jl
# (A) Outcome after 30 free-running cycles in three settings, every run pooled (classes in r2_common.jl):
#       benchmark  13 cells: 4 simulated systems × 2 regimes (30 training cycles without noise; 12 with noise) and 5
#                  human gait recordings (self_driving_benchmark.jl; Fourier in phase and LSTM from benchmark_more.jl)
#       SLIP       the walker of Figure 3: 8 conditions × 10 held-out walks, 30 strides from the first test start of
#                  each walk (predictable_more.jl; predictable_ssm.py)
#       running    the whole signal of Figure 4: 22 trials, 150 training strides, real and stride-normalised time
#                  (maus_whole_r2.jl); the neural ODE and LSTM were not run freely on the whole running signal
# (B) Benchmark: error over the first free-running cycle, per system, against the phase-aligned mean-cycle template.
# (C) Benchmark: time each free run stays within ε of the true cycle (r2_tube.jl).
# (D) Running: free runs of the deviation models (maus_requirements.jl), size of the deviation relative to its SD.
include(joinpath(@__DIR__, "r2_common.jl"))

fig = Figure(size = (COL2, 118MM))
gB = fig[1, 1] = GridLayout()
gC = fig[2, 1] = GridLayout()

# ── (A) outcome after 30 free-running cycles, three settings ─────────────────
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
        ("SLIP walker (Figure 3)", m -> get(slipf, m, String[])),
        ("running, whole signal (Figure 4)", m -> get(runf, m, String[]))]
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
Label(gB[1, 1, TopLeft()], "A"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (B) first-cycle error per benchmark system ───────────────────────────────
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
Label(gC[1, 1, TopLeft()], "B"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))

# ── (C) time within ε of the true cycle ──────────────────────────────────────
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
    Label(gC[1, 2, TopLeft()], "C"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))
end

# ── (D) running: free runs of the deviation models ───────────────────────────
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
Label(gC[1, 3, TopLeft()], "D"; fontsize = 10, font = :bold, padding = (0, 6, 2, 0))
Legend(gC[2, 1:3], [[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 5)] for (_, _, col, mk, ls) in MODELS],
       [nm for (_, nm, _, _, _) in MODELS]; orientation = :horizontal, tellwidth = false, framevisible = false, labelsize = 6.5, colgap = 10)
colgap!(gC, 16); rowgap!(gC, 2)

rowsize!(fig.layout, 1, Relative(0.45)); rowgap!(fig.layout, 10)
save_paper(fig, "Figure_06")

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
