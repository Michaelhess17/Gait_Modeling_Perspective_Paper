#!/usr/bin/env julia
# Floquet multipliers read out of fitted models, against known truth
#
# For each system and data regime (floquet_from_models.jl), the transverse
# multiplier read from the model's own limit cycle by chaining one-step
# Jacobians over one period (neural ODE, ESN), or by the consistent variational
# probe on data-delay windows (Hankel-DMD, coupled FHN only).
#   Neural ODE: 4 seeds on one data set.      ESN: 8 seeds on one data set
#   plus one ESN per data realisation (10).   DMD: 8 analysis settings
#   (window ½ or 1 cycle × rank 16/30/50/80, least squares) × 10 data realisations.
# Points are drawn only for runs that settled on a cycle; runs that collapsed or
# whose learned period is more than 10 % off the data period are marked.
#
# DMD uses the least-squares fit with windows of ½ and 1 cycle (the
# forward–backward fit and quarter-cycle windows were unstable; see
# floquet_from_models.jl).
#
# Inputs: cache/floquet/models_*.jls

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Random

const DIR = joinpath(dirname(@__DIR__), "cache", "floquet")
load(f) = deserialize(joinpath(DIR, f))

CASES = [("cfhn", "excited", "Coupled FHN, 30 cycles"), ("cfhn", "realistic", "Coupled FHN, 12 noisy cycles"),
         ("sl", "excited", "Stuart–Landau, 30 cycles"), ("sl", "realistic", "Stuart–Landau, 12 noisy cycles")]

"All rows for a (system, regime) and model, with the truth."
function rows_for(sys, regime, model)
    out = NamedTuple[]; truth = NaN
    for f in readdir(DIR)
        startswith(f, "models_$(sys)_$(regime)_") || continue
        o = load(f); truth = o.truth
        for r in o.rows
            r.model == model || continue
            model == "DMD" && !(endswith(r.variant, " ls") && !startswith(r.variant, "τ=27")) && continue
            push!(out, (; r..., file = f))
        end
    end
    return out, truth
end

usable(r) = r.status == "cycle" && abs(r.period - 1) < 0.1
fig = Figure(size = (COL2, 120MM))
g = fig[1, 1] = GridLayout()
rng = MersenneTwister(2)
MODELS = [("NODE", "neural ODE", CAT[3]), ("ESN", "ESN, 8 seeds", CAT[2]),
          ("ESN1", "ESN, 10 data sets", CAT[2]), ("DMD", "Hankel-DMD", CAT[1])]
YMAX = 1.6

for (k, (sys, regime, title)) in enumerate(CASES)
    ax = Axis(g[div(k - 1, 2) + 1, mod(k - 1, 2) + 1]; title = title,
              ylabel = "Transverse |μ|", xticks = (1:4, ["neural ODE\n4 seeds", "ESN\n8 seeds", "ESN\n10 data sets", "DMD\n88 settings"]),
              limits = (0.4, 4.6, -0.05, YMAX), xticklabelsize = 6.5)
    truth = NaN
    for (j, (m, _, col)) in enumerate(MODELS)
        rs, truth_m = m == "ESN1" ? rows_for(sys, regime, "ESN") : rows_for(sys, regime, m == "ESN" ? "ESN" : m)
        isnan(truth_m) || (truth = truth_m)
        if m == "ESN"      # the 8-seed run on data set 1
            rs = [r for r in rs if occursin("seed", r.variant)]
        elseif m == "ESN1"
            rs = [r for r in rs if occursin("data", r.variant)]
        end
        ok = [r for r in rs if (m == "DMD" ? r.status == "probe2" : usable(r))]
        bad = [r for r in rs if !(m == "DMD" ? r.status == "probe2" : usable(r))]
        vals = [r.μ for r in ok]
        isempty(vals) || scatter!(ax, j .+ 0.28 .* (rand(rng, length(vals)) .- 0.5),
                                  clamp.(vals, -0.03, YMAX - 0.02); color = (col, 0.7), markersize = 5)
        isempty(vals) || lines!(ax, [j - 0.3, j + 0.3], fill(median(vals), 2); color = C_TRUTH, linewidth = 1.4)
        # runs that did not give a usable cycle
        if m == "DMD" && sys == "sl"
            text!(ax, j, 0.8; text = "n/a\n(cycle fills\nthe plane)", fontsize = 6, color = C_REF,
                  align = (:center, :center))
        end
        if !isempty(bad) && m != "DMD"
            text!(ax, j, YMAX - 0.04; text = "$(length(bad)) no cycle", fontsize = 6, color = C_REF,
                  align = (:center, :top))
        end
        n_over = count(>(YMAX - 0.02), vals)
        n_over > 0 && text!(ax, j, YMAX - 0.12; text = "↑ $(n_over) above", fontsize = 6, color = C_REF,
                            align = (:center, :top))
    end
    hlines!(ax, truth; color = C_TRUTH, linestyle = :dash, linewidth = 1.0)
    hlines!(ax, 1.0; color = C_REF, linestyle = :dot, linewidth = 0.8)
    Label(g[div(k - 1, 2) + 1, mod(k - 1, 2) + 1, TopLeft()], string('A' + k - 1); fontsize = 10,
          font = :bold, padding = (0, 6, 2, 0), halign = :right)
    k == 1 && text!(ax, 4.5, truth + 0.03; text = "truth", fontsize = 6.5, align = (:right, :bottom))
    k == 1 && text!(ax, 4.5, 1.03; text = "marginal", fontsize = 6.5, color = C_REF, align = (:right, :bottom))
end
colgap!(g, 14); rowgap!(g, 12)
save_paper(fig, "Fig_floquet_models")

# numbers for the text
for (sys, regime, title) in CASES
    println("\n", title)
    for (m, lab, _) in MODELS
        rs, truth = rows_for(sys, regime, m == "ESN1" ? "ESN" : m)
        rs = m == "ESN" ? [r for r in rs if occursin("seed", r.variant)] :
             m == "ESN1" ? [r for r in rs if occursin("data", r.variant)] : rs
        ok = [r for r in rs if (m == "DMD" ? r.status == "probe2" : usable(r))]
        v = [r.μ for r in ok]
        isempty(v) && (println("  $lab: no usable runs of $(length(rs))"); continue)
        println("  $lab: n=$(length(v))/$(length(rs))  median $(round(median(v), digits = 2))  range $(round(minimum(v), digits = 2))–$(round(maximum(v), digits = 2))  (truth $(round(truth, digits = 2)))")
    end
end
