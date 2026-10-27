#!/usr/bin/env julia
# Tabulate the coupling benchmark (cache/coupling/*.jls) and write a tidy CSV.
#
# For each (condition, config, model): median and range of the asymmetry index
# AI and cross-talk index CI across that model's nuisance variants (seeds for
# ESN/NODE, delay × rank for DMD), alongside the truth, plus the fraction of
# variants with the correct sign of AI for directed configs.
#
# Usage: julia --project=. scripts/summarize_coupling.jl [condition-filter]

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using Serialization, Statistics, Printf, CSV, DataFrames

const DIR = joinpath(dirname(@__DIR__), "cache", "coupling")
filt = length(ARGS) >= 1 ? ARGS[1] : ""

rows = NamedTuple[]
for f in sort(readdir(DIR))
    endswith(f, ".jls") || continue
    occursin(filt, f) || continue
    append!(rows, deserialize(joinpath(DIR, f)))
end
df = DataFrame(rows)
df = df[:, [:model, :variant, :cond, :cfg, :AI, :CI, :fc]]
CSV.write(joinpath(dirname(@__DIR__), "cache", "coupling_all.csv"), df)

truth = Dict((r.cond, r.cfg) => r for r in eachrow(df) if r.model == "truth")
models = ["DMD", "DMD-resolvent", "DMD-fullrank", "ESN", "NODE"]

@printf("%-12s %-9s %6s %5s | %-14s %6s %13s %5s %5s %5s\n",
        "condition", "config", "AI*", "CI*", "model", "AI", "range", "sign", "CI", "fc")
summary = NamedTuple[]
for key in sort(collect(keys(truth)))
    t = truth[key]
    for m in models
        g = df[(df.model .== m) .& (df.cond .== key[1]) .& (df.cfg .== key[2]), :]
        g = g[isfinite.(g.AI), :]
        nrow(g) == 0 && continue
        sgn = abs(t.AI) > 0.3 ? mean(sign.(g.AI) .== sign(t.AI)) : NaN
        fcs = filter(isfinite, g.fc)
        s = (; cond = key[1], cfg = key[2], model = m, AI_true = t.AI, CI_true = t.CI,
               AI_med = median(g.AI), AI_lo = minimum(g.AI), AI_hi = maximum(g.AI),
               sign_ok = sgn, CI_med = median(g.CI), fc_med = isempty(fcs) ? NaN : median(fcs),
               n = nrow(g))
        push!(summary, s)
        @printf("%-12s %-9s %6.2f %5.2f | %-14s %6.2f [%5.2f,%5.2f] %5s %5.2f %5.2f\n",
                s.cond, s.cfg, s.AI_true, s.CI_true, m, s.AI_med, s.AI_lo, s.AI_hi,
                isnan(sgn) ? "  -" : @sprintf("%.2f", sgn), s.CI_med, s.fc_med)
    end
end
CSV.write(joinpath(dirname(@__DIR__), "cache", "coupling_summary.csv"), DataFrame(summary))
