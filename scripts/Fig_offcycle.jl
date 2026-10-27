#!/usr/bin/env julia
# Off-cycle relaxation of models trained on decays from perturbed starts (offcycle_v2.jl).
# The TRUE system is started off its cycle; each model, warmed up on 170 samples of that trajectory, runs
# freely. Deviation = distance to the nearest point of the model's own attractor (phase-independent), RMS
# over each cycle, relative to the first cycle. Truth: distance to the true cycle; decays by |μ| per cycle.
# One line per fitted model (DMD: tuned setting; ESN, neural ODE: one line per seed), median over 4 start
# phases. Runs whose free run blew up are not drawn.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics

const DIR = joinpath(dirname(@__DIR__), "cache", "floquet")
runs(sys, model) = [deserialize(joinpath(DIR, f)) for f in readdir(DIR) if startswith(f, "offv2_$(sys)_$(model)_")]
MODELS = [("dmd", "Hankel-DMD", CAT[1]), ("esn", "ESN", CAT[2]), ("node", "neural ODE", CAT[3])]
fig = Figure(size = (COL2, 62MM)); g = fig[1, 1] = GridLayout()
for (c, (sys, title)) in enumerate((("cfhn", "Coupled FHN (|μ| = 0.69)"), ("sl", "Stuart–Landau (|μ| = 0.60)")))
    ax = Axis(g[1, c]; yscale = log10, xlabel = "Cycles after warm-up", ylabel = c == 1 ? "Distance to cycle\n(relative)" : "",
              title, xticks = 0:4, limits = (-0.2, 4.2, 0.05, 2))
    tr = nothing
    for (m, lab, col) in MODELS, o in runs(sys, m)
        tr = o.truth
        all(isfinite, o.own) || continue
        y = clamp.(o.own, 0.05, 2)
        lines!(ax, 0:4, y; color = (col, 0.8), linewidth = 1.0)
        scatter!(ax, 0:4, y; color = (col, 0.8), markersize = 3.5)
    end
    tr === nothing || lines!(ax, 0:4, tr; color = C_TRUTH, linewidth = 1.8, linestyle = :dash)
    Label(g[1, c, TopLeft()], string('A' + c - 1); fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
end
Legend(g[2, 1:2], vcat([[LineElement(color = c), MarkerElement(color = c, marker = :circle, markersize = 4)] for (_, _, c) in MODELS],
                       [LineElement(color = C_TRUTH, linestyle = :dash, linewidth = 1.8)]),
       vcat([l for (_, l, _) in MODELS], ["truth"]); orientation = :horizontal, tellwidth = false)
colgap!(g, 12); rowgap!(g, 6)
save_paper(fig, "Fig_offcycle")
