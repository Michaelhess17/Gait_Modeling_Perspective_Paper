#!/usr/bin/env julia
# Hankel-DMD off-cycle relaxation rate vs amount of transient training data (dmd_delay_sweep.jl logs,
# parsed into cache/floquet/dmd_datasize_rows.jls).
# Per-cycle rate = (deviation ratio after 4 cycles)^(1/4), least-squares fits, 16 settings
# (window 1–2.5 cycles × rank 20–160) per data size; decays of 3 cycles each, process noise as elsewhere.
using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Statistics
rx = r"^(\w+)\s+(\d+)×3 τ=\s*(\d+) \(([\d.]+) cyc\) r=\s*(\d+) ls .*per-cycle rate ([\d.]+) \(truth ([\d.]+)\)"
using Serialization
const ROWS_CACHE = joinpath(dirname(@__DIR__), "cache", "floquet", "dmd_datasize_rows.jls")
const LOGDIR = joinpath(dirname(@__DIR__), "logs")
rows = NamedTuple[]
for f in (isdir(LOGDIR) ? readdir(LOGDIR; join = true) : String[])
    (occursin("dsweep_", f) || occursin("dsize_", f)) || continue
    for l in eachline(f)
        m = match(rx, l); m === nothing && continue
        push!(rows, (; sys = m[1], n = parse(Int, m[2]), win = parse(Float64, m[4]), r = parse(Int, m[5]),
                       rate = parse(Float64, m[6]), truth = parse(Float64, m[7])))
    end
end
rows = unique(rows)
# sweep logs are parsed once and cached, so the figure does not depend on logs/
isempty(rows) ? (rows = deserialize(ROWS_CACHE)) : serialize(ROWS_CACHE, rows)
NS = [10, 30, 100, 300]
fig = Figure(size = (COL2, 65MM)); g = fig[1, 1] = GridLayout()
for (c, (sys, title)) in enumerate((("cfhn", "Coupled FHN"), ("sl", "Stuart–Landau")))
    ax = Axis(g[1, c]; xscale = log10, xticks = (NS, string.(NS)), xlabel = "Training decays (3 cycles each)",
              ylabel = c == 1 ? "Per-cycle relaxation rate" : "", title, limits = (8, 380, 0.45, 1.1))
    tr = [r.truth for r in rows if r.sys == sys]
    band!(ax, [8, 380], fill(minimum(tr), 2), fill(maximum(tr), 2); color = (C_TRUTH, 0.12))
    groups = sys == "cfhn" ? [("all windows", w -> true, CAT[1])] :
             [("window 1 cycle", w -> w < 1.2, CAT[4]), ("window ≥ 1.5 cycles", w -> w > 1.2, CAT[1])]
    for (lab, sel, col) in groups
        S = [[r.rate for r in rows if r.sys == sys && r.n == n && sel(r.win)] for n in NS]
        for (n, s) in zip(NS, S)
            scatter!(ax, fill(n, length(s)) .* exp.(0.06 .* randn(length(s))), min.(s, 1.09); color = (col, 0.45), markersize = 4)
        end
        lines!(ax, NS, median.(S); color = col, linewidth = 1.6, label = lab)
    end
    hlines!(ax, 1.0; color = C_REF, linestyle = :dot, linewidth = 0.8)
    text!(ax, 370, minimum(tr) - 0.01; text = "truth", fontsize = 7, align = (:right, :top))
    text!(ax, 370, 1.0 + 0.01; text = "no relaxation", fontsize = 7, color = C_REF, align = (:right, :bottom))
    axislegend(ax; position = (:left, :bottom), framevisible = false, labelsize = 7)
    Label(g[1, c, TopLeft()], string('A' + c - 1); fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
end
colgap!(g, 14)
save_paper(fig, "Fig_dmd_datasize")
