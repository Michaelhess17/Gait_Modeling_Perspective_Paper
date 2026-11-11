#!/usr/bin/env julia
# Figure_S02.jl — Validate by rollout, not one-step error (manuscript: @suppfig-validation). Run: julia --project=. scripts/figures/Figure_S02.jl
# Figure S7 — Validate by rollout, not by one-step error (OUTLINE.md §3 ¶1, R1; scripts/validation_criteria.jl).
#
# Every Hankel-DMD setting (60) and ESN setting (36 × 3 seeds) of the benchmark's tuning grids, in each of the 13
# system × regime cells, fitted on the training window minus its last two cycles and scored on those two cycles by
#   in-sample 1-step    one-step error on the fitting data (a training loss)
#   held-out 1-step     teacher-forced one-step error on the two held-out cycles
#   held-out rollout    one free run from the end of the fitting data over the two held-out cycles
#   multistart rollout  mean error of 1-cycle free runs from 5 starts across the held-out cycles
# then refitted on the full training window and run freely (test: first held-out test cycle, and 30-cycle fate).
#   (A) Rank correlation across settings between each validation error and the test forecast error, one dot per cell.
#   (B) Test forecast skill (1 − error / mean-cycle reference error) of the setting each criterion would choose, one
#       dot per cell (clipped at the bottom; open: clipped); "best on test" is the setting with the lowest test error.
#       Text: number of cells where the chosen setting keeps the right cycle over 30 cycles.
#
# Usage: julia --project=. scripts/figures/Figure_S02.jl      (reads cache/selfdrive/validation_criteria.jls)

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Random

const ROOT = dirname(dirname(@__DIR__))
cells = deserialize(joinpath(ROOT, "cache", "selfdrive", "validation_criteria.jls"))
const CRIT = [(:train1, "in-sample\n1-step"), (:val1, "held-out\n1-step"), (:valroll, "held-out\nrollout"),
              (:valmulti, "multistart\nrollout")]
const MODELS = [("DMD", CAT[1], :circle, -0.17), ("ESN", CAT[2], :utriangle, 0.17)]
const FLOOR = -3.0
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
fin(x) = isfinite(x) ? x : 1e9
function rk(v)                                         # average ranks (as validation_criteria.jl)
    o = sortperm(v); r = zeros(length(v)); i = 1
    while i <= length(v)
        j = i; while j < length(v) && v[o[j+1]] == v[o[i]]; j += 1; end
        r[o[i:j]] .= (i + j) / 2; i = j + 1
    end
    r
end
jit = MersenneTwister(3)

fig = Figure(size = (COL2, 72MM))
axA = Axis(fig[1, 1]; ylabel = "Rank correlation with test\nforecast error, across settings",
           title = "Rollout errors track test forecasts", xticks = (1:4, last.(CRIT)), limits = (0.4, 4.6, -1.05, 1.05))
axB = Axis(fig[1, 2]; ylabel = "Test forecast skill of the\nchosen setting (vs mean cycle)",
           title = "What each criterion chooses", xticks = (1:5, vcat(last.(CRIT), ["best\non test"])),
           limits = (0.4, 5.6, FLOOR - 0.3, 1.85))
hlines!(axA, 0.0; color = C_REF, linewidth = 0.6); hlines!(axB, 0.0; color = C_REF, linewidth = 0.6)
for (m, col, mk, dx) in MODELS
    for (j, (k, _)) in enumerate(CRIT)
        ρ = Float64[]
        for c in cells
            R = [r for r in c.rows if r.model == m]
            push!(ρ, cor(rk([fin(getfield(r, k)) for r in R]), rk([fin(r.e1) for r in R])))
        end
        scatter!(axA, j + dx .+ 0.05 .* randn(jit, length(ρ)), ρ; color = (col, 0.55), marker = mk, markersize = 5)
        lines!(axA, [j + dx - 0.12, j + dx + 0.12], fill(median(ρ), 2); color = col, linewidth = 2)
    end
    for j in 1:5
        sk = Float64[]; nok = 0
        for c in cells
            R = [r for r in c.rows if r.model == m]
            x = j == 5 ? R[argmin([fin(r.e1) for r in R])] : R[argmin([fin(getfield(r, CRIT[j][1])) for r in R])]
            push!(sk, isfinite(x.e1) ? 1 - x.e1 / c.template_e1 : -Inf); nok += x.fate == "ok"
        end
        xs = j + dx .+ 0.05 .* randn(jit, length(sk)); low = sk .< FLOOR
        scatter!(axB, xs[.!low], sk[.!low]; color = (col, 0.55), marker = mk, markersize = 5)
        any(low) && scatter!(axB, xs[low], fill(FLOOR, count(low)); color = :white, strokecolor = col, strokewidth = 0.8,
                             marker = mk, markersize = 5)
        lines!(axB, [j + dx - 0.12, j + dx + 0.12], fill(clamp(median(sk), FLOOR, 2), 2); color = col, linewidth = 2)
        text!(axB, j + dx, 1.17; text = "$nok", fontsize = 6.5, color = col, align = (:center, :bottom))
    end
end
text!(axB, 0.45, 1.82; text = "cells where the choice keeps the cycle (of 13):", fontsize = 6.5, color = C_REF,
      align = (:left, :top))
text!(axB, 5.55, FLOOR - 0.25; text = "open: below $(Int(FLOOR))", fontsize = 6.5, color = C_REF, align = (:right, :bottom))
axislegend(axA, [MarkerElement(color = col, marker = mk, markersize = 6) for (_, col, mk, _) in MODELS],
           first.(MODELS); position = :lb, framevisible = false, labelsize = 7, padding = (2, 2, 2, 2))
plabel!(fig[1, 1], "A"); plabel!(fig[1, 2], "B")
colsize!(fig.layout, 1, Relative(0.44)); colgap!(fig.layout, 18)
save_paper(fig, "Figure_S02")
