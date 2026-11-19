#!/usr/bin/env julia
# scorecard.jl — the comparison table of the manuscript (@tbl-scorecard), assembled from the headline numbers the
# requirement figures record (scorecard! in paper_style.jl → cache/scorecard/<column>.jls). Run after the figures
# (make_figures.jl runs it last): julia --project=. scripts/scorecard.jl → _scorecard.md (included by index.qmd)
using Serialization

const ROOT = dirname(@__DIR__)
const DIR = joinpath(ROOT, "cache", "scorecard")
const ROWS = [("FIP", "Fourier in phase"), ("LPV", "phase-binned linear"), ("DMD", "Hankel-DMD (time-invariant)"),
              ("ESN", "ESN"), ("LSTM", "LSTM"), ("NODE", "neural ODE"), ("SSM", "SSMLearnPy")]
#           column file   header
const COLS = [("R1", "R1 forecast"), ("R2", "R2 keeps the cycle"), ("R3", "R3 seed dependence"),
              ("R4", "R4 short, partial records"), ("R5", "R5 multipliers"), ("R6", "R6 between people")]
const CAPTION = "Headline numbers for each requirement, in the settings tested here (—, not tested). " *
    "R1: remaining variance of the running deviations one eighth of a stride ahead, 150 training strides (median over " *
    "trials; 1, no better than the average stride); in parentheses, the share of the attainable variance reduction over " *
    "strides 1–3 of the SLIP walker with feedback, 60 training steps (@fig-running, @fig-predictable). " *
    "R2: share of free runs that keep the cycle for 30 cycles in the benchmark · on the SLIP walker · on the running whole " *
    "signal (@fig-selfdriving A). " *
    "R3: squared difference between forecasts from different seeds, relative to the deviation's variance · benchmark cells " *
    "in which the seeds end in different fates (@fig-reproducible). " *
    "R4: remaining variance of the running deviations with 16 training strides · the SLIP share with the height alone · " *
    "seconds to choose the settings and fit one running trial (@fig-realistic). " *
    "R5: distance of the three leading per-cycle Floquet multipliers from the truth, 64 cycles of a simulated linear " *
    "time-periodic system (@fig-recovery). " *
    "R6: share of a per-trial quantity's variance between runners, intraclass correlation (@fig-heterogeneity A)."

load(k) = (f = joinpath(DIR, "$k.jls"); isfile(f) ? deserialize(f) : Dict{String, String}())
cols = [(k, h, load(k)) for (k, h) in COLS]
r1run, r1slip = load("R1run"), load("R1slip")                  # R1 from two figures: running (Figure 4), SLIP (Figure 3)
cols[1] = ("R1", COLS[1][2], Dict(m => string(get(r1run, m, "—"), " (", get(r1slip, m, "—"), ")") for (m, _) in ROWS))
io = IOBuffer()
println(io, "| Model | ", join([h for (_, h, _) in cols], " | "), " |")
println(io, "|:--|", join([":-:" for _ in cols], "|"), "|")
for (m, nm) in ROWS
    println(io, "| ", nm, " | ", join([get(d, m, "—") for (_, _, d) in cols], " | "), " |")
end
println(io, "\n: ", CAPTION, " {#tbl-scorecard tbl-colwidths=\"[18,14,17,15,18,9,9]\"}")
write(joinpath(ROOT, "_scorecard.md"), take!(io))
print(read(joinpath(ROOT, "_scorecard.md"), String))
