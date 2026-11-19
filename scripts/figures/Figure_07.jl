#!/usr/bin/env julia
# Figure_07.jl — Reproducible answers (R3) (manuscript: @fig-reproducible). Run: julia --project=. scripts/figures/Figure_07.jl
# Models that depend on a random seed (ESN reservoir; LSTM and neural ODE initialisation and minibatches) refitted with
# several seeds on the same data; linear models and SSMLearnPy have no seed (the same data and settings give the same fit).
#   A  running deviations (maus_requirements.jl, maus_ssm_req.py; 22 trials, 150 training strides): forecasts ⅛ stride
#      ahead from 5 seeds, squared difference between seeds relative to the deviation's variance (mean over seed pairs;
#      colour) against the model's remaining variance (median over seeds; grey)
#   B  the remaining variance ⅛ stride ahead of every trial and seed: trials ordered by their median; lines span the seeds
#   C  a quantity a study would compare between people (the share of the current deviation's variance the model expects
#      to remain ⅛ stride later): the spread between seeds as a share of the spread between trials
#   D  the free-running benchmark (self_driving_benchmark.jl; Figures 5–6): first-cycle remaining variance of every seed in
#      every system × regime cell (ESN 8 seeds, LSTM 2, neural ODE 4); numbers: cells whose seeds end in different fates
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf, NPZ, Random

const ROOT = dirname(dirname(@__DIR__)); const RQ = joinpath(ROOT, "cache", "maus", "req"); const SD = joinpath(ROOT, "cache", "selfdrive")
const C_LTP = colorant"#1a4f99"; const C_LPV = colorant"#78b0f5"
MODELS = [(:ltp, "Fourier in phase", C_LTP, :diamond, :solid), (:lpv, "phase-binned", C_LPV, :rect, :solid),
          (:dmd, "time-invariant (DMD)", CAT[1], :circle, :solid), (:esn, "ESN", CAT[2], :utriangle, :dash),
          (:lstm, "LSTM", CAT[5], :dtriangle, :dash), (:node, "neural ODE", CAT[3], :hexagon, :dash),
          (:ssm, "SSMLearnPy", CAT[4], :star5, :dash)]
const SEEDED = (:esn, :lstm, :node)
TR = deserialize(joinpath(RQ, "truth.jls")); NT = length(TR.ids)
function load(m)
    if m == :ssm
        fs = [joinpath(RQ, @sprintf("ssm_trial_%02d.npz", i)) for i in 1:NT]
        all(isfile, fs) || return nothing
        return [[(Z = npzread(f); (; r2 = Z["r2"], fc = Dict(12 => Z["fc12"]))) for f in fs]]
    end
    files = m in SEEDED ? [joinpath(RQ, "$(m)_seed$(s).jls") for s in 1:5] : [joinpath(RQ, "$(m).jls")]
    out = filter(R -> length(R) == NT && all(!isnothing, R), [deserialize(f) for f in files if isfile(f)])
    isempty(out) ? nothing : out
end
RUN = Dict(m => load(m) for (m, _, _, _, _) in MODELS)
present = [x for x in MODELS if RUN[x[1]] !== nothing]
rrv(R, i) = 1 - R[i].r2[2]
disagree(Rs, i) = length(Rs) < 2 ? 0.0 :
    mean(sum(abs2, Float64.(Rs[a][i].fc[12]) .- Float64.(Rs[b][i].fc[12])) / sum(abs2, TR.d[12][i]) for a in eachindex(Rs) for b in a+1:length(Rs))
persist(R, i) = sum(abs2, Float64.(R[i].fc[12])) / sum(abs2, Float64.(TR.d0[i]))
"Benchmark: per cell (system, regime) the first-cycle error of each seed of ESN and neural ODE, its fate, and the template's."
function bench()
    cells = Dict{Tuple{String, String}, Any}()
    for f in readdir(SD)
        m = match(r"^(.+)_(realistic|long)_dmd-esn-template\.jls$", f); m === nothing && continue
        o = deserialize(joinpath(SD, f)); t = only([r for r in o.rows if r.model == "template"]).e1
        esn = isfile(joinpath(SD, "esntuned_$(m[1])_$(m[2]).jls")) ? deserialize(joinpath(SD, "esntuned_$(m[1])_$(m[2]).jls")).rows : []
        nf = joinpath(SD, "$(m[1])_$(m[2])_node.jls"); node = isfile(nf) ? [r for r in deserialize(nf).rows if r.model == "NODE"] : []
        lf = joinpath(SD, "lstm_$(m[1])_$(m[2]).jls"); lstm = isfile(lf) ? deserialize(lf).rows : []
        cells[(m[1], m[2])] = (; t, esn = [(r.e1, r.fate) for r in esn], node = [(r.e1, r.fate) for r in node], lstm = [(r.e1, r.fate) for r in lstm])
    end
    cells
end
B = bench(); cellkeys = sort(collect(keys(B)); by = k -> (k[2], k[1]))

fig = Figure(size = (COL2, 118MM))
# A: forecast disagreement between seeds
axA = Axis(fig[1, 1]; xticks = (1:length(present), [x[2] for x in present]), xticklabelrotation = π / 5, xticklabelsize = 6,
           ylabel = "share of the deviation's variance", title = "running: forecasts from different seeds", titlesize = 7,
           ylabelsize = 6.5, yticklabelsize = 6, limits = (0.4, length(present) + 0.6, -0.03, 1.15))
rng = MersenneTwister(1)
for (k, (m, nm, col, mk, _)) in enumerate(present)
    Rs = RUN[m]; v = [disagree(Rs, i) for i in 1:NT]; e = [median(rrv(R, i) for R in Rs) for i in 1:NT]
    scatter!(axA, fill(k - 0.14, NT) .+ 0.04 .* randn(rng, NT), min.(e, 1.1); color = (C_REF, 0.45), marker = mk, markersize = 3.5)
    xs = fill(k + 0.14, NT) .+ 0.04 .* randn(rng, NT); cl = v .> 1.1
    scatter!(axA, xs[.!cl], v[.!cl]; color = (col, 0.75), marker = mk, markersize = 4)
    any(cl) && scatter!(axA, xs[cl], fill(1.1, count(cl)); color = :white, strokecolor = col, strokewidth = 0.8, marker = mk, markersize = 4.5)
    m in SEEDED || text!(axA, k + 0.14, 0.03; text = "no\nseed", fontsize = 5.5, color = C_REF, align = (:center, :bottom))
end
text!(axA, 0.45, 1.12; text = "grey: remaining variance, ⅛ stride\ncolour: difference between seeds (open: above 1.1)", fontsize = 5.5,
      color = C_REF, align = (:left, :top))
panel_label!(fig[1, 1], "A")
# B: every trial and seed
axB = Axis(fig[1, 2]; yscale = log10, ylabel = "remaining variance, ⅛ stride", title = "running: each trial, each seed", titlesize = 7,
           ylabelsize = 6.5, yticklabelsize = 6, xticklabelsize = 6, limits = (0, 3NT + 6, 0.2, 80),
           yticks = ([0.3, 0.5, 1, 2, 5, 20, 50], ["0.3", "0.5", "1", "2", "5", "20", "50"]))
hlines!(axB, 1; color = C_REF, linewidth = 0.6, linestyle = :dash)
sm = [x for x in present if x[1] in SEEDED]; axB.xticks = ([(j - 1) * (NT + 3) + NT / 2 for j in eachindex(sm)], [x[2] for x in sm])
for (j, (m, nm, col, mk, _)) in enumerate(sm)
    Rs = RUN[m]; E = [rrv(R, i) for i in 1:NT, R in Rs]; o = sortperm(vec(median(E; dims = 2)))
    x0 = (j - 1) * (NT + 3)
    for (k, i) in enumerate(o)
        lines!(axB, [x0 + k, x0 + k], [minimum(E[i, :]), maximum(E[i, :])]; color = (col, 0.6), linewidth = 0.8)
        scatter!(axB, fill(x0 + k, size(E, 2)), E[i, :]; color = (col, 0.8), marker = mk, markersize = 2.8)
    end
    text!(axB, x0 + NT / 2, 70; text = "$(length(Rs)) seed" * (length(Rs) == 1 ? "" : "s"), fontsize = 5.5, color = col, align = (:center, :top))
end
text!(axB, 3NT + 5, 1.05; text = "the average stride", fontsize = 5.5, color = C_REF, align = (:right, :bottom))
panel_label!(fig[1, 2], "B")
# C: seed spread of a quantity compared between people, as a share of the spread between trials
axC = Axis(fig[2, 1]; xticks = (1:length(present), [x[2] for x in present]), xticklabelrotation = π / 5, xticklabelsize = 6,
           ylabel = "spread between seeds /\nspread between trials", title = "a quantity compared between people", titlesize = 7,
           ylabelsize = 6.5, yticklabelsize = 6, limits = (0.4, length(present) + 0.6, 0, nothing))
ratio = Dict{Symbol, Float64}()
for (k, (m, nm, col, mk, _)) in enumerate(present)
    Rs = RUN[m]; Q = [persist(R, i) for i in 1:NT, R in Rs]
    ratio[m] = size(Q, 2) < 2 ? 0.0 : median(std(Q[i, :]) for i in 1:NT) / std(vec(median(Q; dims = 2)))
    barplot!(axC, [k], [ratio[m]]; color = (col, 0.85), width = 0.6)
    m in SEEDED || text!(axC, k, 0.0; text = "0", fontsize = 5.5, color = C_REF, align = (:center, :bottom), offset = (0, 2))
end
panel_label!(fig[2, 1], "C")
# D: benchmark seeds
axD = Axis(fig[2, 2]; yscale = log10, ylabel = "first-cycle remaining variance", title = "free-running benchmark: each cell, each seed",
           titlesize = 7, ylabelsize = 6.5, yticklabelsize = 6, xticks = (1:length(cellkeys), ["$(k[1]) $(k[2] == "long" ? "L" : "R")" for k in cellkeys]),
           xticklabelrotation = π / 3, xticklabelsize = 5.5, limits = (0.3, length(cellkeys) + 0.7, 1e-10, 1e4),
           yticks = ([1e-8, 1e-4, 1, 1e4], ["10⁻⁸", "10⁻⁴", "1", "10⁴"]))
hlines!(axD, 1; color = C_REF, linewidth = 0.6, linestyle = :dash)
ndiff = Dict("ESN" => 0, "NODE" => 0, "LSTM" => 0)
BSEED = ((:esn, "ESN", CAT[2], :utriangle, -0.2), (:lstm, "LSTM", CAT[5], :dtriangle, 0.0), (:node, "NODE", CAT[3], :hexagon, 0.2))
for (c, k) in enumerate(cellkeys), (key, lab, col, mk, dx) in BSEED
    rows = getfield(B[k], key); isempty(rows) && continue
    v = [isfinite(e) ? clamp((e / B[k].t)^2, 1e-10, 1e4) : 1e4 for (e, _) in rows]
    lines!(axD, [c + dx, c + dx], [minimum(v), maximum(v)]; color = (col, 0.6), linewidth = 0.8)
    scatter!(axD, fill(c + dx, length(v)), v; color = (col, 0.75), marker = mk, markersize = 3)
    length(unique(last.(rows))) > 1 && (ndiff[lab] += 1)
end
hasl = any(!isempty(B[k].lstm) for k in cellkeys)
text!(axD, length(cellkeys) + 0.6, 2e-10; text = "seeds end in different fates in $(ndiff["ESN"]) (ESN)" * (hasl ? ", $(ndiff["LSTM"]) (LSTM)\n" : "\n") * "and $(ndiff["NODE"]) (neural ODE) of $(length(cellkeys)) cells",
      fontsize = 5.5, color = C_REF, align = (:right, :bottom))
panel_label!(fig[2, 2], "D")
Legend(fig[3, 1:2], [[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 5)] for (_, _, col, mk, ls) in MODELS],
       [x[2] for x in MODELS]; orientation = :horizontal, framevisible = false, labelsize = 6.5, patchsize = (12, 5), colgap = 9)
colsize!(fig.layout, 1, Relative(0.42)); colgap!(fig.layout, 16); rowgap!(fig.layout, 10); rowgap!(fig.layout, 2, 2)
save_paper(fig, "Figure_07")

for (m, nm, _, _, _) in present
    Rs = RUN[m]
    @printf("%-22s seeds %d | disagreement %.3f | fits worse than the average stride %d/%d | seed/trial spread of the persistence %.2f\n", nm, length(Rs),
            median(disagree(Rs, i) for i in 1:NT), sum(count(i -> rrv(R, i) > 1, 1:NT) for R in Rs), NT * length(Rs), ratio[m])
end
@printf("benchmark: cells with seeds in different fates: ESN %d, LSTM %d, NODE %d of %d\n", ndiff["ESN"], ndiff["LSTM"], ndiff["NODE"], length(cellkeys))
for (key, lab) in ((:esn, "ESN"), (:lstm, "LSTM"), (:node, "NODE"))
    r = [(v = [isfinite(e) ? e / B[k].t : Inf for (e, _) in getfield(B[k], key)]; isempty(v) ? NaN : maximum(v)^2 / minimum(v)^2) for k in cellkeys]
    all(isnan, r) && continue
    @printf("%s: worst / best seed first-cycle remaining variance, median over cells %.1f (range %.1f–%s)\n", lab, median(filter(isfinite, r)),
            minimum(filter(isfinite, r)), any(!isfinite, r) ? "∞" : @sprintf("%.0f", maximum(r)))
end
KEY = Dict(:ltp => "FIP", :lpv => "LPV", :dmd => "DMD", :esn => "ESN", :lstm => "LSTM", :node => "NODE", :ssm => "SSM")
sc = Dict(KEY[m] => "no seed" for (m, _, _, _, _) in MODELS if !(m in SEEDED))
for (m, _, _, _, _) in present
    m in SEEDED || continue
    lab = KEY[m]; nb = any(!isempty(getfield(B[k], m)) for k in cellkeys)
    sc[lab] = @sprintf("%.2f", median(disagree(RUN[m], i) for i in 1:NT)) * (nb ? " · $(ndiff[lab])/$(length(cellkeys))" : "")
end
scorecard!("R3", sc)
