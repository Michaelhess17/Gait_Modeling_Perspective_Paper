#!/usr/bin/env julia
# Recovering known Floquet multipliers with the Fourier-in-phase operator (slip_ltp_recovery.jl): kicked SLIP walks,
# truth = eigenvalues of the exact step-to-step Jacobian (κ = 0: 0.628 ± 0.713i; κ = 0.1: 0.758 ± 0.431i). Error = the
# two largest monodromy eigenvalues matched to the true pair, mean distance in the complex plane; medians over 40 walks
# × chunks, both gains pooled; measurement noise 0.1 × the stride-to-stride SD unless stated.
#   (A) Error by strides of data: setting chosen by held-out prediction (R² 12% of a step ahead; band = IQR), a fixed
#       small-ridge setting (delay 8, rank 3, 2 harmonics, ridge 1e-3), the per-trial human setting (8, 12, 4, ridge
#       0.1), and the one-section return map (affine LS in the top-2 PCs at phase 0).
#   (B) Error by POD rank (delay 8; best harmonics ≥ 1 and ridge: filled; without phase: open), 16 and 128 strides.
#   (C) Bias of the leading modulus by ridge (delay 8, rank 12, 8 harmonics; 128 strides), by noise level. (Noise-free
#       data at this rank and ridge 1e-6 give spurious unstable modes, median bias +7.7: the deviations fill only 2
#       of the 12 dimensions; not shown.)
#   (D) Estimates in the complex plane, 16 strides: chosen by prediction (coloured by gain) and ridge 0.1 (grey); ×: truth.
# Usage: julia --project=. scripts/Fig_slip_ltp_recovery.jl      (writes figures/Fig_slip_ltp_recovery.*)

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, LinearAlgebra, Printf
const ROOT = dirname(@__DIR__)
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const R = deserialize(joinpath(ROOT, "cache", "slip_ltp", "recovery.jls"))
pair_err(ev, tr) = min(abs(ev[1] - tr[1]) + abs(ev[2] - tr[2]), abs(ev[1] - tr[2]) + abs(ev[2] - tr[1])) / 2
rows = R.rows; SETS = R.SETS; NS = R.NS
err(x, k) = pair_err(x.ev[:, k], R.truth[x.kappa])
sel(x) = argmax(replace(x.sk, NaN => -Inf))
cell(n, r) = [x for x in rows if x.n == n && x.noise == r]
kof(s) = findfirst(==(s), SETS)
KCOL = Dict(0.0 => CAT[1], 0.1 => CAT[2])

fig = Figure(size = (COL2, 64MM)); g = fig[1, 1] = GridLayout()

aA = Axis(g[1, 1]; xscale = log2, yscale = log10, xticks = NS, xlabel = "Strides of data", ylabel = "Multiplier error",
          title = "Strides needed", limits = ((6.5, 620), (0.01, 1.2)), yticks = ([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"]))
vspan!(aA, 6, 12; color = C_SHADE); text!(aA, 8.9, 0.012; text = "15-s\nwalk", fontsize = 6, color = C_REF, align = (:center, :bottom))
q = [quantile([err(x, sel(x)) for x in cell(n, 0.1)], [0.25, 0.5, 0.75]) for n in NS]
band!(aA, NS, getindex.(q, 1), getindex.(q, 3); color = (C_TRUTH, 0.12))
lines!(aA, NS, getindex.(q, 2); color = C_TRUTH, linewidth = 1.4); scatter!(aA, NS, getindex.(q, 2); color = C_TRUTH, markersize = 5)
for (s, col, ls) in (((8, 3, 2, 1e-3), C_TRUTH, :dash), ((8, 12, 4, 1e-1), CAT[2], :dot))
    k = kof(s); lines!(aA, NS, [median(err(x, k) for x in cell(n, 0.1)) for n in NS]; color = col, linewidth = 1.2, linestyle = ls)
end
lines!(aA, NS, [median(pair_err(x.sec, R.truth[x.kappa]) for x in cell(n, 0.1)) for n in NS]; color = C_REF, linewidth = 1.2)
text!(aA, 600, 0.75; text = "ridge 0.1", color = CAT[2], fontsize = 6, align = (:right, :top))
text!(aA, 600, 0.11; text = "chosen by prediction\n(band: IQR)", fontsize = 6, align = (:right, :bottom))
text!(aA, 600, 0.013; text = "rank 3, ridge 1e-3 (dashed)\nsection map (grey)", fontsize = 6, align = (:right, :bottom))
plabel!(g[1, 1], "A")

aB = Axis(g[1, 2]; xscale = log2, yscale = log10, xticks = [2, 3, 4, 6, 12, 24], xlabel = "Rank (delay 8)", ylabel = "Multiplier error",
          title = "Rank > true dimension", limits = ((1.7, 30), (0.01, 1.2)), yticks = ([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"]))
rks = [2, 3, 4, 6, 12, 24]
for (n, col) in ((16, C_REF), (128, C_TRUTH))
    C = cell(n, 0.1)
    best(r, hs) = minimum(median(err(x, k) for x in C) for k in eachindex(SETS) if SETS[k][1] == 8 && SETS[k][2] == r && SETS[k][3] in hs)
    v1 = [best(r, (1, 2, 4, 8, 16)) for r in rks]; v0 = [best(r, (0,)) for r in rks]
    lines!(aB, rks, v1; color = col, linewidth = 1.3); scatter!(aB, rks, v1; color = col, markersize = 5)
    lines!(aB, rks, v0; color = col, linewidth = 0.8, linestyle = :dash); scatter!(aB, rks, v0; color = :white, strokecolor = col, strokewidth = 0.9, markersize = 5)
end
text!(aB, 25, 1.1; text = "filled: with phase\nopen: without\ngrey: 16 strides\nblack: 128", fontsize = 6, align = (:right, :top))
text!(aB, 2.05, 0.019; text = "true\ndim. 2", fontsize = 6, color = C_REF, align = (:left, :top))
plabel!(g[1, 2], "B")

aC = Axis(g[1, 3]; xscale = log10, xlabel = "Ridge (× mean Gram diagonal)", ylabel = "Leading |multiplier| − truth",
          title = "Ridge shrinks multipliers", limits = ((5e-7, 0.2), (-0.65, 0.05)), xticks = ([1e-6, 1e-3, 1e-2, 1e-1], ["1e-6", "1e-3", "0.01", "0.1"]))
hlines!(aC, 0; color = C_REF, linewidth = 0.6)
λs = sort(unique(last.(SETS)))
for (r, col) in zip(R.NOISE[2:end], (C_REF, colorant"#555452", C_TRUTH))
    v = [median(abs(x.ev[1, kof((8, 12, 8, λ))]) - abs(R.truth[x.kappa][1]) for x in cell(128, r)) for λ in λs]
    lines!(aC, λs, v; color = col, linewidth = 1.2); scatter!(aC, λs, v; color = col, markersize = 4.5)
    text!(aC, 7e-7, v[1] - 0.012; text = @sprintf("%.2f", r), color = col, fontsize = 6, align = (:left, :top))
end
text!(aC, 7e-7, -0.4; text = "measurement noise\n(× stride-to-stride SD)\n128 strides", fontsize = 6, align = (:left, :top))
plabel!(g[1, 3], "C")

aD = Axis(g[1, 4]; aspect = DataAspect(), xlabel = "Re", ylabel = "Im", title = "16 strides", limits = ((-0.15, 1.15), (-1.05, 1.05)))
lines!(aD, cos.(range(0, 2π; length = 200)), sin.(range(0, 2π; length = 200)); color = C_REF, linewidth = 0.6)
k01 = kof((8, 12, 4, 1e-1))
for x in cell(16, 0.1)
    scatter!(aD, real.(x.ev[1:2, k01]), imag.(x.ev[1:2, k01]); color = (C_REF, 0.5), markersize = 2.5)
end
for x in cell(16, 0.1)
    k = sel(x); scatter!(aD, real.(x.ev[1:2, k]), imag.(x.ev[1:2, k]); color = (KCOL[x.kappa], 0.55), markersize = 2.5)
end
for κ in (0.0, 0.1)
    scatter!(aD, real.(R.truth[κ]), imag.(R.truth[κ]); color = C_TRUTH, marker = :xcross, markersize = 8)
end
text!(aD, -0.1, -1.0; text = "κ 0   κ 0.1   ridge 0.1", fontsize = 6, align = (:left, :bottom), color = :transparent)
text!(aD, -0.1, -1.0; text = "κ 0", fontsize = 6, color = KCOL[0.0], align = (:left, :bottom))
text!(aD, 0.2, -1.0; text = "κ 0.1", fontsize = 6, color = KCOL[0.1], align = (:left, :bottom))
text!(aD, 0.6, -1.0; text = "ridge 0.1", fontsize = 6, color = C_REF, align = (:left, :bottom))
plabel!(g[1, 4], "D")
colgap!(g, 14); colsize!(g, 4, Relative(0.2))
save_paper(fig, "Fig_slip_ltp_recovery")
