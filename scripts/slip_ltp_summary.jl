#!/usr/bin/env julia
# Summary of slip_ltp_recovery.jl: how well the Fourier-in-phase operator's one-step monodromy recovers the SLIP
# walker's true step-to-step multipliers, by strides of data, measurement noise, rank, harmonics and ridge.
# Error: the two largest monodromy eigenvalues matched to the two true ones, mean distance in the complex plane
# (truth: κ = 0: 0.628 ± 0.713i; κ = 0.1: 0.758 ± 0.431i). Pooled over both gains unless stated.
# Usage: julia --project=. scripts/slip_ltp_summary.jl

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using Serialization, Statistics, LinearAlgebra, Printf
const ROOT = dirname(@__DIR__)
const R = deserialize(joinpath(ROOT, "cache", "slip_ltp", "recovery.jls"))
pair_err(ev, tr) = min(abs(ev[1] - tr[1]) + abs(ev[2] - tr[2]), abs(ev[1] - tr[2]) + abs(ev[2] - tr[1])) / 2
rows = R.rows; SETS = R.SETS; ns = length(SETS)
E = [Float64[isnan(real(x.ev[1, k])) ? NaN : pair_err(x.ev[:, k], R.truth[x.kappa]) for k in 1:ns] for x in rows]
Es = [pair_err(x.sec, R.truth[x.kappa]) for x in rows]
ρt(x) = abs(R.truth[x.kappa][1])
sel(x) = argmax(replace(x.sk, NaN => -Inf))
q(v) = @sprintf("%.3f [%.3f, %.3f]", median(v), quantile(v, 0.25), quantile(v, 0.75))
cell(n, r) = [i for i in eachindex(rows) if rows[i].n == n && rows[i].noise == r]
fixed = [(8, 24, 4, 1e-2), (8, 12, 4, 1e-1), (8, 4, 2, 1e-3), (4, 6, 4, 1e-3)]

println("Multiplier error (median [IQR]) and leading modulus bias ρ̂ − ρ, by strides and noise; both gains pooled")
for r in R.NOISE
    @printf("noise %.2f\n", r)
    for n in R.NS
        I = cell(n, r); isempty(I) && continue
        ks = sel.(rows[I]); es = [E[i][k] for (i, k) in zip(I, ks)]; bs = [abs(rows[i].ev[1, k]) - ρt(rows[i]) for (i, k) in zip(I, ks)]
        med = [median(filter(!isnan, [E[i][k] for i in I])) for k in 1:ns]; kb = argmin(replace(med, NaN => Inf))
        @printf("  n %4d | section %s | selected by prediction %s (bias %+.3f, within 0.1: %3.0f%%) | best fixed %s %.3f\n", n, q(Es[I]), q(es),
                median(bs), 100mean(es .< 0.1), string(SETS[kb]), med[kb])
    end
end

println("\nFixed settings (noise 0.1): error by strides")
for s in fixed
    k = findfirst(==(s), SETS)
    @printf("  %-22s %s\n", string(s), join([@sprintf("n%d %.3f (ρ̂−ρ %+.2f)", n, median(E[i][k] for i in cell(n, 0.1)),
            median(abs(rows[i].ev[1, k]) - ρt(rows[i]) for i in cell(n, 0.1))) for n in R.NS], "  "))
end

for n in (16, 128), r in (0.1,)
    I = cell(n, r)
    @printf("\nError by rank (rows, with delay) × harmonics (columns), best ridge, n = %d, noise %.2f\n", n, r)
    @printf("  %-8s %s\n", "(d, r)", join([@sprintf("H%-5d", h) for h in (0, 1, 2, 4, 8, 16)], " "))
    for dr in unique([(s[1], s[2]) for s in SETS])
        v = [minimum(median(E[i][k] for i in I) for k in eachindex(SETS) if (SETS[k][1], SETS[k][2]) == dr && SETS[k][3] == h) for h in (0, 1, 2, 4, 8, 16)]
        @printf("  %-8s %s\n", string(dr), join([@sprintf("%.3f ", x) for x in v], " "))
    end
end

println("\nRidge: leading modulus bias ρ̂ − ρ (median) at (8, 12, 8), noise 0.1")
for λ in sort(unique(last.(SETS)))
    k = findfirst(==((8, 12, 8, λ)), SETS)
    @printf("  λ %.0e: %s\n", λ, join([@sprintf("n%d %+.3f", n, median(abs(rows[i].ev[1, k]) - ρt(rows[i]) for i in cell(n, 0.1))) for n in R.NS], "  "))
end

println("\nSettings chosen by prediction (noise 0.1): most common (d, r, H, λ) by strides")
for n in R.NS
    ks = sel.(rows[cell(n, 0.1)]); c = Dict{Int, Int}(); for k in ks; c[k] = get(c, k, 0) + 1; end
    top = sort(collect(c), by = last, rev = true)[1:min(3, end)]
    @printf("  n %4d: %s | median rank %d, harmonics %d, ridge %.0e\n", n, join(["$(SETS[k]) ×$m" for (k, m) in top], ", "),
            median([SETS[k][2] for k in ks]), median([SETS[k][3] for k in ks]), median([SETS[k][4] for k in ks]))
end
