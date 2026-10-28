#!/usr/bin/env julia
# SLIP step multiplier, modulus AND rotation, read from each model's off-cycle runs (Fig 6C; OUTLINE.md §6 ¶2).
#
# Fig 6A/B score only the relaxation rate. The SLIP step map has a complex pair (truth 0.748 ± 0.181i: |λ| 0.769,
# 13.6° per step), so a model can match the rate and still not rotate. Same runs as Fig 6B (offcycle_v2.jl caches:
# each model's 30-step free run after the 4 off-cycle warm-ups; truth: slip_rotation_truth.jl, same starts and
# window), read the same way for truth and every model:
#   section   hyperplane through the true cycle at phase p, normal to the cycle there (standardised units);
#             crossings (− → +, within R_SEC of the section point) by linear interpolation, in-plane coordinates
#   map       affine least squares z_{k+1} ≈ A z_k + b over the first K_STEPS steps of all 4 runs; the affine
#             term absorbs a model's own (offset) cycle, as in Fig 3's section map
#   sections  PHASES, where the protocol recovers the exact multiplier on true runs at ×0.25 perturbation (within
#             0.4° and 0.01). Sections near VLO are degenerate in CoM coordinates (the CoM projection folds there).
# Checks: on true runs at the test perturbation the protocol reads 0.78 ∠ 11.6–12.4° (the perturbation is mildly
# nonlinear). Pairing each run with the model's own on-cycle run instead of the affine term (refits of DMD and
# ESN) gives the same values. DMD reads ≈ 1: its fundamental is pinned on the unit circle, so cycle amplitude is
# neutral and its off-cycle deviation plateaus at 28–49% of the initial deviation after about 8 steps.
#
# Usage: julia --project=. scripts/slip_rotation.jl     (writes cache/fig6/slip_rotation.jls)

using Serialization, Statistics, LinearAlgebra, Printf

const ROOT = dirname(@__DIR__)
const FDIR = joinpath(ROOT, "cache", "floquet")
const PHASES = (56, 67, 89, 100)
const K_STEPS = 15
const R_SEC = 1.0

"In-plane coordinates of the section crossings of X (rows = samples)."
function crossings(X, c, n, B)
    z = Vector{Float64}[]
    for i in 1:size(X, 1)-1
        a, b = dot(X[i, :] .- c, n), dot(X[i+1, :] .- c, n)
        (a < 0 <= b && norm(X[i, :] .- c) < R_SEC) || continue
        x = X[i, :] .+ (-a / (b - a)) .* (X[i+1, :] .- X[i, :]); push!(z, B' * (x .- c))
    end
    z
end

"Affine step map from the crossings of several runs; eigenvalues."
function step_eigs(runs, C, p; K = K_STEPS)
    P = size(C, 1); c = C[p, :]; n = normalize(C[mod1(p + 1, P), :] .- C[mod1(p - 1, P), :]); B = nullspace(n')
    Z0 = Vector{Float64}[]; Z1 = Vector{Float64}[]
    for X in runs
        z = crossings(X, c, n, B)[1:min(end, K + 1)]
        append!(Z0, z[1:end-1]); append!(Z1, z[2:end])
    end
    Za = hcat(reduce(hcat, Z0)', ones(length(Z0))); A = (Za \ reduce(hcat, Z1)')'[:, 1:2]
    eigvals(A)
end
lead(e) = e[argmax(abs.(e) .+ 1e-12 .* imag.(e))]          # largest |λ|; the upper one of a complex pair
row(model, seed, evs) = (; model, seed, ev = ComplexF64.(lead.(evs)), mod = median(abs.(lead.(evs))),
                         ang = median(rad2deg.(abs.(angle.(lead.(evs))))))

if abspath(PROGRAM_FILE) == @__FILE__
    D = deserialize(joinpath(ROOT, "cache", "truth", "slip_decays.jls"))
    Lg = deserialize(joinpath(ROOT, "cache", "truth", "slip_long_starts.jls"))
    allX = reduce(vcat, D.decays[1]); μ = mean(allX, dims = 1); σ = std(allX, dims = 1)   # offcycle_v2's standardisation
    std_(X) = (X .- μ) ./ σ
    C = std_(D.cycle); after(X) = std_(X)[Lg.w_warm+1:end, :]
    rows = NamedTuple[]
    push!(rows, row("truth ×0.25", 0, [step_eigs(after.(Lg.runs[0.25]), C, p) for p in PHASES]))
    push!(rows, row("truth", 0, [step_eigs(after.(Lg.runs[1.0]), C, p) for p in PHASES]))
    for m in ("dmd", "esn", "node"), f in sort(filter(f -> startswith(f, "offv2_slip_$(m)_"), readdir(FDIR)))
        o = deserialize(joinpath(FDIR, f))
        push!(rows, row(m, o.seed, [step_eigs([fc for fc in o.fcs if fc !== nothing], C, p) for p in PHASES]))
    end
    @printf("exact: |λ| %.3f, %.1f° per step\n", maximum(abs, D.λ), rad2deg(maximum(angle.(D.λ))))
    for r in rows
        @printf("%-11s seed %d: |λ| %.3f  %5.1f° per step   sections: %s\n", r.model, r.seed, r.mod, r.ang,
                join([@sprintf("%.3f∠%.1f°", abs(e), rad2deg(abs(angle(e)))) for e in r.ev], " "))
    end
    mkpath(joinpath(ROOT, "cache", "fig6"))
    serialize(joinpath(ROOT, "cache", "fig6", "slip_rotation.jls"), (; rows, exact = D.λ, phases = PHASES, k_steps = K_STEPS))
end
