#!/usr/bin/env julia
# Data-driven Floquet maps on SLIP walking (Fig 3; OUTLINE.md §4 ¶2).
#
# Data: kicked SLIP walks (scripts/slip_truth.jl): stride-level kicks to (y, θ) at VLO, noise-free steps in
# between; truth = eigenvalues of the exact single-step Jacobian (open loop 0.628 ± 0.713i, ρ = 0.950;
# speed feedback κ = 0.1: 0.758 ± 0.431i, ρ = 0.872). Observed record: CoM height and velocities (y, ẋ, ẏ),
# each step resampled onto 50 points of its own duration (phase 0 = VLO, just after the kick).
# Energy is conserved, so deviations live in a 2-D plane: every map is fitted in the top-2 principal
# components of the deviations at its phase. Measurement noise: i.i.d., SD = ratio × the stride-to-stride SD
# of each variable (averaged over phase).
#
# Estimators (spectral radius ρ̂ of the estimated one-step map):
#   section     affine least-squares map of the phase-0 state from one step to the next
#   phase       the same at 10 phases across the step, median (Revzen & Guckenheimer 2012 phase-indexed maps)
#   piecewise K K affine sector-to-sector maps (K − 1 within the step, one across the kick), composed
#
# Usage: julia --project=. scripts/fig3_slip_maps.jl     (writes cache/fig3/slip_maps.jls)

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using Serialization, Statistics, LinearAlgebra, Random, Printf

const ROOT = dirname(@__DIR__)
const WALKDIR = joinpath(ROOT, "cache", "truth", "slip_walks")
const NS = [10, 20, 30, 60, 120, 250, 500, 1000, 2000]
const NOISE = [0.0, 0.1, 0.25, 0.5, 1.0]
const KS = [2, 4, 8, 16]
const NPH = 50

"Least-squares affine map between deviation samples (rows), in the top-2 PCA bases at each end."
function pca2(X)
    Xc = X .- mean(X, dims = 1)
    V = svd(Xc).V[:, 1:2]
    return Xc * V, V
end
function fitmap(Xa, Xb)                       # rows = samples (already reduced, centred)
    Z = hcat(Xa, ones(size(Xa, 1)))
    B = (Z \ Xb)'                              # Xb ≈ Z B'
    return B[:, 1:2]
end
specrad(A) = maximum(abs, eigvals(A))
function section_map(S; φ = 1)
    Xall, _ = pca2(S[:, φ, :]); fitmap(Xall[1:end-1, :], Xall[2:end, :])
end

function est_section(S; φ = 1)
    # one basis for both ends (same phase): all phase-φ samples
    Xall, V = pca2(S[:, φ, :]); Xa = Xall[1:end-1, :]; Xb = Xall[2:end, :]
    specrad(fitmap(Xa, Xb))
end
est_phase(S) = median([est_section(S; φ) for φ in 1:5:NPH])
function est_piecewise(S, K)
    b = [round(Int, 1 + (j - 1) * NPH / K) for j in 1:K]
    red = [pca2(S[:, bj, :])[1] for bj in b]           # reduced coordinates at each boundary, all steps
    M = Matrix(1.0I, 2, 2)
    for j in 1:K
        if j < K
            A = fitmap(red[j], red[j+1])                 # within the step
        else
            A = fitmap(red[K][1:end-1, :], red[1][2:end, :])   # last boundary → next step's phase 0 (kick)
        end
        M = A * M
    end
    specrad(M)
end

function noisy(S, ratio, rng)
    ratio == 0 && return S
    sd = vec(mean(std(S, dims = 1), dims = 2))           # stride-to-stride SD per variable, averaged over phase
    S .+ ratio .* reshape(sd, 1, 1, :) .* randn(rng, size(S))
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = NamedTuple[]
    for f in sort(readdir(WALKDIR))
        w = deserialize(joinpath(WALKDIR, f)); w.fallen && continue
        truth = maximum(abs, eigvals(w.J))
        for N in NS, r in NOISE
            N > size(w.S, 1) && continue
            rng = MersenneTwister(hash((w.seed, N, r)))
            S = noisy(w.S[1:N, :, :], r, rng)
            ev = eigvals(section_map(S))
            push!(rows, (; kappa = w.kappa, seed = w.seed, N, noise = r, truth, method = "section", K = 0, ρ = maximum(abs, ev),
                           ev = ComplexF64.(ev)))
            push!(rows, (; kappa = w.kappa, seed = w.seed, N, noise = r, truth, method = "phase", K = 0, ρ = est_phase(S),
                           ev = ComplexF64[]))
            for K in KS
                push!(rows, (; kappa = w.kappa, seed = w.seed, N, noise = r, truth, method = "piecewise", K, ρ = est_piecewise(S, K),
                               ev = ComplexF64[]))
            end
        end
    end
    mkpath(joinpath(ROOT, "cache", "fig3"))
    truths = Dict(κ => ComplexF64.(eigvals(deserialize(joinpath(WALKDIR, "walk_k$(κ)_s1.jls")).J)) for κ in (0.0, 0.1))
    serialize(joinpath(ROOT, "cache", "fig3", "slip_maps.jls"), (; rows, truths))
    med(sel) = (v = [x.ρ for x in rows if sel(x)]; isempty(v) ? NaN : median(v))
    for κ in (0.0, 0.1), m in ("section", "phase", "piecewise")
        tr = med(x -> x.kappa == κ); K = m == "piecewise" ? 8 : 0
        @printf("κ=%.1f %-9s%s truth %.3f | noise 0: N=30 %.3f N=250 %.3f N=2000 %.3f | N=250: noise .25 %.3f .5 %.3f 1 %.3f\n",
                κ, m, K > 0 ? "(K=8)" : "     ", [x.truth for x in rows if x.kappa == κ][1],
                [med(x -> x.kappa == κ && x.method == m && x.K == K && x.noise == 0 && x.N == N) for N in (30, 250, 2000)]...,
                [med(x -> x.kappa == κ && x.method == m && x.K == K && x.N == 250 && x.noise == r) for r in (0.25, 0.5, 1.0)]...)
    end
end
