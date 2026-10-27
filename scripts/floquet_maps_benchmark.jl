#!/usr/bin/env julia
# Data-driven Floquet estimation: how many cycles, and how much noise?
#
# Ground truth: two coupled FitzHugh–Nagumo relaxation oscillators
#   v̇ᵢ = vᵢ − vᵢ³/3 − wᵢ + 0.6 + k (vⱼ − vᵢ),  ẇᵢ = ε (vᵢ + 0.7 − 0.8 wᵢ)
# with ε = 0.08, k = 0.02. The synchronous cycle has one slow transverse
# multiplier — the relative-phase (coordination) mode, |μ| = 0.688, set by the
# coupling — and two that are effectively zero (each unit's own amplitude
# relaxes within a cycle). Process noise on v₁, v₂ (cycle-to-cycle variability)
# excites the slow mode; measurement noise is added on top, quoted relative to
# the size of the deviations from the cycle that the process noise produces.
#
# Estimators of the slow multiplier, as they are used in practice:
#   section     Poincaré return map: affine fit x_{k+1} = M x_k + c of the state
#               at successive upward crossings of v₁ (section coordinates
#               w₁, v₂, w₂); largest |eig(M)|.
#   phase-map   phase-indexed next-cycle map in deviation coordinates (Revzen &
#               Guckenheimer): phase from interpolated crossings, mean cycle by
#               phase bin, δ(φ, cycle k+1) = M_φ δ(φ, cycle k) per phase bin;
#               median over bins of the largest |eig(M_φ)|.
#   raw τ-map   one global affine map x(t+T) = A x(t) + b in raw coordinates
#               (what a next-cycle DMD or ESN readout implies). Directions the
#               cycle itself occupies map to themselves (unit eigenvalues), so we
#               drop as many eigenvalues as the mean cycle has dimensions and
#               report the largest remaining one. This works here only because
#               the synchronous cycle spans 2 of 4 dimensions; a cycle that fills
#               the observed space leaves nothing to report.
#   piecewise   K phase sectors; one affine map per sector from the state at its
#               entry boundary to the state at the next boundary (same cycle);
#               the K maps compose into a monodromy spanning exactly one period.
#               Boundary states are sampled at fixed phase, so the phase direction
#               is projected out (as for the section map): report the largest
#               |eigenvalue|.
#   piecewise-1step (diagnostic) the same with one-step maps per sector, raised
#               to the sector dwell time — ~110 maps composed per cycle.
#
# Usage: julia --project=. scripts/floquet_maps_benchmark.jl <noise_ratio> [reps]

include("common.jl")

const EPS_F, K_CPL = 0.08, 0.02
const SAMPLES_PER_CYCLE = 110
const N_GRID = [8, 15, 30, 60, 120, 250, 500, 1000, 2000]
const K_GRID = [4, 8, 16, 32, 64]
const SIGMA_P = 0.02

function cfhn!(du, u, p, t)
    v1, w1, v2, w2 = u
    du[1] = v1 - v1^3/3 - w1 + 0.6 + K_CPL * (v2 - v1); du[2] = EPS_F * (v1 + 0.7 - 0.8w1)
    du[3] = v2 - v2^3/3 - w2 + 0.6 + K_CPL * (v1 - v2); du[4] = EPS_F * (v2 + 0.7 - 0.8w2)
end

"Deterministic period, a point on the cycle, and the true multipliers."
const TRUTH = let
    sol = solve(ODEProblem(cfhn!, [-1.0, -0.3, -0.6, -0.25], (0.0, 3000.0)), Tsit5();
                saveat = 0.01, abstol = 1e-11, reltol = 1e-11)
    t = sol.t; v = sol[1, :]; i0 = length(t) ÷ 2; m = mean(v[i0:end])
    c = [t[i] for i in i0+1:length(t) if v[i-1] < m <= v[i]]
    T = mean(diff(c)); k = findlast(<=(c[end-1]), t)
    μ = sort(abs.(numerical_floquet(cfhn!, sol[:, k], T).values), rev = true)
    (; T, u0 = sol[:, k], μ_slow = μ[2], μ_all = μ, vmid = m)
end
const DT = TRUTH.T / SAMPLES_PER_CYCLE

"Simulate n_cyc cycles with process noise on v₁, v₂ (Euler–Maruyama)."
function simulate(n_cyc; σp = SIGMA_P, seed = 1)
    g!(du, u, p, t) = (du .= 0; du[1] = σp; du[3] = σp)
    n = n_cyc * SAMPLES_PER_CYCLE + 2SAMPLES_PER_CYCLE
    sol = solve(SDEProblem(cfhn!, g!, TRUTH.u0, (0.0, (n - 1) * DT); seed = seed), EM();
                dt = DT / 20, saveat = DT)
    return permutedims(Array(sol))
end

# ── phase from crossings ─────────────────────────────────────────────────────
"Upward crossings of v₁ through the mid-level (on a lightly smoothed copy)."
function crossings(X)
    v = X[:, 1]; w = 3
    vs = [mean(v[max(1, i-w):min(end, i+w)]) for i in eachindex(v)]
    c = Float64[]
    for i in 2:length(vs)
        if vs[i-1] < TRUTH.vmid <= vs[i]
            push!(c, (i - 1) + (TRUTH.vmid - vs[i-1]) / (vs[i] - vs[i-1]))
        end
    end
    return c                                   # fractional sample indices
end
interp_row(X, s) = (i = clamp(floor(Int, s), 1, size(X, 1) - 1); f = s - i;
                    (1 - f) .* X[i, :] .+ f .* X[i+1, :])

# ── estimators ───────────────────────────────────────────────────────────────
affine_fit(Xa, Xb) = (Z = vcat(Xa, ones(1, size(Xa, 2))); (Xb * Z' / (Z * Z' + 1e-10I))[:, 1:end-1])
drop_unit(ev) = (m = abs.(ev); i = argmin(abs.(m .- 1)); maximum(m[setdiff(eachindex(m), i)]))

"""
Split a monodromy estimate into its phase eigenvalue (eigenvector most aligned
with the flow direction `tang` at the section, which should be exactly 1) and
the largest remaining |eigenvalue| (the slow transverse multiplier).
"""
function split_phase(M, tang)
    F = eigen(M)
    align = [abs(dot(normalize(real(F.vectors[:, i])), normalize(tang))) for i in axes(M, 1)]
    ip = argmax(align)
    rest = [abs(F.values[i]) for i in axes(M, 1) if i != ip]
    return abs(F.values[ip]), maximum(rest)
end

function est_section(X, c)
    S = reduce(hcat, [interp_row(X, s)[[2, 3, 4]] for s in c])
    size(S, 2) < 5 && return NaN
    maximum(abs.(eigvals(affine_fit(S[:, 1:end-1], S[:, 2:end]))))
end

function est_phasemap(X, c; nbins = 20)
    length(c) < 5 && return NaN
    # phase-matched samples: for cycle k and bin b, the state at φ = (b - 0.5)/nbins
    ncyc = length(c) - 1
    states = [interp_row(X, c[k] + (b - 0.5) / nbins * (c[k+1] - c[k])) for b in 1:nbins, k in 1:ncyc]
    vals = Float64[]
    for b in 1:nbins
        D = reduce(hcat, states[b, :]); D = D .- mean(D, dims = 2)     # deviation from mean cycle
        push!(vals, maximum(abs.(eigvals(affine_fit(D[:, 1:end-1], D[:, 2:end])))))
    end
    median(vals)
end

"Number of dimensions the mean cycle occupies (singular values > 2 % of the largest)."
function cycle_span(X, c; nbins = 50)
    ncyc = length(c) - 1
    G = [mean(interp_row(X, c[k] + (b - 0.5) / nbins * (c[k+1] - c[k])) for k in 1:ncyc)
         for b in 1:nbins]
    G = reduce(hcat, G); S = svdvals(G .- mean(G, dims = 2))
    return count(S .> 0.02 * S[1])
end

function est_rawmap(X, c)
    T = SAMPLES_PER_CYCLE
    m = sort(abs.(eigvals(affine_fit(X[1:end-T, :]', X[1+T:end, :]'))), rev = true)
    k = cycle_span(X, c)
    k >= length(m) ? NaN : m[k+1]
end

"""
Piecewise phase-sector model: one affine map per sector, from the state at the
sector's entry boundary to the state at the next boundary in the same cycle;
the K maps compose into a monodromy spanning exactly one period.
"""
function est_piecewise(X, c, K; return_fit = false)
    length(c) < 5 && return return_fit ? (NaN, NaN, NaN) : NaN
    ncyc = length(c) - 1
    B = [interp_row(X, c[k] + (j - 1) / K * (c[k+1] - c[k])) for j in 1:K+1, k in 1:ncyc]
    M = Matrix{Float64}(I, size(X, 2), size(X, 2)); fit_err = Float64[]
    for j in 1:K
        Xa = reduce(hcat, B[j, :]); Xb = reduce(hcat, B[j+1, :])
        A = affine_fit(Xa, Xb)
        b = mean(Xb .- A * Xa, dims = 2)
        push!(fit_err, sqrt(mean(abs2, Xb .- (A * Xa .+ b))))
        M = A * M
    end
    μ = maximum(abs.(eigvals(M)))          # phase direction already projected out
    return return_fit ? (μ, median(fit_err), NaN) : μ
end

"Diagnostic: one-step affine maps per sector, raised to the dwell time (~110 composed)."
function est_piecewise_1step(X, c, K; return_fit = false)
    length(c) < 3 && return return_fit ? (NaN, NaN, NaN) : NaN
    ph = fill(NaN, size(X, 1))
    for k in 1:length(c)-1
        i0, i1 = ceil(Int, c[k]), floor(Int, c[k+1])
        for i in i0:i1
            ph[i] = (i - c[k]) / (c[k+1] - c[k])
        end
    end
    sec = [isnan(p) ? 0 : clamp(floor(Int, p * K) + 1, 1, K) for p in ph]
    M = Matrix{Float64}(I, size(X, 2), size(X, 2)); fit_err = Float64[]
    dwell = SAMPLES_PER_CYCLE / K
    for j in 1:K
        idx = [i for i in 1:size(X, 1)-1 if sec[i] == j && sec[i+1] != 0]
        length(idx) < 8 && return return_fit ? (NaN, NaN, NaN) : NaN
        A = affine_fit(X[idx, :]', X[idx .+ 1, :]')
        b = mean(X[idx .+ 1, :]' .- A * X[idx, :]', dims = 2)
        push!(fit_err, sqrt(mean(abs2, X[idx .+ 1, :]' .- (A * X[idx, :]' .+ b))))
        M = real(exp(dwell * log(complex(A)))) * M      # A^dwell, fractional power
    end
    # flow direction at the section (phase 0), from the data
    tang = mean(reduce(hcat, [X[ceil(Int, s)+1, :] .- X[ceil(Int, s), :] for s in c[1:end-1]]), dims = 2)
    μ_phase, μ = split_phase(M, vec(tang))
    return return_fit ? (μ, median(fit_err), μ_phase) : μ
end

# ── one replicate ────────────────────────────────────────────────────────────
function replicate(noise_ratio, seed)
    Xc = simulate(maximum(N_GRID); seed)
    # deviation scale: spread of the section states across cycles (process-noise driven)
    c0 = crossings(Xc)
    dev = mean(std(reduce(hcat, [interp_row(Xc, s) for s in c0]), dims = 2))
    X = Xc .+ noise_ratio * dev .* randn(MersenneTwister(seed + 999), size(Xc))
    out = NamedTuple[]
    for N in N_GRID
        Xn = X[1:N*SAMPLES_PER_CYCLE + SAMPLES_PER_CYCLE, :]
        c = crossings(Xn)
        push!(out, (; N, est = "section", μ = est_section(Xn, c)))
        push!(out, (; N, est = "phase-map", μ = est_phasemap(Xn, c)))
        push!(out, (; N, est = "raw τ-map", μ = est_rawmap(Xn, c)))
        for K in K_GRID
            μ, fe, μp = est_piecewise(Xn, c, K; return_fit = true)
            push!(out, (; N, est = "piecewise K=$K", μ, fit_err = fe, μ_phase = μp))
        end
        μ1 = est_piecewise_1step(Xn, c, 16)
        push!(out, (; N, est = "piecewise-1step K=16", μ = μ1))
    end
    return (; noise_ratio, seed, dev, rows = out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ratio = parse(Float64, ARGS[1]); reps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    @info "truth" TRUTH.T TRUTH.μ_all ratio reps
    res = [replicate(ratio, s) for s in 1:reps]
    mkpath(cache_path("floquet"))
    serialize(cache_path("floquet/maps_ratio$(ratio).jls"), (; truth = TRUTH, res))
    for est in ["section", "phase-map", "raw τ-map", "piecewise K=8", "piecewise K=16", "piecewise K=64"],
        N in (8, 30, 120, 500, 2000)
        v = filter(isfinite, [r.μ for x in res for r in x.rows if r.est == est && r.N == N])
        isempty(v) && continue
        @printf("ratio %.2f %-15s N=%5d  median %.3f  IQR [%.3f, %.3f]  (truth %.3f)\n",
                ratio, est, N, median(v), quantile(v, 0.25), quantile(v, 0.75), TRUTH.μ_slow)
    end
end
