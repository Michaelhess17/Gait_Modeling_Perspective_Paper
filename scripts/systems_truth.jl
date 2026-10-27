#!/usr/bin/env julia
# Ground truth for Figs 3–4 (OUTLINE.md §4): Floquet multipliers and how much eigenvalues miss.
#
# Systems: Van der Pol ẋ = y, ẏ = μ(1 − x²)y − x (μ sweep); Stuart–Landau ż = (λ + i)z − |z|²z (normal
# reference); coupled FitzHugh–Nagumo (as in floquet_maps_benchmark.jl, 3 transverse directions).
# SLIP is computed in its own environment (scripts/slip_truth.jl).
#
# Everything is measured in OBSERVED coordinates: each state variable scaled by its SD along the cycle (the
# norm a data-driven model of standardised data uses). Non-normality is norm-dependent; these are statements
# about that norm. For each system, from the fundamental matrix Φ(t) along one period (M = Φ(T)):
#   multipliers         eigenvalues of M (one equals 1: the phase direction)
#   κV                  eigenvector condition number of the scaled monodromy
#   full(n)             worst-case (over start phase s) growth of a perturbation relative to the UNPERTURBED
#                       trajectory after n cycles, ‖D⁻¹Φ(s+nT, s)D‖ — includes lasting phase shifts
#   trans(n)            worst-case growth of the distance FROM THE CYCLE (phase direction projected out)
#                       after n cycles; eigenvalues predict ρⁿ (ρ = largest transverse |multiplier|)
#   within(τ)           worst-case transverse amplification τ after a perturbation, τ ∈ [0, 2T]
#                       ("locally unstable but orbitally stable")

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using OrdinaryDiffEq, LinearAlgebra, Statistics, Printf, Serialization, ForwardDiff

const NGRID = 220                        # samples per period

# ── systems ──────────────────────────────────────────────────────────────────
vdp!(du, u, μ, t) = (du[1] = u[2]; du[2] = μ * (1 - u[1]^2) * u[2] - u[1])
function sl!(du, u, λ, t)
    r2 = u[1]^2 + u[2]^2
    du[1] = λ * u[1] - u[2] - r2 * u[1]; du[2] = u[1] + λ * u[2] - r2 * u[2]
end
const EPS_F, K_CPL = 0.08, 0.02           # floquet_maps_benchmark.jl
function cfhn!(du, u, p, t)
    v1, w1, v2, w2 = u
    du[1] = v1 - v1^3/3 - w1 + 0.6 + K_CPL * (v2 - v1); du[2] = EPS_F * (v1 + 0.7 - 0.8w1)
    du[3] = v2 - v2^3/3 - w2 + 0.6 + K_CPL * (v1 - v2); du[4] = EPS_F * (v2 + 0.7 - 0.8w2)
end
field(f!, p) = u -> (du = similar(u); f!(du, u, p, 0.0); du)

const OPT = (; abstol = 1e-11, reltol = 1e-11)

"Settle onto the cycle; period from upward crossings of coordinate 1 through its mean."
function find_cycle(f!, p, u0; tsettle = 400.0)
    u = solve(ODEProblem(f!, u0, (0.0, tsettle), p), Vern9(); OPT...).u[end]
    sol = solve(ODEProblem(f!, u, (0.0, 200.0), p), Vern9(); OPT..., saveat = 0.002)
    x = sol[1, :]; m = mean(x)
    c = [sol.t[i-1] + (m - x[i-1]) / (x[i] - x[i-1]) * (sol.t[i] - sol.t[i-1]) for i in 2:length(x) if x[i-1] < m <= x[i]]
    T = mean(diff(c[2:end]))
    ustar = solve(ODEProblem(f!, u, (0.0, c[end-1]), p), Vern9(); OPT...).u[end]
    return T, ustar
end

"Trajectory U (NGRID+1 × d) and fundamental matrices Φ(t) over one period."
function fundamental(f!, p, ustar, T)
    d = length(ustar); F = field(f!, p)
    function var!(dz, z, q, t)
        u = z[1:d]; Φ = reshape(z[d+1:end], d, d)
        dz[1:d] .= F(u); dz[d+1:end] .= vec(ForwardDiff.jacobian(F, u) * Φ)
    end
    sol = solve(ODEProblem(var!, vcat(ustar, vec(Matrix(1.0I, d, d))), (0.0, T)), Vern9(); OPT...,
                saveat = range(0, T; length = NGRID + 1))
    U = reduce(hcat, [z[1:d] for z in sol.u])'
    Φs = [reshape(z[d+1:end], d, d) for z in sol.u]
    return U, Φs
end

"trans(n)/ρⁿ, only where ρⁿ is above roundoff (for 2-D systems it is ≡ 1: the transverse map is scalar)."
trans_ratio(trans, ρ) = [ρ^n > 1e-8 ? trans[n+1] / ρ^n : NaN for n in 0:length(trans)-1] |> v -> filter(isfinite, v)

"Orthonormal basis of the complement of v."
complement(v) = (Q = nullspace(reshape(v / norm(v), 1, :)); Q)

"Forward propagators Φ(s+τ, s), τ ∈ [0, 2T] on the period grid, integrated from phase index i (no inverses:
Φ(t)Φ(s)⁻¹ is numerically useless when the cycle contracts strongly, e.g. coupled FHN, |μ| ~ 1e-17)."
function propagators_from(f!, p, u_s, T)
    d = length(u_s); F = field(f!, p)
    function var!(dz, z, q, t)
        u = z[1:d]; Φ = reshape(z[d+1:end], d, d)
        dz[1:d] .= F(u); dz[d+1:end] .= vec(ForwardDiff.jacobian(F, u) * Φ)
    end
    sol = solve(ODEProblem(var!, vcat(u_s, vec(Matrix(1.0I, d, d))), (0.0, 2T)), Vern9(); OPT...,
                saveat = range(0, 2T; length = 2NGRID + 1))
    [reshape(z[d+1:end], d, d) for z in sol.u]
end

function analyse(name, f!, p, u0; nmax = 8, phase_step = 2)
    T, ustar = find_cycle(f!, p, u0)
    U, Φs = fundamental(f!, p, ustar, T)
    M = Φs[end]; F = field(f!, p)
    D = Diagonal(vec(std(U[1:end-1, :], dims = 1)))
    ev = eigvals(M); iu = argmin(abs.(ev .- 1))
    trans_ev = ev[setdiff(eachindex(ev), iu)]; ρ = maximum(abs, trans_ev)
    κV = cond(eigvecs(D \ M * D))
    nph = size(U, 1) - 1
    scaled_basis(u) = (fsc = D \ F(u); (I - fsc * fsc' / dot(fsc, fsc), complement(fsc)))
    PQ = [scaled_basis(U[k, :]) for k in 1:nph]          # (Π, Q) at each grid phase
    full = zeros(nmax + 1); trans = zeros(nmax + 1); within = zeros(2nph + 1)
    curves = Dict{Int, Vector{Float64}}(); worst_val = 0.0; worst_i = 1
    for i in 1:phase_step:nph
        P = propagators_from(f!, p, U[i, :], T)
        Ms = D \ P[nph+1] * D                            # one-period map from phase i, scaled
        Πi, Qi = PQ[i]
        for n in 0:nmax
            Mn = Ms^n
            full[n+1] = max(full[n+1], opnorm(Mn))
            trans[n+1] = max(trans[n+1], opnorm(Qi' * Πi * Mn * Qi))
        end
        c = map(0:2nph) do m
            Πj, Qj = PQ[(i - 1 + m) % nph + 1]
            opnorm(Qj' * Πj * (D \ P[m+1] * D) * Qi)
        end
        within .= max.(within, c)
        maximum(c) > worst_val && (worst_val = maximum(c); worst_i = i)
        curves[i] = c
    end
    tr = [ρ^n > 1e-8 ? trans[n+1] / ρ^n : NaN for n in 0:nmax]
    @printf("%-14s T=%6.3f  |transverse μ| = %s  κV=%6.2f  max full=%5.2f  max trans/ρⁿ=%5.2f  max within=%6.2f\n",
            name, T, join([@sprintf("%.3g", abs(x)) for x in sort(trans_ev; by = abs, rev = true)], ", "), κV,
            maximum(full), maximum(filter(isfinite, tr)), maximum(within))
    (; name, p, T, ustar, U, M, D, multipliers = ev, trans_ev, ρ, κV, full, trans, ρn = ρ .^ (0:nmax),
       trans_ratio = tr, within, curve = curves[worst_i], worst_phase = (worst_i - 1) / nph, τ = (0:2nph) ./ nph)
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = Dict{String, Any}()
    for μ in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)     # beyond μ ≈ 2, Φ(t) is numerically singular
        out["vdp_$μ"] = analyse("VdP μ=$μ", vdp!, μ, [2.0, 0.0])
    end
    for λ in (0.05, 0.1, 0.2)
        out["sl_$λ"] = analyse("SL λ=$λ", sl!, λ, [1.0, 0.0])
    end
    out["cfhn"] = analyse("coupled FHN", cfhn!, nothing, [-1.0, -0.3, -0.6, -0.25])
    mkpath(joinpath(dirname(@__DIR__), "cache", "truth"))
    serialize(joinpath(dirname(@__DIR__), "cache", "truth", "systems_truth.jls"), out)
end
