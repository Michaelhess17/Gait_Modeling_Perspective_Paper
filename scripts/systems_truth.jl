#!/usr/bin/env julia
# Ground truth for the Floquet / non-normality figures (OUTLINE.md, Figs 3–4).
#
#   vdp(μ)   Van der Pol  ẋ = y, ẏ = μ(1 − x²)y − x            (2-D, one transverse multiplier)
#   sl(λ)    Stuart–Landau  ż = (λ + i)z − |z|²z                (normal reference)
# SLIP is handled in scripts/slip_truth.jl, run in SLIP.jl's own environment (it needs OrdinaryDiffEq v7).
#
# For each: period / fixed point, monodromy (or step Jacobian), Floquet multipliers, and non-normality in
# OBSERVED coordinates (state scaled by its SD along the cycle, the norm a data-driven model would use):
# eigenvector condition number κ(V), Henrici departure, and the worst-case n-cycle growth max‖Mⁿ‖ vs ρⁿ.
# Non-normality is norm-dependent; these are statements about the observed-coordinate norm only.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using OrdinaryDiffEq, LinearAlgebra, Statistics, Printf, Serialization


# ── smooth oscillators ───────────────────────────────────────────────────────
vdp!(du, u, μ, t) = (du[1] = u[2]; du[2] = μ * (1 - u[1]^2) * u[2] - u[1])
function sl!(du, u, λ, t)
    r2 = u[1]^2 + u[2]^2
    du[1] = λ * u[1] - u[2] - r2 * u[1]; du[2] = u[1] + λ * u[2] - r2 * u[2]
end
jac_vdp(u, μ) = [0.0 1.0; -2μ * u[1] * u[2] - 1 μ * (1 - u[1]^2)]
function jac_sl(u, λ)
    x, y = u; r2 = x^2 + y^2
    [λ - r2 - 2x^2  -1 - 2x * y; 1 - 2x * y  λ - r2 - 2y^2]
end

"Period (upward x-crossings of 0 after transients) and a point on the cycle."
function find_cycle(f!, p, u0; tsettle = 200.0)
    sol = solve(ODEProblem(f!, u0, (0.0, tsettle), p), Vern9(); abstol = 1e-12, reltol = 1e-12)
    u = sol.u[end]
    cr = Float64[]; pts = Vector{Float64}[]           # section y = 0 with x > 0 (either crossing direction)
    rec!(i) = (push!(cr, i.t); push!(pts, copy(i.u)))
    cb = ContinuousCallback((u, t, i) -> u[2], rec!, rec!)
    solve(ODEProblem(f!, u, (0.0, 60.0), p), Vern9(); abstol = 1e-12, reltol = 1e-12, callback = cb)
    # keep crossings with x > 0
    keep = [k for k in eachindex(pts) if pts[k][1] > 0]
    T = mean(diff(cr[keep])); return T, pts[keep[end]]
end

"Monodromy and the fundamental matrix Φ(t) along one period (variational equations)."
function variational!(dz, z, q, t)
    f!, jac, p = q
    u = z[1:2]; Φ = reshape(z[3:6], 2, 2)
    du = similar(u); f!(du, u, p, t)
    dz[1:2] .= du; dz[3:6] .= vec(jac(u, p) * Φ)
end
function monodromy(f!, jac, p, u0, T; nsave = 400)
    z0 = vcat(u0, vec(Matrix(1.0I, 2, 2)))
    sol = solve(ODEProblem(variational!, z0, (0.0, T), (f!, jac, p)), Vern9(); abstol = 1e-12, reltol = 1e-12,
                saveat = range(0, T; length = nsave))
    U = reduce(hcat, [z[1:2] for z in sol.u])'; Φs = [reshape(z[3:6], 2, 2) for z in sol.u]
    return Φs[end], U, Φs, sol.t
end

"Non-normality of M in a scaled norm (D = diag of observation SDs)."
function nonnormality(M, D; nmax = 30)
    Ms = D \ M * D
    F = eigen(Ms); V = F.vectors
    κV = cond(V)
    henrici = sqrt(max(norm(Ms)^2 - sum(abs2, F.values), 0)) / norm(Ms)
    ρ = maximum(abs, F.values)
    growth = [opnorm(Ms^n) for n in 0:nmax]
    (; λ = F.values, κV, henrici, ρ, growth, ρn = ρ .^ (0:nmax))
end

"Transverse multiplier, tangent-removed transverse growth ‖P Φ(t)‖ within a cycle."
function smooth_truth(name, f!, jac, p, u0)
    T, ustar = find_cycle(f!, p, u0)
    M, U, Φs, ts = monodromy(f!, jac, p, ustar, T)
    D = Diagonal(vec(std(U, dims = 1)))
    ev = eigen(M).values
    μF = real(ev[argmin(abs.(ev))])                     # the non-unit (transverse) multiplier
    nn = nonnormality(M, D)
    # within-cycle transverse deviation: start normal to the flow at u*, follow Φ(t)δ, measure the
    # component normal to the local flow direction, in observed coordinates
    du = similar(ustar); f!(du, ustar, p, 0.0)
    n0 = D \ [-du[2], du[1]]; n0 = D * (n0 / norm(n0))   # unit normal in scaled coordinates
    trans = map(eachindex(ts)) do k
        v = Φs[k] * n0; fk = similar(ustar); f!(fk, U[k, :], p, 0.0)
        vs = D \ v; fs = D \ fk; fs ./= norm(fs)
        norm(vs - dot(vs, fs) * fs)
    end
    trans ./= trans[1]
    (; name, p, T, ustar, M, μF, nn, U, ts, trans, D)
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = Dict{String, Any}()
    for μ in (0.2, 0.5, 1.0, 2.0, 4.0)
        t = smooth_truth("vdp", vdp!, jac_vdp, μ, [2.0, 0.0]); out["vdp_$μ"] = t
        @printf("VdP μ=%.1f  T=%.3f  μF=%.4f  κ(V)=%.2f  Henrici=%.2f  max‖Mⁿ‖=%.2f  max within-cycle transverse growth=%.2f\n",
                μ, t.T, t.μF, t.nn.κV, t.nn.henrici, maximum(t.nn.growth), maximum(t.trans))
    end
    for λ in (0.05, 0.1, 0.2)
        t = smooth_truth("sl", sl!, jac_sl, λ, [1.0, 0.0]); out["sl_$λ"] = t
        @printf("SL λ=%.2f  T=%.3f  μF=%.4f  κ(V)=%.2f  Henrici=%.2f  max within-cycle growth=%.2f\n",
                λ, t.T, t.μF, t.nn.κV, t.nn.henrici, maximum(t.trans))
    end
    serialize(joinpath(dirname(@__DIR__), "cache", "truth", "systems_truth.jls"), out)
end
