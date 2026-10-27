#!/usr/bin/env julia
# Extract a Floquet multiplier from an arbitrary vector field — the true one or
# one learned by a neural ODE — exactly as a practitioner would:
#
#   1. integrate the field long enough to land on whatever attractor it has
#   2. decide whether a sustained oscillation exists at all
#   3. build a Poincare section, find the fixed point of the return map
#   4. finite-difference the return map to get the monodromy, and take the
#      non-trivial eigenvalue
#
# Step 2 matters as much as step 4: a learned field that collapses to a point or
# runs away has no multiplier to report, and that outcome has to be counted, not
# silently dropped.
#
# Run directly to validate against a system whose multiplier is known.

include("common.jl")
using Printf

"""
    field_attractor(vf, u0; ...)

Integrate `vf` from `u0` past its transient and classify what it settles on.
Returns (kind, t, X) with kind ∈ (:cycle, :fixed_point, :divergent, :failed).
"""
function field_attractor(vf, u0; T_trans=200.0, T_obs=200.0, dt=0.02,
                         amp_tol=1e-3, blow_tol=1e3)
    prob = ODEProblem((u, p, t) -> vf(u), collect(float.(u0)), (0.0, T_trans + T_obs))
    sol = solve(prob, Tsit5(); saveat=dt, abstol=1e-9, reltol=1e-9, maxiters=10^7)
    sol.retcode == ReturnCode.Success || return (:failed, Float64[], zeros(0, 0))
    X = permutedims(Array(sol)); t = collect(sol.t)
    i0 = findfirst(>=(T_trans), t)
    i0 === nothing && return (:failed, t, X)
    Xo = X[i0:end, :]
    maximum(abs, Xo) > blow_tol && return (:divergent, t, X)
    amp = maximum(std(Xo, dims=1))
    amp < amp_tol && return (:fixed_point, t, X)
    return (:cycle, t[i0:end] .- t[i0], Xo)
end

"""
    cycle_period(t, X; chan=1)

Period from interpolated up-crossings of the mid-level of one channel.
"""
function cycle_period(t, X; chan=1)
    x = X[:, chan]
    x̄ = (maximum(x) + minimum(x)) / 2
    dt = t[2] - t[1]
    cr = Float64[]
    for i in 2:length(x)
        if x[i-1] < x̄ && x[i] >= x̄
            w = (x̄ - x[i-1]) / (x[i] - x[i-1])
            push!(cr, t[i-1] + w * dt)
        end
    end
    length(cr) < 3 && return NaN
    return mean(diff(cr))
end

"""
    monodromy_multiplier(vf, u_on_cycle, P; ε=1e-4)

Finite-difference monodromy of `vf` about a point on its own cycle: perturb each
coordinate, integrate one period, and read off how the perturbation maps. The
phase direction gives an eigenvalue near 1; the reported multiplier is the
largest of the remaining ones.
"""
function monodromy_multiplier(vf, u_on_cycle, P; ε=1e-4)
    n = length(u_on_cycle)
    flow(u) = begin
        s = solve(ODEProblem((x, p, t) -> vf(x), collect(float.(u)), (0.0, P)),
                  Tsit5(); saveat=[P], abstol=1e-11, reltol=1e-11, maxiters=10^7)
        s.retcode == ReturnCode.Success ? s[end] : fill(NaN, n)
    end
    base = flow(u_on_cycle)
    all(isfinite, base) || return (NaN, fill(NaN, n))
    M = zeros(n, n)
    for j in 1:n
        up = copy(collect(float.(u_on_cycle))); up[j] += ε
        um = copy(collect(float.(u_on_cycle))); um[j] -= ε
        fp, fm = flow(up), flow(um)
        (all(isfinite, fp) && all(isfinite, fm)) || return (NaN, fill(NaN, n))
        M[:, j] = (fp .- fm) ./ (2ε)
    end
    ev = eigvals(M)
    mags = abs.(ev)
    # drop the eigenvalue closest to 1 (the phase direction), report the next
    iphase = argmin(abs.(mags .- 1))
    rest = [mags[i] for i in eachindex(mags) if i != iphase]
    return (isempty(rest) ? NaN : maximum(rest), mags)
end

"""
    floquet_of_field(vf, u0; chan=1)

Full pipeline. Returns a NamedTuple with the classification, period and
multiplier, so failures are reportable rather than silently dropped.
"""
function floquet_of_field(vf, u0; chan=1)
    kind, t, X = field_attractor(vf, u0)
    kind == :cycle || return (; kind, period=NaN, μ=NaN, mags=Float64[])
    P = cycle_period(t, X; chan=chan)
    isfinite(P) || return (; kind=:no_period, period=NaN, μ=NaN, mags=Float64[])
    μ, mags = monodromy_multiplier(vf, X[1, :], P)
    return (; kind, period=P, μ, mags)
end

# ═════════════════════════════════════════════════════════════════════════════
# Validation: the extractor must recover a multiplier we already know
# ═════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    println("="^72)
    println("VALIDATING the extractor on fields whose multiplier is known")
    println("="^72)

    # weakly nonlinear Van der Pol: |mu| ~ exp(-2*pi*mu_vdp), tunable
    for target in [0.9, 0.6, 0.4]
        μv = -log(target) / (2π)
        vf = u -> [u[2], μv * (1 - u[1]^2) * u[2] - u[1]]
        r = floquet_of_field(vf, [2.0, 0.0])
        # independent check with the existing variational-equation routine
        f!(du, u, p, t) = (du[1] = u[2]; du[2] = μv * (1 - u[1]^2) * u[2] - u[1])
        kind, t, X = field_attractor(vf, [2.0, 0.0])
        P = cycle_period(t, X)
        ref = minimum(abs.(numerical_floquet(f!, X[1, :], P).values))
        @printf("VdP target %.2f | %-11s P=%7.4f | extractor %.4f | variational %.4f\n",
                target, String(r.kind), r.period, r.μ, ref)
    end

    # a field with NO cycle: must be reported as such, not given a number
    sink = u -> [-0.5u[1] - 2π*u[2], 2π*u[1] - 0.5u[2]]
    rs = floquet_of_field(sink, [1.0, 0.0])
    @printf("stable spiral   | %-11s | mu = %s  (correct: no cycle to report)\n",
            String(rs.kind), isnan(rs.μ) ? "n/a" : @sprintf("%.4f", rs.μ))

    blow = u -> [0.3u[1] - 2π*u[2], 2π*u[1] + 0.3u[2]]
    rb = floquet_of_field(blow, [1.0, 0.0])
    @printf("unstable spiral | %-11s | mu = %s  (correct: no cycle to report)\n",
            String(rb.kind), isnan(rb.μ) ? "n/a" : @sprintf("%.4f", rb.μ))
    println("="^72)
end
