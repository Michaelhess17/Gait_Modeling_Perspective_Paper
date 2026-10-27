#!/usr/bin/env julia
# SLIP ground truth and noisy walking data for Figs 3–4 (OUTLINE.md §4).
#
# Runs in SLIP.jl's own environment (it needs OrdinaryDiffEq v7):
#   julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_truth.jl
#
# Bipedal SLIP walker without hip springs, Rummel et al. 2010 pattern A (k̃ = 20, Ē = 1.05, angle of attack
# 69° from horizontal). Optional feedback of forward speed onto the landing angle (kappa: a simple closed
# loop). Truth: fixed point and Jacobian of the single-step VLO → VLO map in (y, θ) (θ = velocity angle), by
# SLIP.floquet_multipliers_step. Data: "kicked" walks: at every VLO the reduced state (y, θ) is perturbed
# (energy kept fixed), then one step is integrated noise-free to the next VLO. This is a stochastic return map
# with exactly known Jacobian (SLIP.jl's own eta_contact noise is redrawn at every event-function evaluation,
# which makes touchdown erratic; the walker then falls within ~20 steps). Saved as plain arrays (continuous
# CoM state and per-step VLO states) in cache/truth/slip_truth.jls.

using SLIP, LinearAlgebra, Statistics, Serialization, Printf, Random

const ROOT = dirname(@__DIR__)
const EBAR = 1.05

function pattern_params(; kappa = 0.0, ktilde = 20.0, aoa = 69.0, eta = 0.0)
    p = biped_slip_default_params()
    p.kth = [0.0, 0.0]; p.th0 = [0.0, 0.0]
    p.k = ktilde .* p.m .* p.g ./ p.L0
    p.th_contact = deg2rad.([90 - aoa, 90 - aoa])
    p.kappa = kappa; p.eta_contact = eta
    return p
end

function truth(kappa)
    p = pattern_params(; kappa)
    r = gait_stability_step(0.98, 0.0, EBAR, RIGHT, p; maxiter = 30)
    λ, J = floquet_multipliers_step(r.z_star, EBAR, RIGHT, p)
    (; kappa, zstar = r.z_star, J, λ, ρ = maximum(abs, λ))
end

const STEP_OPTS = SimOptions(t_max = 5.0, max_steps = 3, dt = 0.0005)   # as SLIP.poincare_step

"One VLO → opposite-leg VLO step from (y, θ): next (y, θ), the continuous segment [t y ẋ ẏ] up to that VLO."
function step_segment(y, θ, stance, p)
    E = EBAR * p.L0[stance] * p.m * p.g
    res = simulate_biped_slip(p, biped_slip_energy_ic(p, stance, y, E, θ), STEP_OPTS)
    res.fallen && return nothing
    opp = stance == LEFT ? RIGHT : LEFT
    th = opp == LEFT ? res.thL : res.thR
    idx_td = findfirst(!=(stance == LEFT ? SS_LEFT : SS_RIGHT), res.mode); idx_td === nothing && return nothing
    cross = findfirst(i -> th[i] > 0 && th[i+1] <= 0, idx_td:(length(res.mode) - 1)); cross === nothing && return nothing
    i = idx_td - 1 + cross; f = th[i] / (th[i] - th[i+1])
    Xc = res.X[i, :] .+ f .* (res.X[i+1, :] .- res.X[i, :]); tc = res.T[i] + f * (res.T[i+1] - res.T[i])
    seg = hcat(res.T[1:i], res.X[1:i, 2:4])
    return (Xc[2], atan(Xc[4], Xc[3]), seg, tc, opp)
end

"""Kicked walk of n steps from the fixed point: kicks σ .* randn added to (y, θ) at every VLO.
Returns VLO states z (n+1 × 2, before each kick), kicks, and the stitched continuous record [t y ẋ ẏ]."""
function kicked_walk(kappa, σ, n; seed = 1)
    rng = MersenneTwister(seed); t0 = truth(kappa); p = pattern_params(; kappa)
    z = copy(t0.zstar); stance = RIGHT; Z = [z']; K = zeros(0, 2); C = zeros(0, 4); toff = 0.0; starts = Int[]
    for k in 1:n
        kick = σ .* randn(rng, 2); zk = z .+ kick
        r = step_segment(zk[1], zk[2], stance, p)
        r === nothing && return (; kappa, σ, seed, fallen = true, steps = k - 1, Z = reduce(vcat, Z), K, C, starts,
                                   zstar = t0.zstar, J = t0.J)
        y1, θ1, seg, tc, stance = r
        seg[:, 1] .+= toff; push!(starts, size(C, 1) + 1); C = vcat(C, seg); toff += tc
        z = [y1, θ1]; push!(Z, z'); K = vcat(K, kick')
    end
    (; kappa, σ, seed, fallen = false, steps = n, Z = reduce(vcat, Z), K, C, starts, zstar = t0.zstar, J = t0.J)
end

"""Kicked walk stored compactly: VLO states Z, kicks K, and every step resampled onto NPHASE points of its
own duration (phase 0 = VLO), S[k, φ, :] = (y, ẋ, ẏ); plus the first 20 steps of the raw record for plots."""
const NPHASE = 50
function phase_walk(kappa, σ, n; seed = 1)
    w = kicked_walk(kappa, σ, n; seed)
    C = w.C; t = C[:, 1]
    starts = w.starts                                  # first sample of each step (at its VLO)
    S = zeros(length(starts), NPHASE, 3)
    for (k, i0) in enumerate(starts)
        i1 = k < length(starts) ? starts[k+1] - 1 : size(C, 1)
        tk = t[i0:i1]; τ = range(tk[1], tk[end]; length = NPHASE)
        for d in 1:3, (j, τj) in enumerate(τ)
            m = clamp(searchsortedlast(tk, τj), 1, length(tk) - 1)
            f = (τj - tk[m]) / max(tk[m+1] - tk[m], eps())
            S[k, j, d] = (1 - f) * C[i0+m-1, d+1] + f * C[i0+m, d+1]
        end
    end
    nraw = length(starts) > 20 ? starts[21] - 1 : size(C, 1)
    (; kappa, σ, seed, fallen = w.fallen, steps = w.steps, Z = w.Z, K = w.K, S, raw = C[1:nraw, :],
       zstar = w.zstar, J = w.J)
end

if abspath(PROGRAM_FILE) == (@__FILE__) && length(ARGS) >= 2
    # data generation: julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_truth.jl <kappa> <seed> [nsteps]
    κ = parse(Float64, ARGS[1]); seed = parse(Int, ARGS[2]); n = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2000
    w = phase_walk(κ, [0.0005, 0.001], n; seed)
    mkpath(joinpath(ROOT, "cache", "truth", "slip_walks"))
    serialize(joinpath(ROOT, "cache", "truth", "slip_walks", "walk_k$(κ)_s$(seed).jls"), w)
    @printf("kappa %.2f seed %d: %d steps, fallen=%s, S %s\n", κ, seed, w.steps, w.fallen, size(w.S))
elseif abspath(PROGRAM_FILE) == (@__FILE__)
    out = Dict{String, Any}()
    for κ in (0.0, 0.05, 0.1, 0.2)
        try
            t = truth(κ); out["truth_$κ"] = t
            @printf("kappa=%.2f  z*=(%.4f, %.5f)  λ=%s  ρ=%.3f\n", κ, t.zstar..., string(round.(t.λ, digits = 3)), t.ρ)
        catch e
            println("kappa=$κ: no stable fixed point found (", sprint(showerror, e)[1:min(end, 120)], ")")
        end
    end
    # test walks: survival, spread and linearity for a few kick sizes (y in m, θ in rad)
    for σ in ([0.0002, 0.0005], [0.0005, 0.001], [0.001, 0.002])
        w = kicked_walk(0.0, σ, 400)            # diagnostics only (not saved: the raw records are large)
        nk = size(w.K, 1)
        dev = [norm(w.Z[k, :] .+ w.K[k, :] .- w.zstar) for k in 1:nk]
        err = [norm(w.zstar .+ w.J * (w.Z[k, :] .+ w.K[k, :] .- w.zstar) .- w.Z[k+1, :]) for k in 1:nk]
        @printf("kicked walk σ=(%.4f m, %.4f rad): %d steps, fallen=%s, SD y %.5f θ %.5f, median |linear error|/|deviation| %.3f\n",
                σ..., w.steps, w.fallen, std(w.Z[:, 1]), std(w.Z[:, 2]), median(err ./ dev))
    end
    mkpath(joinpath(ROOT, "cache", "truth"))
    serialize(joinpath(ROOT, "cache", "truth", "slip_truth.jls"), out)
end
