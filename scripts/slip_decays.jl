#!/usr/bin/env julia
# SLIP decay data for the off-cycle model comparison (Fig 6A; OUTLINE.md §6 ¶2), in the protocol of
# floquet_decays.jl / offcycle_v2.jl.
#
# Runs in SLIP.jl's own environment (it needs OrdinaryDiffEq v7):
#   julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_decays.jl      (writes cache/truth/slip_decays.jls)
#
# Walker: bipedal SLIP without hip springs, parameterised as in the companion paper (Paper3, 03_slip_locomotor_
# intuition.jl): angle of attack AOA from horizontal, k̃ = k L₀ / (m g), Ē = Ē(k̃). AOA = 73°, k̃ = 21 gives a stable
# walk with a gait-like step-to-step spectral radius (≈ 0.77; Paper3 grid). Truth: Jacobian of the VLO → VLO step
# map in (y, θ) (SLIP.floquet_multipliers_step).
#
# Observed state: centre-of-mass height and velocities (y, ẋ, ẏ), continuous through touchdown and liftoff,
# resampled to SPC = 110 samples per step (one step = one period of the CoM motion), the samples-per-cycle of the
# coupled-FHN benchmark, so the model code is shared unchanged.
#   decays    per data seed: 10 trajectories of 3 steps from VLO states perturbed by (2 mm, 2°) × randn, each
#             starting at a random phase, with small stride kicks (0.2 mm, 0.2°; process noise) at every VLO
#   starts    4 off-cycle test runs, perturbation ±2 mm in y or ±3° in θ, warm-up starting at 4 phases, no kicks
# Perturbation sizes are balanced in the observed coordinates: at fixed energy 1° of θ moves the CoM state about as
# much as 1 mm of y (≈ 0.13 vs 0.27 SD of the cycle), and 3 mm of y is already nonlinear (decay ≈ 0.91 per step
# instead of 0.77). The step map is strongly non-normal in (y, θ): J ≈ [1.31 0.03; −12.4 0.19].
#   oncycle   the same 4 runs from the fixed point (the models' on-cycle reference)

using SLIP, LinearAlgebra, Statistics, Serialization, Printf, Random

const ROOT = dirname(@__DIR__)
const AOA, KTILDE = 73.0, 21.0
const EBAR = -0.054 * log10(KTILDE)^2 + 0.189 * log10(KTILDE) + 0.898
const SPC = 110
const N_TRAJ, TRAJ_CYC = 10, 3
const PERT = (0.002, deg2rad(2.0))
const KICK = (0.0002, deg2rad(0.2))
const W_WARM = 170
const SEEDS = vcat(1:10, 101)
const OPTS = SimOptions(t_max = 5.0, max_steps = 3, dt = 0.0005)

function params()
    p = biped_slip_default_params()
    p.kth = [0.0, 0.0]; p.th0 = [0.0, 0.0]
    p.k = KTILDE .* p.m .* p.g ./ p.L0
    p.th_contact = deg2rad.([90 - AOA, 90 - AOA])
    p
end

"One VLO → next-VLO step from (y, θ): next (y, θ), the raw CoM segment [t y ẋ ẏ] up to that VLO, its duration."
function step_segment(y, θ, stance, p)
    E = EBAR * p.L0[stance] * p.m * p.g
    res = simulate_biped_slip(p, biped_slip_energy_ic(p, stance, y, E, θ), OPTS)
    res.fallen && return nothing
    opp = stance == LEFT ? RIGHT : LEFT
    th = opp == LEFT ? res.thL : res.thR
    idx_td = findfirst(!=(stance == LEFT ? SS_LEFT : SS_RIGHT), res.mode); idx_td === nothing && return nothing
    cross = findfirst(i -> th[i] > 0 && th[i+1] <= 0, idx_td:(length(res.mode) - 1)); cross === nothing && return nothing
    i = idx_td - 1 + cross; f = th[i] / (th[i] - th[i+1])
    Xc = res.X[i, :] .+ f .* (res.X[i+1, :] .- res.X[i, :]); tc = res.T[i] + f * (res.T[i+1] - res.T[i])
    (Xc[2], atan(Xc[4], Xc[3]), hcat(res.T[1:i], res.X[1:i, 2:4]), tc, opp)
end

"Walk n steps from VLO state z (kicks σ .* randn at every later VLO); CoM (y, ẋ, ẏ) on a uniform grid of step dt."
function walk(z, n, dt; σ = (0.0, 0.0), rng = MersenneTwister(0))
    p = params(); stance = RIGHT; t0 = 0.0; C = zeros(0, 4)
    for k in 1:n
        k > 1 && (z = z .+ [σ[1], σ[2]] .* randn(rng, 2))
        r = step_segment(z[1], z[2], stance, p); r === nothing && return nothing
        y1, θ1, seg, tc, stance = r
        seg[:, 1] .+= t0; C = vcat(C, k == 1 ? seg : seg[2:end, :]); t0 += tc; z = [y1, θ1]
    end
    tq = range(0, C[end, 1]; step = dt); out = zeros(length(tq), 3)
    for (j, t) in enumerate(tq)
        i = clamp(searchsortedlast(C[:, 1], t), 1, size(C, 1) - 1); a = (t - C[i, 1]) / max(C[i+1, 1] - C[i, 1], eps())
        out[j, :] = (1 - a) .* C[i, 2:4] .+ a .* C[i+1, 2:4]
    end
    out
end

if abspath(PROGRAM_FILE) == @__FILE__
    p = params()
    gs = gait_stability_step(0.98, 0.0, EBAR, RIGHT, p; opts = OPTS, maxiter = 30)
    zstar = gs.z_star; λ, J = floquet_multipliers_step(zstar, EBAR, RIGHT, p); ρ = maximum(abs, λ)
    Tstep = step_segment(zstar[1], zstar[2], RIGHT, p)[4]; dt = Tstep / SPC
    @printf("AOA %.0f°, k̃ %.0f, Ē %.4f: z* = (%.5f, %.5f), λ = %s, ρ = %.4f, step %.4f s\n", AOA, KTILDE, EBAR, zstar...,
            string(round.(λ, digits = 4)), ρ, Tstep)
    cycle = walk(zstar, 2, dt)[1:SPC, :]
    decays = Dict{Int, Vector{Matrix{Float64}}}()
    for s in SEEDS
        rng = MersenneTwister(s); trajs = Matrix{Float64}[]
        while length(trajs) < N_TRAJ
            X = walk(zstar .+ [PERT[1], PERT[2]] .* randn(rng, 2), TRAJ_CYC + 2, dt; σ = KICK, rng)
            X === nothing && (println("seed $s: a decay fell, redrawn"); continue)
            o = rand(rng, 0:SPC-1); push!(trajs, X[o+1:o+TRAJ_CYC*SPC, :])
        end
        decays[s] = trajs
    end
    nrun = W_WARM + 6SPC + 2; offs = (0, 27, 54, 82)
    dirs = ([0.002, 0.0], [0.0, deg2rad(3.0)], [-0.002, 0.0], [0.0, -deg2rad(3.0)])
    starts = [walk(zstar .+ d, 10, dt)[o+1:o+nrun, :] for (d, o) in zip(dirs, offs)]
    oncycle = [walk(zstar, 10, dt)[o+1:o+nrun, :] for o in offs]
    # truth check: RMS distance to the cycle per step after warm-up
    dist(X) = [minimum(sum(abs2, (cycle .- X[i:i, :]) ./ std(cycle, dims = 1), dims = 2))^0.5 for i in axes(X, 1)]
    for X in starts
        d = dist(X)[W_WARM+1:end]; e = [sqrt(mean(abs2, d[k*SPC+1:(k+1)*SPC])) for k in 0:4]
        @printf("  start: initial RMS deviation %.2f SD, per-step ratios %s, rate %.3f\n", e[1], join([@sprintf("%.2f", e[k+1] / e[k]) for k in 1:4], " "), (e[5] / e[1])^0.25)
    end
    mkpath(joinpath(ROOT, "cache", "truth"))
    serialize(joinpath(ROOT, "cache", "truth", "slip_decays.jls"),
              (; aoa = AOA, ktilde = KTILDE, Ebar = EBAR, zstar, J, λ = ComplexF64.(λ), ρ, Tstep, dt, spc = SPC, cycle,
                 decays, starts, oncycle))
    println("saved: ", length(decays), " data seeds × ", N_TRAJ, " decays of ", TRAJ_CYC * SPC, " samples")
end
