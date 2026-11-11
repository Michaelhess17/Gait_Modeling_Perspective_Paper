#!/usr/bin/env julia
# A cohort of SLIP walkers with small individual differences, for individual against group fits.
#
# Runs in SLIP.jl's own environment:
#   julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_family_data.jl
#
# Ten "individuals": speed-feedback gain κ ∈ [0.06, 0.14], dimensionless leg stiffness k̃ ∈ [19, 21] and angle of
# attack ∈ [68.5°, 69.5°] (drawn once, seeded), each with a stable gait. Eight trials per individual of 36 steps
# (about 15 s, as the human recordings), each after a 50-step burn-in so that trials start in the stationary regime;
# kicks at every VLO as in slip_truth.jl (σ = 0.0005 m in height, 0.001 rad in velocity angle). Every step is
# resampled onto NPHASE points (stride-normalised time). Trials 1–6 train, 7–8 test. For every test start (steps
# 9, 13, …, 29 at a random phase) the noise-free continuation from that step's kicked VLO state is stored (8 steps):
# the ceiling of predictable_response.jl. Per individual: fixed point, step Jacobian, multipliers.
# → cache/slip_family/data.jls

include("slip_truth.jl")                 # pattern_params, step_segment, NPHASE, EBAR

const NIND, NTRIAL, NSTEP, BURN, NCONT = 10, 8, 36, 50, 8
const KICK = [0.0005, 0.001]

params(κ, kt, aoa) = pattern_params(; kappa = κ, ktilde = kt, aoa)
function truth_p(p)
    r = gait_stability_step(0.98, 0.0, EBAR, RIGHT, p; maxiter = 30)
    λ, J = floquet_multipliers_step(r.z_star, EBAR, RIGHT, p)
    (; zstar = r.z_star, J, λ, ρ = maximum(abs, λ))
end
function resample_step(seg)
    tk = seg[:, 1]; τ = range(tk[1], tk[end]; length = NPHASE); out = zeros(NPHASE, 3)
    for d in 1:3, (j, τj) in enumerate(τ)
        m = clamp(searchsortedlast(tk, τj), 1, length(tk) - 1)
        f = (τj - tk[m]) / max(tk[m+1] - tk[m], eps())
        out[j, d] = (1 - f) * seg[m, d+1] + f * seg[m+1, d+1]
    end
    out
end
"Kicked walk of BURN + NSTEP steps; returns the last NSTEP steps (S: NSTEP × NPHASE × 3), their VLO states and kicks."
function walk(p, zstar, rng)
    z = copy(zstar); stance = RIGHT; S = zeros(BURN + NSTEP, NPHASE, 3); Z = zeros(BURN + NSTEP, 2); K = zeros(BURN + NSTEP, 2)
    for k in 1:BURN+NSTEP
        kick = KICK .* randn(rng, 2); Z[k, :] = z; K[k, :] = kick
        r = step_segment(z[1] + kick[1], z[2] + kick[2], stance, p); r === nothing && return nothing
        y1, θ1, seg, _, stance = r
        S[k, :, :] = resample_step(seg); z = [y1, θ1]
    end
    (; S = S[BURN+1:end, :, :], Z = Z[BURN+1:end, :], K = K[BURN+1:end, :], stance0 = isodd(BURN) ? LEFT : RIGHT)
end
function continuation(p, z, stance)
    out = zeros(NCONT, NPHASE, 3)
    for s in 1:NCONT
        r = step_segment(z[1], z[2], stance, p); r === nothing && return nothing
        y1, θ1, seg, _, stance = r
        out[s, :, :] = resample_step(seg); z = [y1, θ1]
    end
    out
end

rng = MersenneTwister(2026)
cohort = Any[]
while length(cohort) < NIND
    κ = 0.06 + 0.08rand(rng); kt = 19 + 2rand(rng); aoa = 68.5 + rand(rng)
    p = params(κ, kt, aoa)
    tr = try truth_p(p) catch; nothing end
    (tr === nothing || !(tr.ρ < 0.98)) && continue
    trials = Any[]
    for j in 1:NTRIAL
        w = walk(p, tr.zstar, MersenneTwister(100 * length(cohort) + j)); w === nothing && break
        push!(trials, w)
    end
    length(trials) < NTRIAL && continue
    # test starts and their noise-free continuations (trials 7–8)
    tests = Any[]
    for j in NTRIAL-1:NTRIAL, k in 9:4:29
        w = trials[j]; ph = rand(rng, 1:NPHASE)
        stance = isodd(k) ? w.stance0 : (w.stance0 == RIGHT ? LEFT : RIGHT)
        C = continuation(p, w.Z[k, :] .+ w.K[k, :], stance)
        C === nothing && continue
        @assert maximum(abs.(C[1, :, :] .- w.S[k, :, :])) < 1e-9 "continuation does not reproduce step $k"
        push!(tests, (; trial = j, k, j_phase = ph, C))
    end
    push!(cohort, (; id = length(cohort) + 1, κ, ktilde = kt, aoa, tr..., trials, tests))
    @printf("individual %2d: κ %.3f k̃ %.2f aoa %.2f | ρ %.3f per step, λ %s | %d test starts\n", length(cohort), κ, kt, aoa,
            tr.ρ, string(round.(tr.λ, digits = 3)), length(tests))
end
mkpath(joinpath(ROOT, "cache", "slip_family"))
serialize(joinpath(ROOT, "cache", "slip_family", "data.jls"), cohort)
println("→ cache/slip_family/data.jls")
