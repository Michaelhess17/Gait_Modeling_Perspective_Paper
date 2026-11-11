#!/usr/bin/env julia
# A cohort of coupled FitzHugh–Nagumo oscillators with small individual differences (smooth regime of
# cfhn_smooth_system.jl), for individual against group fits.
#
# Ten "individuals": ε ∈ [0.23, 0.27], drive ∈ [0.58, 0.62], and coupling set so that the coordination multiplier
# (the antisymmetric Floquet multiplier) is ∈ [0.60, 0.78] per cycle (reference system: ε 0.25, drive 0.6, 0.69).
# Realistic regime: white process noise on both fast variables (σ = 0.05, as the benchmark's realistic regime),
# measurement noise added at fitting time. Eight trials per individual of 15 cycles (as the human recordings), 50
# samples per cycle of the individual's own period, each after a 10-cycle burn-in from a random phase. Trials 1–6
# train, 7–8 test. For every test start (cycles 3, 5, …, 11 at a random phase) the noise-free continuation of the
# true state for 4 cycles is stored: the ceiling of predictable_response.jl.
# Usage: julia -t 8 --project=. scripts/cfhn_family_data.jl   → cache/cfhn_family/data.jls

include("cfhn_smooth_system.jl")
using DifferentialEquations, Printf

const NIND, NTRIAL, NCYC, BURN, SPC, NCONT = 10, 8, 15, 10, 50, 4
const SIGP = 0.05

function trial(p, c, rng)
    T = c.T; dt = T / SPC
    u0 = c.C[rand(rng, 1:size(c.C, 1)-1), :]
    f!(du, u, p_, t) = smooth_fhn!(du, u, p, t)
    g!(du, u, p_, t) = (du .= 0; du[1] = SIGP; du[3] = SIGP)
    sol = solve(SDEProblem(f!, g!, u0, (0.0, (BURN + NCYC) * T); seed = rand(rng, UInt32)), EM(); dt = dt / 20,
                saveat = dt, save_start = true)
    X = permutedims(Array(sol)); i0 = BURN * SPC + 1
    X[i0:i0+NCYC*SPC-1, :]
end

rng = MersenneTwister(2027)
cohort = Vector{Any}(undef, NIND)
draws = [(ε = 0.23 + 0.04rand(rng), drive = 0.58 + 0.04rand(rng), target = 0.60 + 0.18rand(rng), seed = rand(rng, UInt32))
         for _ in 1:NIND]
Threads.@threads for i in 1:NIND
    dr = draws[i]; rl = MersenneTwister(dr.seed)
    c = smooth_cycle(dr.ε; drive = dr.drive)
    k = smooth_match_coupling(c; target = dr.target)
    p = (dr.ε, k, dr.drive)
    trials = [trial(p, c, rl) for _ in 1:NTRIAL]
    tests = Any[]
    for j in NTRIAL-1:NTRIAL, cyc in 3:2:11
        s = cyc * SPC + rand(rl, 1:SPC)                                  # start sample (state observed up to s)
        u = trials[j][s, :]
        sol = solve(ODEProblem(smooth_fhn!, u, (0.0, NCONT * c.T), p), Vern9(); SMOOTH_OPT..., saveat = c.T / SPC)
        C = permutedims(Array(sol))[2:end, :]                            # samples s+1, s+2, …
        push!(tests, (; trial = j, s, C))
    end
    cohort[i] = (; id = i, dr.ε, dr.drive, k, coord = dr.target, T = c.T, cycle = c.C, trials, tests)
end
for c in cohort
    @printf("individual %2d: ε %.3f drive %.3f coupling %.4f | coordination multiplier %.2f | period %.2f | %d tests\n",
            c.id, c.ε, c.drive, c.k, c.coord, c.T, length(c.tests))
end
mkpath(joinpath(dirname(@__DIR__), "cache", "cfhn_family"))
serialize(joinpath(dirname(@__DIR__), "cache", "cfhn_family", "data.jls"), cohort)
println("→ cache/cfhn_family/data.jls")
