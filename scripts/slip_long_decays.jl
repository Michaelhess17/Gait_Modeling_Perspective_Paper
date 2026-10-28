#!/usr/bin/env julia
# Long SLIP decays for the limit-cycle representation tests (scripts/floquet_lifted.jl; OUTLINE.md §6 ¶7).
#
# Same walker, perturbations and stride kicks as slip_decays.jl, but each decay lasts LONG_CYC steps instead of 3:
# a harmonic and its Floquet sideband are 13.6°/360° = 0.038 cycles per step apart, so separating them needs
# records long compared with 1/0.038 ≈ 26 steps. Data seeds 1, 2 with stride kicks (which dominate after about
# 10 steps), 11, 12 kick-free.
#
#   julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_long_decays.jl   (writes cache/truth/slip_long_decays.jls)

include("slip_decays.jl")
const LONG_CYC = 30

if abspath(PROGRAM_FILE) == @__FILE__
    D = deserialize(joinpath(ROOT, "cache", "truth", "slip_decays.jls"))
    decays = Dict{Int, Vector{Matrix{Float64}}}()          # 1, 2: with stride kicks; 11, 12: kick-free
    for s in (1, 2, 11, 12)
        rng = MersenneTwister(1000 + s); trajs = Matrix{Float64}[]
        while length(trajs) < N_TRAJ
            X = walk(D.zstar .+ [PERT[1], PERT[2]] .* randn(rng, 2), LONG_CYC + 2, D.dt; σ = s > 10 ? (0.0, 0.0) : KICK, rng)
            X === nothing && continue
            o = rand(rng, 0:SPC-1); push!(trajs, X[o+1:o+LONG_CYC*SPC, :])
        end
        decays[s] = trajs
    end
    serialize(joinpath(ROOT, "cache", "truth", "slip_long_decays.jls"), (; decays, long_cyc = LONG_CYC))
    println("saved ", length(decays), " data seeds × ", N_TRAJ, " decays of ", LONG_CYC * SPC, " samples")
end
