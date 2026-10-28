#!/usr/bin/env julia
# Stationary, noise-driven SLIP walks for validating the lattice fit on gait-like data (scripts/gait_lattice.jl).
#
# Same walker as slip_decays.jl (AOA 73°, k̃ 21; step multiplier 0.769 ∠ 13.6°), started on the cycle and driven
# only by stride kicks at every VLO (σ_y, σ_θ) = KICK_W. Each step, VLO to VLO, is resampled onto 100 points of
# its own duration, as the human strides are (fig5_gait_data.jl: strides): re-anchoring the phase at every VLO is a
# Poincaré section, so the per-step multiplier is unchanged.
#
#   julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_kicked_walks.jl   (writes cache/truth/slip_kicked_walks.jls)

include("slip_decays.jl")
const KICK_W = (0.001, deg2rad(1.0))
const N_WALK, N_STEP, NPT = 20, 15, 100

"Per-step resampled CoM (n × NPT × 3) of a kicked walk from z."
function kicked_walk(z, n; rng)
    p = params(); stance = RIGHT; out = zeros(n, NPT, 3)
    for k in 1:n
        k > 1 && (z = z .+ [KICK_W[1], KICK_W[2]] .* randn(rng, 2))
        r = step_segment(z[1], z[2], stance, p); r === nothing && return nothing
        y1, θ1, seg, tc, stance = r
        for (j, t) in enumerate(range(0, tc; length = NPT + 1)[1:NPT])
            i = clamp(searchsortedlast(seg[:, 1], t), 1, size(seg, 1) - 1); a = (t - seg[i, 1]) / max(seg[i+1, 1] - seg[i, 1], eps())
            out[k, j, :] = (1 - a) .* seg[i, 2:4] .+ a .* seg[i+1, 2:4]
        end
        z = [y1, θ1]
    end
    out
end

if abspath(PROGRAM_FILE) == @__FILE__
    D = deserialize(joinpath(ROOT, "cache", "truth", "slip_decays.jls"))
    rng = MersenneTwister(77); walks = Array{Float64, 3}[]
    while length(walks) < N_WALK
        W = kicked_walk(D.zstar, N_STEP + 5; rng); W === nothing && continue
        push!(walks, W[6:end, :, :])                                     # drop the first 5 steps (settle into noise)
    end
    serialize(joinpath(ROOT, "cache", "truth", "slip_kicked_walks.jls"), (; walks, kick = KICK_W, λ = D.λ))
    println("saved ", length(walks), " walks × ", N_STEP, " steps × ", NPT, " points")
end
