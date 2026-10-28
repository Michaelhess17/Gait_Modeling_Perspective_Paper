#!/usr/bin/env julia
# True SLIP off-cycle runs over the models' 30-step window, for slip_rotation.jl (Fig 6C).
#
# Same 4 starts and start phases as slip_decays.jl (±2 mm in y, ±3° in θ), warm-up W_WARM then 30 steps, at the
# test perturbation (×1, what the models were warmed up on) and at ×0.25 (linear regime: validates the section
# protocol against the exact multiplier).
#
#   julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_rotation_truth.jl   (writes cache/truth/slip_long_starts.jls)

include("slip_decays.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    D = deserialize(joinpath(ROOT, "cache", "truth", "slip_decays.jls"))
    dirs = ([0.002, 0.0], [0.0, deg2rad(3.0)], [-0.002, 0.0], [0.0, -deg2rad(3.0)]); offs = (0, 27, 54, 82)
    nrun = W_WARM + 30SPC
    runs = Dict(s => [walk(D.zstar .+ s .* d, 33, D.dt)[o+1:o+nrun, :] for (d, o) in zip(dirs, offs)] for s in (1.0, 0.25))
    serialize(joinpath(ROOT, "cache", "truth", "slip_long_starts.jls"), (; runs, w_warm = W_WARM, offs))
    println("saved 2 × ", length(runs[1.0]), " runs of ", nrun, " samples")
end
