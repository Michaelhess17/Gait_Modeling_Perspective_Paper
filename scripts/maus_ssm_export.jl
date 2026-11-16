#!/usr/bin/env julia
# Exports the running trials' three representations for maus_ssm.py (SSMLearnPy): the deviation (clean, and with
# measurement noise η = 0.5 and 1 as maus_learning_all.jl), and the whole signal in real time and stride-normalised
# time with its deviation and stride boundaries (maus_ladder.jl's rung_data). Same 22 trials, same order.
# Usage: julia --project=. scripts/maus_ssm_export.jl   → cache/maus/ssm_input/trial_NN.npz
empty!(ARGS)
include("maus_ladder.jl")

if abspath(PROGRAM_FILE) == (@__FILE__)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    SS = [pair_strides(t.X) for t in A.trials]; keep = qc_run.(SS) .< 5
    SS = SS[keep]; T = A.trials[keep]
    dir = joinpath(ROOT, "cache", "maus", "ssm_input"); mkpath(dir)
    for (i, (S, t)) in enumerate(zip(SS, T))
        D, _ = deviations(S, 0.0); _, D05 = deviations(S, 0.5); _, D10 = deviations(S, 1.0); R = rung_data(t.X)
        npzwrite(joinpath(dir, @sprintf("trial_%02d.npz", i)),
                 Dict("D" => D, "Dn05" => D05, "Dn10" => D10, "nstr" => size(S, 1),
                      "Yrt" => R.Yrt, "Drt" => R.Drt, "bidx_rt" => R.bidx_rt, "Ysn" => R.Y, "Dsn" => R.D, "bidx_sn" => R.bidx_sn))
    end
    @printf("%d trials → %s\n", length(SS), dir)
end
