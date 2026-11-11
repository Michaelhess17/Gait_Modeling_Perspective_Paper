#!/usr/bin/env julia
# Paired intervention on the same DMD fit: selected training data, rank, delay,
# ridge, initial history and forecast duration are identical across modes.
# Selection is inherited from the existing train-selected unit-circle fit;
# this is a conditional ablation, not separate retuning of the raw estimator.
include("dmd_tuning.jl")

results = map((91, 231)) do trial
    D = npzread(cache_path("selfdrive/data_gait$(trial)_realistic.npz"))
    R = deserialize(cache_path("selfdrive/dmdtuned_gait$(trial)_realistic.jls"))
    c = R.chosen
    @assert c.variant == "unit"
    Y, P = D["Ytr"], D["P"]
    n = round(Int, 20.0 / 0.02)
    predict(mode) = fit_dmd_variant(Y; τ=c.τ, r=c.r, variant=c.variant, P,
                                  stabilization=mode)(Y, n)
    raw, clipped, unit = predict(:none), predict(:clip), predict(:unit)
    # Ensure the intervention reproduces the already published selected fit.
    mismatch = maximum(abs.(unit[1:size(R.Fc, 1), :] .- Float64.(R.Fc)))
    @assert mismatch < 1e-5 "stabilised fit does not reproduce the cached rollout"
    @printf("trial %d: cache mismatch %.3g; hip ranges raw %s, clipped %s, unit %s\n",
            trial, mismatch, extrema(raw[:, 1]), extrema(clipped[:, 1]), extrema(unit[:, 1]))
    (; trial, dt=0.02, P, chosen=c, Yte=D["Yte"], raw, clipped, unit, mismatch)
end
serialize(cache_path("stabilization_gait.jls"), results)
