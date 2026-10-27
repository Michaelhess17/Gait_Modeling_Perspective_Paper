#!/usr/bin/env julia
# Give the echo state network the same train-only hyperparameter search as DMD
# (dmd_tuning.jl): fit on all but the last two training cycles, free-run,
# choose by error over those two cycles; refit on the full window and score.
# Grid: spectral radius ρ, leak α, reservoir size N, ridge λ; 3 seeds per
# setting so the choice is not a lucky seed (the chosen setting is then scored
# over 8 seeds, as in the benchmark).
#
# Usage: julia --project=. scripts/esn_tuning.jl <system> <regime>

include("self_driving_benchmark.jl")

function tune_esn(system, regime)
    D = npzread(cache_path("selfdrive/data_$(replace(system, ":" => ""))_$(regime).npz"))
    Ytr, Yte, P, n_run = D["Ytr"], D["Yte"], D["P"], Int(D["n_run"])
    nv = round(Int, 2P)
    Yfit, Yval = Ytr[1:end-nv, :], Ytr[end-nv+1:end, :]
    grid = [(; ρ, α, N, λ) for ρ in (0.5, 0.9, 1.2), α in (0.2, 0.5, 1.0),
                               N in (200, 800), λ in (1e-6, 1e-3)]
    verr(fc) = any(!isfinite, fc) ? Inf : sqrt(mean(abs2, fc .- Yval)) / sqrt(mean(abs2, Yval))
    vals = map(grid) do g
        median([verr(fit_esn(Yfit; g..., seed)(Yfit, nv)) for seed in 1:3])
    end
    chosen = grid[argmin(vals)]
    rows = map(1:8) do seed
        fc = fit_esn(Ytr; chosen..., seed)(Ytr, n_run)
        (; seed, score(fc, Yte, Ytr, P)..., Fc = Float32.(fc[1:round(Int, 8P), :]))
    end
    return (; system, regime, chosen, val = minimum(vals), rows, P)
end

if abspath(PROGRAM_FILE) == @__FILE__
    system, regime = ARGS[1], ARGS[2]
    out = tune_esn(system, regime)
    serialize(cache_path("selfdrive/esntuned_$(replace(system, ":" => ""))_$(regime).jls"), out)
    fates = [r.fate for r in out.rows]
    @printf("%-10s %-9s ESN tuned ρ=%.1f α=%.1f N=%d λ=%.0e | val %.2f | median e1 %.2f e3 %.2f | %s\n",
            system, regime, out.chosen.ρ, out.chosen.α, out.chosen.N, out.chosen.λ, out.val,
            median([r.e1 for r in out.rows]), median([r.e3 for r in out.rows]),
            join(["$(count(==(f), fates)) $f" for f in unique(fates)], ", "))
end
