#!/usr/bin/env julia
# LSTM in the self-driving benchmark, with the same train-only selection as the ESN (esn_tuning.jl): fit on all but
# the last two training cycles, free-run from the end of that window, choose the hidden size by error over the two
# held-back cycles (median of 2 seeds); refit on the full window with 4 seeds and score with score() (first-cycle
# error, 30-cycle fate), keeping 8 cycles of each free run for r2_tube.jl.
# Training windows: one cycle of teacher-forced warm-up, one cycle of free rollout (lstm_model.jl).
#
# Usage: julia -t 13 --project=. scripts/lstm_tuning.jl [system regime]   → cache/selfdrive/lstmtuned_<system>_<regime>.jls

include("self_driving_benchmark.jl")
include("lstm_model.jl")
using LinearAlgebra; BLAS.set_num_threads(1)

const HS = (32, 64, 128)

function tune_lstm(system, regime)
    D = npzread(cache_path("selfdrive/data_$(system)_$(regime).npz"))
    Ytr, Yte, P, n_run = D["Ytr"], D["Yte"], D["P"], Int(D["n_run"])
    W = K = round(Int, P); nv = round(Int, 2P)
    Yfit, Yval = Ytr[1:end-nv, :], Ytr[end-nv+1:end, :]
    verr(fc) = any(!isfinite, fc) ? Inf : sqrt(mean(abs2, fc .- Yval)) / sqrt(mean(abs2, Yval))
    vals = [median([verr(forecast_lstm(fit_lstm(Yfit; H, seed, W, K), Yfit[end-W+1:end, :], nv)) for seed in 1:2]) for H in HS]
    H = HS[argmin(vals)]
    t0 = time()
    rows = map(1:4) do seed
        m = fit_lstm(Ytr; H, seed, W, K)
        fc = forecast_lstm(m, Ytr[end-W+1:end, :], n_run)
        (; seed, score(fc, Yte, Ytr, P)..., Fc = Float32.(fc[1:round(Int, 8P), :]))
    end
    (; system, regime, chosen = (; H, W, K), vals = collect(zip(HS, vals)), rows, P, secs_per_fit = (time() - t0) / 4)
end

cells = length(ARGS) >= 2 ? [(ARGS[1], ARGS[2])] :
        [(m[1], m[2]) for f in readdir(cache_path("selfdrive")) for m in (match(r"^dmdtuned_(.+)_(realistic|long)\.jls$", f),) if m !== nothing]
Threads.@threads for (system, regime) in cells
    out = tune_lstm(system, regime)
    serialize(cache_path("selfdrive/lstmtuned_$(system)_$(regime).jls"), out)
    @printf("%-9s %-9s H=%3d  val %s | fates %s | e1 median %.3f | %.0f s per fit\n", system, regime, out.chosen.H,
            join([@sprintf("%d:%.2f", h, v) for (h, v) in out.vals], " "), join([r.fate for r in out.rows], ","),
            median([r.e1 for r in out.rows]), out.secs_per_fit)
    flush(stdout)
end
