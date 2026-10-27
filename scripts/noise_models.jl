#!/usr/bin/env julia
# Off-cycle relaxation under measurement noise, all models on the same decay data.
#
# Training: n decays × 3 cycles from perturbed starts (process noise as elsewhere) + i.i.d. measurement
# noise σm (SD units). Test: true system started off its cycle at 4 phases (dmd_delay_sweep.jl), model warmed
# up on the first W samples and run freely; distance to the nearest point of the same model's run from an
# on-cycle warm-up, RMS per cycle, per-cycle rate = (ratio after 4 cycles)^(1/4).
# Each setting is scored with a CLEAN warm-up (model property) and a NOISY warm-up (σm added to the
# warm-up of both the perturbed and the on-cycle run, independent draws: what one could do with data).
#
#   dmd   grid: fit ls / tls × window × rank × ridge
#   esn   24 random settings (floquet_tuned.jl ranges, N ≤ 800), seed 1
#   node  decay-tuned setting (floquet_decays.jl search), one seed per call
#
# Usage: julia --project=. scripts/noise_models.jl <sys> <model> <σm> <n> [seed]
include("dmd_noise_variants.jl")

"Score one predictor: pred(Wwarm matrix) -> free run (rows = time) or nothing."
function score_pred(pred, sys, W, σm, μ, σ, Ctrue; rng)
    std_(X) = (X .- μ) ./ σ
    out = Dict{Symbol, Float64}()
    runs = truth_runs(sys, W, OFFSETS.cfhn)
    for (lab, noisy) in ((:clean, false), (:noisy, true))
        et = Vector{Float64}[]; em = Vector{Float64}[]
        for (T, T0) in runs
            Ts = std_(T); T0s = std_(T0)
            e = env(nearest(Ts[W+1:end, :], Ctrue)); push!(et, e ./ e[1])
            if noisy
                Ts = copy(Ts); T0s = copy(T0s)
                Ts[1:W, :] .+= σm .* randn(rng, W, size(Ts, 2)); T0s[1:W, :] .+= σm .* randn(rng, W, size(Ts, 2))
            end
            fc = pred(Ts[1:W, :]); f0 = pred(T0s[1:W, :])
            ok = fc !== nothing && f0 !== nothing && all(isfinite, fc) && all(isfinite, f0) &&
                 maximum(abs, fc) < 50 && maximum(abs, f0) < 50
            ok || (push!(em, fill(NaN, 5)); continue)
            a = env(nearest(fc, f0)); push!(em, a ./ a[1])
        end
        medn(x) = isempty(x) ? NaN : median(x); med(v) = [medn(filter(isfinite, [x[k] for x in v])) for k in 1:5]
        m = filter(isfinite, [x[5] for x in em])
        out[lab] = isempty(m) ? NaN : median(m)^(1/4)
        out[:truth] = med(et)[5]^(1/4)
    end
    out
end

if abspath(PROGRAM_FILE) == @__FILE__
    sys, model = ARGS[1], ARGS[2]; σm, ntraj = parse(Float64, ARGS[3]), parse(Int, ARGS[4])
    seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
    Ys, μ, σ = decays(sys, ntraj, 3)
    rng = MersenneTwister(77); Ys = [Y .+ σm .* randn(rng, size(Y)) for Y in Ys]
    Ctrue = (cycle_states(sys) .- μ) ./ σ; dt = sys_dt(sys); n = 6SPC_F
    line(tag, s) = @printf("%s %s σm=%.2f n=%3d %-40s | clean %.2f | noisy %.2f | truth %.2f\n",
                           sys, model, σm, ntraj, tag, s[:clean], s[:noisy], s[:truth])
    if model == "dmd"
        wins = sys == "sl" ? (1.5, 2.0, 2.5) : (1.0, 1.5, 2.0)
        for τc in wins, r in (20, 40, 80, 160), ridge in (1e-6, 1e-4, 1e-2), variant in ("ls", "tls")
            τ = round(Int, τc * SPC_F); W = τ + 10
            m = try dmd_fit(Ys; τ, r, variant, ridge) catch; continue end
            s = score_pred(Wm -> dmd_free(m, Wm, n), sys, W, σm, μ, σ, Ctrue; rng = MersenneTwister(9))
            line(@sprintf("τ=%.1fcyc r=%d ridge=%.0e %s", τc, r, ridge, variant), s)
        end
    elseif model == "esn"
        hrng = MersenneTwister(11)
        for i in 1:24
            st = esn_settings(hrng); st = (; st..., N = min(st.N, 800))
            m = try fit_esn_multi(Ys; st..., seed) catch; continue end
            pred = Wm -> (S = run_steps(esn_step(m), esn_tf(m, Wm, size(Wm, 1)), n);
                          S === nothing ? nothing : permutedims(S[m.N+1:end, :]))
            s = score_pred(pred, sys, 180, σm, μ, σ, Ctrue; rng = MersenneTwister(9))
            line(@sprintf("N=%d ρ=%.1f α=%.2f σin=%.1f λ=%.0e", st.N, st.ρ, st.α, st.σin, st.λ), s)
        end
    else
        hp = chosen_hparams(sys, "decays", "node")
        f, _ = fit_node_multi(Ys, dt; seed, hp...); stp = step_of_vf(f, dt)
        pred = Wm -> (S = run_steps(stp, Wm[end, :], n); S === nothing ? nothing : permutedims(S))
        line("seed=$seed $(hp)", score_pred(pred, sys, 180, σm, μ, σ, Ctrue; rng = MersenneTwister(9)))
    end
end
