#!/usr/bin/env julia
# Give the neural ODE every advantage on real gait, then ask the two questions
# that matter for stability analysis:
#
#   1. does the learned field have a limit cycle at all?
#   2. if it does, what is its period and its monodromy multiplier?
#
# The period is the honest check: we do not know the true Floquet multiplier of a
# walking trial, but we DO know its stride period. A learned cycle at the wrong
# period has not recovered the rhythm, let alone its stability.
#
# The model is deliberately overparameterised (3 x 128 on 6 channels ~ 35k
# parameters against ~3k training values), trained with multiple shooting and
# reverse-mode AD so capacity is not the limiting factor.
#
# Usage:  julia --project=. scripts/gait_overparam.jl <trial_idx> <seed>

include("common.jl")
include("floquet_from_field.jl")

using Lux, ComponentArrays, Optimization, OptimizationOptimisers
using Zygote, SciMLSensitivity
using FFTW
using Printf

const TRIAL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 91
const SEED  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1

const DECIM   = 2
const DT      = DECIM / 100.0
const T_TRAIN = 10.0
const SHOOT   = 25
const EPOCHS  = parse(Int, get(ENV, "NODE_EPOCHS", "400"))   # override only for smoke tests
const WIDTH   = 128
const NLAYER  = 3

big_model(n) = Lux.Chain(Lux.Dense(n, WIDTH, tanh),
                         Lux.Dense(WIDTH, WIDTH, tanh),
                         Lux.Dense(WIDTH, WIDTH, tanh),
                         Lux.Dense(WIDTH, n))

"Stride period from the autocorrelation peak of the first channel."
function stride_period(x, fs)
    xc = x .- mean(x)
    ac = real.(ifft(abs2.(fft(xc)))); ac ./= ac[1]
    lo, hi = round(Int, 0.5fs), min(round(Int, 2.5fs), length(ac) - 2)
    lo >= hi && return NaN
    seg = lo:hi
    pk = [i for i in seg if ac[i] > ac[i-1] && ac[i] > ac[i+1]]
    isempty(pk) && return NaN
    i = pk[argmax(ac[pk])]
    # parabolic refinement
    α, β, γ = ac[i-1], ac[i], ac[i+1]
    δ = 0.5 * (α - γ) / (α - 2β + γ + eps())
    return (i + δ) / fs
end

function fit_big(t_tr, Y_tr; seed=1, epochs=EPOCHS, lr=1e-3)
    n_t, n_d = size(Y_tr)
    model  = big_model(n_d)
    ps, st = Lux.setup(Lux.replicate(MersenneTwister(seed)), model)
    ps_ca  = ComponentArray(Float64.(ComponentArray(ps)))
    n_par  = length(ps_ca)
    @info "model" n_par n_data=(n_t * n_d) ratio=round(n_par / (n_t * n_d), digits=2)
    dudt(u, p, t) = first(model(u, p, st))
    # every segment the same length: Zygote's map pullback cannot accumulate
    # across ragged segments
    starts = collect(1:SHOOT:(n_t - SHOOT))
    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())

    function loss(pv, _)
        tot = sum(map(starts) do s0
            s1 = s0 + SHOOT
            prob = ODEProblem(dudt, Y_tr[s0, :], (t_tr[s0], t_tr[s1]), pv)
            sol = solve(prob, Tsit5(); saveat=t_tr[s0:s1], abstol=1e-6, reltol=1e-6,
                        sensealg=sensealg)
            sum(abs2, permutedims(Array(sol)) .- Y_tr[s0:s1, :])
        end)
        return tot / (n_t * n_d)
    end

    hist = Float64[]
    sol = solve(OptimizationProblem(
                  OptimizationFunction(loss, Optimization.AutoZygote()), ps_ca),
                OptimizationOptimisers.Adam(lr); maxiters=epochs,
                callback=function (st, l)
                    push!(hist, l)
                    if length(hist) % 20 == 0 || length(hist) == 1
                        @printf(stderr, "  epoch %4d/%d  loss %.6f\n",
                                length(hist), epochs, l)
                        flush(stderr)
                    end
                    return false
                end)
    psf = sol.u
    return (u -> first(model(u, psf, st))), hist, n_par
end

"Autonomous rollout of a learned field, sampled every `dt`; (ok, t, n_t x n_d)."
function rollout(vf, u0, T; dt=DT)
    s = solve(ODEProblem((u, p, t) -> vf(u), collect(float.(u0)), (0.0, T)), Tsit5();
              saveat=dt, abstol=1e-8, reltol=1e-8, maxiters=10^7)
    ok = s.retcode == ReturnCode.Success
    return ok, collect(s.t), permutedims(Array(s))
end

data_all, speeds, groups, subjects = load_gait_data()
# v2 also stores the data, the held-out forecast, a long rollout and the
# attractor orbit, so figures can show what the learned field actually does
key = @sprintf("gait_overparam_v2_t%d_s%d%s.jls", TRIAL, SEED,
              EPOCHS == 400 ? "" : "_smoke$(EPOCHS)")

if isfile(cache_path(key))
    @info "already cached" key
else
    td = Float64.(data_all[TRIAL, :, :])[1:DECIM:end, :]
    n_tr = round(Int, T_TRAIN / DT)
    P_stride = stride_period(td[1:n_tr, 1], 1 / DT)
    μs = mean(td[1:n_tr, :], dims=1); σs = std(td[1:n_tr, :], dims=1)
    Z = (td .- μs) ./ σs
    Y_tr = Z[1:n_tr, :]
    t_tr = collect((0:n_tr-1) .* DT)

    @info "fitting" TRIAL SEED group=groups[TRIAL] P_stride; flush(stderr)
    vf, hist, n_par = fit_big(t_tr, Y_tr; seed=SEED)
    @info "trained" final_loss=hist[end]; flush(stderr)

    # what did the learned field actually settle onto?
    fl = floquet_of_field(vf, Float64.(Y_tr[end, :]))
    @info "attractor" kind=fl.kind period=fl.period μ=fl.μ; flush(stderr)

    # held-out autonomous forecast from the last training sample
    n_te = min(round(Int, 5.0 / DT), size(Z, 1) - n_tr)
    Y_te = Z[n_tr+1:n_tr+n_te, :]
    ok_f, t_f, F = rollout(vf, Float64.(Y_tr[end, :]), n_te * DT)
    rms_te = sqrt(mean(abs2, Y_te))
    # cumulative normalised error up to each held-out sample
    err_t = ok_f ? [sqrt(mean(abs2, F[2:i+1, :] .- Y_te[1:i, :])) / rms_te
                    for i in 1:min(n_te, size(F, 1) - 1)] : Float64[]
    # long rollout: does the forecast settle onto a cycle, a point, or run off?
    ok_l, t_l, L = rollout(vf, Float64.(Y_tr[end, :]), 60.0)
    # one stretch of the attractor the learned field settles onto
    kind_a, t_a, X_a = field_attractor(vf, Float64.(Y_tr[end, :]); T_obs=20.0)

    r = (; trial=TRIAL, seed=SEED, group=String(groups[TRIAL]), speed=speeds[TRIAL], n_par,
           loss=hist[end], hist, P_stride, kind=fl.kind, P_learned=fl.period, μ=fl.μ,
           mags=fl.mags, period_ratio=fl.period / P_stride,
           dt=DT, Y_tr, Y_te, forecast_ok=ok_f, t_f, F, err_t,
           long_ok=ok_l, t_l, L, attractor_kind=kind_a, t_a, X_a)
    serialize(cache_path(key), r)

    println("\n" * "="^74)
    println("OVERPARAMETERISED NEURAL ODE ON REAL GAIT")
    println("="^74)
    @printf("trial %d (%s), seed %d | %d parameters vs %d training values (%.1fx)\n",
            TRIAL, groups[TRIAL], SEED, n_par, length(Y_tr), n_par / length(Y_tr))
    @printf("final training loss     : %.5f\n", hist[end])
    @printf("attractor of the learned field : %s\n", String(fl.kind))
    @printf("stride period in the data      : %.3f s\n", P_stride)
    if fl.kind == :cycle
        @printf("period of the learned cycle    : %.3f s  (%.2fx the stride period)\n",
                fl.period, fl.period / P_stride)
        @printf("monodromy multiplier |mu|      : %.4f\n", fl.μ)
    else
        println("no limit cycle -> no monodromy to report")
    end
    println("="^74)
end
