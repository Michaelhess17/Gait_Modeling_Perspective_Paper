#!/usr/bin/env julia
# Off-cycle relaxation test for every model.
#
# The TRUE system is started off its cycle (Stuart–Landau: radius ×1.3; coupled
# FHN: unit 2 offset by 0.3 × SD of its cycle values). Each fitted model is
# warmed up on the first W = 170 samples of that true trajectory (DMD: its delay
# window; ESN: teacher-forced reservoir; NODE: the state), then run freely. The
# deviation from the cycle (SL: |r − r*|; cFHN: distance from the synchronous
# manifold, √((v₁−v₂)²+(w₁−w₂)²)) is read at W + kP for k = 0…4 and normalised by
# its value at W. Truth decays by |μ| per cycle. Median over 4 starting phases.
#
# Models are refit with their tuned hyperparameters (floquet_tuned.jl /
# floquet_decays.jl searches), trained on either the single long trajectory
# ("excited") or the decay data ("decays").
#
# Usage: julia --project=. scripts/offcycle_relaxation.jl <sys> <train> <model> [seed]
#   train: excited | decays      model: dmd | esn | node

include("floquet_decays.jl")

const W_WARM = 170
const OFFSETS = (cfhn = 0.3, sl = 1.3)

function raw_training(sys, train; seed = 1)
    if train == "excited"
        σp = sys == "cfhn" ? SIGMA_P : 0.02 * sqrt(SL_SIGMA)
        X = sys == "cfhn" ? simulate(30; σp, seed) : sl_simulate(30; σp, seed)
        μ = mean(X, dims = 1); σ = std(X, dims = 1)
        return [((X .- μ) ./ σ)[1:30*SPC_F, :]], μ, σ
    elseif sys == "slip"
        raw = slipd().decays[seed]; allX = reduce(vcat, raw); μ = mean(allX, dims = 1); σ = std(allX, dims = 1)
        return [(X .- μ) ./ σ for X in raw], μ, σ
    else
        σp = sys == "cfhn" ? SIGMA_P : 0.02 * sqrt(SL_SIGMA)
        cs = cycle_states(sys); sdc = vec(std(cs, dims = 1)); rng = MersenneTwister(seed)
        raw = [sim_from(sys, cs[rand(rng, 1:size(cs, 1)), :] .+ PERT .* sdc .* randn(rng, length(sdc)), LEN;
                        σp, seed = seed * 1000 + k) for k in 1:N_TRAJ]
        allX = reduce(vcat, raw); μ = mean(allX, dims = 1); σ = std(allX, dims = 1)
        return [(X .- μ) ./ σ for X in raw], μ, σ
    end
end

function deviation(sys, Zraw)       # Zraw: n × d in raw units
    if sys == "sl"
        [abs(sqrt(Zraw[i, 1]^2 + Zraw[i, 2]^2) - sqrt(SL_SIGMA)) for i in axes(Zraw, 1)]
    else
        [sqrt((Zraw[i, 1] - Zraw[i, 3])^2 + (Zraw[i, 2] - Zraw[i, 4])^2) for i in axes(Zraw, 1)]
    end
end

"True off-cycle trajectories (raw units), 4 starting phases."
function true_starts(sys)
    sys == "slip" && return slipd().starts
    cs = cycle_states(sys); f! = sys == "cfhn" ? cfhn! : sl!; dt = sys_dt(sys)
    sdc = vec(std(cs, dims = 1)); out = Matrix{Float64}[]
    for ph in (1, 28, 55, 83)
        u0 = copy(cs[ph, :])
        if sys == "sl"
            u0 .*= OFFSETS.sl
        else
            u0[3] += OFFSETS.cfhn * sdc[3]; u0[4] += OFFSETS.cfhn * sdc[4]
        end
        sol = solve(ODEProblem(f!, u0, (0.0, (W_WARM + 5SPC_F + 2) * dt)), Tsit5(); saveat = dt, abstol = 1e-10, reltol = 1e-10)
        push!(out, permutedims(Array(sol)))
    end
    out
end

function chosen_hparams(sys, train, model)
    pre = train == "excited" ? "tuned_$(sys)_excited" : "decays_$(sys)"
    if model == "node"
        res = [deserialize(cache_path("floquet/$(pre)_nodesearch_$i.jls")) for i in 1:10
               if isfile(cache_path("floquet/$(pre)_nodesearch_$i.jls"))]
        return res[argmin([r.val for r in res])].st
    end
    deserialize(cache_path("floquet/$(pre)_$(model).jls")).chosen
end

function predict(sys, train, model, Ys, hp, seed, Ts, dt)
    n = 5SPC_F
    if model == "dmd"
        m = train == "excited" ? dmd_model(Ys[1]; hp...) : dmd_model_multi(Ys; hp...)
        return dmd_free(m, Ts[1:W_WARM, :], n)
    elseif model == "esn"
        m = train == "excited" ? fit_esn_full(Ys[1]; hp..., seed) : fit_esn_multi(Ys; hp..., seed)
        S = run_steps(esn_step(m), esn_tf(m, Ts, W_WARM), n)
        return S === nothing ? nothing : S[m.N+1:end, :]'
    else
        f, _ = train == "excited" ? fit_node_vf(Ys[1], dt; seed, hp...) : fit_node_multi(Ys, dt; seed, hp...)
        S = run_steps(step_of_vf(f, dt), Ts[W_WARM, :], n)
        return S === nothing ? nothing : S'
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    sys, train, model = ARGS[1], ARGS[2], ARGS[3]
    seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    dt = sys_dt(sys)
    Ys, μ, σ = raw_training(sys, train; seed = 1)
    hp = chosen_hparams(sys, train, model)
    starts = true_starts(sys)
    truth = [deviation(sys, T) for T in starts]
    ratios_true = [[td[W_WARM+k*SPC_F] / td[W_WARM] for k in 0:4] for td in truth]
    rat = Vector{Float64}[]; 
    for (T, td) in zip(starts, truth)
        Ts = (T .- μ) ./ σ
        fc = predict(sys, train, model, Ys, hp, seed, Ts, dt)
        if fc === nothing
            push!(rat, fill(NaN, 5)); continue
        end
        full = vcat(Ts[1:W_WARM, :], fc)
        d = deviation(sys, full .* vec(σ)' .+ vec(μ)')
        push!(rat, [d[W_WARM+k*SPC_F] / td[W_WARM] for k in 0:4])
    end
    tr = [median([r[k] for r in ratios_true]) for k in 1:5]
    md = [median(filter(isfinite, [r[k] for r in rat])) for k in 1:5]
    mkpath(cache_path("floquet"))
    serialize(cache_path("floquet/offcycle_$(sys)_$(train)_$(model)_$(seed).jls"),
              (; sys, train, model, seed, hp, truth = tr, model_ratio = md, all = rat, all_true = ratios_true))
    @printf("%s %s %s seed %d  truth %s  model %s\n", sys, train, model, seed,
            join([@sprintf("%.2f", v) for v in tr], " "), join([@sprintf("%.2f", v) for v in md], " "))
end
