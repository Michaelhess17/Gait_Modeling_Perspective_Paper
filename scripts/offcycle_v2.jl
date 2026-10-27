#!/usr/bin/env julia
# Off-cycle relaxation, decay-trained models only, phase-independent scoring.
#
# True system started off its cycle (offcycle_relaxation.jl: SL radius ×1.3, cFHN unit-2 offset), model
# warmed up on the first W samples and run freely for 30 cycles. Deviation at each time = Euclidean distance
# (standardised units) to the NEAREST point of a reference cycle, so timing/phase drift does not count.
#   own:   reference = the SAME model run from the true ON-cycle warm-up at the same phase, over the same
#          window (its own cycle; no offset, no settling assumption) → measures contraction rate
#   true:  reference = the true limit cycle → also includes any offset of the model's cycle
# Truth uses the true cycle for both. Score per cycle k = 0…4: RMS deviation over the k-th cycle after
# warm-up, divided by cycle 0. Median over 4 start phases. Trajectories are saved for rescoring.
# Usage: julia --project=. scripts/offcycle_v2.jl <sys> <model: dmd|esn|node> [seed]
include("offcycle_relaxation.jl")

const NLONG = 30SPC_F
nearest(X, C) = [minimum(sum(abs2, C .- X[i:i, :], dims = 2))^0.5 for i in axes(X, 1)]
env(d) = [sqrt(mean(abs2, d[k*SPC_F+1:(k+1)*SPC_F])) for k in 0:4]

"Fit once; return Ts -> free run (NLONG × d) after warm-up on Ts[1:W]."
function fit_predictor(sys, model, Ys, hp, seed, dt)
    if model == "dmd"
        m = dmd_model_multi(Ys; hp...); return Ts -> dmd_free(m, Ts[1:W_WARM, :], NLONG)
    elseif model == "esn"
        m = fit_esn_multi(Ys; hp..., seed)
        return Ts -> (S = run_steps(esn_step(m), esn_tf(m, Ts, W_WARM), NLONG); S === nothing ? nothing : permutedims(S[m.N+1:end, :]))
    else
        f, _ = fit_node_multi(Ys, dt; seed, hp...); st = step_of_vf(f, dt)
        return Ts -> (S = run_steps(st, Ts[W_WARM, :], NLONG); S === nothing ? nothing : permutedims(S))
    end
end

"True ON-cycle trajectories from the same 4 starting phases (raw units)."
function true_oncycle(sys)
    sys == "slip" && return slipd().oncycle
    cs = cycle_states(sys); f! = sys == "cfhn" ? cfhn! : sl!; dt = sys_dt(sys)
    [permutedims(Array(solve(ODEProblem(f!, copy(cs[ph, :]), (0.0, (W_WARM + 5SPC_F + 2) * dt)), Tsit5();
                             saveat = dt, abstol = 1e-10, reltol = 1e-10))) for ph in (1, 28, 55, 83)]
end

if abspath(PROGRAM_FILE) == @__FILE__
    sys, model = ARGS[1], ARGS[2]
    seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    dt = sys_dt(sys)
    Ys, μ, σ = raw_training(sys, "decays"; seed = 1)
    hp = chosen_hparams(sys, "decays", model)
    std_(X) = (X .- μ) ./ σ
    Ctrue = std_(cycle_states(sys))
    starts = true_starts(sys); ons = true_oncycle(sys)
    pred = fit_predictor(sys, model, Ys, hp, seed, dt)
    et = Vector{Float64}[]; eown = Vector{Float64}[]; etrue = Vector{Float64}[]; offs = Float64[]; fcs = []
    for (T, T0) in zip(starts, ons)
        Ts = std_(T)
        e = env(nearest(Ts[W_WARM+1:end, :], Ctrue)); push!(et, e ./ e[1])
        fc = pred(Ts); f0 = pred(std_(T0)); push!(fcs, fc)
        if fc === nothing || f0 === nothing || maximum(abs, fc) > 50 || maximum(abs, f0) > 50
            push!(eown, fill(NaN, 5)); push!(etrue, fill(NaN, 5)); push!(offs, NaN); continue
        end
        Cown = f0[1:6SPC_F, :]                 # model's own unperturbed run over the same window
        a = env(nearest(fc, Cown)); push!(eown, a ./ a[1])
        b = env(nearest(fc, Ctrue)); push!(etrue, b ./ b[1])
        push!(offs, sqrt(mean(abs2, nearest(f0[end-2SPC_F+1:end, :], Ctrue))) / e[1])   # own-cycle offset / initial deviation
    end
    med(v) = [median(filter(isfinite, [r[k] for r in v])) for k in 1:5]
    serialize(cache_path("floquet/offv2_$(sys)_$(model)_$(seed).jls"),
              (; sys, model, seed, hp, truth = med(et), own = med(eown), vs_true = med(etrue), offset = offs,
                 all_own = eown, all_vs_true = etrue, all_truth = et, fcs, starts))
    f(v) = join([@sprintf("%.2f", x) for x in v], " ")
    println(sys, " ", model, " ", seed, " | truth ", f(med(et)), " | own-cycle ", f(med(eown)),
            " | vs true cycle ", f(med(etrue)), " | offset/dev0 ", f(offs))
end
