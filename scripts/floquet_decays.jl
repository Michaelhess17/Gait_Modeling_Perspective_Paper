#!/usr/bin/env julia
# Many decays from perturbed starts instead of one long trajectory.
#
# Same systems, models, wide train-only hyperparameter searches and both
# readouts as floquet_tuned.jl, but the training data are N_TRAJ = 10 short
# trajectories (TRAJ_CYC = 3 cycles each, 30 cycles in total, so the same budget
# as the "excited" regime) started from random phases of the limit cycle plus a
# perturbation of PERT × (state SD) in every coordinate. Process noise is on, as
# in the "excited" regime; no measurement noise. This is what a perturbation
# experiment would provide (it is outside typical gait capture).
#
# Validation for the searches (training data only): fit on the first 8
# trajectories; for each of the last 2, forecast samples 171–330 from the state at
# sample 170 (all models forecast the same horizon); plus the free-run amplitude
# penalty from the end of the last fit trajectory (10 cycles). Floquet multipliers
# are never used for selection.
#
# Free-run outcomes (the self-driving classification) are recorded for every fit:
#   ok        cycle, period within 10 %, amplitude 0.75–1.33
#   decays    cycle with the right period but amplitude < 0.75
#   wrong     cycle with wrong period/amplitude
#   collapse / blow-up / no-cycle
#
# Usage: julia --project=. scripts/floquet_decays.jl <sys> <task> [arg]
#   task: dmd | esn | nodesearch:<i> | nodefinal:<seed> | dmdfate:<regime>

include("floquet_tuned.jl")

const N_TRAJ, TRAJ_CYC, PERT = 10, 3, 0.2
const LEN = TRAJ_CYC * SPC_F
const EVAL_FROM = 170

# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════
function cycle_states(sys)
    f! = sys == "cfhn" ? cfhn! : sl!
    u0 = sys == "cfhn" ? TRUTH.u0 : [sqrt(SL_SIGMA), 0.0]
    T = sys == "cfhn" ? TRUTH.T : 1.0
    sol = solve(ODEProblem(f!, u0, (0.0, T)), Tsit5(); saveat = sys_dt(sys), abstol = 1e-10, reltol = 1e-10)
    permutedims(Array(sol))
end

function sim_from(sys, u0, n; σp, seed)
    f! = sys == "cfhn" ? cfhn! : sl!; dt = sys_dt(sys)
    g!(du, u, p, t) = (du .= 0; sys == "cfhn" ? (du[1] = σp; du[3] = σp) : (du .= σp))
    sol = solve(SDEProblem(f!, g!, u0, (0.0, (n - 1) * dt); seed = seed), EM(); dt = dt / 20, saveat = dt)
    permutedims(Array(sol))
end

function make_decay_data(sys; seed = 1, σm = 0.0)
    σp = sys == "cfhn" ? SIGMA_P : 0.02 * sqrt(SL_SIGMA)
    cs = cycle_states(sys); sdc = vec(std(cs, dims = 1)); rng = MersenneTwister(seed)
    trajs = [sim_from(sys, cs[rand(rng, 1:size(cs, 1)), :] .+ PERT .* sdc .* randn(rng, length(sdc)), LEN;
                      σp, seed = seed * 1000 + k) for k in 1:N_TRAJ]
    allX = reduce(vcat, trajs); μ = mean(allX, dims = 1); σ = std(allX, dims = 1)
    [(X .- μ) ./ σ .+ σm .* randn(rng, size(X)) for X in trajs]
end

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════
"Free-run outcome from an observed trajectory (d × n)."
function classify_free(Sobs)
    any(!isfinite, Sobs) && return (; status = "blow-up", fate = "blow-up")
    tail = Sobs[:, end-15*SPC_F+1:end]; amp = mean(std(tail, dims = 2))
    amp < 0.25 && return (; status = "collapse", fate = "collapse", amp)
    c = cross_idx(tail[1, :])
    length(c) < 4 && return (; status = "no-cycle", fate = "no-cycle", amp)
    per = mean(diff(c)) / SPC_F
    fate = abs(per - 1) < 0.1 ? (amp > 1.33 ? "wrong" : amp >= 0.75 ? "ok" : "decays") : "wrong"
    return (; status = "cycle", fate, amp, period = per)
end
fate_of(r) = hasproperty(r, :fate) ? r.fate :
    r.status != "cycle" ? r.status :
    abs(r.period - 1) < 0.1 ? (r.amp > 1.33 ? "wrong" : r.amp >= 0.75 ? "ok" : "decays") : "wrong"

function run_steps(step, s0, n)
    out = zeros(length(s0), n); s = copy(s0)
    for k in 1:n
        s = step(s); (any(!isfinite, s) || maximum(abs, s) > 1e4) && return nothing
        out[:, k] = s
    end
    out
end

function multi_val(fcs, Yhs, amp_run, Yfit_all)
    (any(isnothing, fcs) || amp_run === nothing) && return Inf
    num = 0.0; den = 0.0
    for (fc, Yh) in zip(fcs, Yhs)
        tgt = Yh[EVAL_FROM+1:end, :]
        any(!isfinite, fc) && return Inf
        num += sum(abs2, fc .- tgt); den += sum(abs2, tgt)
    end
    amp = mean(std(amp_run[:, end-3SPC_F+1:end], dims = 2)) / mean(std(Yfit_all, dims = 1))
    return sqrt(num / den) + 0.5 * abs(log(max(amp, 1e-3)))
end
split_fit_hold(Ys) = (Ys[1:end-2], Ys[end-1:end])

# ═════════════════════════════════════════════════════════════════════════════
# DMD (several trajectories)
# ═════════════════════════════════════════════════════════════════════════════
function dmd_model_multi(Ys; τ, r, variant, ridge)
    Hs = [build_hankel_multi(Y, τ) for Y in Ys]
    r = min(r, size(Hs[1], 1) - 1); U = svd(reduce(hcat, Hs)).U[:, 1:r]
    X = reduce(hcat, [U' * H[:, 1:end-1] for H in Hs]); Z = reduce(hcat, [U' * H[:, 2:end] for H in Hs])
    reg(M) = M + ridge * tr(M) / r * I
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    A = variant == "unit" ? stabilise_unit(A, SPC_F) : stabilise_unit_clip(A)
    (; A, U, τ, d = size(Ys[1], 2))
end

function dmd_probe_multi(m, Ys)
    τ = m.τ; W = Matrix{Float64}[]; W1 = Matrix{Float64}[]
    for Y in Ys
        H = build_hankel_multi(Y, τ)
        ck = [i for i in section_idx(Y) if i - τ >= 1 && i - τ + 1 <= size(H, 2)]
        isempty(ck) && continue
        push!(W, H[:, ck .- τ]); push!(W1, H[:, ck .- τ .+ 1])
    end
    isempty(W) && return (; status = "too-few-cycles")
    Wm = reduce(hcat, W); W1m = reduce(hcat, W1)
    size(Wm, 2) < 8 && return (; status = "too-few-cycles")
    V = deviation_basis(Wm, vec(mean(W1m .- Wm, dims = 2)))
    V === nothing && return (; status = "no-deviation")
    Mm = V' * (m.U * (m.A^SPC_F * (m.U' * V)))
    (; status = "probe", μ_win = maximum(abs.(eigvals(Mm))), dimV = size(V, 2))
end

function tune_dmd_multi(Ys)
    fit, hold = split_fit_hold(Ys)
    grid = [(; τ, r, variant, ridge) for τ in unique(max.(4, round.(Int, SPC_F .* (1/8, 1/4, 1/2, 3/4, 1, 1.5)))),
                                         r in (8, 16, 30, 50, 80, 120), variant in ("ls", "fb", "unit"),
                                         ridge in (1e-8, 1e-6, 1e-4)]
    sc = map(grid) do g
        try
            m = dmd_model_multi(fit; g...)
            fcs = [dmd_free(m, Yh[1:EVAL_FROM, :], LEN - EVAL_FROM) for Yh in hold]
            amp = dmd_free(m, fit[end], 10SPC_F)'
            multi_val(fcs, hold, amp, reduce(vcat, fit))
        catch
            Inf
        end
    end
    grid[argmin(sc)], minimum(sc), length(grid)
end

# ═════════════════════════════════════════════════════════════════════════════
# ESN (several trajectories)
# ═════════════════════════════════════════════════════════════════════════════
function fit_esn_multi(Ys; N, ρ, α, σin, λ, seed, washout = 30)
    rng = MersenneTwister(seed); d = size(Ys[1], 2)
    W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= ρ / maximum(abs.(eigvals(W)))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    Rsl = Matrix{Float64}[]; Fc = Matrix{Float64}[]; Tc = Matrix{Float64}[]
    for Y in Ys
        n = size(Y, 1); Rs = zeros(N, n); r = zeros(N)
        for t in 1:n
            r = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * Y[t, :] .+ b); Rs[:, t] = r
        end
        push!(Rsl, Rs)
        push!(Fc, vcat(Rs[:, washout:n-1], Y[washout:n-1, :]', ones(1, n - washout)))
        push!(Tc, Y[washout+1:n, :]')
    end
    F = reduce(hcat, Fc); T = reduce(hcat, Tc)
    Wout = T * F' / (F * F' + λ * I)
    (; W, Win, b, α, N, d, Wr = Wout[:, 1:N], Wu = Wout[:, N+1:N+d], c = Wout[:, end],
       s0 = vcat(Rsl[end][:, end-1], Ys[end][end, :]), Rs = Rsl)
end
function esn_tf(m, Y, upto)       # teacher-forced reservoir state through sample `upto`
    r = zeros(m.N)
    for t in 1:upto
        r = (1 - m.α) .* r .+ m.α .* tanh.(m.W * r .+ m.Win * Y[t, :] .+ m.b)
    end
    vcat(r, Y[upto, :])
end

function esn_val_multi(Ys, st; seeds = (1, 2))
    fit, hold = split_fit_hold(Ys)
    median(map(seeds) do sd
        try
            m = fit_esn_multi(fit; st..., seed = sd); step = esn_step(m)
            fcs = [(S = run_steps(step, esn_tf(m, Yh, EVAL_FROM), LEN - EVAL_FROM);
                    S === nothing ? nothing : S[m.N+1:end, :]') for Yh in hold]
            amp = run_steps(step, m.s0, 10SPC_F)
            multi_val(fcs, hold, amp === nothing ? nothing : amp[m.N+1:end, :], reduce(vcat, fit))
        catch
            Inf
        end
    end)
end
function tune_esn_multi(Ys)
    rng = MersenneTwister(77); cands = [esn_settings(rng) for _ in 1:48]
    sc = [esn_val_multi(Ys, c) for c in cands]
    cands[argmin(sc)], minimum(sc)
end

function esn_read_multi(Ys, st, seed)
    m = fit_esn_multi(Ys; st..., seed); d = m.d
    Sk = Vector{Float64}[]; Tg = Vector{Float64}[]
    for (Y, Rs) in zip(Ys, m.Rs)
        for i in section_idx(Y)
            31 < i < size(Y, 1) - 1 || continue
            push!(Sk, vcat(Rs[:, i], Y[i, :])); push!(Tg, vcat(Rs[:, i+1] .- Rs[:, i], Y[i+1, :] .- Y[i, :]))
        end
    end
    V = length(Sk) >= 8 ? deviation_basis(reduce(hcat, Sk), vec(mean(reduce(hcat, Tg), dims = 2))) : nothing
    read_both(esn_step(m), esn_jac(m), m.s0, m.N+1:m.N+d, V)
end

# ═════════════════════════════════════════════════════════════════════════════
# Neural ODE (several trajectories)
# ═════════════════════════════════════════════════════════════════════════════
function fit_node_multi(Ys, dt; seed = 1, hidden = 64, depth = 2, epochs = 300, lr = 2e-3, shoot = 25)
    d = size(Ys[1], 2); t = collect((0:size(Ys[1], 1)-1) .* dt)
    layers = Any[Lux.Dense(d, hidden, tanh)]
    for _ in 2:depth; push!(layers, Lux.Dense(hidden, hidden, tanh)); end
    push!(layers, Lux.Dense(hidden, d)); model = Lux.Chain(layers...)
    ps, st = Lux.setup(Lux.replicate(MersenneTwister(seed)), model)
    p0 = ComponentArray(Float64.(ComponentArray(ps)))
    dudt(u, p, _) = first(model(u, p, st))
    segs = [(k, s0) for k in eachindex(Ys) for s0 in 1:shoot:(size(Ys[k], 1) - shoot)]
    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())
    function loss(p, _)
        tot = sum(map(segs) do (k, s0)
            s1 = s0 + shoot
            sol = solve(ODEProblem(dudt, Ys[k][s0, :], (t[s0], t[s1]), p), Tsit5();
                        saveat = t[s0:s1], abstol = 1e-6, reltol = 1e-6, sensealg = sensealg)
            sum(abs2, permutedims(Array(sol)) .- Ys[k][s0:s1, :])
        end)
        tot / (length(segs) * (shoot + 1) * d)
    end
    hist = Float64[]
    sol = solve(OptimizationProblem(OptimizationFunction(loss, Optimization.AutoZygote()), p0),
                OptimizationOptimisers.Adam(lr); maxiters = epochs, callback = (s, l) -> (push!(hist, l); false))
    pf = sol.u
    (u -> first(model(u, pf, st))), hist[end]
end

function node_read_multi(Ys, dt, st, seed)
    d = size(Ys[1], 2)
    f, l = fit_node_multi(Ys, dt; seed, st...); step = step_of_vf(f, dt)
    Sk = Vector{Float64}[]; Tg = Vector{Float64}[]
    for Y in Ys, i in section_idx(Y)
        1 < i < size(Y, 1) - 1 || continue
        push!(Sk, Y[i, :]); push!(Tg, Y[i+1, :] .- Y[i, :])
    end
    V = length(Sk) >= 8 ? deviation_basis(reduce(hcat, Sk), vec(mean(reduce(hcat, Tg), dims = 2))) : nothing
    r = read_both(step, s -> ForwardDiff.jacobian(step, s), Ys[end][end, :], 1:d, V)
    (; loss = l, r...)
end

# ═════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    sys, task = ARGS[1], ARGS[2]
    dt = sys_dt(sys); truth = sys_truth(sys)
    mkpath(cache_path("floquet"))
    out(tag, obj) = serialize(cache_path("floquet/decays_$(sys)_$(tag).jls"), obj)
    Ytune = make_decay_data(sys; seed = TUNE_SEED)
    if task == "dmd"
        g, sc, ng = tune_dmd_multi(Ytune)
        rows = NamedTuple[]
        for ds in 1:NEVAL
            Ys = make_decay_data(sys; seed = ds); m = dmd_model_multi(Ys; g...)
            fr = classify_free(dmd_free(m, Ys[end], N_FREE)')
            push!(rows, (; model = "DMD", data = ds, dmd_probe_multi(m, Ys)..., fate = fr.fate, fr_amp = get(fr, :amp, NaN),
                           fr_period = get(fr, :period, NaN)))
        end
        out("dmd", (; sys, truth, chosen = g, val = sc, rows)); println(g, " val ", sc); foreach(println, rows)
    elseif task == "esn"
        st, sc = tune_esn_multi(Ytune)
        rows = NamedTuple[]
        for ds in 1:NEVAL
            push!(rows, (; model = "ESN", kind = "data", data = ds, esn_read_multi(make_decay_data(sys; seed = ds), st, ds)...))
        end
        Y1 = make_decay_data(sys; seed = 1)
        for sd in 1:8
            push!(rows, (; model = "ESN", kind = "seed", data = 1, esn_read_multi(Y1, st, 100 + sd)...))
        end
        out("esn", (; sys, truth, chosen = st, val = sc, rows)); println(st, " val ", sc); foreach(println, rows)
    elseif startswith(task, "nodesearch")
        i = parse(Int, split(task, ":")[2]); st = node_settings(i)
        fit, hold = split_fit_hold(Ytune)
        f, l = fit_node_multi(fit, dt; seed = 1, st...); step = step_of_vf(f, dt)
        fcs = [(S = run_steps(step, Yh[EVAL_FROM, :], LEN - EVAL_FROM); S === nothing ? nothing : S') for Yh in hold]
        sc = multi_val(fcs, hold, run_steps(step, fit[end][end, :], 10SPC_F), reduce(vcat, fit))
        out("nodesearch_$i", (; st, loss = l, val = sc)); println(i, " ", st, " loss ", l, " val ", sc)
    elseif startswith(task, "nodefinal")
        seed = parse(Int, split(task, ":")[2])
        res = [deserialize(cache_path("floquet/decays_$(sys)_nodesearch_$i.jls")) for i in 1:10
               if isfile(cache_path("floquet/decays_$(sys)_nodesearch_$i.jls"))]
        best = res[argmin([r.val for r in res])]
        r = node_read_multi(make_decay_data(sys; seed = 1), dt, best.st, seed)
        out("nodefinal_$seed", (; sys, truth, chosen = best.st, val = best.val, rows = [(; model = "NODE", seed, r...)]))
        println(best.st, " ", r)
    elseif startswith(task, "dmdfate")      # free-run fate of the tuned single-trajectory DMD
        regime = split(task, ":")[2]
        g = deserialize(cache_path("floquet/tuned_$(sys)_$(regime)_dmd.jls")).chosen
        rows = NamedTuple[]
        for ds in 1:NEVAL
            Y = make_data(sys, regime; seed = ds); m = dmd_model(Y; g...)
            fr = classify_free(dmd_free(m, Y, N_FREE)')
            push!(rows, (; model = "DMD", data = ds, fate = fr.fate, fr_amp = get(fr, :amp, NaN),
                           fr_period = get(fr, :period, NaN)))
        end
        out("dmdfate_$regime", (; sys, regime, rows)); foreach(println, rows)
    end
end
