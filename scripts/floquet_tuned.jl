#!/usr/bin/env julia
# Floquet multipliers from models with wide, training-data-only hyperparameter
# searches, read out two ways.
#
# Tuning (data realisation 101, separate from the evaluation realisations):
# fit on all but the last two cycles; free-run; score = error over the held-out
# two cycles + 0.5·|log(free-run amplitude ratio)| measured over cycles 8–10 of a
# 10-cycle free run (both use training data only). The Floquet multiplier is
# never used for selection.
#   DMD   grid: window τ ∈ {⅛,¼,½,¾,1,1.5} cycles, rank {8,16,30,50,80,120},
#         fit {ls, fb, unit}, ridge {1e-8,1e-6,1e-4}
#   ESN   random 48 settings × 2 seeds: ρ {0.3…1.3}, leak {0.05…1}, N {200…1200},
#         ridge {1e-8…1e-2}, input scale {0.1…2}
#   NODE  random 10 settings, 1 seed each: width {32,64,128}, depth {2,3,4},
#         lr {5e-4…5e-3}, shooting length {10,25,50,110}, epochs {200,400}
#
# Readouts (same for every model):
#   chain   transverse multiplier of the monodromy of the model's own cycle
#           (product of one-step Jacobians; the eigenvalue nearest 1 is dropped)
#   window  the same monodromy compressed onto the subspace spanned by the data's
#           own cycle-to-cycle deviations at the section (windows/states the data
#           actually visited, along-orbit direction removed): largest |eig| of
#           Vᵀ M V. For DMD (no orbit of its own) this is Vᵀ A^n V in delay space.
#
# Usage: julia --project=. scripts/floquet_tuned.jl <sys> <regime> <task> [arg]
#   task: dmd | esn | nodesearch:<i> | nodefinal:<seed>

include("floquet_from_models.jl")

const TUNE_SEED = 101
const NEVAL = 10

# ═════════════════════════════════════════════════════════════════════════════
# Shared pieces
# ═════════════════════════════════════════════════════════════════════════════
function val_score(fc, Yval, Yfit)
    any(!isfinite, fc) && return Inf
    nv = size(Yval, 1)
    e = sqrt(mean(abs2, fc[1:nv, :] .- Yval)) / sqrt(mean(abs2, Yval))
    amp = mean(std(fc[end-3SPC_F+1:end, :], dims = 1)) / mean(std(Yfit, dims = 1))
    return e + 0.5 * abs(log(max(amp, 1e-3)))
end

"Smoothed up-crossings of channel 1 (robust to measurement noise)."
function section_idx(Y)
    x = Y[:, 1]; w = 3
    xs = [mean(x[max(1, i-w):min(end, i+w)]) for i in eachindex(x)]
    round.(Int, cross_idx(xs))
end

"Orthonormal basis of the data's section deviations (columns of Sk), tangent removed."
function deviation_basis(Sk, tang; energy = 0.995, kmax = 8)
    D = Sk .- mean(Sk, dims = 2); t = tang ./ norm(tang)
    D = D .- t * (t' * D)
    F = svd(D); tot = sum(F.S .^ 2)
    tot < 1e-14 && return nothing
    k = clamp(something(findfirst(cumsum(F.S .^ 2) ./ tot .>= energy), 1), 1, min(kmax, length(F.S)))
    return F.U[:, 1:k]
end

function chain2(step, jac, s0, n, obs)
    M = Matrix{Float64}(I, length(s0), length(s0)); s = copy(s0)
    for _ in 1:n; M = jac(s) * M; s = step(s); end
    return M, norm(s[obs] .- s0[obs]) / sqrt(length(obs))
end

"Free-run, classify, and read the transverse multiplier both ways."
function read_both(step, jac, s0, obs, V)
    S = free_run(step, s0)
    S === nothing && return (; status = "blow-up")
    tail = S[obs, end-15*SPC_F+1:end]
    amp = mean(std(tail, dims = 2))
    amp < 0.25 && return (; status = "collapse", amp)
    c = cross_idx(tail[1, :])
    length(c) < 4 && return (; status = "no-cycle", amp)
    P = mean(diff(c)); n = round(Int, P)
    istart = size(S, 2) - 15 * SPC_F + round(Int, c[end-1])
    M, clos = chain2(step, jac, S[:, istart], n, obs)
    μw = V === nothing ? NaN : maximum(abs.(eigvals(V' * M * V)))
    return (; status = "cycle", amp, period = P / SPC_F, closure = clos, transverse(M)...,
              μ_win = μw, dimV = V === nothing ? 0 : size(V, 2))
end

# ═════════════════════════════════════════════════════════════════════════════
# DMD
# ═════════════════════════════════════════════════════════════════════════════
function stabilise_unit(A, P)
    F = schur(complex(A)); T = copy(F.T)
    for i in axes(T, 1)
        m = abs(T[i, i]); (m > 1 || m^P > 0.5) && (T[i, i] /= m)
    end
    real(F.Z * T * F.Z')
end

function dmd_model(Y; τ, r, variant, ridge)
    Hk = build_hankel_multi(Y, τ)
    r = min(r, size(Hk, 1) - 1); U = svd(Hk).U[:, 1:r]
    a = U' * Hk; X, Z = a[:, 1:end-1], a[:, 2:end]
    reg(M) = M + ridge * tr(M) / r * I
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    A = variant == "unit" ? stabilise_unit(A, SPC_F) : stabilise_unit_clip(A)
    return (; A, U, τ, d = size(Y, 2))
end
function stabilise_unit_clip(A)
    F = schur(complex(A)); T = copy(F.T)
    for i in axes(T, 1); abs(T[i, i]) > 1 && (T[i, i] /= abs(T[i, i])); end
    real(F.Z * T * F.Z')
end
function dmd_free(m, Yhist, n)
    h = vec(permutedims(Yhist[end-m.τ:end, :])); α = m.U' * h
    out = zeros(n, m.d)
    for k in 1:n; α = m.A * α; out[k, :] = (m.U * α)[m.τ*m.d+1:end]; end
    out
end

function dmd_window_probe(m, Y)
    d = m.d; τ = m.τ; Hk = build_hankel_multi(Y, τ)
    ck = [i for i in section_idx(Y) if i - τ >= 1 && i - τ + 1 <= size(Hk, 2)]
    length(ck) < 8 && return (; status = "too-few-cycles")
    W = Hk[:, ck .- τ]; W1 = Hk[:, ck .- τ .+ 1]
    V = deviation_basis(W, vec(mean(W1 .- W, dims = 2)))
    V === nothing && return (; status = "no-deviation")
    n = round(Int, mean(diff(ck)))
    Mm = V' * (m.U * (m.A^n * (m.U' * V)))
    return (; status = "probe", μ_win = maximum(abs.(eigvals(Mm))), dimV = size(V, 2), n)
end

function tune_dmd(Y)
    P = SPC_F; nv = 2P; Yfit, Yval = Y[1:end-nv, :], Y[end-nv+1:end, :]
    grid = [(; τ, r, variant, ridge) for τ in unique(max.(4, round.(Int, P .* (1/8, 1/4, 1/2, 3/4, 1, 1.5)))),
                                         r in (8, 16, 30, 50, 80, 120), variant in ("ls", "fb", "unit"),
                                         ridge in (1e-8, 1e-6, 1e-4)]
    sc = map(grid) do g
        try
            m = dmd_model(Yfit; g...)
            val_score(dmd_free(m, Yfit, 10P), Yval, Yfit)
        catch
            Inf
        end
    end
    return grid[argmin(sc)], minimum(sc), length(grid)
end

# ═════════════════════════════════════════════════════════════════════════════
# ESN
# ═════════════════════════════════════════════════════════════════════════════
function esn_settings(rng)
    (; ρ = rand(rng, (0.3, 0.6, 0.9, 1.1, 1.3)), α = rand(rng, (0.05, 0.1, 0.2, 0.5, 1.0)),
       N = rand(rng, (200, 400, 800, 1200)), λ = rand(rng, (1e-8, 1e-6, 1e-4, 1e-2)),
       σin = rand(rng, (0.1, 0.3, 0.5, 1.0, 2.0)))
end

function esn_val(Y, st; seeds = (1, 2))
    P = SPC_F; nv = 2P; Yfit, Yval = Y[1:end-nv, :], Y[end-nv+1:end, :]
    vals = map(seeds) do sd
        try
            m = fit_esn_full(Yfit; st..., seed = sd)
            S = free_run(esn_step(m), m.s0)
            S === nothing ? Inf : val_score(S[m.N+1:end, 1:10P]', Yval, Yfit)
        catch
            Inf
        end
    end
    return median(vals)
end

function tune_esn(Y)
    rng = MersenneTwister(77)
    cands = [esn_settings(rng) for _ in 1:48]
    sc = [esn_val(Y, c) for c in cands]
    return cands[argmin(sc)], minimum(sc)
end

function esn_read(Y, st, seed)
    d = size(Y, 2)
    m = fit_esn_full(Y; st..., seed)
    ck = [i for i in section_idx(Y) if 101 < i < size(Y, 1) - 1]
    V = if length(ck) >= 8
        Sk = reduce(hcat, [vcat(m.Rs[:, i], Y[i, :]) for i in ck])
        tg = mean(reduce(hcat, [vcat(m.Rs[:, i+1] .- m.Rs[:, i], Y[i+1, :] .- Y[i, :]) for i in ck]), dims = 2)
        deviation_basis(Sk, vec(tg))
    else nothing end
    read_both(esn_step(m), esn_jac(m), m.s0, m.N+1:m.N+d, V)
end

# ═════════════════════════════════════════════════════════════════════════════
# Neural ODE
# ═════════════════════════════════════════════════════════════════════════════
function node_settings(i)
    rng = MersenneTwister(5)
    list = [(; hidden = rand(rng, (32, 64, 128)), depth = rand(rng, (2, 3, 4)),
              lr = rand(rng, (5e-4, 1e-3, 2e-3, 5e-3)), shoot = rand(rng, (10, 25, 50, 110)),
              epochs = rand(rng, (200, 400))) for _ in 1:10]
    list[i]
end

function node_read(Y, dt, st, seed)
    d = size(Y, 2)
    f, l = fit_node_vf(Y, dt; seed, st...)
    step = step_of_vf(f, dt)
    ck = [i for i in section_idx(Y) if 1 < i < size(Y, 1) - 1]
    V = if length(ck) >= 8
        Sk = permutedims(Y[ck, :]); tg = vec(mean(permutedims(Y[ck .+ 1, :] .- Y[ck, :]), dims = 2))
        deviation_basis(Sk, tg)
    else nothing end
    r = read_both(step, s -> ForwardDiff.jacobian(step, s), Y[end, :], 1:d, V)
    return (; loss = l, r...)
end

# ═════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    sys, regime, task = ARGS[1], ARGS[2], ARGS[3]
    dt = sys_dt(sys); d = sys == "cfhn" ? 4 : 2
    mkpath(cache_path("floquet"))
    out(tag, obj) = serialize(cache_path("floquet/tuned_$(sys)_$(regime)_$(tag).jls"), obj)
    Ytune = make_data(sys, regime; seed = TUNE_SEED)
    truth = sys_truth(sys)
    if task == "dmd"
        g, sc, ng = tune_dmd(Ytune)
        @info "DMD tuned" g sc ng
        rows = NamedTuple[]
        for ds in 1:NEVAL
            Y = make_data(sys, regime; seed = ds)
            m = dmd_model(Y; g...)
            push!(rows, (; model = "DMD", data = ds, dmd_window_probe(m, Y)...))
        end
        out("dmd", (; sys, regime, truth, chosen = g, val = sc, rows))
        println(g, " val ", sc); foreach(println, rows)
    elseif task == "esn"
        st, sc = tune_esn(Ytune)
        @info "ESN tuned" st sc
        rows = NamedTuple[]
        for ds in 1:NEVAL                                    # one fit per data realisation
            push!(rows, (; model = "ESN", kind = "data", data = ds, esn_read(make_data(sys, regime; seed = ds), st, ds)...))
        end
        Y1 = make_data(sys, regime; seed = 1)
        for sd in 1:8                                        # model seeds on data set 1
            push!(rows, (; model = "ESN", kind = "seed", data = 1, esn_read(Y1, st, 100 + sd)...))
        end
        out("esn", (; sys, regime, truth, chosen = st, val = sc, rows))
        println(st, " val ", sc); foreach(println, rows)
    elseif startswith(task, "nodesearch")
        i = parse(Int, split(task, ":")[2]); st = node_settings(i)
        P = SPC_F; nv = 2P; Yfit, Yval = Ytune[1:end-nv, :], Ytune[end-nv+1:end, :]
        f, l = fit_node_vf(Yfit, dt; seed = 1, st...)
        step = step_of_vf(f, dt); S = free_run(step, Yfit[end, :])
        sc = S === nothing ? Inf : val_score(S[:, 1:10P]', Yval, Yfit)
        out("nodesearch_$i", (; st, loss = l, val = sc))
        println(i, " ", st, " loss ", l, " val ", sc)
    elseif startswith(task, "nodefinal")
        seed = parse(Int, split(task, ":")[2])
        res = [deserialize(cache_path("floquet/tuned_$(sys)_$(regime)_nodesearch_$i.jls")) for i in 1:10
               if isfile(cache_path("floquet/tuned_$(sys)_$(regime)_nodesearch_$i.jls"))]
        best = res[argmin([r.val for r in res])]
        r = node_read(make_data(sys, regime; seed = 1), dt, best.st, seed)
        out("nodefinal_$seed", (; sys, regime, truth, chosen = best.st, val = best.val,
                                  rows = [(; model = "NODE", seed, r...)]))
        println(best.st, " val ", best.val, "  ", r)
    end
end
