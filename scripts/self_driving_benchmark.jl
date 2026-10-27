#!/usr/bin/env julia
# Self-driving benchmark: can a learned model run on its own and stay a walker?
#
# Each model is fitted on a training window, then run FREELY (no data fed back)
# from the end of that window. We score
#   * free-run error at 1, 3 and 5 cycles against the held-out data
#     (cumulative RMSE over the window, in training-SD units), and
#   * where the model ends up after 30 cycles on its own:
#       ok        amplitude within ×0.75–1.33 of the data, period within 10 %
#       decays    right rhythm (period within 10 %) but amplitude 0.25–0.75×
#       collapse  amplitude below 0.25× (falls to a point)
#       blow-up   non-finite or amplitude above 4×
#       wrong     wrong rhythm (period off by > 10 %) or too large
#
# Systems (≈ 60–130 samples per cycle, like gait at 50–100 Hz):
#   fhn   FitzHugh–Nagumo relaxation oscillator, (v, w) observed
#   cfhn  two coupled FitzHugh–Nagumo units, (v1, w1, v2, w2)
#   vdp2  two Van der Pol oscillators, A drives B (coupling_benchmark.jl)
#   slip  bipedal SLIP walking from a perturbation, CoM (y, ẋ, ẏ)
#         (../Julia/SLIP.jl/CSV/DMD_dataset_grid, stable gait, |ρ| = 0.94)
#   gait  six joint angles of one human treadmill trial (50 Hz)
# Regimes:
#   long       30 training cycles, clean
#   realistic  12 training cycles, 1 % measurement noise; process noise on the
#              ODE systems (stride-to-stride variability)
#
# Models: Hankel-DMD (linear; unit-circle stabilised, the lab's standard),
# echo state network (8 seeds), neural ODE with multiple shooting (4 seeds),
# and the phase-aligned mean-cycle template as a reference.
#
# Usage: julia --project=. scripts/self_driving_benchmark.jl <system> <regime> [models]
#   gait takes the trial index as the system: gait:91

include("common.jl")
using Lux, ComponentArrays, Optimization, OptimizationOptimisers
using Zygote, SciMLSensitivity
using SparseArrays, FFTW

const N_TEST_CYC = 5
const N_LONG_CYC = 30

# ═════════════════════════════════════════════════════════════════════════════
# Systems
# ═════════════════════════════════════════════════════════════════════════════
"Dominant period (samples) of a multichannel signal from the autocorrelation."
function period_samples(Y; lo = 20)
    x = Y[:, 1] .- mean(Y[:, 1]); n = length(x)
    ac = real.(ifft(abs2.(fft(vcat(x, zeros(n)))))); ac = ac[1:n] ./ ac[1]
    pk = [i for i in lo:n÷2 if ac[i] > ac[i-1] && ac[i] >= ac[i+1] && ac[i] > 0.3]
    isempty(pk) && return NaN
    i = pk[1]; α, β, γ = ac[i-1], ac[i], ac[i+1]
    return (i - 1) + 0.5 * (α - γ) / (α - 2β + γ)
end

function simulate_ode(f!, u0, p, n, dt; σp = 0.0, noise_idx = Int[], burn = 400.0, seed = 1)
    T = burn + n * dt
    sol = if σp > 0
        g!(du, u, p, t) = (du .= 0; du[noise_idx] .= σp)
        solve(SDEProblem(f!, g!, u0, (0.0, T), p; seed = seed), EM(); dt = dt / 20,
              saveat = dt, maxiters = 10^8)
    else
        solve(ODEProblem(f!, u0, (0.0, T), p), Tsit5(); saveat = dt,
              abstol = 1e-10, reltol = 1e-10)
    end
    X = permutedims(Array(sol)); i0 = round(Int, burn / dt) + 1
    return X[i0:i0+n-1, :]
end

function fhn!(du, u, p, t)
    v, w = u
    du[1] = v - v^3 / 3 - w + 0.5
    du[2] = 0.08 * (v + 0.7 - 0.8w)
end
function cfhn!(du, u, p, t)                     # as in ablation_run.jl
    v1, w1, v2, w2 = u; k = 0.08
    du[1] = v1 - v1^3/3 - w1 + 0.6 + k * (v2 - v1); du[2] = 0.5 * (v1 + 0.7 - 0.8w1)
    du[3] = v2 - v2^3/3 - w2 + 0.6 + k * (v1 - v2); du[4] = 0.5 * (v2 + 0.7 - 0.8w2)
end
function vdp2!(du, u, p, t)                     # as in coupling_benchmark.jl, A→B
    xA, vA, xB, vB = u
    du[1] = vA; du[2] = (1 - xA^2) * vA - xA
    du[3] = vB; du[4] = (1 - xB^2) * vB - 1.05^2 * xB + 0.2xA
end

const ODE_SYSTEMS = Dict(
    "fhn"  => (f = fhn!,  u0 = [-1.0, 1.0],              T = 36.6, noise_idx = [1]),
    "cfhn" => (f = cfhn!, u0 = [-1.0, -0.3, -0.6, -0.25], T = 12.9, noise_idx = [1, 3]),
    "vdp2" => (f = vdp2!, u0 = [2.0, 0.0, -1.0, 0.5],    T = 6.66, noise_idx = [2, 4]),
)

const SLIP_CSV = expanduser("~/Documents/Julia/SLIP.jl/CSV/DMD_dataset_grid/" *
                            "traj_aoa59.0_ktilde7.0_pert1_continuous.csv")

"Raw data (n × d), sample step, and period in samples for a system."
function load_system(name, n_cyc, σp)
    if haskey(ODE_SYSTEMS, name)
        s = ODE_SYSTEMS[name]; dt = s.T / 110        # ≈ 110 samples per cycle
        n = round(Int, (n_cyc + N_TEST_CYC + 2) * 110)
        X = simulate_ode(s.f, s.u0, nothing, n, dt; σp, noise_idx = s.noise_idx)
        return X, dt
    elseif name == "slip"
        df = CSV.read(SLIP_CSV, DataFrame)
        X = Matrix{Float64}(df[1:2:end, [:y, :xdot, :ydot]])   # 100 Hz
        return X, 0.01
    elseif startswith(name, "gait")
        idx = parse(Int, split(name, ":")[2])
        data_all, _, _, _ = load_gait_data()
        return Float64.(data_all[idx, 1:2:end, :]), 0.02          # 50 Hz
    end
    error("unknown system $name")
end

# ═════════════════════════════════════════════════════════════════════════════
# Models — each returns a forecaster  Ytr, n -> n × d
# ═════════════════════════════════════════════════════════════════════════════
"Hankel-DMD on a rank-r POD subspace; eigenvalues outside the unit circle are
pulled onto it (in the Schur basis), the usual stabilisation for oscillators."
function fit_dmd(Y; τ, r, ridge = 1e-6)
    d = size(Y, 2); H = build_hankel_multi(Y, τ)
    U = svd(H).U[:, 1:min(r, size(H, 1))]
    a = U' * H; X, Z = a[:, 1:end-1], a[:, 2:end]
    G = X * X'; A = Z * X' / (G + ridge * tr(G) / size(U, 2) * I)
    F = schur(complex(A))
    T = F.T; for i in axes(T, 1); abs(T[i, i]) > 1 && (T[i, i] /= abs(T[i, i])); end
    A = real(F.Z * T * F.Z')
    function forecast(Ytr, n)
        h = vec(permutedims(Ytr[end-τ:end, :])); α = U' * h
        out = zeros(n, d)
        for k in 1:n
            α = A * α; out[k, :] = (U * α)[τ*d+1:end]
        end
        out
    end
    return forecast
end

function fit_esn(Y; N = 400, ρ = 0.9, σin = 0.5, α = 0.5, λ = 1e-6, seed = 1, washout = 100)
    rng = MersenneTwister(seed); d = size(Y, 2)
    W = sprandn(rng, N, N, 10 / N); W .*= ρ / maximum(abs.(eigvals(Matrix(W))))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    step(r, u) = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * u .+ b)
    n = size(Y, 1); Rs = zeros(N, n); r = zeros(N)
    for t in 1:n; r = step(r, Y[t, :]); Rs[:, t] = r; end
    F = vcat(Rs[:, washout:n-1], Y[washout:n-1, :]', ones(1, n - washout))
    Wout = Y[washout+1:n, :]' * F' / (F * F' + λ * I)
    function forecast(Ytr, nf)
        r = Rs[:, end-1]; u = Ytr[end, :]; out = zeros(nf, d)
        for k in 1:nf
            r = step(r, u); u = Wout * vcat(r, u, 1.0); out[k, :] = u
            if any(!isfinite, u) || maximum(abs, u) > 1e6
                out[k:end, :] .= NaN; break
            end
        end
        out
    end
    return forecast
end

function fit_node(Y, dt; seed = 1, hidden = 64, epochs = 300, lr = 2e-3, shoot = 25)
    n_t, d = size(Y); t = collect((0:n_t-1) .* dt)
    model = Lux.Chain(Lux.Dense(d, hidden, tanh), Lux.Dense(hidden, hidden, tanh),
                      Lux.Dense(hidden, d))
    ps, st = Lux.setup(Lux.replicate(MersenneTwister(seed)), model)
    p0 = ComponentArray(Float64.(ComponentArray(ps)))
    dudt(u, p, _) = first(model(u, p, st))
    # equal-length segments; reverse-mode adjoint (as in gait_overparam.jl)
    starts = collect(1:shoot:(n_t - shoot))
    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())
    function loss(p, _)
        tot = sum(map(starts) do s0
            s1 = s0 + shoot
            sol = solve(ODEProblem(dudt, Y[s0, :], (t[s0], t[s1]), p), Tsit5();
                        saveat = t[s0:s1], abstol = 1e-6, reltol = 1e-6, sensealg = sensealg)
            sum(abs2, permutedims(Array(sol)) .- Y[s0:s1, :])
        end)
        tot / (length(starts) * (shoot + 1) * d)
    end
    hist = Float64[]
    sol = solve(OptimizationProblem(OptimizationFunction(loss, Optimization.AutoZygote()), p0),
                OptimizationOptimisers.Adam(lr); maxiters = epochs,
                callback = (s, l) -> (push!(hist, l); false))
    pf = sol.u
    function forecast(Ytr, n)
        s = solve(ODEProblem((u, p, _) -> first(model(u, pf, st)), Ytr[end, :], (0.0, n * dt)),
                  Tsit5(); saveat = dt, abstol = 1e-7, reltol = 1e-7, maxiters = 10^6)
        A = permutedims(Array(s))
        (s.retcode == ReturnCode.Success && size(A, 1) >= n + 1) || return fill(NaN, n, d)
        A[2:n+1, :]
    end
    return forecast, hist[end]
end

"Phase-aligned mean-cycle template (reference, not a dynamical model)."
function fit_template(Y, P)
    n, d = size(Y); nb = max(10, min(100, floor(Int, P)))
    b = min.(floor.(Int, mod.((0:n-1) ./ P, 1) .* nb), nb - 1) .+ 1
    T = [mean(Y[b .== k, j]) for k in 1:nb, j in 1:d]
    T = (circshift(T, 1) .+ T .+ circshift(T, -1)) ./ 3
    at(φ) = (x = mod(φ, 1) * nb - 0.5; i0 = floor(Int, x); w = x - i0;
             (1 - w) .* T[mod(i0, nb) + 1, :] .+ w .* T[mod(i0 + 1, nb) + 1, :])
    L = round(Int, P); rel = ((0:L-1) .- (L - 1)) ./ P
    seg = Y[end-L+1:end, :]
    sh = range(0, 1, length = 401)[1:end-1]
    φ0 = sh[argmin([mean(abs2, reduce(hcat, [at(s + r) for r in rel])' .- seg) for s in sh])]
    return (Ytr, nf) -> reduce(hcat, [at(φ0 + k / P) for k in 1:nf])'
end

# ═════════════════════════════════════════════════════════════════════════════
# Scoring
# ═════════════════════════════════════════════════════════════════════════════
function score(Fc, Yte, Ytr, P)
    errs = map((1, 3, 5)) do h
        n = min(round(Int, h * P), size(Yte, 1))
        any(!isfinite, Fc[1:n, :]) ? Inf :
            sqrt(mean(abs2, Fc[1:n, :] .- Yte[1:n, :])) / sqrt(mean(abs2, Yte[1:n, :]))
    end
    # long-run behaviour: last 10 cycles of a 30-cycle free run
    tail = Fc[end-round(Int, 10P)+1:end, :]
    amp_data = mean(std(Ytr, dims = 1))
    if any(!isfinite, tail)
        return (; e1 = errs[1], e3 = errs[2], e5 = errs[3], amp = Inf, per = NaN, fate = "blow-up")
    end
    amp = mean(std(tail, dims = 1)) / amp_data
    per = amp > 0.05 ? period_samples(tail) / P : NaN
    rhythm = isfinite(per) && abs(per - 1) < 0.1
    fate = amp > 4 ? "blow-up" : amp < 0.25 ? "collapse" :
           !rhythm || amp > 1.33 ? "wrong" : amp >= 0.75 ? "ok" : "decays"
    return (; e1 = errs[1], e3 = errs[2], e5 = errs[3], amp, per, fate)
end

const REGIMES = Dict("long" => (n_cyc = 30, σm = 0.0, σp = 0.0),
                     "realistic" => (n_cyc = 12, σm = 0.01, σp = 0.05))

function run_cell(system, regime; models = ["dmd", "esn", "node", "template"])
    rg = REGIMES[regime]
    is_ode = haskey(ODE_SYSTEMS, system)
    X, dt = load_system(system, rg.n_cyc, is_ode ? rg.σp : 0.0)
    P = period_samples(X)
    n_tr = startswith(system, "gait") ? round(Int, 10 / dt) :    # 10 s, as elsewhere
           min(round(Int, rg.n_cyc * P), size(X, 1) - round(Int, N_TEST_CYC * P) - 1)
    μ = mean(X[1:n_tr, :], dims = 1); s = std(X[1:n_tr, :], dims = 1)
    Y = (X .- μ) ./ s
    Y .+= rg.σm .* randn(MersenneTwister(11), size(Y))
    Ytr = Y[1:n_tr, :]; Yte = Y[n_tr+1:end, :]
    n_run = round(Int, N_LONG_CYC * P)
    @info "cell" system regime n_tr P dt n_test = size(Yte, 1); flush(stderr)
    if "export" in models          # identical data for the Python SSMLearnPy run
        npzwrite(cache_path("selfdrive/data_$(replace(system, ":" => ""))_$(regime).npz"),
                 Dict("Ytr" => Ytr, "Yte" => Yte, "P" => P, "dt" => dt, "n_run" => n_run))
        return nothing
    end

    rows = NamedTuple[]
    add!(model, variant, Fc; loss = NaN) = begin
        sc = score(Fc, Yte, Ytr, P)
        push!(rows, (; system, regime, model, variant, sc..., loss,
                       Fc = Float32.(Fc[1:min(end, round(Int, 8P)), :])))
        @info model variant sc.e1 sc.e3 sc.fate sc.amp sc.per; flush(stderr)
    end
    "template" in models && add!("template", "", fit_template(Ytr, P)(Ytr, n_run))
    if "dmd" in models
        τ = max(4, round(Int, P / 10))
        for r in (8, 12, 20)
            add!("DMD", "τ=$τ r=$r", fit_dmd(Ytr; τ, r)(Ytr, n_run))
        end
    end
    if "esn" in models
        for seed in 1:8
            add!("ESN", "seed $seed", fit_esn(Ytr; seed)(Ytr, n_run))
        end
    end
    if "node" in models
        for seed in 1:4
            fc, l = fit_node(Ytr, dt; seed)
            add!("NODE", "seed $seed", fc(Ytr, n_run); loss = l)
        end
    end
    return (; rows, Ytr, Yte = Yte[1:min(end, round(Int, 8P)), :], P, dt)
end

if abspath(PROGRAM_FILE) == @__FILE__
    system = length(ARGS) >= 1 ? ARGS[1] : "fhn"
    regime = length(ARGS) >= 2 ? ARGS[2] : "realistic"
    models = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["dmd", "esn", "node", "template"]
    mkpath(cache_path("selfdrive"))
    tag = replace(system, ":" => "") * "_" * regime * "_" * join(models, "-")
    out = run_cell(system, regime; models)
    out === nothing && exit(0)
    serialize(cache_path("selfdrive/$tag.jls"), out)
    @printf("\n%-9s %-12s | %6s %6s %6s | %6s %5s  %s\n", "model", "variant",
            "e@1", "e@3", "e@5", "amp", "per", "fate")
    for r in out.rows
        @printf("%-9s %-12s | %6.2f %6.2f %6.2f | %6.2f %5.2f  %s\n", r.model, r.variant,
                r.e1, r.e3, r.e5, r.amp, r.per, r.fate)
    end
end
