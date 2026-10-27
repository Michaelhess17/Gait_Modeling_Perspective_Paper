#!/usr/bin/env julia
# Can a data-driven model tell you WHO DRIVES WHOM between two oscillators?
#
# Ground truth: two detuned Van der Pol oscillators with directed forcing
# coupling (A drives B with gain kBA, B drives A with gain kAB):
#   ẍ_A − μ(1 − x_A²)ẋ_A + ω_A² x_A = kAB x_B
#   ẍ_B − μ(1 − x_B²)ẋ_B + ω_B² x_B = kBA x_A
# With ω_B/ω_A = 1.05 and k ≥ 0.1 the coupled pair phase-locks 1:1, like limbs;
# uncoupled it drifts (quasi-periodic). At dt = 0.05 one cycle is ~130 samples,
# close to a gait stride sampled at 100 Hz.
#
# Every model is read out the SAME way — a model-agnostic perturbation test:
# kick position channel j of the learned model by ε (in SD units) at an
# on-orbit state, run the model forward, and measure the RMS deviation of
# position channel i over the horizon, averaged over phases along the orbit.
# The true system's response to the same kick is the ground truth.
#   R[i, j]   response of oscillator i to a kick in oscillator j
#   AI = (R[B,A] − R[A,B]) / (R[B,A] + R[A,B])   directional asymmetry (+1: A→B)
#   CI = (R[B,A] + R[A,B]) / (R[A,A] + R[B,B])   cross-talk relative to self-response
#
# Models: Hankel-DMD (nuisance variation = delays × rank), echo state network
# (nuisance = reservoir seed), neural ODE with multiple shooting (nuisance =
# init seed). Each also gets a held-out free-run forecast score, so prediction
# skill and coupling recovery can be compared directly.
#
# Usage: julia --project=. scripts/coupling_benchmark.jl <condition> <config> [models]
#   condition ∈ keys(CONDITIONS), config ∈ keys(CONFIGS), models e.g. "dmd,esn,node"

include("common.jl")
using Lux, ComponentArrays, Optimization, OptimizationOptimisers
using SparseArrays

# ═════════════════════════════════════════════════════════════════════════════
# Ground-truth system
# ═════════════════════════════════════════════════════════════════════════════
const MU, OMEGA_A, OMEGA_B = 1.0, 1.0, 1.05
const DT = 0.05
const T_CYCLE = 2π / OMEGA_A * 1.03          # ≈ VdP(μ=1) period for ω = 1 (≈ 6.66 s)
const CYC = round(Int, T_CYCLE / DT)         # samples per cycle (≈ 133)

const CONFIGS = Dict(
    "none"    => (kAB = 0.0, kBA = 0.0),
    "AtoB"    => (kAB = 0.0, kBA = 0.2),
    "BtoA"    => (kAB = 0.2, kBA = 0.0),
    "bi"      => (kAB = 0.2, kBA = 0.2),
    "AtoB_k05" => (kAB = 0.0, kBA = 0.05),
    "AtoB_k10" => (kAB = 0.0, kBA = 0.1),
    "AtoB_k40" => (kAB = 0.0, kBA = 0.4),
)

# n_cyc: training cycles; σm: measurement noise (SD units); σp: process noise
# (stochastic forcing on both velocities); obs: :full (x, ẋ for both) or :pos
const CONDITIONS = Dict(
    "ideal"     => (n_cyc = 50, σm = 0.0,   σp = 0.0,  obs = :full),
    "gaitlike"  => (n_cyc = 12, σm = 0.004, σp = 0.05, obs = :full),
    "short"     => (n_cyc = 12, σm = 0.0,   σp = 0.0,  obs = :full),
    "noisy"     => (n_cyc = 50, σm = 0.05,  σp = 0.0,  obs = :full),
    "process"   => (n_cyc = 50, σm = 0.0,   σp = 0.1,  obs = :full),
    "partial"   => (n_cyc = 50, σm = 0.0,   σp = 0.0,  obs = :pos),
    "realistic" => (n_cyc = 12, σm = 0.01,  σp = 0.05, obs = :pos),
    # is direction recoverable once the data leave the attractor?
    "excited"      => (n_cyc = 50,  σm = 0.0, σp = 0.3, obs = :full),
    "excited_long" => (n_cyc = 200, σm = 0.0, σp = 0.3, obs = :full),
    # deterministic, but each oscillator is kicked at random times (like
    # treadmill perturbations): ~one kick every 2 cycles, random target
    "kicked"       => (n_cyc = 50,  σm = 0.0, σp = 0.0, obs = :full, kicks = true),
    "kicked_gait"  => (n_cyc = 12,  σm = 0.004, σp = 0.05, obs = :full, kicks = true),
)

function cvdp!(du, u, p, t)
    kAB, kBA = p
    xA, vA, xB, vB = u
    du[1] = vA; du[2] = MU * (1 - xA^2) * vA - OMEGA_A^2 * xA + kAB * xB
    du[3] = vB; du[4] = MU * (1 - xB^2) * vB - OMEGA_B^2 * xB + kBA * xA
end
function noise!(du, u, p, t)
    σp = p[3]
    du[1] = 0; du[2] = σp; du[3] = 0; du[4] = σp
end

"Simulate n samples after a 300-s burn-in; returns the 4-D state (n × 4)."
function simulate(cfg, n; σp = 0.0, seed = 1, kicks = false)
    u0 = [2.0, 0.0, -1.0, 0.5]
    burn = 300.0; T = burn + n * DT
    p = (cfg.kAB, cfg.kBA, σp)
    cb = nothing
    if kicks
        rng = MersenneTwister(seed + 100)
        tk = sort(burn .+ rand(rng, round(Int, n / (2CYC))) .* (n * DT))
        targ = rand(rng, (2, 4), length(tk)); amp = 0.8 .* randn(rng, length(tk))
        kick!(integ) = (k = findfirst(≈(integ.t), tk);
                        k === nothing || (integ.u[targ[k]] += amp[k]))
        cb = PresetTimeCallback(tk, kick!; save_positions = (false, false))
    end
    sol = if σp > 0
        # fixed-step Euler–Maruyama (additive noise); adaptive SDE solvers
        # stall over the long burn-in
        solve(SDEProblem(cvdp!, noise!, u0, (0.0, T), p; seed = seed), EM();
              dt = DT / 10, saveat = DT, maxiters = 10^8, callback = cb)
    else
        solve(ODEProblem(cvdp!, u0, (0.0, T), p), Tsit5(); saveat = DT,
              abstol = 1e-10, reltol = 1e-10, callback = cb)
    end
    X = permutedims(Array(sol))
    i0 = round(Int, burn / DT) + 1
    return X[i0:i0+n-1, :]
end

const POS_FULL = [1, 3]     # position channels within the full observation
const POS_PART = [1, 2]     # ... within the position-only observation

# ═════════════════════════════════════════════════════════════════════════════
# The common readout
# ═════════════════════════════════════════════════════════════════════════════
const EPS = 0.1             # kick size, SD units
const H = CYC ÷ 2           # response horizon: half a cycle
const N_PHASE = 20

"Assemble R and the two indices from per-phase deviation traces."
function indices(R)
    AI = (R[2, 1] - R[1, 2]) / (R[2, 1] + R[1, 2])
    CI = (R[2, 1] + R[1, 2]) / (R[1, 1] + R[2, 2])
    return (; R, AI, CI)
end

"Ground truth: kick x_j of the TRUE (noise-free) system at on-orbit states."
function true_response(cfg, sd_pos)
    X = simulate(cfg, 4CYC)
    starts = round.(Int, range(1, 2CYC, length = N_PHASE))
    R = zeros(2, 2)
    for s in starts, (jj, j) in enumerate((1, 3))
        u0 = X[s, :]
        up = copy(u0); up[j] += EPS * sd_pos[jj]
        f(u) = permutedims(Array(solve(ODEProblem(cvdp!, u, (0.0, H * DT), (cfg.kAB, cfg.kBA)),
                                       Tsit5(); saveat = DT, abstol = 1e-10, reltol = 1e-10)))
        D = f(up) .- f(u0)
        for (ii, i) in enumerate((1, 3))
            R[ii, jj] += sqrt(mean(abs2, D[2:end, i])) / (EPS * sd_pos[ii]) / N_PHASE
        end
    end
    return indices(R)
end

# ═════════════════════════════════════════════════════════════════════════════
# Model 1: Hankel-DMD
# ═════════════════════════════════════════════════════════════════════════════
function fit_dmd(Y; τ, r, ridge = 1e-6)
    Hk = build_hankel_multi(Y, τ)
    U, Σ, _ = svd(Hk); r = min(r, size(U, 2)); U = U[:, 1:r]
    a = U' * Hk; X, Z = a[:, 1:end-1], a[:, 2:end]
    G = X * X'
    A = Z * X' / (G + ridge * tr(G) / r * I)
    return (; A, U, τ, nch = size(Y, 2))
end
"Newest-block index of channel c in the delay vector."
newest(m, c) = m.τ * m.nch + c

function dmd_response(m, pos)
    R = zeros(2, 2)
    for (jj, j) in enumerate(pos)
        δ = zeros(size(m.U, 1)); δ[newest(m, j)] = EPS
        a = m.U' * δ
        traj = zeros(2, H)
        for k in 1:H
            a = m.A * a
            h = m.U * a
            traj[:, k] = [h[newest(m, i)] for i in pos]
        end
        for ii in 1:2
            R[ii, jj] = sqrt(mean(abs2, traj[ii, :])) / EPS
        end
    end
    return indices(R)
end

"""
Coupling readout used in our DMD papers: input–output resolvent gain at the
stride frequency, G[i, j] = |C_i (zI − A)⁻¹ B_j| with z = 1.02·e^{iω dt}
(evaluated just off the unit circle so neutral poles don't sit on the contour).
B_j injects into channel j of the newest delay block, C_i reads channel i there.
"""
function dmd_resolvent(m, pos, f_stride; radius = 1.02)
    z = radius * exp(im * 2π * f_stride * DT)
    Rz = inv(z * I - m.A)
    G = zeros(2, 2)
    for (jj, j) in enumerate(pos), (ii, i) in enumerate(pos)
        b = m.U[newest(m, j), :]; c = m.U[newest(m, i), :]
        G[ii, jj] = abs(c' * Rz * b)
    end
    return indices(G)
end

"Dominant frequency (Hz) of channel 1 in the training data."
function stride_freq(Y)
    x = Y[:, 1] .- mean(Y[:, 1]); n = length(x)
    P = abs2.([sum(x .* cis.(-2π * k .* (0:n-1) ./ n)) for k in 1:n÷8])
    return argmax(P) / (n * DT)
end

function dmd_forecast(m, Y_hist, n_ahead)
    h = vec(permutedims(Y_hist[end-m.τ:end, :]))     # newest block last
    a = m.U' * h
    out = zeros(n_ahead, m.nch)
    for k in 1:n_ahead
        a = m.A * a; hk = m.U * a
        out[k, :] = hk[m.τ*m.nch+1:end]
    end
    return out
end

# ═════════════════════════════════════════════════════════════════════════════
# Model 2: echo state network (hand-rolled, leaky tanh reservoir, ridge readout)
# ═════════════════════════════════════════════════════════════════════════════
function fit_esn(Y; N = 400, ρ = 0.9, σin = 0.5, α = 0.5, λ = 1e-6, seed = 1, washout = 100)
    rng = MersenneTwister(seed); d = size(Y, 2)
    W = sprandn(rng, N, N, 10 / N)
    W .*= ρ / maximum(abs.(eigvals(Matrix(W))))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    step(r, u) = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * u .+ b)
    n = size(Y, 1); Rs = zeros(N, n); r = zeros(N)
    for t in 1:n
        r = step(r, Y[t, :]); Rs[:, t] = r
    end
    F = vcat(Rs[:, washout:n-1], Y[washout:n-1, :]', ones(1, n - washout))
    T = Y[washout+1:n, :]'
    Wout = T * F' / (F * F' + λ * I)
    readout(r, u) = Wout * vcat(r, u, 1.0)
    return (; step, readout, Rs, N)
end

"Closed-loop run of the ESN for `n` steps from reservoir state r after input u."
function esn_run(m, r, u, n)
    out = zeros(n, length(u))
    for k in 1:n
        r = m.step(r, u)
        u = m.readout(r, u)
        out[k, :] = u
        any(!isfinite, u) && (out[k:end, :] .= NaN; break)
    end
    return out
end

function esn_response(m, Y, pos)
    starts = round.(Int, range(size(Y, 1) - 2CYC, size(Y, 1) - 2, length = N_PHASE))
    R = zeros(2, 2)
    for s in starts, (jj, j) in enumerate(pos)
        r0 = m.Rs[:, s - 1]                 # state after inputs up to s-1
        u0 = Y[s, :]; up = copy(u0); up[j] += EPS
        D = esn_run(m, r0, up, H) .- esn_run(m, r0, u0, H)
        # the kicked input itself is step 0; the response is what the model produces
        for (ii, i) in enumerate(pos)
            R[ii, jj] += sqrt(mean(abs2, D[:, i])) / EPS / N_PHASE
        end
    end
    return indices(R)
end
esn_forecast(m, Y, n) = esn_run(m, m.Rs[:, end-1], Y[end, :], n)

# ═════════════════════════════════════════════════════════════════════════════
# Model 3: neural ODE, multiple shooting (the recipe that works on toy systems)
# ═════════════════════════════════════════════════════════════════════════════
const SHOOT = 25
function fit_node(Y; seed = 1, hidden = 64, epochs = 300, lr = 2e-3)
    n_t, d = size(Y); t = collect((0:n_t-1) .* DT)
    model = Lux.Chain(Lux.Dense(d, hidden, tanh), Lux.Dense(hidden, hidden, tanh),
                      Lux.Dense(hidden, d))
    ps, st = Lux.setup(Lux.replicate(MersenneTwister(seed)), model)
    p0 = ComponentArray(Float64.(ComponentArray(ps)))
    dudt(u, p, _) = first(model(u, p, st))
    starts = 1:SHOOT:(n_t - SHOOT)
    function loss(p, _)
        tot = 0.0
        for s0 in starts
            s1 = s0 + SHOOT
            sol = solve(ODEProblem(dudt, Y[s0, :], (t[s0], t[s1]), p), Tsit5();
                        saveat = t[s0:s1], abstol = 1e-6, reltol = 1e-6)
            sol.retcode == ReturnCode.Success || return Inf
            tot += sum(abs2, permutedims(Array(sol)) .- Y[s0:s1, :])
        end
        tot / (length(starts) * (SHOOT + 1) * d)
    end
    hist = Float64[]
    sol = solve(OptimizationProblem(OptimizationFunction(loss,
                    Optimization.AutoForwardDiff(; chunksize = 32)), p0),
                OptimizationOptimisers.Adam(lr); maxiters = epochs,
                callback = (s, l) -> (push!(hist, l); false))
    pf = sol.u
    vf(u) = first(model(u, pf, st))
    return (; vf, loss = hist[end])
end

function node_run(m, u0, n)
    sol = solve(ODEProblem((u, p, t) -> m.vf(u), collect(u0), (0.0, n * DT)), Tsit5();
                saveat = DT, abstol = 1e-8, reltol = 1e-8, maxiters = 10^6)
    sol.retcode == ReturnCode.Success || return fill(NaN, n, length(u0))
    A = permutedims(Array(sol))
    size(A, 1) < n + 1 && return fill(NaN, n, length(u0))
    return A[2:n+1, :]
end

function node_response(m, Y, pos)
    starts = round.(Int, range(size(Y, 1) - 2CYC, size(Y, 1) - 2, length = N_PHASE))
    R = zeros(2, 2)
    for s in starts, (jj, j) in enumerate(pos)
        u0 = Y[s, :]; up = copy(u0); up[j] += EPS
        D = node_run(m, up, H) .- node_run(m, u0, H)
        for (ii, i) in enumerate(pos)
            R[ii, jj] += sqrt(mean(abs2, D[:, i])) / EPS / N_PHASE
        end
    end
    return indices(R)
end
node_forecast(m, Y, n) = node_run(m, Y[end, :], n)

# ═════════════════════════════════════════════════════════════════════════════
# One (condition, config) cell
# ═════════════════════════════════════════════════════════════════════════════
"Held-out forecast error over the first cycle, normalised by the test signal RMS."
forecast_err(P, Yte) = any(!isfinite, P) ? NaN :
    sqrt(mean(abs2, P[1:CYC, :] .- Yte[1:CYC, :])) / sqrt(mean(abs2, Yte[1:CYC, :]))

function run_cell(cond_name, cfg_name; models = ["dmd", "esn", "node"], data_seed = 1)
    cond = CONDITIONS[cond_name]; cfg = CONFIGS[cfg_name]
    n_tr = cond.n_cyc * CYC; n_te = 2CYC
    X = simulate(cfg, n_tr + n_te; σp = cond.σp, seed = data_seed,
                 kicks = get(cond, :kicks, false))
    chans = cond.obs == :full ? [1, 2, 3, 4] : [1, 3]
    pos = cond.obs == :full ? POS_FULL : POS_PART
    Yraw = X[:, chans]
    μ = mean(Yraw[1:n_tr, :], dims = 1); s = std(Yraw[1:n_tr, :], dims = 1)
    Y = (Yraw .- μ) ./ s
    Y .+= cond.σm .* randn(MersenneTwister(data_seed + 7), size(Y))
    Ytr, Yte = Y[1:n_tr, :], Y[n_tr+1:end, :]

    truth = true_response(cfg, vec(s)[pos])
    rows = NamedTuple[]
    push!(rows, (; model = "truth", variant = "", cond = cond_name, cfg = cfg_name,
                 truth.AI, truth.CI, R = truth.R, fc = 0.0, loss = NaN))

    if "dmd" in models
        for τ in (6, 13, 26), r in (6, 10, 16, 24)
            m = fit_dmd(Ytr; τ, r)
            o = dmd_response(m, pos)
            fc = forecast_err(dmd_forecast(m, Ytr, n_te), Yte)
            push!(rows, (; model = "DMD", variant = "τ=$τ r=$r", cond = cond_name, cfg = cfg_name,
                         o.AI, o.CI, o.R, fc, loss = NaN))
        end
    end
    if "dmdx" in models     # extra DMD readouts: resolvent (as in our papers), full rank
        fs = stride_freq(Ytr)
        for τ in (6, 13, 26), r in (6, 10, 16, 24, 0)
            rr = r == 0 ? length(chans) * (τ + 1) : r          # 0 → full rank
            m = fit_dmd(Ytr; τ, r = rr, ridge = r == 0 ? 1e-4 : 1e-6)
            o = dmd_resolvent(m, pos, fs)
            push!(rows, (; model = "DMD-resolvent", variant = "τ=$τ r=$(r == 0 ? "full" : r)",
                         cond = cond_name, cfg = cfg_name, o.AI, o.CI, o.R, fc = NaN, loss = NaN))
            if r == 0
                o2 = dmd_response(m, pos)
                fc = forecast_err(dmd_forecast(m, Ytr, n_te), Yte)
                push!(rows, (; model = "DMD-fullrank", variant = "τ=$τ", cond = cond_name,
                             cfg = cfg_name, o2.AI, o2.CI, o2.R, fc, loss = NaN))
            end
        end
    end
    if "esn" in models
        for seed in 1:8
            m = fit_esn(Ytr; seed)
            o = esn_response(m, Ytr, pos)
            fc = forecast_err(esn_forecast(m, Ytr, n_te), Yte)
            push!(rows, (; model = "ESN", variant = "seed $seed", cond = cond_name, cfg = cfg_name,
                         o.AI, o.CI, o.R, fc, loss = NaN))
        end
    end
    if "node" in models
        for seed in 1:4
            m = fit_node(Ytr; seed)
            o = node_response(m, Ytr, pos)
            fc = forecast_err(node_forecast(m, Ytr, n_te), Yte)
            push!(rows, (; model = "NODE", variant = "seed $seed", cond = cond_name, cfg = cfg_name,
                         o.AI, o.CI, o.R, fc, m.loss))
            @info "node" cond_name cfg_name seed AI = o.AI CI = o.CI fc; flush(stderr)
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    cond_name = length(ARGS) >= 1 ? ARGS[1] : "ideal"
    cfg_name  = length(ARGS) >= 2 ? ARGS[2] : "AtoB"
    models    = length(ARGS) >= 3 ? split(ARGS[3], ",") : ["dmd", "esn", "node"]
    mkpath(cache_path("coupling"))
    out = cache_path("coupling/$(cond_name)_$(cfg_name)_$(join(models, "-")).jls")
    rows = run_cell(cond_name, cfg_name; models)
    serialize(out, rows)
    @printf("\n%-6s %-14s | %6s %6s | %6s\n", "model", "variant", "AI", "CI", "fc@1cyc")
    for r in rows
        @printf("%-6s %-14s | %6.2f %6.2f | %6.2f\n", r.model, r.variant, r.AI, r.CI, r.fc)
    end
end
