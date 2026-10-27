#!/usr/bin/env julia
# Floquet multipliers read out of fitted models, against known truth.
#
# Systems (110 samples per cycle, process noise excites the transverse mode):
#   cfhn  two coupled FitzHugh–Nagumo relaxation oscillators (ε = 0.08,
#         k = 0.02): slow coordination multiplier |μ| = 0.688, others ≈ 0
#   sl    Stuart–Landau tuned to |μ| = 0.6
# Regimes:
#   excited    30 cycles, process noise, no measurement noise
#   realistic  12 cycles, process noise + 1 % measurement noise (SD units)
#
# Readout, the same for every dynamical model: run the model freely until it
# settles, take a state on its own orbit, and multiply the one-step Jacobians
# of the model's own step map around one period to get the monodromy M.
# The eigenvalue nearest 1 is the phase direction; the transverse multiplier is
# the largest |eigenvalue| of what remains.
#   truth  RK4 step of the true vector field (validates the chain readout)
#   NODE   RK4 step of the learned vector field (multiple shooting, 4 seeds)
#   ESN    closed-loop step of the full state (reservoir, output); analytic
#          Jacobian; 8 seeds
# Linear Hankel-DMD has no orbit of its own, so no eigenvalues are read: kick
# the NEWEST delay block in each observed channel, apply A^n for one period n
# (data period), read the response in the newest block, project onto the
# directions the mean cycle does not occupy, and take the largest eigenvalue of
# that response map (cfhn only: for Stuart–Landau the cycle fills the observed
# plane and nothing is left to read).
#
# Usage: julia --project=. scripts/floquet_from_models.jl <system> <regime> <spec>
#   spec ∈ truth | esn | dmd | node:<seed>

include("floquet_maps_benchmark.jl")      # cfhn!, TRUTH, simulate, crossings
using Lux, ComponentArrays, Optimization, OptimizationOptimisers
using Zygote, SciMLSensitivity, SparseArrays, ForwardDiff

const SPC_F = 110
const SL_SIGMA = -log(0.6) / 2

# ═════════════════════════════════════════════════════════════════════════════
# Systems and data
# ═════════════════════════════════════════════════════════════════════════════
sl!(du, u, p, t) = (r2 = u[1]^2 + u[2]^2;
                    du[1] = SL_SIGMA * u[1] - 2π * u[2] - r2 * u[1];
                    du[2] = 2π * u[1] + SL_SIGMA * u[2] - r2 * u[2])

function vf_true(sys)
    f! = sys == "cfhn" ? cfhn! : sl!
    return u -> (du = similar(u); f!(du, u, nothing, 0.0); du)
end
sys_dt(sys) = sys == "cfhn" ? TRUTH.T / SPC_F : 1 / SPC_F
sys_truth(sys) = sys == "cfhn" ? TRUTH.μ_slow : 0.6

function sl_simulate(n_cyc; σp, seed)
    dt = sys_dt("sl"); burn = 40
    g!(du, u, p, t) = (du .= σp)
    n = (n_cyc + 2) * SPC_F
    sol = solve(SDEProblem(sl!, g!, [sqrt(SL_SIGMA), 0.0], (0.0, (burn + n_cyc + 2) * 1.0); seed = seed),
                EM(); dt = dt / 20, saveat = dt)
    X = permutedims(Array(sol))
    return X[burn*SPC_F+1:burn*SPC_F+n, :]
end

const REGIMES_F = Dict("excited" => (n_cyc = 30, σm = 0.0), "realistic" => (n_cyc = 12, σm = 0.01))

"Standardised training data (n_tr × d) for a (system, regime)."
function make_data(sys, regime; seed = 1)
    rg = REGIMES_F[regime]
    X = sys == "cfhn" ? simulate(rg.n_cyc; σp = SIGMA_P, seed) :
                        sl_simulate(rg.n_cyc; σp = 0.02 * sqrt(SL_SIGMA), seed)
    Y = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)
    Y .+= rg.σm .* randn(MersenneTwister(seed + 5), size(Y))
    return Y[1:rg.n_cyc*SPC_F, :]
end

# ═════════════════════════════════════════════════════════════════════════════
# Chain readout
# ═════════════════════════════════════════════════════════════════════════════
rk4(f, u, h) = (k1 = f(u); k2 = f(u .+ h / 2 .* k1); k3 = f(u .+ h / 2 .* k2);
                k4 = f(u .+ h .* k3); u .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4))
step_of_vf(f, dt; m = 4) = u -> (for _ in 1:m; u = rk4(f, u, dt / m); end; u)

"Monodromy over n steps from s0: product of step Jacobians; also the closure error."
function chain(step, jac, s0, n, d)
    M = Matrix{Float64}(I, length(s0), length(s0)); s = copy(s0)
    for _ in 1:n
        M = jac(s) * M; s = step(s)
    end
    return M, norm(s[1:d] .- s0[1:d]) / sqrt(d)
end

"Transverse multiplier from a monodromy: drop the eigenvalue nearest 1."
function transverse(M)
    ev = eigvals(M); m = abs.(ev); i = argmin(abs.(ev .- 1))
    rest = sort(m[setdiff(eachindex(m), i)], rev = true)
    return (; μ = rest[1], μ2 = length(rest) > 1 ? rest[2] : NaN, phase = abs(ev[i]),
              n_unstable = count(>(1.02), rest))
end

"Validate the readout on the true vector field."
function truth_row(sys)
    f = vf_true(sys); dt = sys_dt(sys); step = step_of_vf(f, dt)
    s0 = sys == "cfhn" ? TRUTH.u0 : [sqrt(SL_SIGMA), 0.0]
    for _ in 1:3000; s0 = step(s0); end
    # period from crossings of channel 1 (mid-level of one settled stretch)
    traj = reduce(hcat, accumulate((s, _) -> step(s), 1:4 * SPC_F; init = s0))
    c = cross_idx(traj[1, :])
    n = round(Int, mean(diff(c)))
    M, clos = chain(step, s -> ForwardDiff.jacobian(step, s), s0, n, length(s0))
    return (; transverse(M)..., period = mean(diff(c)), closure = clos)
end

"Up-crossings of a channel through its mid-level (fractional indices)."
function cross_idx(x)
    lev = (maximum(x) + minimum(x)) / 2
    [(i - 1) + (lev - x[i-1]) / (x[i] - x[i-1]) for i in 2:length(x) if x[i-1] < lev <= x[i]]
end

# ═════════════════════════════════════════════════════════════════════════════
# Models
# ═════════════════════════════════════════════════════════════════════════════
"Echo state network; returns everything needed to run and linearise it."
function fit_esn_full(Y; N = 800, ρ = 0.9, α = 0.2, σin = 0.5, λ = 1e-3, seed = 1, washout = 100)
    rng = MersenneTwister(seed); d = size(Y, 2)
    W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= ρ / maximum(abs.(eigvals(W)))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    n = size(Y, 1); Rs = zeros(N, n); r = zeros(N)
    for t in 1:n
        r = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * Y[t, :] .+ b); Rs[:, t] = r
    end
    F = vcat(Rs[:, washout:n-1], Y[washout:n-1, :]', ones(1, n - washout))
    Wout = Y[washout+1:n, :]' * F' / (F * F' + λ * I)
    return (; W, Win, b, α, N, d, Wr = Wout[:, 1:N], Wu = Wout[:, N+1:N+d], c = Wout[:, end],
              s0 = vcat(Rs[:, end-1], Y[end, :]), Rs)
end
function esn_step(m)
    return s -> begin
        r = s[1:m.N]; u = s[m.N+1:end]
        r2 = (1 - m.α) .* r .+ m.α .* tanh.(m.W * r .+ m.Win * u .+ m.b)
        vcat(r2, m.Wr * r2 .+ m.Wu * u .+ m.c)
    end
end
function esn_jac(m)
    return s -> begin
        r = s[1:m.N]; u = s[m.N+1:end]
        D = 1 .- tanh.(m.W * r .+ m.Win * u .+ m.b) .^ 2
        Jrr = (1 - m.α) .* Matrix{Float64}(I, m.N, m.N) .+ m.α .* (D .* m.W)
        Jru = m.α .* (D .* m.Win)
        [Jrr Jru; m.Wr * Jrr (m.Wr * Jru .+ m.Wu)]
    end
end

"Neural ODE (multiple shooting, reverse-mode adjoint), returning the vector field."
function fit_node_vf(Y, dt; seed = 1, hidden = 64, depth = 2, epochs = 300, lr = 2e-3, shoot = 25)
    n_t, d = size(Y); t = collect((0:n_t-1) .* dt)
    layers = Any[Lux.Dense(d, hidden, tanh)]
    for _ in 2:depth; push!(layers, Lux.Dense(hidden, hidden, tanh)); end
    push!(layers, Lux.Dense(hidden, d))
    model = Lux.Chain(layers...)
    ps, st = Lux.setup(Lux.replicate(MersenneTwister(seed)), model)
    p0 = ComponentArray(Float64.(ComponentArray(ps)))
    dudt(u, p, _) = first(model(u, p, st))
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
    return (u -> first(model(u, pf, st))), hist[end]
end

# ═════════════════════════════════════════════════════════════════════════════
# Free run, cycle detection, readout
# ═════════════════════════════════════════════════════════════════════════════
const N_FREE = 40 * SPC_F

"Free-run a step map; returns the states (n_state × N_FREE) or nothing on blow-up."
function free_run(step, s0)
    S = zeros(length(s0), N_FREE); s = copy(s0)
    for k in 1:N_FREE
        s = step(s)
        (any(!isfinite, s) || maximum(abs, s) > 1e4) && return nothing
        S[:, k] = s
    end
    S
end

"Classify a free run and read the transverse multiplier from its own orbit."
function read_floquet(step, jac, s0, d)
    S = free_run(step, s0)
    S === nothing && return (; status = "blow-up")
    tail = S[1:d, end-15*SPC_F+1:end]
    amp = mean(std(tail, dims = 2))
    amp < 0.25 && return (; status = "collapse", amp)
    c = cross_idx(tail[1, :])
    length(c) < 4 && return (; status = "no-cycle", amp)
    P = mean(diff(c)); n = round(Int, P)
    istart = size(S, 2) - 15 * SPC_F + round(Int, c[end-1])
    M, clos = chain(step, jac, S[:, istart], n, d)
    return (; status = "cycle", amp, period = P / SPC_F, closure = clos, transverse(M)...)
end

# ── DMD probe (no eigenvalues of A) ─────────────────────────────────────────
function dmd_probe(Y; τ, r, variant, ridge = 1e-8)
    d = size(Y, 2); Hk = build_hankel_multi(Y, τ)
    r = min(r, size(Hk, 1) - 1); U = svd(Hk).U[:, 1:r]
    a = U' * Hk; X, Z = a[:, 1:end-1], a[:, 2:end]
    reg(M) = M + ridge * tr(M) / r * I
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    # directions the mean cycle occupies
    c = cross_idx(Y[:, 1]); nb = 50; ncyc = length(c) - 1
    G = reduce(hcat, [mean(Y[clamp(round(Int, c[k] + (b - 0.5) / nb * (c[k+1] - c[k])), 1, size(Y, 1)), :]
                           for k in 1:ncyc) for b in 1:nb])
    UG, SG, _ = svd(G .- mean(G, dims = 2)); k = count(SG .> 0.02 * SG[1])
    k >= d && return (; status = "no-room", span = k)
    Q = UG[:, k+1:end]
    n = SPC_F; new = τ * d
    B = zeros(size(U, 1), d); for j in 1:d; B[new+j, j] = 1.0; end
    R = (U * (A^n * (U' * B)))[new+1:end, :]
    ev = eigvals(Q' * R * Q)
    return (; status = "probe", μ = maximum(abs.(ev)), span = k)
end

"""
Consistent variational probe for Hankel-DMD. Perturbations are delay windows the
data themselves contain: at each upward crossing (the section) the window of
the last τ+1 samples; deviations from the mean-cycle window span the transverse
subspace in delay coordinates (the along-orbit tangent is projected out).
The model's one-period map A^n is compressed onto that subspace; its largest
|eigenvalue| is the model's transverse multiplier. The same compression of the
data's own cycle-to-cycle deviation map (least squares) is returned for
comparison.
"""
function dmd_probe2(Y; τ, r, variant, ridge = 1e-8, energy = 0.995)
    d = size(Y, 2); Hk = build_hankel_multi(Y, τ)
    r = min(r, size(Hk, 1) - 1); U = svd(Hk).U[:, 1:r]
    a = U' * Hk; X, Z = a[:, 1:end-1], a[:, 2:end]
    reg(M) = M + ridge * tr(M) / r * I
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    c = [ci for ci in cross_idx(Y[:, 1]) if round(Int, ci) - τ >= 1 && round(Int, ci) - τ <= size(Hk, 2)]
    length(c) < 8 && return (; status = "too-few-cycles")
    cols = [round(Int, ci) - τ for ci in c]
    W = Hk[:, cols]; Wn = Hk[:, min.(cols .+ 0, size(Hk, 2))]
    Dm = W .- mean(W, dims = 2)
    tang = mean(Hk[:, min.(cols .+ 1, size(Hk, 2))] .- W, dims = 2); tang ./= norm(tang)
    Dm = Dm .- tang * (tang' * Dm)                       # remove the along-orbit direction
    F = svd(Dm); k = clamp(findfirst(cumsum(F.S .^ 2) ./ sum(F.S .^ 2) .>= energy), 1, 8)
    V = F.U[:, 1:k]
    n = round(Int, mean(diff(c)))
    Mmod = V' * (U * (A^n * (U' * V)))
    # the data's own map: deviation at cycle j -> deviation at cycle j+1 in the same basis
    Dv = V' * Dm
    Mdat = Dv[:, 2:end] * Dv[:, 1:end-1]' / (Dv[:, 1:end-1] * Dv[:, 1:end-1]' + 1e-10I)
    return (; status = "probe2", μ = maximum(abs.(eigvals(Mmod))), μ_data = maximum(abs.(eigvals(Mdat))),
              dim = k, n)
end

# ═════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    sys, regime, spec = ARGS[1], ARGS[2], ARGS[3]
    dseed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1      # data realisation
    dt = sys_dt(sys); d = sys == "cfhn" ? 4 : 2
    mkpath(cache_path("floquet"))
    rows = NamedTuple[]
    tag = replace("$(sys)_$(regime)_$(spec)", ":" => "") * (dseed == 1 ? "" : "_d$(dseed)")
    if spec == "truth"
        push!(rows, (; model = "truth", variant = "", truth_row(sys)...))
    else
        Y = make_data(sys, regime; seed = dseed)
        if spec == "esn1"            # one ESN per data realisation
            m = fit_esn_full(Y; seed = dseed)
            push!(rows, (; model = "ESN", variant = "data $dseed",
                         read_floquet(esn_step(m), esn_jac(m), m.s0, d)...))
        elseif spec == "dmd2ls"
            P = SPC_F
            for τ in (P ÷ 2, P), r in (16, 30, 50, 80)
                o = dmd_probe2(Y; τ, r, variant = "ls")
                push!(rows, (; model = "DMD", variant = "τ=$τ r=$r ls", o...))
            end
        elseif spec == "esn"
            for seed in 1:8
                m = fit_esn_full(Y; seed)
                push!(rows, (; model = "ESN", variant = "seed $seed",
                             read_floquet(esn_step(m), esn_jac(m), m.s0, d)...))
                @info rows[end]; flush(stderr)
            end
        elseif startswith(spec, "node")
            seed = parse(Int, split(spec, ":")[2])
            f, l = fit_node_vf(Y, dt; seed)
            step = step_of_vf(f, dt)
            r = read_floquet(step, s -> ForwardDiff.jacobian(step, s), Y[end, :], d)
            push!(rows, (; model = "NODE", variant = "seed $seed", loss = l, r...))
            @info rows[end]
        elseif spec == "dmd2" && sys == "cfhn"
            P = SPC_F
            for τ in (P ÷ 4, P ÷ 2, P), r in (16, 30, 50, 80), variant in ("ls", "fb")
                o = dmd_probe2(Y; τ, r, variant)
                push!(rows, (; model = "DMD", variant = "τ=$τ r=$r $variant", o...))
            end
        elseif spec == "dmd" && sys == "cfhn"
            P = SPC_F
            for τ in (P ÷ 4, P ÷ 2), r in (16, 30, 50), variant in ("ls", "fb")
                o = dmd_probe(Y; τ, r, variant)
                push!(rows, (; model = "DMD", variant = "τ=$τ r=$r $variant", o...))
            end
        end
    end
    serialize(cache_path("floquet/models_$tag.jls"), (; sys, regime, truth = sys_truth(sys), rows))
    for r in rows
        println(r)
    end
end
