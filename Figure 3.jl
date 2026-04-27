#!/usr/bin/env julia
# Figure 3: Neural ODE Failure — Limit Cycle with Transients
# Shows that when training data includes perturbations away from the orbit,
# the neural ODE still cannot learn the transverse contraction dynamics.
#
# 4 panels:
#   (a) Training data: Stuart-Landau trajectories from perturbed ICs
#   (b) Learned vs true vector field
#   (c) Forward simulation from perturbed IC (spiral-in failure)
#   (d) Floquet multiplier estimates from neural ODE Jacobian vs truth

include("common.jl")

using Lux, ComponentArrays, Optimization, OptimizationOptimisers
using OrdinaryDiffEq: Tsit5, ODEProblem, solve
using ForwardDiff

# ── Generate Stuart-Landau training data with transients ─────────────────────
const SL3_σ  = 0.1
const SL3_ω  = 2π
const SL3_dt = 0.02
const SL3_T  = 5.0      # shorter trajectories, more ICs
const N_ICS  = 25

rng_ic = MersenneTwister(42)

# Generate ICs: some on the cycle, some perturbed inward/outward
r_cycle = sqrt(SL3_σ)
initial_conditions = ComplexF64[]
for _ in 1:N_ICS
    θ = 2π * rand(rng_ic)
    r = r_cycle * (1.0 + 0.5 * randn(rng_ic))  # ±50% radial perturbation
    r = max(r, 0.05)  # keep positive
    push!(initial_conditions, r * exp(im * θ))
end

# Simulate all trajectories
all_trajectories = @cached cache_path("fig3_trajectories.jls") begin
    trajs = []
    for (i, z0) in enumerate(initial_conditions)
        t, x, y = simulate_stuart_landau(σ=SL3_σ, ω=SL3_ω, T=SL3_T, dt=SL3_dt,
                                          z0=z0, noise=0.005)
        push!(trajs, (; t, x, y))
    end
    trajs
end

# ── Train Neural ODE on collection of trajectories ──────────────────────────
function train_node_sl(; hidden=64, n_layers=2, seed=1, lr=5e-4, epochs=200,
                         shoot_len=15)
    rng_lux = Lux.replicate(Random.MersenneTwister(seed))
    model = Lux.Chain(Lux.Dense(2, hidden, tanh),
                      Lux.Dense(hidden, hidden, tanh),
                      Lux.Dense(hidden, 2))
    ps, st = Lux.setup(rng_lux, model)
    ps_ca = ComponentArray(Float64.(ComponentArray(ps)))

    function dudt(u, p, t)
        pred, _ = model(u, p, st)
        return pred
    end

    # Collect all trajectory segments for training
    function loss(p_vec, _)
        total_loss = 0.0
        n_points = 0
        for traj in all_trajectories
            t_traj = traj.t
            data_traj = hcat(traj.x, traj.y)
            n_t = length(t_traj)
            for seg_start in 1:shoot_len:(n_t - 1)
                seg_end = min(seg_start + shoot_len, n_t)
                tspan = (t_traj[seg_start], t_traj[seg_end])
                u0_seg = Float64.(data_traj[seg_start, :])
                prob = ODEProblem(dudt, u0_seg, tspan, p_vec)
                sol = solve(prob, Tsit5(); saveat=t_traj[seg_start:seg_end],
                            abstol=1e-6, reltol=1e-6)
                if sol.retcode != :Success && sol.retcode != ReturnCode.Success
                    return Inf
                end
                for (k, idx) in enumerate(seg_start:seg_end)
                    pred = sol[:, k]
                    target = Float64.(data_traj[idx, :])
                    total_loss += sum((pred .- target).^2)
                    n_points += 1
                end
            end
        end
        return total_loss / max(n_points, 1)
    end

    loss_history = Float64[]
    callback = function(state, l)
        push!(loss_history, l)
        return false
    end

    opt_f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_f, ps_ca)
    sol = solve(opt_prob, OptimizationOptimisers.Adam(lr);
                maxiters=epochs, callback=callback)

    return sol.u, st, model, loss_history
end

# ── Run training ─────────────────────────────────────────────────────────────
trained = @cached cache_path("fig3_node_trained.jls") begin
    ps_final, st_final, model_final, loss_hist = train_node_sl(;
        hidden=64, n_layers=2, seed=42, lr=5e-4, epochs=200)
    (; ps=ps_final, st=st_final, loss_history=loss_hist)
end

# Reconstruct model for evaluation
model_eval = Lux.Chain(Lux.Dense(2, 64, tanh),
                       Lux.Dense(64, 64, tanh),
                       Lux.Dense(64, 2))
rng_eval = Lux.replicate(Random.MersenneTwister(42))
_, st_eval = Lux.setup(rng_eval, model_eval)

function learned_dudt(u, p, t)
    pred, _ = model_eval(u, p, st_eval)
    return pred
end

# True vector field for Stuart-Landau
function true_dudt(x, y)
    z = x + im * y
    dz = (SL3_σ + im * SL3_ω) * z - z * abs2(z)
    return real(dz), imag(dz)
end

# ── Panel (a): Training trajectories ────────────────────────────────────────
θ_lc = range(0, 2π, length=200)
p_a = plot(r_cycle .* cos.(θ_lc), r_cycle .* sin.(θ_lc);
           lw=2, ls=:dash, color=:black, label="Limit cycle",
           xlabel="x", ylabel="y", aspect_ratio=:equal,
           title="(A) Training trajectories",
           size=(PUB_W, PUB_H), xlims=(-0.6, 0.6), ylims=(-0.6, 0.6))
for (i, traj) in enumerate(all_trajectories)
    plot!(p_a, traj.x, traj.y; lw=0.6, alpha=0.5,
          color=RGB(0.27, 0.51, 0.71), label=(i == 1 ? "Trajectories" : ""))
end

# ── Panel (b): Vector field comparison ──────────────────────────────────────
grid_pts = range(-0.55, 0.55, length=15)
arrows_true_dx = Float64[]
arrows_true_dy = Float64[]
arrows_learned_dx = Float64[]
arrows_learned_dy = Float64[]
arrow_x = Float64[]
arrow_y = Float64[]

for gx in grid_pts, gy in grid_pts
    push!(arrow_x, gx)
    push!(arrow_y, gy)
    dx_t, dy_t = true_dudt(gx, gy)
    push!(arrows_true_dx, dx_t)
    push!(arrows_true_dy, dy_t)
    u_in = Float64[gx, gy]
    pred, _ = model_eval(u_in, trained.ps, st_eval)
    push!(arrows_learned_dx, pred[1])
    push!(arrows_learned_dy, pred[2])
end

# Normalize arrows for display
scale_true = 0.03 / max(maximum(abs.(arrows_true_dx)), maximum(abs.(arrows_true_dy)), 1e-10)
scale_learned = 0.03 / max(maximum(abs.(arrows_learned_dx)), maximum(abs.(arrows_learned_dy)), 1e-10)

p_b = plot(; xlabel="x", ylabel="y", aspect_ratio=:equal,
             title="(B) Vector fields: true vs learned",
             size=(PUB_W, PUB_H), xlims=(-0.6, 0.6), ylims=(-0.6, 0.6))
quiver!(p_b, arrow_x, arrow_y;
        quiver=(arrows_true_dx .* scale_true, arrows_true_dy .* scale_true),
        color=:black, lw=0.8, label="True")
quiver!(p_b, arrow_x, arrow_y;
        quiver=(arrows_learned_dx .* scale_learned, arrows_learned_dy .* scale_learned),
        color=RGB(0.86, 0.08, 0.24), lw=0.8, alpha=0.7, label="Learned")
plot!(p_b, r_cycle .* cos.(θ_lc), r_cycle .* sin.(θ_lc);
      lw=1.5, ls=:dash, color=:gray, label="")

# ── Panel (c): Forward simulation from perturbed IC ─────────────────────────
# Pick a perturbed IC: 1.4× the limit cycle radius
z0_test = 1.4 * r_cycle * exp(im * 0.0)
u0_test = [real(z0_test), imag(z0_test)]
T_pred = 10.0

# True trajectory
t_true, x_true_fwd, y_true_fwd = simulate_stuart_landau(
    σ=SL3_σ, ω=SL3_ω, T=T_pred, dt=SL3_dt, z0=z0_test)

# Neural ODE prediction
prob_pred = ODEProblem(learned_dudt, u0_test, (0.0, T_pred), trained.ps)
sol_pred = solve(prob_pred, Tsit5(); saveat=SL3_dt, abstol=1e-6, reltol=1e-6,
                 maxiters=1_000_000)

p_c = plot(x_true_fwd, y_true_fwd; lw=2, color=:black, label="True",
           xlabel="x", ylabel="y", aspect_ratio=:equal,
           title="(C) Forward sim from perturbed IC",
           size=(PUB_W, PUB_H))
if sol_pred.retcode == :Success || sol_pred.retcode == ReturnCode.Success
    pred_x_fwd = [sol_pred[1, i] for i in 1:length(sol_pred.t)]
    pred_y_fwd = [sol_pred[2, i] for i in 1:length(sol_pred.t)]
    plot!(p_c, pred_x_fwd, pred_y_fwd; lw=2, color=RGB(0.86, 0.08, 0.24),
          ls=:dash, label="Neural ODE")
else
    annotate!(p_c, 0, 0, text("Solver failed", 8, :red))
end
plot!(p_c, r_cycle .* cos.(θ_lc), r_cycle .* sin.(θ_lc);
      lw=1, ls=:dot, color=:gray, label="Cycle")
scatter!(p_c, [u0_test[1]], [u0_test[2]]; ms=6, color=:green, label="IC")

# ── Panel (d): Floquet multiplier comparison ────────────────────────────────
# Extract Jacobian of learned vector field at a point on the limit cycle
# and integrate around one cycle to estimate monodromy matrix

function learned_jacobian_at(x_pt, y_pt, ps)
    u_in = [x_pt, y_pt]
    J = ForwardDiff.jacobian(u -> begin
        pred, _ = model_eval(u, ps, st_eval)
        return pred
    end, u_in)
    return J
end

# Numerical Floquet via learned field: integrate variational equation
T_period = 2π / SL3_ω
n_cycle_pts = round(Int, T_period / SL3_dt)

# Sample points around the limit cycle
θ_cycle = range(0, 2π, length=n_cycle_pts+1)[1:end-1]
x_cycle = r_cycle .* cos.(θ_cycle)
y_cycle = r_cycle .* sin.(θ_cycle)

# Accumulate Jacobian product of the discrete-time map
# For continuous-time: M ≈ prod(I + dt*J(x_k)) over one cycle
M_learned = Matrix{Float64}(I, 2, 2)
for k in 1:n_cycle_pts
    J_k = learned_jacobian_at(x_cycle[k], y_cycle[k], trained.ps)
    M_learned = (I + SL3_dt * J_k) * M_learned
end
μ_learned = eigvals(M_learned)

# Ground truth
μ_true_phase, μ_true_trans = ground_truth_floquet_sl(σ=SL3_σ, ω=SL3_ω)

p_d = plot(; xlabel="Re(μ)", ylabel="Im(μ)", aspect_ratio=:equal,
             title="(D) Floquet multiplier estimates",
             size=(PUB_W, PUB_H))
plot!(p_d, cos.(θ_lc), sin.(θ_lc); color=:gray, ls=:dash, lw=0.5, label="Unit circle")
scatter!(p_d, [μ_true_trans, μ_true_phase], [0.0, 0.0];
         ms=8, color=:black, markershape=:diamond, label="True μ")
scatter!(p_d, real.(μ_learned), imag.(μ_learned);
         ms=6, color=RGB(0.86, 0.08, 0.24), label="Neural ODE μ")
annotate!(p_d, μ_true_trans + 0.05, 0.1,
          text("μ_trans=$(round(μ_true_trans; digits=3))", 7, :left))

# ── Compose ──────────────────────────────────────────────────────────────────
fig3 = plot(p_a, p_b, p_c, p_d; layout=(2, 2), size=(2PUB_W, 2PUB_H))
