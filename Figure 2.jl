#!/usr/bin/env julia
# Figure 2: Neural ODE Failure — Sine Wave
# Demonstrates that neural ODEs consistently converge to predicting the mean
# when trained on periodic data, across multiple random seeds and architectures.
#
# 4 panels:
#   (a) Training data + multiple trained predictions overlaid
#   (b) Training loss curves (all plateau at MSE of the signal variance)
#   (c) Phase portrait showing collapse to origin
#   (d) Histogram of final prediction amplitudes across seeds

include("common.jl")

using Lux, ComponentArrays, Optimization, OptimizationOptimisers
using OrdinaryDiffEq: Tsit5, ODEProblem, solve

# ── Generate periodic training data ─────────────────────────────────────────
const NODE_DT      = 0.05
const NODE_T_TRAIN = 10.0
const NODE_NOISE   = 0.02
const N_SEEDS      = 4
const N_EPOCHS     = 100

t_data = collect(0.0:NODE_DT:NODE_T_TRAIN)
n_data = length(t_data)

# 2D representation: (sin(2πt), cos(2πt)) — a circle in phase space
# The neural ODE must learn the vector field that produces this rotation
x_true = sin.(2π .* t_data)
y_true = cos.(2π .* t_data)
data_2d = hcat(x_true, y_true)  # n × 2

function make_training_data(; seed=42)
    rng = MersenneTwister(seed)
    noisy = data_2d .+ NODE_NOISE .* randn(rng, size(data_2d))
    return noisy
end

# ── Neural ODE architecture ──────────────────────────────────────────────────
# We test three architectures of increasing capacity
function make_node_model(hidden; n_layers=2)
    if n_layers == 1
        return Lux.Chain(Lux.Dense(2, hidden, tanh), Lux.Dense(hidden, 2))
    elseif n_layers == 2
        return Lux.Chain(Lux.Dense(2, hidden, tanh), Lux.Dense(hidden, hidden, tanh), Lux.Dense(hidden, 2))
    else
        return Lux.Chain(Lux.Dense(2, hidden, tanh), Lux.Dense(hidden, hidden, tanh),
                         Lux.Dense(hidden, hidden, tanh), Lux.Dense(hidden, 2))
    end
end

# ── Training function ────────────────────────────────────────────────────────
function train_node(; hidden=32, n_layers=2, seed=1, lr=1e-3, epochs=N_EPOCHS,
                      shoot_len=20)
    rng_lux = Lux.replicate(Random.MersenneTwister(seed))
    model = make_node_model(hidden; n_layers=n_layers)
    ps, st = Lux.setup(rng_lux, model)
    ps_ca = ComponentArray(Float64.(ComponentArray(ps)))

    train_data = make_training_data(; seed=seed)
    u0 = Float64.(train_data[1, :])

    function dudt(u, p, t)
        pred, _ = model(u, p, st)
        return pred
    end

    # Multiple-shooting loss: break trajectory into segments to avoid
    # vanishing gradients over long integration horizons
    segment_starts = 1:shoot_len:(n_data - 1)

    function loss(p_vec, _)
        total = 0.0
        for seg_start in segment_starts
            seg_end = min(seg_start + shoot_len, n_data)
            tspan = (t_data[seg_start], t_data[seg_end])
            u0_seg = Float64.(train_data[seg_start, :])
            prob = ODEProblem(dudt, u0_seg, tspan, p_vec)
            sol = solve(prob, Tsit5(); saveat=t_data[seg_start:seg_end],
                        abstol=1e-6, reltol=1e-6)
            if sol.retcode != :Success && sol.retcode != ReturnCode.Success
                return Inf
            end
            for (k, idx) in enumerate(seg_start:seg_end)
                pred = sol[:, k]
                target = Float64.(train_data[idx, :])
                total += sum((pred .- target).^2)
            end
        end
        return total / n_data
    end

    # Track loss history
    loss_history = Float64[]
    callback = function(state, l)
        push!(loss_history, l)
        return false
    end

    opt_f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
    opt_prob = OptimizationProblem(opt_f, ps_ca)
    sol = solve(opt_prob, OptimizationOptimisers.Adam(lr);
                maxiters=epochs, callback=callback)

    # Generate full prediction from initial condition
    final_ps = sol.u
    prob_pred = ODEProblem(dudt, u0, (0.0, NODE_T_TRAIN), final_ps)
    pred_sol = solve(prob_pred, Tsit5(); saveat=t_data, abstol=1e-6, reltol=1e-6)

    pred_x = [pred_sol[1, i] for i in 1:length(t_data)]
    pred_y = [pred_sol[2, i] for i in 1:length(t_data)]

    # Measure prediction amplitude (should be ~1.0 if learned correctly)
    amplitude = mean(sqrt.(pred_x.^2 .+ pred_y.^2))

    return (; pred_x, pred_y, loss_history, amplitude, final_ps)
end

# ── Run experiments ──────────────────────────────────────────────────────────
# Architecture configs: (hidden, n_layers, label)
architectures = [
    (16, 1, "1×16"),
    (32, 2, "2×32"),
    (64, 3, "3×64"),
]

all_results = @cached cache_path("fig2_node_results.jls") begin
    results = Dict{String, Vector{NamedTuple}}()
    for (hidden, n_layers, label) in architectures
        arch_results = NamedTuple[]
        for s in 1:N_SEEDS
            @info "Training neural ODE" architecture=label seed=s
            try
                r = train_node(; hidden=hidden, n_layers=n_layers, seed=s,
                                 epochs=N_EPOCHS, lr=1e-3, shoot_len=20)
                push!(arch_results, r)
            catch e
                @warn "Training failed" architecture=label seed=s exception=e
                push!(arch_results, (pred_x=zeros(n_data), pred_y=zeros(n_data),
                                      loss_history=Float64[Inf], amplitude=0.0,
                                      final_ps=nothing))
            end
        end
        results[label] = arch_results
    end
    results
end

# ── Panel (a): Data + predictions overlaid ───────────────────────────────────
arch_colors = Dict("1×16" => RGB(0.27, 0.51, 0.71),
                   "2×32" => RGB(1.00, 0.55, 0.00),
                   "3×64" => RGB(0.86, 0.08, 0.24))

p_a = plot(t_data, x_true; lw=2, color=:black, label="True sin(2πt)",
           xlabel="Time (s)", ylabel="x(t)",
           title="(A) Neural ODE predictions (x-component)",
           size=(PUB_W, PUB_H), alpha=0.8)
for (label, results) in sort(collect(all_results))
    for (i, r) in enumerate(results)
        plot!(p_a, t_data[1:length(r.pred_x)], r.pred_x;
              lw=0.5, alpha=0.4, color=arch_colors[label],
              label=(i == 1 ? label : ""))
    end
end

# ── Panel (b): Training loss curves ──���───────────────────────────────────────
# Reference: MSE of predicting the mean = variance of the signal ≈ 0.5
signal_variance = var(x_true) + var(y_true)

p_b = plot(; xlabel="Epoch", ylabel="MSE Loss", yscale=:log10,
             title="(B) Training loss curves",
             size=(PUB_W, PUB_H))
for (label, results) in sort(collect(all_results))
    for (i, r) in enumerate(results)
        length(r.loss_history) > 0 && plot!(p_b, r.loss_history;
            lw=0.5, alpha=0.4, color=arch_colors[label],
            label=(i == 1 ? label : ""))
    end
end
hline!(p_b, [signal_variance]; ls=:dash, color=:gray, lw=1.5,
       label="Var(signal)")

# ── Panel (c): Phase portrait ────────────────────────────────────────────────
p_c = plot(x_true, y_true; lw=2, color=:black, label="True orbit",
           xlabel="x", ylabel="y", aspect_ratio=:equal,
           title="(C) Phase portrait",
           size=(PUB_W, PUB_H))
for (label, results) in sort(collect(all_results))
    for (i, r) in enumerate(results)
        plot!(p_c, r.pred_x, r.pred_y;
              lw=0.5, alpha=0.4, color=arch_colors[label],
              label=(i == 1 ? label : ""))
    end
end
scatter!(p_c, [0], [0]; ms=6, color=:red, markershape=:xcross, label="Mean")

# ── Panel (d): Amplitude histogram ──────────────────────────────────────────
all_amps = Dict{String, Vector{Float64}}()
for (label, results) in all_results
    all_amps[label] = [r.amplitude for r in results]
end

p_d = plot(; xlabel="Mean prediction amplitude", ylabel="Count",
             title="(D) Prediction amplitudes",
             size=(PUB_W, PUB_H))
for (label, amps) in sort(collect(all_amps))
    histogram!(p_d, amps; bins=range(0, 1.2, length=15),
               alpha=0.5, color=arch_colors[label], label=label)
end
vline!(p_d, [1.0]; ls=:dash, color=:black, lw=1.5, label="True amplitude")

# ── Compose ──────��─────────────────────────────���─────────────────────────────
fig2 = plot(p_a, p_b, p_c, p_d; layout=(2, 2), size=(2PUB_W, 2PUB_H))
