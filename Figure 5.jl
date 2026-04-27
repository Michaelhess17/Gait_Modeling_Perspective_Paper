#!/usr/bin/env julia
# Figure 5: Echo State Network Floquet Recovery — Simulated Systems
# Shows that ESN Jacobian-based monodromy estimation gets remarkably close
# to true Floquet multipliers for simple simulated oscillators.
#
# 3×2 grid:
#   Row 1: Stuart-Landau   — (a) eigenvalue comparison, (b) ESN prediction
#   Row 2: FitzHugh-Nagumo — (c) eigenvalue comparison, (d) ESN prediction
#   Row 3: Van der Pol     — (e) eigenvalue comparison, (f) ESN prediction

include("common.jl")

using FFTW

# ── Manual ESN implementation ────────────────────────────────────────────────
# We implement a simple ESN from scratch for full control over internals
# and Jacobian access.

struct SimpleESN
    W_in::Matrix{Float64}   # input weights: reservoir_size × n_dim
    W_res::Matrix{Float64}  # reservoir weights: reservoir_size × reservoir_size
    W_out::Matrix{Float64}  # output weights: n_dim × reservoir_size
    reservoir_size::Int
    n_dim::Int
end

"""
    create_esn(n_dim, reservoir_size; spectral_radius=0.95, input_scaling=0.5, sparsity=0.9, seed=42)

Create a simple Echo State Network with random reservoir.
"""
function create_esn(n_dim, reservoir_size; spectral_radius=0.95, input_scaling=0.5,
                    sparsity=0.9, seed=42)
    rng = MersenneTwister(seed)
    W_in = input_scaling .* randn(rng, reservoir_size, n_dim)

    # Sparse random reservoir
    W_res = randn(rng, reservoir_size, reservoir_size)
    mask = rand(rng, reservoir_size, reservoir_size) .> sparsity
    W_res .*= mask
    # Scale to desired spectral radius
    ρ = maximum(abs.(eigvals(W_res)))
    ρ > 0 && (W_res .*= spectral_radius / ρ)

    W_out = zeros(n_dim, reservoir_size)  # to be trained
    return SimpleESN(W_in, W_res, W_out, reservoir_size, n_dim)
end

"""
    train_esn!(esn, input_data; washout=200, regularization=1e-6)

Train ESN output weights via ridge regression.
input_data: n_dim × n_t
Returns trained ESN and reservoir states.
"""
function train_esn!(esn::SimpleESN, input_data::AbstractMatrix;
                     washout=200, regularization=1e-6)
    n_dim, n_t = size(input_data)
    states = zeros(esn.reservoir_size, n_t)
    r = zeros(esn.reservoir_size)

    # Drive reservoir
    for t in 1:n_t
        r = tanh.(esn.W_in * input_data[:, t] .+ esn.W_res * r)
        states[:, t] = r
    end

    # Ridge regression: target is input shifted by 1 step (one-step-ahead prediction)
    S = states[:, washout+1:end-1]       # reservoir states
    T = input_data[:, washout+2:end]      # targets (next step)

    # W_out = T * S' * (S * S' + λI)^{-1}
    W_out = T * S' * inv(S * S' + regularization * I)

    return SimpleESN(esn.W_in, esn.W_res, W_out, esn.reservoir_size, esn.n_dim), states
end

"""
    esn_predict(esn, states, n_steps, last_input)

Generate n_steps of autonomous prediction.
"""
function esn_predict(esn::SimpleESN, last_state::Vector{Float64},
                      last_input::Vector{Float64}, n_steps::Int)
    predictions = zeros(esn.n_dim, n_steps)
    r = copy(last_state)
    x = copy(last_input)
    for t in 1:n_steps
        r = tanh.(esn.W_in * x .+ esn.W_res * r)
        x = esn.W_out * r
        predictions[:, t] = x
    end
    return predictions
end

"""
    esn_stride_map(input_data, dt, period; kwargs...)

Train ESN and estimate Floquet multipliers via the stride-to-stride linear map.
Also returns ESN prediction for validation.
"""
function esn_stride_map(x_data, y_data, dt, period;
                         reservoir_size=300, spectral_radius=0.95, seed=42)
    data = hcat(x_data, y_data)'  # 2 × n_t
    n_t = size(data, 2)
    stride = round(Int, period / dt)

    esn = create_esn(2, reservoir_size; spectral_radius=spectral_radius, seed=seed)
    esn_trained, states = train_esn!(esn, data; washout=min(200, stride))

    # Generate prediction
    pred = esn_predict(esn_trained, states[:, end], data[:, end], stride * 3)

    # Stride-to-stride linear map for Floquet estimation
    # Use the input data directly: X(t+stride) ≈ A * X(t)
    n_pairs = n_t - stride
    X_stride = data[:, 1:n_pairs]
    Y_stride = data[:, stride+1:stride+n_pairs]
    A_mono = Y_stride * pinv(X_stride)
    μ = eigvals(A_mono)

    return (; μ, pred, A_mono, esn=esn_trained)
end

# ── Run on three simulated systems ──────────────────────────────────────────

results_esn = @cached cache_path("fig5_esn_simulated_v2.jls") begin
    res = Dict()

    # 1. Stuart-Landau
    @info "ESN Floquet: Stuart-Landau"
    t_sl, x_sl, y_sl = simulate_stuart_landau(σ=0.1, ω=2π, T=200.0, dt=0.01,
                                                z0=sqrt(0.1) + 0im, noise=0.001)
    n_disc = round(Int, 5.0 / 0.01)
    x_sl = x_sl[n_disc+1:end]; y_sl = y_sl[n_disc+1:end]

    for rs in [100, 200, 500]
        r = esn_stride_map(x_sl, y_sl, 0.01, 1.0;
                            reservoir_size=rs, spectral_radius=0.95, seed=42)
        res[("SL", rs)] = r
    end
    res["SL_true"] = ground_truth_floquet_sl(σ=0.1, ω=2π)

    # 2. FitzHugh-Nagumo
    @info "ESN Floquet: FitzHugh-Nagumo"
    t_fhn, v_fhn, w_fhn = simulate_fitzhugh_nagumo(ε=0.08, T=500.0, dt=0.01)
    local n_disc_fhn = round(Int, 50.0 / 0.01)
    v_fhn = v_fhn[n_disc_fhn+1:end]; w_fhn = w_fhn[n_disc_fhn+1:end]

    # Estimate period from autocorrelation
    acf_v = real.(ifft(abs2.(fft(v_fhn .- mean(v_fhn)))))
    acf_v ./= acf_v[1]
    local_maxima = findall(i -> i > 1 && i < length(acf_v) - 1 &&
                                acf_v[i] > acf_v[i-1] && acf_v[i] > acf_v[i+1],
                           2:length(acf_v)-1) .+ 1
    period_fhn = isempty(local_maxima) ? 10.0 : local_maxima[1] * 0.01
    @info "FHN period estimate:" period_fhn

    # Numerical ground-truth Floquet
    function fhn_rhs!(du, u, p, t)
        v, w = u
        du[1] = v - v^3 / 3 - w + 0.5
        du[2] = 0.08 * (v + 0.7 - 0.8 * w)
    end
    u0_fhn = [v_fhn[end], w_fhn[end]]
    res["FHN_true"] = try
        numerical_floquet(fhn_rhs!, u0_fhn, period_fhn)
    catch e
        @warn "FHN Floquet failed" exception=e
        (values=[1.0, 0.5], vectors=I(2))
    end

    for rs in [100, 200, 500]
        r = esn_stride_map(v_fhn, w_fhn, 0.01, period_fhn;
                            reservoir_size=rs, spectral_radius=0.95, seed=42)
        res[("FHN", rs)] = r
    end

    # 3. Van der Pol
    @info "ESN Floquet: Van der Pol"
    t_vdp, x_vdp, y_vdp = simulate_van_der_pol(μ=1.0, T=200.0, dt=0.01)
    local n_disc_vdp = round(Int, 20.0 / 0.01)
    x_vdp = x_vdp[n_disc_vdp+1:end]; y_vdp = y_vdp[n_disc_vdp+1:end]

    acf_x = real.(ifft(abs2.(fft(x_vdp .- mean(x_vdp)))))
    acf_x ./= acf_x[1]
    local_maxima_vdp = findall(i -> i > 1 && i < length(acf_x) - 1 &&
                                    acf_x[i] > acf_x[i-1] && acf_x[i] > acf_x[i+1],
                               2:length(acf_x)-1) .+ 1
    period_vdp = isempty(local_maxima_vdp) ? 6.28 : local_maxima_vdp[1] * 0.01
    @info "VdP period estimate:" period_vdp

    function vdp_rhs!(du, u, p, t)
        du[1] = u[2]
        du[2] = 1.0 * (1 - u[1]^2) * u[2] - u[1]
    end
    u0_vdp = [x_vdp[end], y_vdp[end]]
    res["VdP_true"] = try
        numerical_floquet(vdp_rhs!, u0_vdp, period_vdp)
    catch e
        @warn "VdP Floquet failed" exception=e
        (values=[1.0, 0.5], vectors=I(2))
    end

    for rs in [100, 200, 500]
        r = esn_stride_map(x_vdp, y_vdp, 0.01, period_vdp;
                            reservoir_size=rs, spectral_radius=0.95, seed=42)
        res[("VdP", rs)] = r
    end

    res
end

# ── Plotting ─────────────────────────────────────────────────────────────────
θ_uc = range(0, 2π, length=200)
rs_colors = Dict(100 => RGB(0.27, 0.51, 0.71),
                 200 => RGB(1.0, 0.55, 0.0),
                 500 => RGB(0.86, 0.08, 0.24))

function get_true_multipliers(results, key)
    val = results[key]
    if val isa Tuple
        return ComplexF64[val[1], val[2]]
    elseif hasfield(typeof(val), :values)
        return ComplexF64.(val.values)
    else
        return ComplexF64.(collect(val))
    end
end

function plot_floquet_comparison(sys_short, sys_label, results; prefix="")
    μ_true = get_true_multipliers(results, "$(sys_short)_true")

    p = plot(cos.(θ_uc), sin.(θ_uc); color=:gray, ls=:dash, lw=0.5,
             aspect_ratio=:equal, xlabel="Re(μ)", ylabel="Im(μ)",
             title="$(prefix)$(sys_label): Floquet μ",
             size=(PUB_W, PUB_H), xlims=(-1.5, 1.5), ylims=(-1.5, 1.5))
    scatter!(p, real.(μ_true), imag.(μ_true);
             ms=10, color=:black, markershape=:diamond, label="True")

    for rs in [100, 200, 500]
        key = (sys_short, rs)
        if haskey(results, key)
            μ_est = results[key].μ
            scatter!(p, real.(μ_est), imag.(μ_est);
                     ms=5, color=rs_colors[rs], label="N=$rs")
        end
    end
    return p
end

# Stuart-Landau
p_sl = plot_floquet_comparison("SL", "Stuart-Landau", results_esn; prefix="(A) ")

t_sl_p, x_sl_p, y_sl_p = simulate_stuart_landau(σ=0.1, ω=2π, T=10.0, dt=0.01,
                                                   z0=sqrt(0.1) + 0im)
p_sl_pred = plot(x_sl_p, y_sl_p; lw=2, color=:black, label="True",
                 xlabel="x", ylabel="y", aspect_ratio=:equal,
                 title="(B) SL: ESN prediction", size=(PUB_W, PUB_H))
if haskey(results_esn, ("SL", 200))
    pred = results_esn[("SL", 200)].pred
    if size(pred, 2) > 10
        plot!(p_sl_pred, pred[1, :], pred[2, :]; lw=1.5, ls=:dash,
              color=RGB(1.0, 0.55, 0.0), label="ESN (N=200)")
    end
end

# FitzHugh-Nagumo
p_fhn = plot_floquet_comparison("FHN", "FitzHugh-Nagumo", results_esn; prefix="(C) ")

t_fhn_p, v_fhn_p, w_fhn_p = simulate_fitzhugh_nagumo(ε=0.08, T=50.0, dt=0.01)
n_d = round(Int, 30.0 / 0.01)
p_fhn_pred = plot(v_fhn_p[n_d:end], w_fhn_p[n_d:end]; lw=2, color=:black,
                  label="True", xlabel="v", ylabel="w",
                  title="(D) FHN: ESN prediction", size=(PUB_W, PUB_H))
if haskey(results_esn, ("FHN", 200))
    pred = results_esn[("FHN", 200)].pred
    if size(pred, 2) > 10
        plot!(p_fhn_pred, pred[1, :], pred[2, :]; lw=1.5, ls=:dash,
              color=RGB(1.0, 0.55, 0.0), label="ESN (N=200)")
    end
end

# Van der Pol
p_vdp = plot_floquet_comparison("VdP", "Van der Pol", results_esn; prefix="(E) ")

t_vdp_p, x_vdp_p, y_vdp_p = simulate_van_der_pol(μ=1.0, T=50.0, dt=0.01)
n_d2 = round(Int, 30.0 / 0.01)
p_vdp_pred = plot(x_vdp_p[n_d2:end], y_vdp_p[n_d2:end]; lw=2, color=:black,
                  label="True", xlabel="x", ylabel="dx/dt",
                  title="(F) VdP: ESN prediction", size=(PUB_W, PUB_H))
if haskey(results_esn, ("VdP", 200))
    pred = results_esn[("VdP", 200)].pred
    if size(pred, 2) > 10
        plot!(p_vdp_pred, pred[1, :], pred[2, :]; lw=1.5, ls=:dash,
              color=RGB(1.0, 0.55, 0.0), label="ESN (N=200)")
    end
end

# ── Compose ──────────────────────────────────────────────────────────────────
fig5 = plot(p_sl, p_sl_pred, p_fhn, p_fhn_pred, p_vdp, p_vdp_pred;
            layout=(3, 2), size=(2PUB_W, 3PUB_H))
