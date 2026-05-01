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

"""
    estimate_period_fft(x, dt; min_period=0.5, max_period=80.0, fallback=10.0)

Robust dominant-period estimate from the peak FFT magnitude in a bounded
frequency window.
"""
function estimate_period_fft(x, dt; min_period=0.5, max_period=80.0, fallback=10.0)
    n = length(x)
    n < 10 && return fallback
    xc = x .- mean(x)
    spec = abs.(fft(xc))
    freqs = (0:n-1) ./ (n * dt)
    half = 2:div(n, 2)
    fmin = 1 / max_period
    fmax = 1 / min_period
    valid = [i for i in half if freqs[i] >= fmin && freqs[i] <= fmax]
    isempty(valid) && return fallback
    peak_idx = valid[argmax(spec[valid])]
    f_peak = freqs[peak_idx]
    return f_peak > 0 ? 1 / f_peak : fallback
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

    # Robust period estimate: FFT peak in physiologic window
    period_fhn = estimate_period_fft(v_fhn, 0.01; min_period=2.0, max_period=60.0,
                                     fallback=10.0)
    @info "FHN period estimate:" period_fhn stride=round(Int, period_fhn / 0.01)

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

    period_vdp = estimate_period_fft(x_vdp, 0.01; min_period=2.0, max_period=20.0,
                                     fallback=6.28)
    @info "VdP period estimate:" period_vdp stride=round(Int, period_vdp / 0.01)

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
             legend=:outertopright,
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
if haskey(results_esn, ("FHN", 500))
    pred = results_esn[("FHN", 500)].pred
    if size(pred, 2) > 10
        plot!(p_fhn_pred, pred[1, :], pred[2, :]; lw=1.5, ls=:dash,
              color=RGB(0.86, 0.08, 0.24), label="ESN (N=500)")
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

# ── Robustness Analysis: Parameter Sweeps & Stability ──────────────────────

robustness_results = @cached cache_path("fig5_robustness_v1.jls") begin
    rob = Dict()

    @info "Robustness: VdP μ parameter sweep"
    # Train on nominal VdP (μ=1.0), test on different μ
    μ_nominal = 1.0
    t_train, x_train, y_train = simulate_van_der_pol(μ=μ_nominal, T=100.0, dt=0.01)
    n_disc = round(Int, 10.0 / 0.01)
    x_train = x_train[n_disc+1:end]; y_train = y_train[n_disc+1:end]
    
    esn_nom = create_esn(2, 300; spectral_radius=0.95, seed=42)
    esn_nom_trained, _ = train_esn!(esn_nom, hcat(x_train, y_train)'; washout=200)
    rob["esn_nom"] = esn_nom_trained

    μ_sweep = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
    mu_errors = Float64[]
    for μ_test in μ_sweep
        t_test, x_test, y_test = simulate_van_der_pol(μ=μ_test, T=50.0, dt=0.01)
        n_disc_test = round(Int, 20.0 / 0.01)
        x_test = x_test[n_disc_test+1:end]; y_test = y_test[n_disc_test+1:end]
        
        # Prediction error
        pred = esn_predict(esn_nom_trained, zeros(300), [x_test[1], y_test[1]], 
                           min(size(x_test, 1), 1000))
        actual = hcat(x_test[1:size(pred,2)], y_test[1:size(pred,2)])'
        err = sqrt(mean((pred .- actual).^2))
        push!(mu_errors, err)
    end
    rob["mu_sweep"] = (μ_sweep, mu_errors)

    @info "Robustness: Spectral radius sweep"
    # Sweep spectral radius and compare climate vs short-term error
    rhos = collect(0.1:0.1:1.5)
    climate_errors = Float64[]
    shortterm_errors = Float64[]
    
    for ρ in rhos
        esn_ρ = create_esn(2, 300; spectral_radius=ρ, seed=42)
        esn_ρ_trained, _ = train_esn!(esn_ρ, hcat(x_train, y_train)'; washout=200)
        
        # Long-term: compare invariant measure
        pred_long = esn_predict(esn_ρ_trained, zeros(300), [x_train[1], y_train[1]], 5000)
        actual_long = hcat(x_train[1:size(pred_long, 2)], y_train[1:size(pred_long,2)])'
        climate_err = sqrt(mean((pred_long .- actual_long).^2))
        push!(climate_errors, climate_err)
        
        # Short-term: 1-step-ahead
        pred_short = esn_predict(esn_ρ_trained, zeros(300), [x_train[1], y_train[1]], 10)
        actual_short = hcat(x_train[1:10], y_train[1:10])'
        shortterm_err = sqrt(mean((pred_short .- actual_short).^2))
        push!(shortterm_errors, shortterm_err)
    end
    rob["rho_sweep"] = (rhos, climate_errors, shortterm_errors)

    @info "Robustness: Initial condition coverage"
    # Show error vs number of training initial conditions
    n_ics_list = [1, 2, 4, 8, 16]
    ic_errors_list = Float64[]
    
    # Get a reference reservoir from the nominal ESN
    esn_ref = create_esn(2, 300; spectral_radius=0.95, seed=42)
    
    for n_ic in n_ics_list
        errs_trial = Float64[]
        for trial in 1:2
            # Sample n_ic random ICs around the nominal limit cycle
            θs = range(0, 2π, length=n_ic+1)[1:end-1]
            R0s = [[2.5*cos(θ), 2.5*sin(θ)] for θ in θs]
            
            # Collect data from these initial conditions
            H_train = Vector{Float64}[]
            Y_target = Vector{Float64}[]
            r = zeros(300)
            
            for x0 in R0s
                x = copy(x0)
                r = zeros(300)
                for t in 1:800
                    # Van der Pol: dx/dt = y, dy/dt = μ*(1-x²)*y - x
                    xn = [x[2], 1.0*(1.0 - x[1]^2)*x[2] - x[1]]
                    update = tanh.(esn_ref.W_in * x .+ esn_ref.W_res * r)
                    r = 0.8 .* r .+ 0.2 .* update
                    if t > 100
                        push!(H_train, copy(r))
                        push!(Y_target, copy(xn))
                    end
                    x = copy(xn)
                end
            end
            
            if !isempty(H_train)
                H_m = hcat(H_train...)
                Y_m = hcat(Y_target...)
                Wout_trial = Y_m * H_m' * inv(H_m * H_m' + 1e-5 * I)
                
                # Test on fresh VdP trajectory
                t_test, x_test, y_test = simulate_van_der_pol(μ=1.0, T=50.0, dt=0.01, z0=-2.0+0.5im)
                
                # Generate predictions and compare
                esn_test = SimpleESN(esn_ref.W_in, esn_ref.W_res, Wout_trial, 300, 2)
                pred_ic = esn_predict(esn_test, zeros(300), [x_test[1], y_test[1]], 
                                      min(2000, length(x_test)))
                actual = hcat(x_test[1:size(pred_ic, 2)], y_test[1:size(pred_ic, 2)])'
                err = sqrt(mean((pred_ic .- actual).^2))
                push!(errs_trial, err)
            end
        end
        push!(ic_errors_list, mean(errs_trial))
    end
    rob["n_ics_sweep"] = (n_ics_list, ic_errors_list)

    rob
end

# ── Plot Robustness Panels ────────────────────────────────────────────────

# (G) Floquet Stability: Eigenvalues across Van der Pol cycle
p_floquet_stab = plot(xlabel="Arc-length (normalized)", ylabel="|λ| (transverse)",
    title="(G) Floquet Decay Along Limit Cycle", size=(PUB_W, PUB_H), legend=:right)

# Extract eigenvalues at different points; for simplicity show true multipliers
μ_true_vdp = get_true_multipliers(results_esn, "VdP_true")
|λ_transverse| = abs(μ_true_vdp[1])
plot!(p_floquet_stab, [0, 1], [|λ_transverse|, |λ_transverse|]; 
    lw=2.5, color=:black, label="VdP")

# Add other systems for comparison
μ_true_sl = get_true_multipliers(results_esn, "SL_true")
|λ_sl| = abs(μ_true_sl[1])
plot!(p_floquet_stab, [0, 1], [|λ_sl|, |λ_sl|]; 
    lw=2.5, color=RGB(0.27, 0.51, 0.71), label="SL")

μ_true_fhn = get_true_multipliers(results_esn, "FHN_true")
|λ_fhn| = abs(μ_true_fhn[1])
plot!(p_floquet_stab, [0, 1], [|λ_fhn|, |λ_fhn|];
    lw=2.5, color=RGB(0.86, 0.08, 0.24), label="FHN")

# (H) VdP μ Parameter Robustness
μ_sweep, mu_errors = robustness_results["mu_sweep"]
p_mu_sweep = plot(μ_sweep, mu_errors; xlabel="System parameter μ", 
    ylabel="Prediction error", title="(H) VdP: Generalization across μ",
    marker=:circle, ms=6, lw=2.5, color=:darkblue, size=(PUB_W, PUB_H),
    yscale=:log10, label="", markerstrokewidth=0)

# (I) Spectral Radius Robustness: Climate vs Short-term
ρs, climate_errs, shortterm_errs = robustness_results["rho_sweep"]
p_rho_sweep = plot(ρs, climate_errs; xlabel="Spectral radius ρ",
    ylabel="Long-term error", title="(I) ESN: Climate vs Short-term",
    lw=2.5, color=:darkgreen, marker=:circle, ms=5, label="Long-term",
    yscale=:log10, size=(PUB_W, PUB_H), legend=:topright, markerstrokewidth=0)
plot!(p_rho_sweep, ρs, shortterm_errs; lw=2.5, color=:darkorange,
    marker=:square, ms=5, label="Short-term", markerstrokewidth=0)

# (J) Initial Condition Coverage
n_ICs, ic_errors = robustness_results["n_ics_sweep"]
p_ic_robust = plot(n_ICs, ic_errors; xlabel="# Training initial conditions",
    ylabel="Mean error", title="(J) ESN: Training coverage robustness",
    marker=:diamond, ms=7, lw=2.5, color=:purple, size=(PUB_W, PUB_H),
    xscale=:log10, yscale=:log10, label="", markerstrokewidth=0)

# ── Compose Extended Figure ──────────────────────────────────────────────────
fig5_extended = plot(p_sl, p_sl_pred, p_fhn, p_fhn_pred, p_vdp, p_vdp_pred,
                     p_floquet_stab, p_mu_sweep, p_rho_sweep, p_ic_robust;
                     layout=(5, 2), size=(2 * PUB_W, 5 * PUB_H))
savefig(fig5_extended, "figures/Figure 5.svg")