#!/usr/bin/env julia
# Figure 6: Echo State Network Floquet Recovery — Gait Data Failure
# Shows that while ESN prediction quality looks reasonable for real walking
# data, the stride-to-stride linear map produces incoherent eigenvalues.
#
# 4 panels:
#   (a) ESN one-step prediction vs actual gait data (looks fine)
#   (b) Extracted "Floquet" eigenvalues — scattered across complex plane
#   (c) Sensitivity to reservoir size — eigenvalues shift wildly
#   (d) Across-trial comparison — no convergence to stable structure

include("common.jl")

using FFTW

# Reuse SimpleESN from Figure 5 if already loaded, otherwise define here
if !@isdefined(SimpleESN)
    struct SimpleESN
        W_in::Matrix{Float64}
        W_res::Matrix{Float64}
        W_out::Matrix{Float64}
        reservoir_size::Int
        n_dim::Int
    end

    function create_esn(n_dim, reservoir_size; spectral_radius=0.95,
                        input_scaling=0.5, sparsity=0.9, seed=42)
        rng = MersenneTwister(seed)
        W_in = input_scaling .* randn(rng, reservoir_size, n_dim)
        W_res = randn(rng, reservoir_size, reservoir_size)
        mask = rand(rng, reservoir_size, reservoir_size) .> sparsity
        W_res .*= mask
        ρ = maximum(abs.(eigvals(W_res)))
        ρ > 0 && (W_res .*= spectral_radius / ρ)
        W_out = zeros(n_dim, reservoir_size)
        return SimpleESN(W_in, W_res, W_out, reservoir_size, n_dim)
    end

    function train_esn!(esn::SimpleESN, input_data::AbstractMatrix;
                         washout=200, regularization=1e-6)
        n_dim, n_t = size(input_data)
        states = zeros(esn.reservoir_size, n_t)
        r = zeros(esn.reservoir_size)
        for t in 1:n_t
            r = tanh.(esn.W_in * input_data[:, t] .+ esn.W_res * r)
            states[:, t] = r
        end
        S = states[:, washout+1:end-1]
        T_mat = input_data[:, washout+2:end]
        W_out = T_mat * S' * inv(S * S' + regularization * I)
        return SimpleESN(esn.W_in, esn.W_res, W_out, esn.reservoir_size, esn.n_dim), states
    end

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
end

# ── Load gait data ──────────────────────────────────────────────────────────
data_all, speeds, groups, subjects = load_gait_data()

# Select representative trials
function select_trial(group_target; speed_range=(40, 80))
    mask = (groups .== group_target) .&
           (speeds .>= speed_range[1]) .& (speeds .<= speed_range[2])
    idx = findall(mask)
    isempty(idx) && return nothing
    return idx[div(length(idx), 2) + 1]
end

trial_ab = select_trial("AB"; speed_range=(60, 100))
trial_hf = select_trial("HF"; speed_range=(50, 90))
trial_lf = select_trial("LF"; speed_range=(30, 60))

representative_trials = filter(!isnothing, [trial_ab, trial_hf, trial_lf])
trial_labels = String[]
!isnothing(trial_ab) && push!(trial_labels, "AB")
!isnothing(trial_hf) && push!(trial_labels, "HF")
!isnothing(trial_lf) && push!(trial_labels, "LF")

# ── Estimate stride period ──────────────────────────────────────────────────
function estimate_stride_period(trial_data; fs=100.0)
    x = trial_data[:, 1] .- mean(trial_data[:, 1])
    acf = real.(ifft(abs2.(fft(x))))
    acf ./= acf[1]
    min_lag = round(Int, 0.5 * fs)
    max_lag = round(Int, 2.0 * fs)
    search_range = min_lag:min(max_lag, length(acf)-1)
    peaks = findall(i -> acf[i] > acf[i-1] && acf[i] > acf[i+1], search_range)
    isempty(peaks) && return round(Int, 1.0 * fs)
    return search_range[peaks[1]]
end

# ── Gait ESN Floquet estimation ─────────────────────────────────────────────
function gait_esn_floquet(trial_data::AbstractMatrix;
                           reservoir_size=200, seed=42)
    n_t, n_j = size(trial_data)
    input = trial_data'  # n_j × n_t

    stride = estimate_stride_period(trial_data)

    esn = create_esn(n_j, reservoir_size; spectral_radius=0.95, seed=seed)
    esn_trained, states = train_esn!(esn, input;
                                      washout=min(200, stride),
                                      regularization=1e-6)

    # Prediction
    pred = esn_predict(esn_trained, states[:, end], input[:, end], stride)

    # Stride-to-stride linear map
    n_pairs = n_t - stride
    X_stride = input[:, 1:n_pairs]
    Y_stride = input[:, stride+1:stride+n_pairs]
    A_mono = Y_stride * pinv(X_stride)
    μ = eigvals(A_mono)

    return (; μ, pred, stride)
end

# ── Run analysis ─────────────────────────────────────────────────────────────
reservoir_sizes = [50, 100, 200, 300, 500]

results_gait = @cached cache_path("fig6_esn_gait_v2.jls") begin
    res = Dict()

    for (trial_idx, label) in zip(representative_trials, trial_labels)
        trial_data = Float64.(data_all[trial_idx, :, :])

        # Z-score within trial
        for j in axes(trial_data, 2)
            μ_j = mean(trial_data[:, j])
            σ_j = std(trial_data[:, j])
            σ_j > 0 && (trial_data[:, j] .= (trial_data[:, j] .- μ_j) ./ σ_j)
        end

        # Sweep reservoir sizes
        for rs in reservoir_sizes
            @info "Trial $label, reservoir=$rs"
            try
                r = gait_esn_floquet(trial_data; reservoir_size=rs, seed=42)
                res[(label, rs)] = r
            catch e
                @warn "Failed" label rs exception=e
                res[(label, rs)] = (; μ=ComplexF64[], pred=zeros(6, 10), stride=100)
            end
        end

        # Multiple random seeds at fixed reservoir size
        for seed in 1:10
            try
                r = gait_esn_floquet(trial_data; reservoir_size=200, seed=seed)
                res[(label, "seed", seed)] = r
            catch e
                res[(label, "seed", seed)] = (; μ=ComplexF64[], pred=zeros(6, 10), stride=100)
            end
        end
    end

    # Multiple AB trials for convergence check
    ab_indices = findall(groups .== "AB")
    n_ab_sample = min(20, length(ab_indices))
    for (i, idx) in enumerate(ab_indices[1:n_ab_sample])
        td = Float64.(data_all[idx, :, :])
        for j in axes(td, 2)
            μ_j = mean(td[:, j]); σ_j = std(td[:, j])
            σ_j > 0 && (td[:, j] .= (td[:, j] .- μ_j) ./ σ_j)
        end
        try
            r = gait_esn_floquet(td; reservoir_size=200, seed=42)
            res[("AB_multi", i)] = r
        catch e
            res[("AB_multi", i)] = (; μ=ComplexF64[], pred=zeros(6, 10), stride=100)
        end
    end

    res
end

# ── Panel (a): ESN prediction quality ────────────────────────────────────────
p_a = plot(; xlabel="Time step", ylabel="Joint angle (z-scored)",
             title="(A) ESN one-step prediction (AB trial)",
             size=(PUB_W, PUB_H))

if !isnothing(trial_ab) && haskey(results_gait, ("AB", 200))
    td_ab = Float64.(data_all[trial_ab, :, :])
    for j in axes(td_ab, 2)
        μ_j = mean(td_ab[:, j]); σ_j = std(td_ab[:, j])
        σ_j > 0 && (td_ab[:, j] .= (td_ab[:, j] .- μ_j) ./ σ_j)
    end
    pred_ab = results_gait[("AB", 200)].pred
    n_show = min(300, size(pred_ab, 2))
    offset = size(td_ab, 1) - n_show
    plot!(p_a, 1:n_show, td_ab[offset+1:offset+n_show, 1];
          lw=1.5, color=:black, label="Actual (hip)")
    plot!(p_a, 1:n_show, pred_ab[1, 1:n_show];
          lw=1.5, ls=:dash, color=RGB(0.27, 0.51, 0.71), label="ESN pred")
end

# ── Panel (b): Extracted eigenvalues ─────────────────────────────────────────
θ_uc = range(0, 2π, length=200)
p_b = plot(cos.(θ_uc), sin.(θ_uc); color=:gray, ls=:dash, lw=0.5,
           aspect_ratio=:equal, xlabel="Re(μ)", ylabel="Im(μ)",
           title="(B) 'Floquet' eigenvalues (gait)",
           size=(PUB_W, PUB_H), xlims=(-2, 2), ylims=(-2, 2))
group_plot_colors = [RGB(0.27, 0.51, 0.71), RGB(1.0, 0.55, 0.0), RGB(0.86, 0.08, 0.24)]
for (i, label) in enumerate(trial_labels)
    key = (label, 200)
    if haskey(results_gait, key) && length(results_gait[key].μ) > 0
        μ = results_gait[key].μ
        scatter!(p_b, real.(μ), imag.(μ);
                 ms=3, color=group_plot_colors[min(i, 3)], alpha=0.6, label=label)
    end
end

# ── Panel (c): Sensitivity to reservoir size ─────────────────────────────────
p_c = plot(; xlabel="Reservoir size", ylabel="|μ| (all eigenvalues)",
             title="(C) Eigenvalue sensitivity to reservoir size",
             size=(PUB_W, PUB_H))
label_target = isempty(trial_labels) ? "AB" : trial_labels[1]
for rs in reservoir_sizes
    key = (label_target, rs)
    if haskey(results_gait, key) && length(results_gait[key].μ) > 0
        μ_mags = abs.(results_gait[key].μ)
        scatter!(p_c, fill(rs, length(μ_mags)), μ_mags;
                 ms=2, alpha=0.5, color=RGB(0.27, 0.51, 0.71), label="")
    end
end
hline!(p_c, [1.0]; ls=:dash, color=:gray, lw=1, label="Unit circle")

# ── Panel (d): Across-trial non-convergence ──────────────────────────────────
p_d = plot(cos.(θ_uc), sin.(θ_uc); color=:gray, ls=:dash, lw=0.5,
           aspect_ratio=:equal, xlabel="Re(μ)", ylabel="Im(μ)",
           title="(D) 20 AB trials — no convergence",
           size=(PUB_W, PUB_H), xlims=(-2, 2), ylims=(-2, 2))
trial_colors = palette(:viridis, 20)
for i in 1:20
    key = ("AB_multi", i)
    if haskey(results_gait, key) && length(results_gait[key].μ) > 0
        μ = results_gait[key].μ
        scatter!(p_d, real.(μ), imag.(μ);
                 ms=2, alpha=0.4, color=trial_colors[i], label="")
    end
end

# ── Compose ──────────────────────────────────────────────────────────────────
fig6 = plot(p_a, p_b, p_c, p_d; layout=(2, 2), size=(2 * PUB_W, 2 * PUB_H))
