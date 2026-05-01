#!/usr/bin/env julia
# Figure 4: Piecewise-Linear & Next-Cycle Model Failure
# Demonstrates that phase-partitioned linear models and stride-to-stride
# predictive models break down on noisy trajectories with transients, even
# when there is abundant data near the limit cycle.
#
# 5 panels:
#   (a) Realistic noisy transient trajectory with phase partitions overlaid
#   (b) Forward prediction error compounds on a transient test trajectory
#   (c) Floquet multiplier estimates vs number of phase partitions
#   (d) Next-cycle model (X(t+T)=AX(t)): Floquet estimates vs data length
#   (e) Data fragmentation: per-partition sample size vs total, and its
#       effect on eigenvalue estimation error

include("common.jl")

using ForwardDiff
using FFTW

# ── Use Van der Pol to demonstrate failure on non-circular cycles
const PW_dt = 0.01
# placeholder period (will be estimated from a long VdP sim)
PW_T_period = 1.0
const PW_NOISE = 0.005
const PW_TRAIN_PERIODS = 100
const PW_SHOW_PERIODS = 8

# Estimate VdP period and numerical Floquet multipliers from a long sim
@info "Estimating Van der Pol period and ground-truth Floquet"
t_pre, x_pre, y_pre = simulate_van_der_pol(μ=1.0, T=500.0, dt=PW_dt)
n_disc_vdp = round(Int, 20.0 / PW_dt)
x_clean = x_pre[n_disc_vdp+1:end]; y_clean = y_pre[n_disc_vdp+1:end]
acf_x = real.(ifft(abs2.(fft(x_clean .- mean(x_clean)))))
acf_x ./= acf_x[1]
local_maxima_vdp = findall(i -> i > 1 && i < length(acf_x) - 1 &&
                                acf_x[i] > acf_x[i-1] && acf_x[i] > acf_x[i+1],
                           2:length(acf_x)-1) .+ 1
period_vdp = isempty(local_maxima_vdp) ? 6.28 : local_maxima_vdp[1] * PW_dt
PW_T_period = period_vdp
@info "VdP period estimate:" PW_T_period

function vdp_rhs!(du, u, p, t)
    du[1] = u[2]
    du[2] = 1.0 * (1 - u[1]^2) * u[2] - u[1]
end

u0_vdp = [x_clean[end], y_clean[end]]
res_vdp_true = try
    numerical_floquet(vdp_rhs!, u0_vdp, PW_T_period)
catch e
    @warn "VdP Floquet failed" exception=e
    nothing
end
μ_true_phase = 1.0
μ_true_trans = if res_vdp_true === nothing
    0.5
else
    sort(abs.(res_vdp_true.values))[1]
end
@info "Ground-truth (VdP) transverse μ:" μ_true_trans

# ── Helper: phase-partitioned DMD ────────────────────────────────────────────
"""
    piecewise_dmd(x, y, dt, n_partitions; period)

Partition the limit cycle into `n_partitions` angular sectors, fit a separate
linear model A_j in each sector, and return:
  - A_j matrices
  - Monodromy estimate M = A_K * ... * A_1
  - Per-partition sample counts
  - Forward prediction from the piecewise model
"""
function piecewise_dmd(x, y, dt, n_partitions; period=PW_T_period)
    n = length(x)
    # Compute phase angles
    phases = atan.(y, x)  # [-π, π]
    phases = mod.(phases, 2π)  # [0, 2π)

    # Define partition boundaries
    boundaries = range(0, 2π, length=n_partitions + 1)

    # Assign each point to a partition
    partition_idx = [searchsortedlast(boundaries, φ) for φ in phases]
    partition_idx = clamp.(partition_idx, 1, n_partitions)

    # Fit linear model in each partition: [x_{k+1}; y_{k+1}] = A_j * [x_k; y_k]
    A_list = Matrix{Float64}[]
    sample_counts = Int[]

    for j in 1:n_partitions
        mask = (partition_idx[1:end-1] .== j)
        count_j = sum(mask)
        push!(sample_counts, count_j)

        if count_j < 3
            push!(A_list, Matrix{Float64}(I, 2, 2))
            continue
        end

        # Build data matrices for this partition
        idx_j = findall(mask)
        X_j = hcat([Float64[x[i], y[i]] for i in idx_j]...)  # 2 × count_j
        Y_j = hcat([Float64[x[i+1], y[i+1]] for i in idx_j]...)

        # Least-squares: Y = A * X => A = Y * X'
        A_j = Y_j * pinv(X_j)
        push!(A_list, A_j)
    end

    # Monodromy matrix: product of partition operators in order
    # For one full cycle, multiply A_1 through A_K
    M = Matrix{Float64}(I, 2, 2)
    for j in 1:n_partitions
        M = A_list[j] * M
    end

    # Forward prediction using piecewise model
    x_pred = zeros(n)
    y_pred = zeros(n)
    x_pred[1] = x[1]
    y_pred[1] = y[1]
    for k in 1:(n-1)
        j = partition_idx[k]
        state = Float64[x_pred[k], y_pred[k]]
        next_state = A_list[j] * state
        x_pred[k+1] = next_state[1]
        y_pred[k+1] = next_state[2]
    end

    return (; A_list, M, sample_counts, x_pred, y_pred)
end

"""
    next_cycle_dmd(x, y, dt; period)

Fit a single linear model X(t+T) = A * X(t) mapping states one full period
ahead. Returns A and the estimated Floquet multipliers.
"""
function next_cycle_dmd(x, y, dt; period=PW_T_period)
    stride = round(Int, period / dt)
    n = length(x)
    n_pairs = n - stride

    if n_pairs < 3
        return Matrix{Float64}(I, 2, 2), [1.0, 1.0]
    end

    X = hcat([Float64[x[i], y[i]] for i in 1:n_pairs]...)
    Y = hcat([Float64[x[i + stride], y[i + stride]] for i in 1:n_pairs]...)

    A = Y * pinv(X)
    μ = eigvals(A)
    return A, μ
end

# ── Sweep: varying data length and noise ─────────────────────────────────────
data_lengths = [50, 100, 200, 500, 1000, 2000, 5000]  # number of periods
noise_levels = [0.0, 0.005, 0.01, 0.02]
n_partitions_sweep = [2, 4, 8, 16, 32, 64]

results_pw = @cached cache_path("fig4_piecewise_results.jls") begin
    res = Dict()

    # 1. Partition sweep (fixed data length = PW_TRAIN_PERIODS, varying noise)
    for noise in noise_levels
        T_sim = PW_TRAIN_PERIODS * PW_T_period + 1.0
        t, xd, yd = simulate_van_der_pol(μ=1.0, T=T_sim, dt=PW_dt)
        # add observational noise if requested
        if noise > 0
            xd .+= noise .* randn(length(xd))
            yd .+= noise .* randn(length(yd))
        end
        # discard transient portion
        n_disc_local = round(Int, 20.0 / PW_dt)
        xd = xd[n_disc_local+1:end]
        yd = yd[n_disc_local+1:end]

        for np in n_partitions_sweep
            pw = piecewise_dmd(xd, yd, PW_dt, np)
            μ_pw = eigvals(pw.M)
            # Error: distance from true transverse multiplier
            μ_est = sort(abs.(μ_pw))[1]  # smallest magnitude = transverse
            err = abs(μ_est - μ_true_trans)
            res[("partition", noise, np)] = (; μ_pw, err, sample_counts=pw.sample_counts,
                                               μ_est)
        end
    end

    # 2. Next-cycle model: sweep data length
    for noise in noise_levels
        for n_periods in data_lengths
            T_sim = n_periods * PW_T_period + 1.0
            t, xd, yd = simulate_van_der_pol(μ=1.0, T=T_sim, dt=PW_dt)
            if noise > 0
                xd .+= noise .* randn(length(xd))
                yd .+= noise .* randn(length(yd))
            end
            n_disc_local = round(Int, 20.0 / PW_dt)
            xd = xd[n_disc_local+1:end]
            yd = yd[n_disc_local+1:end]

            A_nc, μ_nc = next_cycle_dmd(xd, yd, PW_dt; period=PW_T_period)
            μ_est_nc = sort(abs.(μ_nc))[1]
            err_nc = abs(μ_est_nc - μ_true_trans)
            res[("nextcycle", noise, n_periods)] = (; μ_nc, err=err_nc, μ_est=μ_est_nc)
        end
    end

    # 3. Forward prediction error compounding (8 partitions, clean data)
    T_sim_fwd = 20 * PW_T_period
    t_fwd, x_fwd, y_fwd = simulate_van_der_pol(μ=1.0, T=T_sim_fwd, dt=PW_dt)
    # Train on long data, predict on this segment
    T_train = PW_TRAIN_PERIODS * PW_T_period + 1.0
    _, x_train, y_train = simulate_van_der_pol(μ=1.0, T=T_train, dt=PW_dt)
    # add observational noise to training data
    x_train .+= PW_NOISE .* randn(length(x_train))
    y_train .+= PW_NOISE .* randn(length(y_train))

    for np in [4, 8, 16, 32]
        pw = piecewise_dmd(x_train, y_train, PW_dt, np)
        # Forward predict on the test segment
        pw_test = piecewise_dmd(x_fwd, y_fwd, PW_dt, np)
        # Compute running error
        errs = sqrt.((pw_test.x_pred .- x_fwd).^2 .+ (pw_test.y_pred .- y_fwd).^2)
        res[("forward_error", np)] = errs
    end

    res
end

# ── Panel (a): Phase partitions schematic ───────────────────────────────────
θ_lc = range(0, 2π, length=300)
partition_colors = palette(:tab10, 8)

# Show a short transient-rich segment of the VdP trajectory
t_show, x_show, y_show = simulate_van_der_pol(μ=1.0, T=PW_SHOW_PERIODS * PW_T_period,
                                                                                            dt=PW_dt)

p_a = plot(x_show, y_show;
                 lw=1.5, color=RGB(0.27, 0.51, 0.71), label="Transient trajectory",
                     aspect_ratio=:equal, xlabel="x", ylabel="y",
                 title="(A) Realistic training data",
                 size=(PUB_W, PUB_H), xlims=(-1.2, 1.2), ylims=(-1.2, 1.2))

# Overlay radial partition regions (schematic) so readers see how regions are defined
n_show = 8
boundaries_show = range(0, 2π, length=n_show + 1)
r_max = 1.1 * maximum(sqrt.(x_show.^2 .+ y_show.^2))
for j in 1:n_show
        θ_start = boundaries_show[j]
        plot!(p_a, [0, r_max * cos(θ_start)], [0, r_max * sin(θ_start)];
                    lw=0.8, ls=:dash, color=:gray, label=(j==1 ? "Partition boundaries" : ""))
end

# ── Panel (b): Forward prediction error compounds ───────────────────────────
t_fwd_ax = range(0, step=PW_dt, length=round(Int, 20 * PW_T_period / PW_dt) + 1)

p_b = plot(; xlabel="Time (periods)", ylabel="log10 prediction error",
             title="(B) Piecewise forward error on transient data",
             size=(PUB_W, PUB_H))
for np in [4, 8, 16, 32]
    key = ("forward_error", np)
    if haskey(results_pw, key)
        errs = results_pw[key]
        # clamp to avoid infinite log values from numerical blowups
        errs_clamped = clamp.(errs, 1e-12, 1e12)
        t_ax = (0:length(errs_clamped)-1) .* PW_dt ./ PW_T_period
        plot!(p_b, t_ax, log10.(errs_clamped);
              lw=PUB_LW, label="$np partitions",
              color=partition_colors[findfirst(==(np), [4, 8, 16, 32])])
    end
end

# ── Panel (c): Floquet error vs number of partitions ────────────────────────
p_c = plot(; xlabel="Number of partitions", ylabel="log10|μ̂_trans − μ_true|",
             title="(C) Piecewise Floquet error from transient data",
             size=(PUB_W, PUB_H), xscale=:log10)
noise_colors = Dict(0.0 => :black, 0.005 => RGB(0.27, 0.51, 0.71),
                    0.01 => RGB(1.0, 0.55, 0.0), 0.02 => RGB(0.86, 0.08, 0.24))
for noise in noise_levels
    errs = Float64[]
    nps = Int[]
    for np in n_partitions_sweep
        key = ("partition", noise, np)
        if haskey(results_pw, key)
            e = clamp(results_pw[key].err, 1e-12, 1e12)
            push!(errs, log10(e))
            push!(nps, np)
        end
    end
    label_str = noise == 0.0 ? "No noise" : "σ_n=$(noise)"
    plot!(p_c, nps, errs; lw=PUB_LW, ms=PUB_MSIZ, markershape=:circle,
          color=noise_colors[noise], label=label_str)
end

# ── Panel (d): Next-cycle Floquet error vs data length ──────────────────────
p_d = plot(; xlabel="Training periods", ylabel="log10|μ̂_trans − μ_true|",
             title="(D) Next-cycle error from transient data",
             size=(PUB_W, PUB_H), xscale=:log10)
for noise in noise_levels
    errs = Float64[]
    lens = Int[]
    for n_per in data_lengths
        key = ("nextcycle", noise, n_per)
        if haskey(results_pw, key)
            e = clamp(results_pw[key].err, 1e-12, 1e12)
            push!(errs, log10(e))
            push!(lens, n_per)
        end
    end
    label_str = noise == 0.0 ? "No noise" : "σ_n=$(noise)"
    plot!(p_d, lens, errs; lw=PUB_LW, ms=PUB_MSIZ, markershape=:circle,
          color=noise_colors[noise], label=label_str)
end
hline!(p_d, [log10(0.01)]; ls=:dash, color=:gray, lw=1, label="1% error")

# ── Panel (e): Data fragmentation ────────────────────────────────────────────
# Show how many samples each partition gets as partitions increase
p_e = plot(; xlabel="Number of partitions", ylabel="Samples per partition",
             title="(E) Data fragmentation",
             size=(PUB_W, PUB_H), xscale=:log10, yscale=:log10)
total_samples = round(Int, PW_TRAIN_PERIODS * PW_T_period / PW_dt)
for np in n_partitions_sweep
    key = ("partition", 0.005, np)
    if haskey(results_pw, key)
        counts = results_pw[key].sample_counts
        scatter!(p_e, fill(np, length(counts)), counts;
                 ms=2, alpha=0.5, color=RGB(0.27, 0.51, 0.71), label="")
    end
end
# Reference: uniform distribution line
plot!(p_e, n_partitions_sweep, total_samples ./ n_partitions_sweep;
      lw=1.5, ls=:dash, color=:black, label="Uniform split")
# Reference: minimum samples for robust 2×2 fit
hline!(p_e, [10]; ls=:dot, color=:red, lw=1, label="Min for 2×2 fit")

# ── Compose ──────────────────────────────────────────────────────────────────
fig4 = plot(p_a, p_b, p_c, p_d, p_e;
            layout=@layout([a b; c d; e{0.5w}]),
            size=(4 * PUB_W, 3 * PUB_H))
savefig(fig4, "figures/Figure 4.svg")