#!/usr/bin/env julia
# Figure 1: Stuart-Landau DMD Anatomy
# Shows that even the simplest oscillator produces a profoundly non-normal
# DMD operator when embedded via Hankel delay coordinates.
#
# 4 panels:
#   (a) Eigenvalue spectrum on the unit circle
#   (b) Transient growth curve ||A^n|| vs n
#   (c) ε-pseudospectral contours
#   (d) Resolvent gain curve at r = 1.01

include("common.jl")

# ── Simulate Stuart-Landau ───────────────────────────────────────────────────
const SL_σ = 0.1
const SL_ω = 2π
const SL_dt = 0.01
const SL_T  = 100.0

# Add measurement noise — the key point is that even modest noise creates
# non-normality in the Hankel-DMD operator for oscillatory data
t_sl, x_sl, y_sl = simulate_stuart_landau(σ=SL_σ, ω=SL_ω, T=SL_T, dt=SL_dt,
                                            z0=sqrt(SL_σ) * exp(im * 0.0),
                                            noise=0.005)

# Discard transient (first 10 periods)
n_discard = round(Int, 10.0 / SL_dt)
t_sl = t_sl[n_discard+1:end]
x_sl = x_sl[n_discard+1:end]
y_sl = y_sl[n_discard+1:end]

# ── Apply Hankel-DMD ─────────────────────────────────────────────────────────
const SL_τ = 30  # larger embedding for richer non-normal structure
data_sl = hcat(x_sl, y_sl)  # n_t × 2

# Compute the RAW (unclamped) DMD operator to show the full non-normality
H_sl = build_hankel_multi(data_sl, SL_τ)
X_sl_h, Y_sl_h = H_sl[:, 1:end-1], H_sl[:, 2:end]
U_svd, Σ_svd, Vt_svd = svd(X_sl_h)
rank_sl = max(2, count(Σ_svd .> Σ_svd[1] * 1e-10))
A_sl_raw = U_svd[:, 1:rank_sl]' * Y_sl_h * Vt_svd[:, 1:rank_sl] * Diagonal(1.0 ./ Σ_svd[1:rank_sl])

# Also compute stability-enforced version for resolvent analysis
Λ_raw, W_raw = eigen(A_sl_raw)
Λ_clamped = copy(Λ_raw)
for i in eachindex(Λ_clamped)
    abs(Λ_clamped[i]) > 1.0 && (Λ_clamped[i] /= abs(Λ_clamped[i]))
end
A_sl = real.(W_raw * Diagonal(Λ_clamped) * inv(W_raw))

# ── Compute metrics on raw operator ─────────────────────────────────────────
Λ_sl = Λ_raw  # show raw eigenvalues (some may be outside unit circle)
κ_V = eigvec_condition_number(A_sl_raw)
δ_H = henrici_departure(A_sl_raw)
growth = transient_growth_curve(A_sl_raw; n_steps=500)
K_sl = kreiss_constant(A_sl_raw)

@info "Stuart-Landau DMD metrics:" rank=rank_sl spectral_radius=maximum(abs.(Λ_sl)) κ_V δ_H peak_growth=maximum(growth) kreiss=K_sl

# Ground-truth Floquet
μ_phase, μ_trans = ground_truth_floquet_sl(σ=SL_σ, ω=SL_ω)
@info "Ground-truth Floquet:" μ_phase μ_transverse=μ_trans

# ── Panel (a): Eigenvalue spectrum ───────────────────────────────────────────
θ_circle = range(0, 2π, length=200)
p_a = plot(cos.(θ_circle), sin.(θ_circle);
           color=:gray, ls=:dash, lw=1, label="Unit circle",
           aspect_ratio=:equal, xlabel="Re(λ)", ylabel="Im(λ)",
           title="(A) Eigenvalue spectrum",
           size=(PUB_W, PUB_H), xlims=(-1.3, 1.3), ylims=(-1.3, 1.3))
scatter!(p_a, real.(Λ_sl), imag.(Λ_sl);
         ms=4, color=RGB(0.27, 0.51, 0.71), markerstrokewidth=0.5, label="DMD eigenvalues")

# ── Panel (b): Transient growth ──────────────────────────────────────────────
ρ_sl = maximum(abs.(Λ_sl))
spectral_bound = [ρ_sl^n for n in 0:500]

p_b = plot(0:500, growth;
           lw=PUB_LW, color=RGB(0.86, 0.08, 0.24), label="||Ã^n||",
           yscale=:log10, xlabel="Power n", ylabel="Operator norm",
           title="(B) Transient growth",
           size=(PUB_W, PUB_H))
plot!(p_b, 0:500, spectral_bound;
      lw=1, ls=:dash, color=:gray, label="ρ(Ã)^n bound")
annotate!(p_b, 250, maximum(growth) * 0.5,
          text("Peak = $(round(Int, maximum(growth)))", 7, :left))

# ── Panel (c): Pseudospectral contours ───────────────────────────────────────
re_grid, im_grid, σ_grid = pseudospectral_contours(A_sl;
    grid_n=150, re_range=(-1.5, 1.5), im_range=(-1.5, 1.5))

# Plot log10(1/σ_min) = -log10(σ_min) for resolvent norm contours
log_resolvent = -log10.(max.(σ_grid, 1e-15))
ε_levels = [1.0, 2.0, 3.0, 4.0, 5.0]  # log10(||resolvent||) levels

p_c = contour(re_grid, im_grid, log_resolvent;
              levels=ε_levels, fill=true, color=:viridis,
              xlabel="Re(z)", ylabel="Im(z)",
              title="(C) Pseudospectra log₁₀‖(zI−Ã)⁻¹‖",
              aspect_ratio=:equal, size=(PUB_W, PUB_H),
              xlims=(-1.5, 1.5), ylims=(-1.5, 1.5))
plot!(p_c, cos.(θ_circle), sin.(θ_circle); color=:white, ls=:dash, lw=1, label="")
scatter!(p_c, real.(Λ_sl), imag.(Λ_sl); ms=2, color=:white, markerstrokewidth=0, label="")

# ── Panel (d): Resolvent gain curve ──────────────────────────────────────────
# Evaluate resolvent at r = 1.01 (just outside unit circle)
r_eval = 1.01
n_angles_res = 2000
thetas_res = range(0, 2π, length=n_angles_res+1)[1:end-1]
freq_hz_res = collect(thetas_res) ./ (2π) .* (1.0 / SL_dt)  # sampling freq = 1/dt
n_dim = size(A_sl, 1)
gains_res = zeros(n_angles_res)
Threads.@threads for i in 1:n_angles_res
    z = r_eval * exp(im * thetas_res[i])
    gains_res[i] = opnorm(inv(z * I(n_dim) - A_sl), 2)
end

# Only show up to Nyquist
nyq_idx = findfirst(freq_hz_res .> (1.0 / SL_dt) / 2)
if nyq_idx !== nothing
    freq_hz_res = freq_hz_res[1:nyq_idx-1]
    gains_res = gains_res[1:nyq_idx-1]
end

p_d = plot(freq_hz_res, gains_res;
           lw=PUB_LW, color=RGB(0.27, 0.51, 0.71),
           xlabel="Frequency (Hz)", ylabel="Resolvent gain",
           title="(D) Resolvent at r = 1.01",
           yscale=:log10, size=(PUB_W, PUB_H), label="")
vline!(p_d, [SL_ω / (2π)]; ls=:dash, color=:gray, lw=1, label="ω₀")
vline!(p_d, [2 * SL_ω / (2π)]; ls=:dot, color=:gray, lw=1, label="2ω₀")

# ── Compose ───────────────────────────────────────��──────────────────────────
fig1 = plot(p_a, p_b, p_c, p_d; layout=(2, 2), size=(2 * PUB_W, 2 * PUB_H))
savefig(fig1, "figures/Figure 1.svg")