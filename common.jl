# common.jl — shared constants, theme, simulators, and DMD pipeline
# include() this at the top of every Figure X.jl
# Guard: only execute the body once per Julia session.
if !@isdefined(COMMON_JL_LOADED)
const COMMON_JL_LOADED = true
ENV["QUARTO_FIG_FORMAT"] = "svg"
using Pkg; Pkg.activate(@__DIR__; io=devnull)
using LinearAlgebra, Statistics, Random, Serialization, Logging
using NPZ, CSV, DataFrames, Printf
using DifferentialEquations
using Plots, Colors
gr(dpi=150)

# ── Cache helpers ─────────────────────────────────────────────────────────────
const CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)

cache_path(name) = joinpath(CACHE_DIR, name)

macro cached(file, expr)
    quote
        local _f = $(esc(file))
        if isfile(_f)
            deserialize(_f)
        else
            local _result = $(esc(expr))
            serialize(_f, _result)
            _result
        end
    end
end

# ── Publication-ready plot defaults ───────────────────────────────────────────
const PUB_MARGIN      = 8Plots.mm
const PUB_GUIDE_FS    = 9
const PUB_TICK_FS     = 8
const PUB_LEGEND_FS   = 8
const PUB_TITLE_FS    = 10
const PUB_LW          = 2.0
const PUB_MSIZ        = 4

const PUB_W  = 300
const PUB_H  = 220

pub_theme!() = theme(:default;
    margin        = PUB_MARGIN,
    guidefontsize  = PUB_GUIDE_FS,
    tickfontsize   = PUB_TICK_FS,
    legendfontsize = PUB_LEGEND_FS,
    titlefontsize  = PUB_TITLE_FS,
    dpi            = 150,
)
pub_theme!()

# ── Group colours ─────────────────────────────────────────────────────────────
const GROUP_ORDER  = ["AB", "HF", "LF"]
const GROUP_COLORS = Dict(
    "AB" => RGB(0.27, 0.51, 0.71),   # steelblue
    "HF" => RGB(1.00, 0.55, 0.00),   # darkorange
    "LF" => RGB(0.86, 0.08, 0.24),   # crimson
)

# ── Data paths ────────────────────────────────────────────────────────────────
const REPO_DATA_DIR = joinpath(@__DIR__, "data")
const DATA_DIR      = get(ENV, "GAIT_DATA_DIR", REPO_DATA_DIR)
const DATA_PATH     = joinpath(DATA_DIR, "all_human_data.npy")
const META_PATH     = joinpath(DATA_DIR, "all_human_data_metadata.csv")

function load_gait_data()
    if !isfile(DATA_PATH) || !isfile(META_PATH)
        missing_files = String[]
        !isfile(DATA_PATH) && push!(missing_files, DATA_PATH)
        !isfile(META_PATH) && push!(missing_files, META_PATH)
        error("Missing gait data:\n  " * join(missing_files, "\n  ") *
              "\nSet GAIT_DATA_DIR or place files in data/")
    end
    data = npzread(DATA_PATH)
    meta = CSV.read(META_PATH, DataFrame)
    return data, meta[!, "speed"], meta[!, "lf_or_hf"], meta[!, "subject"]
end

# ── Oscillator simulators ────────────────────────────────────────────────────

"""
    simulate_stuart_landau(; σ=0.1, ω=2π, T=20.0, dt=0.01, z0=nothing, noise=0.0)

Simulate the Stuart-Landau oscillator: dz/dt = (σ + iω)z - z|z|²
Returns (t, x, y) where x = Re(z), y = Im(z).
"""
function simulate_stuart_landau(; σ=0.1, ω=2π, T=20.0, dt=0.01, z0=nothing, noise=0.0)
    if z0 === nothing
        z0 = sqrt(σ) * exp(im * 0.0) + 0.1 * randn(ComplexF64)
    end
    function sl!(du, u, p, t)
        z = u[1] + im * u[2]
        dz = (σ + im * ω) * z - z * abs2(z)
        du[1] = real(dz)
        du[2] = imag(dz)
    end
    u0 = [real(z0), imag(z0)]
    prob = ODEProblem(sl!, u0, (0.0, T))
    sol = solve(prob, Tsit5(); saveat=dt, abstol=1e-10, reltol=1e-10)
    t = sol.t
    x = [sol[1, i] for i in eachindex(t)]
    y = [sol[2, i] for i in eachindex(t)]
    if noise > 0
        x .+= noise .* randn(length(t))
        y .+= noise .* randn(length(t))
    end
    return t, x, y
end

"""
    simulate_fitzhugh_nagumo(; ε=0.08, a=0.7, b=0.8, I=0.5, T=100.0, dt=0.01)

Simulate the FitzHugh-Nagumo oscillator.
"""
function simulate_fitzhugh_nagumo(; ε=0.08, a=0.7, b=0.8, I_ext=0.5, T=100.0, dt=0.01)
    function fhn!(du, u, p, t)
        v, w = u
        du[1] = v - v^3 / 3 - w + I_ext
        du[2] = ε * (v + a - b * w)
    end
    u0 = [0.0, 0.0]
    prob = ODEProblem(fhn!, u0, (0.0, T))
    sol = solve(prob, Tsit5(); saveat=dt, abstol=1e-10, reltol=1e-10)
    t = sol.t
    v = [sol[1, i] for i in eachindex(t)]
    w = [sol[2, i] for i in eachindex(t)]
    return t, v, w
end

"""
    simulate_van_der_pol(; μ=1.0, T=50.0, dt=0.01)

Simulate the Van der Pol oscillator: x'' - μ(1 - x²)x' + x = 0
"""
function simulate_van_der_pol(; μ=1.0, T=50.0, dt=0.01)
    function vdp!(du, u, p, t)
        du[1] = u[2]
        du[2] = μ * (1 - u[1]^2) * u[2] - u[1]
    end
    u0 = [2.0, 0.0]
    prob = ODEProblem(vdp!, u0, (0.0, T))
    sol = solve(prob, Tsit5(); saveat=dt, abstol=1e-10, reltol=1e-10)
    t = sol.t
    x = [sol[1, i] for i in eachindex(t)]
    y = [sol[2, i] for i in eachindex(t)]
    return t, x, y
end

"""
    ground_truth_floquet_sl(; σ=0.1, ω=2π)

Analytical Floquet multipliers for the Stuart-Landau oscillator.
Returns (μ_phase, μ_transverse) where μ_phase = 1.0 and μ_transverse = exp(-2σT).
"""
function ground_truth_floquet_sl(; σ=0.1, ω=2π)
    T_period = 2π / ω
    μ_phase = 1.0
    μ_transverse = exp(-2σ * T_period)
    return μ_phase, μ_transverse
end

"""
    numerical_floquet(f!, u0_on_cycle, T_period; n_steps=1000)

Compute Floquet multipliers numerically by integrating the variational equation
alongside the nonlinear system for one period.
"""
function numerical_floquet(f!, u0_on_cycle, T_period; dt=0.001)
    n = length(u0_on_cycle)
    # State: [x; vec(Φ)] where Φ is the fundamental matrix
    function augmented!(du, u, p, t)
        x = u[1:n]
        Φ = reshape(u[n+1:end], n, n)
        # Compute f(x)
        dx = similar(x)
        f!(dx, x, nothing, t)
        du[1:n] .= dx
        # Compute Jacobian by finite differences
        J = zeros(n, n)
        ε_fd = 1e-7
        for j in 1:n
            x_plus = copy(x); x_plus[j] += ε_fd
            x_minus = copy(x); x_minus[j] -= ε_fd
            dx_plus = similar(x); dx_minus = similar(x)
            f!(dx_plus, x_plus, nothing, t)
            f!(dx_minus, x_minus, nothing, t)
            J[:, j] .= (dx_plus .- dx_minus) ./ (2ε_fd)
        end
        dΦ = J * Φ
        du[n+1:end] .= vec(dΦ)
    end
    u0_aug = vcat(u0_on_cycle, vec(Matrix{Float64}(I, n, n)))
    prob = ODEProblem(augmented!, u0_aug, (0.0, T_period))
    sol = solve(prob, Tsit5(); abstol=1e-12, reltol=1e-12, saveat=[T_period])
    Φ_T = reshape(sol[end][n+1:end], n, n)
    return eigen(Φ_T)
end

# ── Hankel DMD pipeline ──────────────────────────────────────────────────────
const FS              = 100.0
const TAU_GAIT        = 10
const NOISE_FLOOR     = 1e-6
const RESOLVENT_FLOOR = 1e-12

function build_hankel_multi(trial_data::AbstractMatrix, τ::Int)
    n_t, n_j = size(trial_data)
    H = zeros(n_j * (τ + 1), n_t - τ)
    for col in 1:(n_t - τ), lag in 0:τ
        H[lag*n_j+1 : (lag+1)*n_j, col] = trial_data[col + lag, :]
    end
    return H
end

"""
    build_hankel_1d(x::AbstractVector, τ::Int)

Build a Hankel matrix from a 1D time series.
"""
function build_hankel_1d(x::AbstractVector, τ::Int)
    n = length(x)
    H = zeros(τ + 1, n - τ)
    for col in 1:(n - τ), lag in 0:τ
        H[lag + 1, col] = x[col + lag]
    end
    return H
end

function get_stable_dmd_operator(trial_data::AbstractMatrix;
                                  τ::Int      = TAU_GAIT,
                                  nf::Float64 = NOISE_FLOOR)
    H        = build_hankel_multi(trial_data, τ)
    X, Y     = H[:, 1:end-1], H[:, 2:end]
    U, Σ, Vt = svd(X)
    r        = max(2, count(Σ .> Σ[1] * nf))
    Ã        = U[:, 1:r]' * Y * Vt[:, 1:r] * Diagonal(1.0 ./ Σ[1:r])
    Λ, W     = eigen(Ã)
    for i in eachindex(Λ)
        abs(Λ[i]) > 1.0 && (Λ[i] /= abs(Λ[i]))
    end
    return real.(W * Diagonal(Λ) * inv(W)), U[:, 1:r], r
end

"""
    get_stable_dmd_operator_1d(x; τ, nf)

DMD operator from a 1D time series via Hankel embedding.
"""
function get_stable_dmd_operator_1d(x::AbstractVector;
                                     τ::Int      = TAU_GAIT,
                                     nf::Float64 = NOISE_FLOOR)
    H        = build_hankel_1d(x, τ)
    X, Y     = H[:, 1:end-1], H[:, 2:end]
    U, Σ, Vt = svd(X)
    r        = max(2, count(Σ .> Σ[1] * nf))
    Ã        = U[:, 1:r]' * Y * Vt[:, 1:r] * Diagonal(1.0 ./ Σ[1:r])
    Λ, W     = eigen(Ã)
    for i in eachindex(Λ)
        abs(Λ[i]) > 1.0 && (Λ[i] /= abs(Λ[i]))
    end
    return real.(W * Diagonal(Λ) * inv(W)), U[:, 1:r], r
end

# Non-normality metrics
function henrici_departure(A::AbstractMatrix)
    opnorm(A' * A - A * A', 2) / opnorm(A, 2)^2
end

function eigvec_condition_number(A::AbstractMatrix)
    _, V = eigen(A)
    cond(V)
end

function transient_growth_curve(A::AbstractMatrix; n_steps=500)
    norms = zeros(n_steps + 1)
    An = Matrix{Float64}(I, size(A))
    norms[1] = opnorm(An, 2)
    for k in 1:n_steps
        An = An * A
        norms[k + 1] = opnorm(An, 2)
    end
    return norms
end

function kreiss_constant(A::AbstractMatrix; n_radii=50, n_angles=200)
    K = 0.0
    for r in range(1.001, 2.0, length=n_radii)
        for θ in range(0, 2π, length=n_angles+1)[1:end-1]
            z = r * exp(im * θ)
            resolvent_norm = opnorm(inv(z * I - A), 2)
            K = max(K, (r - 1) * resolvent_norm)
        end
    end
    return K
end

# Resolvent gain on the unit circle
function resolvent_gain_on_circle(A::AbstractMatrix; n_angles=2000, fs=FS)
    thetas  = range(0, 2π, length=n_angles+1)[1:end-1]
    n       = size(A, 1)
    gains   = zeros(n_angles)
    Threads.@threads for i in 1:n_angles
        z        = exp(im * thetas[i])
        gains[i] = 1.0 / max(minimum(svdvals(z * I(n) - A)), RESOLVENT_FLOOR)
    end
    freq_hz = collect(thetas) ./ (2π) .* fs
    nyq     = findfirst(freq_hz .> fs / 2)
    isnothing(nyq) && return freq_hz, gains
    return freq_hz[1:nyq-1], gains[1:nyq-1]
end

function find_stride_freq(freq_hz, gains; dc_cutoff_hz=0.3)
    mask = freq_hz .>= dc_cutoff_hz
    any(mask) || return freq_hz[argmax(gains)]
    idx = findall(mask)
    return freq_hz[idx[argmax(gains[idx])]]
end

"""
    pseudospectral_contours(A; grid_n=200, re_range=(-1.5, 1.5), im_range=(-1.5, 1.5))

Compute σ_min(zI - A) on a complex grid for pseudospectra visualization.
Returns (re_grid, im_grid, sigma_min_grid).
"""
function pseudospectral_contours(A::AbstractMatrix;
                                  grid_n=200,
                                  re_range=(-1.5, 1.5),
                                  im_range=(-1.5, 1.5))
    re_grid = range(re_range..., length=grid_n)
    im_grid = range(im_range..., length=grid_n)
    n = size(A, 1)
    σ_grid = zeros(grid_n, grid_n)
    Threads.@threads for j in 1:grid_n
        for i in 1:grid_n
            z = re_grid[i] + im * im_grid[j]
            σ_grid[j, i] = minimum(svdvals(z * I(n) - A))
        end
    end
    return collect(re_grid), collect(im_grid), σ_grid
end

fmt_pvalue(p) = p < 1e-3 ? "< 0.001" : "= $(round(p; digits=3))"

end # if !@isdefined(COMMON_JL_LOADED)
