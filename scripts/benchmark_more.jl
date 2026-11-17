#!/usr/bin/env julia
# The free-running benchmark (self_driving_benchmark.jl; Figure 7 A–D) for the model classes it did not include: the
# phase-varying linear model (Fourier in phase) and the LSTM. Same cells (systems × regimes), the same exported data
# (cache/selfdrive/data_*.npz: training window, held-out data, period, run length) and the same scoring (score: error over
# the first held-out cycles, and the fate after 30 free-running cycles).
#   fip   Fourier in phase on the whole signal (affine): the leading r components of the last d samples evolve as
#         z₊ = A(θ) z + b(θ), A and b Fourier series in the phase θ. The phase is read from the state, causally: the angle
#         of the two leading principal components of the last half cycle of the signal (during a free run, of the model's
#         own forecast). Settings (d ∈ {P/10, P/4}, r ∈ {8, 16, 32}, harmonics ∈ {2, 4, 8}, ridge ∈ {1e-4, 1e-3, 1e-2})
#         chosen by the benchmark's training-data criterion (dmd_tuning.jl: fitted on all but the last two training
#         cycles; error of a free run over those cycles plus 0.5 |log amplitude ratio| of its cycles 8–10)
#   lstm  LSTM trained as in @winner2023 (lstm_model.jl: lookback of one cycle, 128 units, early stopping on the
#         validation segment after the training segments), 2 seeds; on the CPU, one fit per thread
# Usage: julia -t 13 --project=. scripts/benchmark_more.jl fip    → cache/selfdrive/fip_<system>_<regime>.jls
#        julia -t 12 --project=. scripts/benchmark_more.jl lstm   → cache/selfdrive/lstm_<system>_<regime>.jls
const BM_MODE = get(ARGS, 1, "fip")
empty!(ARGS)
include("self_driving_benchmark.jl")
include("lstm_model.jl")
using NPZ
const ROOT = dirname(@__DIR__)

const CELLS = [(s, r) for s in ("fhn", "cfhn", "vdp2", "slip") for r in ("long", "realistic")] ∪
              [(g, "realistic") for g in ("gait91", "gait243", "gait231", "gait63", "gait267")]
fbθ(θ, H) = vcat(1.0, [f(h * θ) for h in 1:H for f in (cos, sin)])
windows(Y, L) = reduce(hcat, [vec(permutedims(Y[t-L+1:t, :])) for t in L:size(Y, 1)])     # column t-L+1: window ending at t

"Fit the affine Fourier-in-phase model on Y; returns a forecaster (Yhist, n) -> n × p."
function fit_fip(Y; d, r, H, λ, P)
    n, p = size(Y); L = max(4, round(Int, P / 2)); μ = vec(mean(Y, dims = 1))
    Wp = windows(Y .- μ', L); Up = svd(Wp).U[:, 1:2]                                   # phase from the last half cycle
    θof(w) = (z = Up' * w; atan(z[2], z[1]))
    Wd = windows(Y .- μ', d); U = svd(Wd).U[:, 1:min(r, d * p)]; rr = size(U, 2)
    t0 = max(L, d); ts = t0:n-1
    θ = [θof(Wp[:, t-L+1]) for t in ts]; Z = U' * Wd[:, ts .- d .+ 1]; Zn = U' * Wd[:, ts .- d .+ 2]
    Φ = reduce(hcat, [(f = fbθ(θ[k], H); vcat(kron(f, Z[:, k]), f)) for k in eachindex(ts)])
    G = Φ * Φ'; Wt = Zn * Φ' / (G + λ * mean(diag(G)) * I); nf = 2H + 1
    A(θ) = (f = fbθ(θ, H); sum(f[i] .* Wt[:, (i-1)*rr+1:i*rr] for i in 1:nf)); b(θ) = Wt[:, nf*rr+1:end] * fbθ(θ, H)
    function forecast(Yh, nf_)
        buf = Yh[end-max(L, d)+1:end, :] .- μ'; out = zeros(nf_, p)
        z = U' * vec(permutedims(buf[end-d+1:end, :]))
        for k in 1:nf_
            θk = θof(vec(permutedims(buf[end-L+1:end, :])))
            z = A(θk) * z .+ b(θk); y = (U * z)[end-p+1:end]
            (any(!isfinite, y) || maximum(abs, y) > 1e3) && (out[k:end, :] .= NaN; break)
            out[k, :] = y .+ μ; buf = vcat(buf[2:end, :], y')
        end
        out
    end
    forecast
end
"The benchmark's training-data criterion (dmd_tuning.jl)."
function criterion(fc_fit, Yfit, Yval, P, amp_ref)
    fcl = fc_fit(Yfit, round(Int, 10P)); any(!isfinite, fcl) && return Inf
    nv = size(Yval, 1); amp = mean(std(fcl[end-round(Int, 3P)+1:end, :], dims = 1)) / amp_ref
    sqrt(mean(abs2, fcl[1:nv, :] .- Yval)) / sqrt(mean(abs2, Yval)) + 0.5 * abs(log(max(amp, 1e-3)))
end

function cell_fip(sys, reg)
    D = npzread(joinpath(ROOT, "cache", "selfdrive", "data_$(sys)_$(reg).npz"))
    Ytr, Yte, P, n_run = D["Ytr"], D["Yte"], D["P"], Int(D["n_run"])
    nv = round(Int, 2P); Yfit, Yval = Ytr[1:end-nv, :], Ytr[end-nv+1:end, :]; amp_ref = mean(std(Ytr, dims = 1))
    grid = [(; d, r, H, λ) for d in unique(max.(4, round.(Int, P .* (0.1, 0.25)))), r in (8, 16, 32), H in (2, 4, 8), λ in (1e-4, 1e-3, 1e-2)] |> vec
    vals = [something(try criterion(fit_fip(Yfit; g..., P), Yfit, Yval, P, amp_ref) catch; nothing end, Inf) for g in grid]
    g = grid[argmin(vals)]
    Fc = try fit_fip(Ytr; g..., P)(Ytr, n_run) catch; fill(NaN, n_run, size(Ytr, 2)) end
    sc = score(Fc, Yte, Ytr, P)
    (; system = sys, regime = reg, rows = [(; model = "FIP", variant = string(g), sc..., Fc = Float32.(Fc[1:min(end, round(Int, 8P)), :]))])
end
function run_lstm(sys, reg, seed)
    D = npzread(joinpath(ROOT, "cache", "selfdrive", "data_$(sys)_$(reg).npz"))
    Ytr, Yte, P, n_run = D["Ytr"], D["Yte"], D["P"], Int(D["n_run"])
    L = round(Int, P); trainseg = max(1, (size(Ytr, 1) - 1 - L) ÷ (L + 1) - 1)
    m = fit_lstm_winner([Ytr]; H = 128, lookback = L, trainseg, valseg = 1, seed)
    Fc = forecast_lstm(m, Ytr, n_run); sc = score(Fc, Yte, Ytr, P)
    (; model = "LSTM", variant = "seed $seed", sc..., Fc = Float32.(Fc[1:min(end, round(Int, 8P)), :]))
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    if BM_MODE == "fip"
        out = Vector{Any}(undef, length(CELLS))
        Threads.@threads for i in eachindex(CELLS)
            out[i] = try cell_fip(CELLS[i]...) catch e; @warn "fip failed" CELLS[i] e; nothing end
        end
        for o in out
            o === nothing && continue
            serialize(joinpath(ROOT, "cache", "selfdrive", "fip_$(o.system)_$(o.regime).jls"), o)
            r = o.rows[1]; @printf("%-8s %-9s FIP %-40s | e1 %.3f | %s\n", o.system, o.regime, r.variant, r.e1, r.fate)
        end
    else
        jobs = [(c..., seed) for c in CELLS for seed in 1:2]; res = Vector{Any}(undef, length(jobs))
        Threads.@threads :greedy for i in eachindex(jobs)
            t = time()
            res[i] = try run_lstm(jobs[i]...) catch e; @warn "lstm failed" jobs[i] e; nothing end
            res[i] === nothing || @printf("%-8s %-9s LSTM seed %d | e1 %.3f | %s | %.0f min\n", jobs[i]..., res[i].e1, res[i].fate, (time() - t) / 60)
            flush(stdout)
        end
        for (sys, reg) in CELLS
            rows = [r for (j, r) in zip(jobs, res) if j[1:2] == (sys, reg) && r !== nothing]
            serialize(joinpath(ROOT, "cache", "selfdrive", "lstm_$(sys)_$(reg).jls"), (; system = sys, regime = reg, rows))
        end
    end
end
