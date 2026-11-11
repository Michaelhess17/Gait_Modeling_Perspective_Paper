#!/usr/bin/env julia
# Individual against group fits on all 299 gait trials (R1, R2, R4, R6).
#
# Each trial as in the self-driving benchmark: 50 Hz (every 2nd sample), the first 10 s train, the last 5 s are held out;
# forecasts start at the end of the training window and run freely for 30 cycles; scored with score() (first-cycle
# error, 30-cycle fate) against the trial's own z-scored data, with the mean-cycle template as the zero-skill reference,
# and with the ε-tube of r2_tube.jl (distance to the training states, ε = 0.5, 8 cycles). No noise is added.
#
#   individual  one model per trial, z-scored with that trial's training window, settings chosen on its own training data
#               (DMD: dmd_tuning grid and criterion; ESN: esn_tuning grid, 3 seeds; LSTM: Winner et al. training on the
#               trial alone, 128 units)
#   group       one model for all 299 trials, z-scored with the pooled training windows (keeping individual offsets and
#               amplitudes), settings chosen on the held-back last two training cycles of every trial (DMD, ESN) or on each
#               trial's last training segment (LSTM: Winner et al., 512 units, 4 seeds, GPU)
# Group forecasts are converted to each trial's own z-units before scoring. Wall-clock fit times are recorded.
#
# Usage: julia -t 16 --project=. scripts/gait_group_fits.jl <dmd|esn|lstm_ind>
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 8 --project=. scripts/gait_group_fits.jl lstm_ind gpu
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia --project=. scripts/gait_group_fits.jl lstm_grp
# → cache/gait_group/<mode>.jls

const MODE = isempty(ARGS) ? "dmd" : ARGS[1]
if MODE == "lstm_grp" || (length(ARGS) >= 2 && ARGS[2] == "gpu")
    using CUDA, cuDNN
end
include("dmd_tuning.jl")                       # self_driving_benchmark (score, fit_template, fit_esn, period_samples), fit_dmd_variant
include("lstm_model.jl")
using LinearAlgebra, SparseArrays, Random
BLAS.set_num_threads(1)

const OUT = cache_path("gait_group"); mkpath(OUT)
const EPS_TUBE = 0.5

# ── data, exactly as run_cell builds a gait cell (without the added 1% noise) ──
data_all, speed, group, subject = load_gait_data()
const NT = size(data_all, 1)
trials = map(1:NT) do k
    X = Float64.(data_all[k, 1:2:end, :]); P = period_samples(X)
    n_tr = 500; μ = mean(X[1:n_tr, :], dims = 1); s = std(X[1:n_tr, :], dims = 1)
    Y = (X .- μ) ./ s
    (; k, X, P, μ, s, Ytr = Y[1:n_tr, :], Yte = Y[n_tr+1:end, :], n_run = round(Int, N_LONG_CYC * P))
end
const μp = mean(reduce(vcat, [t.X[1:500, :] for t in trials]), dims = 1)
const sp = std(reduce(vcat, [t.X[1:500, :] for t in trials]), dims = 1)
pooled(t) = (t.X .- μp) ./ sp                                   # pooled z-units
to_trial(F, t) = ((F .* sp .+ μp) .- t.μ) ./ t.s                # pooled z-units → the trial's own z-units

nearest(x, R) = sqrt(minimum(sum(abs2, R .- x', dims = 2)) / size(R, 2))
function tube(Fc, t)
    n8 = min(size(Fc, 1), round(Int, 8t.P))
    i = findfirst(j -> !all(isfinite, Fc[j, :]) || nearest(Fc[j, :], t.Ytr) > EPS_TUBE, 1:n8)
    i === nothing ? Inf : (i - 1) / t.P
end
function evaluate(Fc, t)
    sc = try
        any(!isfinite, Fc[1:min(end, round(Int, t.P)), :]) ? (; e1 = Inf, e3 = Inf, e5 = Inf, amp = Inf, per = NaN, fate = "blow-up") :
            score(Fc, t.Yte, t.Ytr, t.P)
    catch
        (; e1 = Inf, e3 = Inf, e5 = Inf, amp = Inf, per = NaN, fate = "blow-up")
    end
    tpl = score(fit_template(t.Ytr, t.P)(t.Ytr, t.n_run), t.Yte, t.Ytr, t.P)
    (; sc..., e1_template = tpl.e1, skill1 = 1 - sc.e1 / tpl.e1, tube = tube(Fc, t))
end

# ── multi-trajectory Hankel-DMD (group) ──
function fit_dmd_group(Ys; τ, r, variant, P, ridge = 1e-8)
    Hs = [build_hankel_multi(Y, τ) for Y in Ys]
    Hc = reduce(hcat, Hs); r = min(r, size(Hc, 1) - 1)
    U = svd(Hc).U[:, 1:r]
    X = reduce(hcat, [U' * H[:, 1:end-1] for H in Hs]); Z = reduce(hcat, [U' * H[:, 2:end] for H in Hs])
    reg(M) = M + ridge * tr(M) / r * I
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    A = stabilise(A, P; mode = variant == "unit" ? :unit : :clip)
    d = size(Ys[1], 2)
    (Yh, n) -> begin
        α = U' * vec(permutedims(Yh[end-τ:end, :])); out = zeros(n, d)
        for k in 1:n; α = A * α; out[k, :] = (U * α)[τ*d+1:end]; end
        out
    end
end

# ── multi-trajectory ESN (group): one reservoir, readout trained on all trials, warm-up on the trial's own history ──
function fit_esn_group(Ys; N, ρ, α, λ, seed, σin = 0.5, washout = 50)
    rng = MersenneTwister(seed); d = size(Ys[1], 2)
    W = sprandn(rng, N, N, 10 / N); W .*= ρ / maximum(abs.(eigvals(Matrix(W))))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    step(r, u) = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * u .+ b)
    G = zeros(N + d + 1, N + d + 1); C = zeros(d, N + d + 1)
    for Y in Ys
        n = size(Y, 1); R = zeros(N, n - 1); r = zeros(N)
        for t in 1:n-1; r = step(r, Y[t, :]); R[:, t] = r; end
        F = vcat(R[:, washout:end], Y[washout:n-1, :]', ones(1, n - washout))
        G .+= F * F'; C .+= Y[washout+1:n, :]' * F'
    end
    Wout = C / (G + λ * I)
    (Yh, n) -> begin
        r = zeros(N)
        for t in 1:size(Yh, 1)-1; r = step(r, Yh[t, :]); end
        u = Yh[end, :]; out = zeros(n, d)
        for k in 1:n
            r = step(r, u); u = Wout * vcat(r, u, 1.0); out[k, :] = u
            (any(!isfinite, u) || maximum(abs, u) > 1e6) && (out[k:end, :] .= NaN; break)
        end
        out
    end
end

amp_ratio(fcl, Ytr, P) = mean(std(fcl[end-round(Int, 3P)+1:end, :], dims = 1)) / mean(std(Ytr, dims = 1))
verr(fc, Yv) = any(!isfinite, fc) ? Inf : sqrt(mean(abs2, fc .- Yv)) / sqrt(mean(abs2, Yv))

results = Dict{String, Any}()
if MODE == "dmd"
    # individual: dmd_tuning's grid and criterion on each trial
    ind = Vector{Any}(undef, NT)
    Threads.@threads for i in 1:NT
        t = trials[i]; P = t.P; nv = round(Int, 2P); Yfit, Yval = t.Ytr[1:end-nv, :], t.Ytr[end-nv+1:end, :]
        grid = [(; τ, r, variant) for τ in unique(max.(4, round.(Int, P .* (0.1, 0.25, 0.5, 1.0)))), r in (8, 16, 30, 50, 80),
                                      variant in ("ls", "fb", "unit")]
        t0 = time()
        vals = map(grid) do g
            try
                fcl = fit_dmd_variant(Yfit; g..., P)(Yfit, round(Int, 10P))
                any(!isfinite, fcl) ? Inf : verr(fcl[1:nv, :], Yval) + 0.5 * abs(log(max(amp_ratio(fcl, t.Ytr, P), 1e-3)))
            catch; Inf end
        end
        g = grid[argmin(vals)]
        t1 = time(); Fc = fit_dmd_variant(t.Ytr; g..., P)(t.Ytr, t.n_run); tfit = time() - t1
        ind[i] = (; k = t.k, chosen = g, evaluate(Fc, t)..., fit_secs = tfit, tune_secs = time() - t0, Fc = Float32.(Fc[1:round(Int, 8P), :]))
    end
    results["individual"] = ind
    # group: one operator for all trials, pooled units
    Pm = median([t.P for t in trials])
    Yp = [pooled(t) for t in trials]
    fitset = [Y[1:500-round(Int, 2t.P), :] for (Y, t) in zip(Yp, trials)]
    grid = [(; τ, r, variant) for τ in unique(round.(Int, Pm .* (0.1, 0.25, 0.5, 1.0))), r in (16, 30, 50, 80, 120, 200),
                                  variant in ("ls", "fb", "unit")]
    t0 = time()
    vals = Vector{Float64}(undef, length(grid))
    Threads.@threads for j in eachindex(grid)
        g = grid[j]
        vals[j] = try
            f = fit_dmd_group(fitset; g..., P = Pm)
            mean(map(1:NT) do i
                t = trials[i]; nv = round(Int, 2t.P)
                fcl = to_trial(f(fitset[i], round(Int, 10t.P)), t)
                any(!isfinite, fcl) ? 10.0 : min(10.0, verr(fcl[1:nv, :], t.Ytr[end-nv+1:end, :]) +
                                                     0.5 * abs(log(max(amp_ratio(fcl, t.Ytr, t.P), 1e-3))))
            end)
        catch; Inf end
    end
    g = grid[argmin(vals)]
    t1 = time(); f = fit_dmd_group([Y[1:500, :] for Y in Yp]; g..., P = Pm); tfit = time() - t1
    grp = map(1:NT) do i
        t = trials[i]; Fc = to_trial(f(Yp[i][1:500, :], t.n_run), t)
        (; k = t.k, chosen = g, evaluate(Fc, t)..., Fc = Float32.(Fc[1:round(Int, 8t.P), :]))
    end
    results["group"] = (; rows = grp, chosen = g, vals = collect(zip(grid, vals)), fit_secs = tfit, tune_secs = time() - t0)
elseif MODE == "esn"
    grid = [(; ρ, α, N, λ) for ρ in (0.5, 0.9, 1.2), α in (0.2, 0.5, 1.0), N in (200, 800), λ in (1e-6, 1e-3)]
    ind = Vector{Any}(undef, NT)
    Threads.@threads for i in 1:NT
        t = trials[i]; nv = round(Int, 2t.P); Yfit, Yval = t.Ytr[1:end-nv, :], t.Ytr[end-nv+1:end, :]
        t0 = time()
        vals = [median([verr(fit_esn(Yfit; g..., seed)(Yfit, nv), Yval) for seed in 1:3]) for g in grid]
        g = grid[argmin(vals)]
        rows = map(1:3) do seed
            t1 = time(); Fc = fit_esn(t.Ytr; g..., seed)(t.Ytr, t.n_run); tfit = time() - t1
            (; seed, evaluate(Fc, t)..., fit_secs = tfit, Fc = Float32.(Fc[1:round(Int, 8t.P), :]))
        end
        ind[i] = (; k = t.k, chosen = g, rows, tune_secs = time() - t0)
    end
    results["individual"] = ind
    Yp = [pooled(t) for t in trials]
    fitset = [Y[1:500-round(Int, 2t.P), :] for (Y, t) in zip(Yp, trials)]
    t0 = time()
    vals = Vector{Float64}(undef, length(grid))
    Threads.@threads for j in eachindex(grid)
        g = grid[j]
        vals[j] = median(map(1:2) do seed
            f = fit_esn_group(fitset; g..., seed)
            mean(map(1:NT) do i
                t = trials[i]; nv = round(Int, 2t.P)
                min(10.0, verr(to_trial(f(fitset[i], nv), t), t.Ytr[end-nv+1:end, :]))
            end)
        end)
    end
    g = grid[argmin(vals)]
    seeds = map(1:3) do seed
        t1 = time(); f = fit_esn_group([Y[1:500, :] for Y in Yp]; g..., seed); tfit = time() - t1
        rows = Vector{Any}(undef, NT)
        Threads.@threads for i in 1:NT
            t = trials[i]; Fc = to_trial(f(Yp[i][1:500, :], t.n_run), t)
            rows[i] = (; k = t.k, evaluate(Fc, t)..., Fc = Float32.(Fc[1:round(Int, 8t.P), :]))
        end
        (; seed, rows, fit_secs = tfit)
    end
    results["group"] = (; seeds, chosen = g, vals = collect(zip(grid, vals)), tune_secs = time() - t0)
elseif MODE == "lstm_ind"
    # one Julia process per shard (`lstm_ind j/n`, single-threaded): threads in one process contend for the garbage
    # collector under Zygote; `lstm_ind merge` collects the shards
    if length(ARGS) >= 2 && ARGS[2] == "gpu"
        # all 299 per-trial models trained at once on the GPU (fit_lstm_winner_many: identical to separate fits)
        t1 = time()
        ms = fit_lstm_winner_many([t.Ytr for t in trials]; H = 128, lookback = 124, trainseg = 3, valseg = 1, seed = 1,
                                  device = gpu_device(), verbose = true)
        tall = time() - t1
        println("lstm_ind (GPU, batched): $(round(tall / 60, digits = 1)) min for $NT models"); flush(stdout)
        rows = Vector{Any}(undef, NT)
        Threads.@threads for i in 1:NT
            t = trials[i]; Fc = forecast_lstm(ms[i], t.Ytr, t.n_run)
            rows[i] = (; k = t.k, evaluate(Fc, t)..., fit_secs = tall / NT, fit_secs_batch = tall, best_epoch = ms[i].best_epoch,
                       val = ms[i].val, Fc = Float32.(Fc[1:round(Int, 8t.P), :]))
        end
        results["individual"] = rows
    elseif length(ARGS) >= 2 && ARGS[2] == "merge"
        parts = [deserialize(f)["individual"] for f in filter(f -> occursin(r"^lstm_ind_part\d+\.jls$", basename(f)), readdir(OUT; join = true))]
        rows = sort(reduce(vcat, parts), by = r -> r.k); @assert [r.k for r in rows] == 1:NT
        results["individual"] = rows
    else
        j, n = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], "/")) : (1, 1)
        rows = Any[]
        for i in j:n:NT
            t = trials[i]; t1 = time()
            m = fit_lstm_winner([t.Ytr]; H = 128, lookback = 124, trainseg = 3, valseg = 1, seed = 1)
            tfit = time() - t1
            Fc = forecast_lstm(m, t.Ytr, t.n_run)
            push!(rows, (; k = t.k, evaluate(Fc, t)..., fit_secs = tfit, best_epoch = m.best_epoch, val = m.val,
                         Fc = Float32.(Fc[1:round(Int, 8t.P), :])))
            println("lstm_ind trial $i: best epoch $(m.best_epoch), $(round(tfit, digits = 1)) s"); flush(stdout)
        end
        if n > 1
            serialize(joinpath(OUT, "lstm_ind_part$j.jls"), Dict("individual" => rows)); exit()
        end
        results["individual"] = rows
    end
elseif MODE == "lstm_grp"
    Yp = [pooled(t) for t in trials]
    dev = gpu_device()
    seeds = map(1:4) do seed
        t1 = time()
        m = fit_lstm_winner([Y[1:500, :] for Y in Yp]; H = 512, lookback = 124, trainseg = 3, valseg = 1, seed,
                            device = dev, verbose = true)
        tfit = time() - t1
        rows = Vector{Any}(undef, NT)
        Threads.@threads for i in 1:NT
            t = trials[i]; Fc = to_trial(forecast_lstm(m, Yp[i][1:500, :], t.n_run), t)
            rows[i] = (; k = t.k, evaluate(Fc, t)..., Fc = Float32.(Fc[1:round(Int, 8t.P), :]))
        end
        println("lstm_grp seed $seed: best epoch $(m.best_epoch), $(round(tfit / 60, digits = 1)) min"); flush(stdout)
        (; seed, rows, fit_secs = tfit, best_epoch = m.best_epoch, history = m.history)
    end
    results["group"] = (; seeds)
end
serialize(joinpath(OUT, "$(MODE).jls"), results)

# summary
summ(rows) = @sprintf("skill1 median %+.2f (IQR %+.2f to %+.2f) | ok %3.0f%% | blow-up %3.0f%% | within ε 8 cycles %3.0f%%",
                      median([r.skill1 for r in rows]), quantile([r.skill1 for r in rows], 0.25), quantile([r.skill1 for r in rows], 0.75),
                      100count(r -> r.fate == "ok", rows) / length(rows), 100count(r -> r.fate == "blow-up", rows) / length(rows),
                      100count(r -> isinf(r.tube), rows) / length(rows))
haskey(results, "individual") && println("$(MODE) individual: ", summ(MODE == "esn" ? [r for x in results["individual"] for r in x.rows] : results["individual"]))
if haskey(results, "group")
    g = results["group"]
    println("$(MODE) group: ", summ(hasproperty(g, :seeds) ? [r for s in g.seeds for r in s.rows] : g.rows))
end
