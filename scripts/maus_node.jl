#!/usr/bin/env julia
# Neural ODEs on the long running trials (Figure 7), with the data, splits, horizons and scores of maus_learning_all.jl
# (deviation) and maus_ladder.jl (whole signal):
#   dev     the deviation from the 61-stride moving-average stride (stride-normalised; the cycle and stride timing are
#           given), with the stride phase as input (cos, sin), as the ESN and LSTM; every training size; noise η (fitted
#           and validated on the noisy deviation, started from it, scored against the clean one)
#   rt, sn  the whole signal in real time and in stride-normalised time, without a phase input (as Figure 5's neural ODE
#           and the whole-signal ESN); 150 training strides
# Model: dz/dτ = f(z, phase), τ in strides, f an MLP with 2 hidden layers of 64 tanh units (as Figure 5), its inputs
# scaled by the training SD of z and its outputs by the training SD of dz/dτ. State z: the channels themselves (d = 1, as
# Figure 5) or the leading r components of the last d = 8 samples (the phase-varying linear model's state); the forecast
# is the newest sample of the reconstructed window. Integrated with fourth-order Runge–Kutta at the sampling step.
# Training: multiple shooting on 25-sample segments (¼ stride, as Figure 5) starting anywhere in the training strides,
# minibatches of BATCH segments, Adam (1e-3, decaying to 1e-4 over the update budget); early stopping on the forecast
# skill at 6–50 samples ahead (HV) on the last fifth of the training strides, which the model is not trained on (one
# stride at the smallest size); the state (d, r) chosen on the same validation strides. Budget (one trial: the deviation
# peaks after 300–900 updates; the whole signal is still improving after 6000): deviation 8000 updates, patience 1500;
# whole signal 20000, patience 5000. Forecasts are clipped to the training range ± half its width, as maus_ladder.jl's
# whole-signal scores (divergence counts as a large, bounded error).
# All trials of a setting are trained at once on the GPU, one independent model per trial along a third array
# dimension (as fit_lstm_winner_many: the loss is the sum of the models' own losses and Adam acts elementwise).
# Usage: NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 16 --project=. scripts/maus_node.jl dev <η>
#            → cache/maus/node_dev_<η>.jls   (η = 0: every training size; otherwise 150 strides)
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 16 --project=. scripts/maus_node.jl whole  → cache/maus/node_whole.jls
#        julia -t 4 --project=. scripts/maus_node.jl test     (two trials on the CPU, a few hundred updates)
const NODE_MODE = get(ARGS, 1, "test")
const NODE_ETA = parse(Float64, get(ARGS, 2, "0"))
empty!(ARGS)                                    # maus_learning_all.jl reads ARGS
if NODE_MODE in ("dev", "whole")
    using CUDA, cuDNN
end
include("maus_ladder.jl")
using Zygote, Optimisers
const bmul = Lux.NNlib.batched_mul

const SEGL = 25; const BATCH = 512; const NHID = 64; const LR = 1e-3; const EVERY = 100
const NODE_STATES = [(1, 6), (8, 16), (8, 32)]  # (delay window d, components r)
const DT = Float32(1 / NPH)

# ── one model (CPU): forecasting and scoring ──
phin(τ) = vcat(cos.(2f0π .* τ'), sin.(2f0π .* τ'))
inp(Z, τ, ph) = ph ? vcat(Z, phin(τ)) : Z
vf(p, Z, τ, nrm, ph) = nrm.σd .* (p.W3 * tanh.(p.W2 * tanh.(p.W1 * inp(Z ./ nrm.σz, τ, ph) .+ p.b1) .+ p.b2) .+ p.b3)
function rk4(p, Z, τ, nrm, ph)
    k1 = vf(p, Z, τ, nrm, ph); k2 = vf(p, Z .+ (DT / 2) .* k1, τ .+ DT / 2, nrm, ph)
    k3 = vf(p, Z .+ (DT / 2) .* k2, τ .+ DT / 2, nrm, ph); k4 = vf(p, Z .+ DT .* k3, τ .+ DT, nrm, ph)
    Z .+ (DT / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end
function init_mlp(rng, nin, nout)
    g(o, i, s = 1.0) = Float32.(randn(rng, o, i) .* (s * sqrt(1 / i)))
    (; W1 = g(NHID, nin), b1 = zeros(Float32, NHID), W2 = g(NHID, NHID), b2 = zeros(Float32, NHID),
       W3 = g(nout, NHID, 0.1), b3 = zeros(Float32, nout))
end
"State model on rows of Yv: delay window d, r components (d = 1: the channels themselves)."
function state_model(Yv, rows, d, r)
    p = size(Yv, 2); μ = vec(mean(Yv[rows, :], dims = 1)); μv = repeat(μ, d)
    U = d == 1 ? Matrix{Float64}(I, p, p) : svd(hankel(Yv[rows, :] .- μ', d)).U[:, 1:min(r, p * d)]
    (; d, U = Float32.(U), μv = Float32.(μv), p)
end
"States (r × starts) at times ts, each the window ending there."
states(M, Yv, ts) = M.U' * (Float32.(reduce(hcat, [vec(permutedims(Yv[t-M.d+1:t, :])) for t in ts])) .- M.μv)
"Forecasts (p × starts) at horizons HZ[hidx] from starts ts; non-finite → 0, clipped to [lo, hi]."
function node_forecast(p, nrm, M, Yv, ts, hidx, ph, lo, hi)
    X = states(M, Yv, ts); τ = Float32.((collect(ts) .- 1) ./ NPH); out = Vector{Matrix{Float64}}(undef, length(hidx)); k = 1
    Ur = M.U[end-M.p+1:end, :]; μr = M.μv[end-M.p+1:end]
    for s in 1:HZ[hidx[end]]
        X = rk4(p, X, τ, nrm, ph); τ = τ .+ DT
        if s == HZ[hidx[k]]
            P = Float64.(Ur * X .+ μr); P[.!isfinite.(P)] .= 0; out[k] = clamp.(P, lo, hi); k += 1
        end
    end
    out
end
"Skill (R² against the deviation, as score_full / score_at) at horizons HZ[hidx] of forecasts F from starts ts."
skill(F, Yv, Dv, ts, hidx) = [1 - sum(abs2, F[j] .- Yv[ts .+ HZ[k], :]') / sum(abs2, Dv[ts .+ HZ[k], :]) for (j, k) in enumerate(hidx)]
"Training rows, validation starts and clipping bounds for ntr training strides with stride boundaries bidx."
function node_split(Yv, bidx, ntr)
    nv = max(1, round(Int, ntr / 5)); tr = 1:bidx[ntr-nv+1]-1; va = bidx[ntr-nv+1]:bidx[ntr+1]-1
    vs = collect(max(first(va), 8):5:(last(va) - HZ[HV[end]])); rows = 1:bidx[ntr+1]-1
    w = maximum(Yv[rows, :], dims = 1) .- minimum(Yv[rows, :], dims = 1)
    (; tr, vs, lo = vec(minimum(Yv[rows, :], dims = 1) .- 0.5w), hi = vec(maximum(Yv[rows, :], dims = 1) .+ 0.5w))
end

# ── many independent models at once (GPU), one per trial along the third dimension ──
inp3(Z, τ, ph) = ph ? vcat(Z, cos.(2f0π .* τ), sin.(2f0π .* τ)) : Z
vf3(p, Z, τ, σz, σd, ph) = σd .* (bmul(p.W3, tanh.(bmul(p.W2, tanh.(bmul(p.W1, inp3(Z ./ σz, τ, ph)) .+ p.b1)) .+ p.b2)) .+ p.b3)
function rk43(p, Z, τ, σz, σd, ph)
    k1 = vf3(p, Z, τ, σz, σd, ph); k2 = vf3(p, Z .+ (DT / 2) .* k1, τ .+ DT / 2, σz, σd, ph)
    k3 = vf3(p, Z .+ (DT / 2) .* k2, τ .+ DT / 2, σz, σd, ph); k4 = vf3(p, Z .+ DT .* k3, τ .+ DT, σz, σd, ph)
    Z .+ (DT / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end
"Sum over models of each model's multiple-shooting mean squared error."
function loss3(p, X0, τ0, Ts, σz, σd, ph)
    X = X0; τ = τ0; tot = 0f0
    for k in 1:SEGL
        X = rk43(p, X, τ, σz, σd, ph); τ = τ .+ DT
        tot += sum(abs2, X .- Ts[k])
    end
    tot / (SEGL * size(X0, 2) * size(X0, 1))
end

"""Train one neural ODE per trial (state d, r) on rows S[i].tr of Ys[i], early stopping on each trial's own validation
starts (skill against Dvs[i] at HV). Returns per-trial CPU models."""
function fit_node_many(Ys, Dvs, S, d, r, ph; maxup, patience, dev, seed = 1)
    M = length(Ys)
    SM = [state_model(Ys[i], S[i].tr, d, r) for i in 1:M]
    tt = [max(first(S[i].tr), d):last(S[i].tr) for i in 1:M]
    Zs = [states(SM[i], Ys[i], tt[i]) for i in 1:M]; rr = size(Zs[1], 1); pp = SM[1].p
    nrms = map(Zs) do Z
        dZ = (Z[:, 2:end] .- Z[:, 1:end-1]) ./ DT; sz = vec(std(Z, dims = 2))
        (; σz = max.(sz, 1f-3 * maximum(sz)), σd = vec(std(dZ, dims = 2)))
    end
    σz = reshape(reduce(hcat, [n.σz for n in nrms]), rr, 1, M) |> dev; σd = reshape(reduce(hcat, [n.σd for n in nrms]), rr, 1, M) |> dev
    inits = [init_mlp(MersenneTwister(seed), rr + 2ph, rr) for _ in 1:M]
    st3(f) = cat([f(q) for q in inits]...; dims = 3)
    p = (; W1 = st3(q -> q.W1), b1 = st3(q -> reshape(q.b1, :, 1)), W2 = st3(q -> q.W2), b2 = st3(q -> reshape(q.b2, :, 1)),
           W3 = st3(q -> q.W3), b3 = st3(q -> reshape(q.b3, :, 1))) |> dev
    Nmax = maximum(size.(Zs, 2)); Zp = zeros(Float32, rr, Nmax, M); τp = zeros(Float32, 1, Nmax, M)
    for i in 1:M; Zp[:, 1:size(Zs[i], 2), i] = Zs[i]; τp[1, 1:size(Zs[i], 2), i] = Float32.((collect(tt[i]) .- 1) ./ NPH); end
    Zfl = reshape(Zp, rr, Nmax * M) |> dev; τfl = reshape(τp, 1, Nmax * M) |> dev
    ncols = [size(Z, 2) - SEGL for Z in Zs]
    # validation starts, padded to the longest
    V = [length(S[i].vs) for i in 1:M]; Vmax = maximum(V)
    Xv = zeros(Float32, rr, Vmax, M); τv = zeros(Float32, 1, Vmax, M)
    for i in 1:M; Xv[:, 1:V[i], i] = states(SM[i], Ys[i], S[i].vs); τv[1, 1:V[i], i] = Float32.((S[i].vs .- 1) ./ NPH); end
    Xv = Xv |> dev; τv = τv |> dev
    Ur3 = cat([SM[i].U[end-pp+1:end, :] for i in 1:M]...; dims = 3) |> dev; μr3 = reshape(reduce(hcat, [SM[i].μv[end-pp+1:end] for i in 1:M]), pp, 1, M) |> dev
    opt = Optimisers.setup(Optimisers.Adam(Float32(LR)), p); rng = MersenneTwister(seed)
    best = fill(-Inf, M); bestu = zeros(Int, M); bestp = Vector{Any}(nothing, M); since = zeros(Int, M); hist = Vector{Float64}[]
    for u in 1:maxup
        cols = reduce(vcat, [rand(rng, 1:ncols[i], BATCH) .+ (i - 1) * Nmax for i in 1:M]) |> dev
        X0 = reshape(Zfl[:, cols], rr, BATCH, M); τ0 = reshape(τfl[:, cols], 1, BATCH, M)
        Ts = [reshape(Zfl[:, cols .+ k], rr, BATCH, M) for k in 1:SEGL]
        g = Zygote.gradient(q -> loss3(q, X0, τ0, Ts, σz, σd, ph), p)[1]
        opt, p = Optimisers.update(opt, p, g)
        u % EVERY == 0 || continue
        Optimisers.adjust!(opt, Float32(LR * 0.1^(u / maxup)))
        X = Xv; τ = τv; P = Dict{Int, Array{Float32, 3}}()
        for s in 1:HZ[HV[end]]
            X = rk43(p, X, τ, σz, σd, ph); τ = τ .+ DT
            s in HZ[HV] && (P[s] = Array(bmul(Ur3, X) .+ μr3))
        end
        pc = nothing; vals = zeros(M)
        for i in 1:M
            F = [(Q = Float64.(P[HZ[k]][:, 1:V[i], i]); Q[.!isfinite.(Q)] .= 0; clamp.(Q, S[i].lo, S[i].hi)) for k in HV]
            v = mean(skill(F, Ys[i], Dvs[i], S[i].vs, HV)); vals[i] = v
            if v > best[i]
                pc === nothing && (pc = map(Array, p))
                best[i] = v; bestu[i] = u; since[i] = 0
                bestp[i] = (; W1 = pc.W1[:, :, i], b1 = vec(pc.b1[:, :, i]), W2 = pc.W2[:, :, i], b2 = vec(pc.b2[:, :, i]),
                              W3 = pc.W3[:, :, i], b3 = vec(pc.b3[:, :, i]))
            else
                since[i] += EVERY
            end
        end
        push!(hist, vals)
        all(since .>= patience) && break
    end
    [(; p = bestp[i], nrm = nrms[i], M = SM[i], val = best[i], best_update = bestu[i], d, r, hist = [h[i] for h in hist]) for i in 1:M]
end

"Fit every state setting for all trials at once; per trial keep the setting with the best validation skill."
function fit_select(Ys, Dvs, S, ph; maxup, patience, dev)
    fits = [fit_node_many(Ys, Dvs, S, d, r, ph; maxup, patience, dev) for (d, r) in NODE_STATES]
    map(eachindex(Ys)) do i
        vs = [f[i].val for f in fits]; k = argmax(vs)
        (; f = fits[k][i], vals = vs, best_updates = [f[i].best_update for f in fits])
    end
end
"Score the chosen models on the test starts (forecasts from Yf, scored against Ys and Ds)."
function score_many(sel, Yf, Ys, Ds, S, tss, ph)
    out = Vector{Any}(undef, length(sel))
    Threads.@threads for i in eachindex(sel)
        f = sel[i].f
        fc = node_forecast(f.p, f.nrm, f.M, Yf[i], tss[i], 1:length(HZ), ph, S[i].lo, S[i].hi)
        out[i] = (; r2 = skill(fc, Ys[i], Ds[i], tss[i], 1:length(HZ)), setting = (f.d, f.r), val = sel[i].vals,
                  best_update = sel[i].best_updates, hist = f.hist)
    end
    out
end

function run_dev(SS, η; sizes, maxup = 8000, patience = 1500, dev)
    DD = [deviations(S, η) for S in SS]; out = Dict{Int, Any}()
    bidx = [[(s - 1) * NPH + 1 for s in 1:size(S, 1)+1] for S in SS]
    tss = [(n = size(S, 1); nh = n ÷ 2; collect((nh * NPH + 1 + HISTL):STEP:(n * NPH - HZ[end]))) for S in SS]
    for ntr in sizes
        ok = [i for i in eachindex(SS) if ntr <= size(SS[i], 1) ÷ 2]
        Ys = [DD[i][2] for i in ok]; S = [node_split(DD[i][2], bidx[i], ntr) for i in ok]
        t0 = time(); sel = fit_select(Ys, Ys, S, true; maxup, patience, dev); tf = time() - t0
        rows = score_many(sel, Ys, [DD[i][1] for i in ok], [DD[i][1] for i in ok], S, tss[ok], true)
        out[ntr] = (; rows, trials = ok, fit_secs = tf)
        v = [x.r2 for x in rows]
        @printf("node dev η %.1f, %3d strides: R² ⅛ / ¼ / ½ stride %.2f / %.2f / %.2f | %.1f min | settings %s\n", η, ntr,
                median(x[2] for x in v), median(x[4] for x in v), median(x[6] for x in v), tf / 60, [x.setting for x in rows]); flush(stdout)
    end
    out
end
function run_whole(T; ntr = 150, maxup = 20000, patience = 5000, dev)
    RR = [rung_data(t.X) for t in T]; out = Dict{Symbol, Any}()
    for rung in (:rt, :sn)
        Ys = [rung == :rt ? R.Yrt : R.Y for R in RR]; Ds = [rung == :rt ? R.Drt : R.D for R in RR]
        bidx = [rung == :rt ? R.bidx_rt : R.bidx_sn for R in RR]
        S = [node_split(Ys[i], bidx[i], ntr) for i in eachindex(RR)]
        tss = [collect((bidx[i][RR[i].n÷2+1] + HISTW):STEP:(size(Ys[i], 1) - HZ[end])) for i in eachindex(RR)]
        t0 = time(); sel = fit_select(Ys, Ds, S, false; maxup, patience, dev); tf = time() - t0
        out[rung] = (; rows = score_many(sel, Ys, Ys, Ds, S, tss, false), fit_secs = tf)
        v = [x.r2 for x in out[rung].rows]
        @printf("node whole %s: R² of the deviation ⅛ / ¼ / 1 stride %.2f / %.2f / %.2f | %.1f min | settings %s\n", rung,
                median(x[2] for x in v), median(x[4] for x in v), median(x[end] for x in v), tf / 60, [x.setting for x in out[rung].rows]); flush(stdout)
    end
    out
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    SS = [pair_strides(t.X) for t in A.trials]; keep = qc_run.(SS) .< 5
    SS = SS[keep]; T = A.trials[keep]; ids = [(t.subj, t.rep) for t in T]
    @printf("%d trials, %d threads, mode %s, η = %.1f\n", length(SS), Threads.nthreads(), NODE_MODE, NODE_ETA); flush(stdout)
    if NODE_MODE == "test"
        o = run_dev(SS[1:2], 0.0; sizes = [16], maxup = 300, patience = 300, dev = identity)
        o = run_whole(T[1:2]; maxup = 300, patience = 300, dev = identity)
    elseif NODE_MODE == "dev"
        sizes = NODE_ETA == 0 ? SIZES_ALL : [150]
        out = run_dev(SS, NODE_ETA; sizes, dev = gpu_device())
        serialize(joinpath(ROOT, "cache", "maus", "node_dev_$(NODE_ETA).jls"), (; out, ids, SIZES = sizes, HZ, ETA = NODE_ETA))
    elseif NODE_MODE == "whole"
        out = run_whole(T; dev = gpu_device())
        serialize(joinpath(ROOT, "cache", "maus", "node_whole.jls"), (; out, ids, HZ))
    end
end
