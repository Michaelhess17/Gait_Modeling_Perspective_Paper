#!/usr/bin/env julia
# When does one model for many recordings beat one model per recording? Group fits on pools of gait trials that span
# heterogeneity (H, heterogeneity.jl) and size (N), against the per-trial fits of gait_group_fits.jl.
#
# Pools (each trial as in the self-driving benchmark: 50 Hz, first 10 s train, last 5 s held out):
#   nearest   a seed trial and its N − 1 nearest trials (pairwise H ratio in the all-trial embedding): homogeneous pools
#   random    N random trials
#   subject   all trials of one participant (5–9 trials, different speeds)
#   speed     one trial per participant from a speed quintile of one group (able-bodied, two post-stroke groups)
#   group     each participant group, and all 299 trials
# For each pool: H in the pool's own pooled z-units; a group DMD (multi-trajectory Hankel-DMD, delay window, rank and
# variant chosen as in gait_group_fits.jl on the held-back last two training cycles of up to 30 members); every member
# forecast from the end of its training window and scored as there (first-cycle skill against the mean-cycle template,
# 30-cycle fate, ε-tube); and a leave-one-out test: the chosen model refitted without a member (z-scored without it)
# and forecasting it, for up to 6 members: generalisation to a recording the group model has not seen.
#
# LSTM mode: Winner et al. group LSTMs (lstm_model.jl, 512 units, GPU) on a subset of the DMD pools (nearest and random
# pools of 3–100 trials, the participants with the lowest and highest H, the three participant groups); one member is
# held out of training, so one fit gives the members' own skill and that of an unseen trial; the baseline is each
# trial's own LSTM (gait_group_fits.jl lstm_ind).
#
# Usage: julia -t 8 --project=. scripts/heterogeneity_pools.jl      → cache/heterogeneity/gait_pools_dmd.jls
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 8 --project=. scripts/heterogeneity_pools.jl lstm
#                                                                     → cache/heterogeneity/gait_pools_lstm.jls

const MODE = isempty(ARGS) ? "dmd" : ARGS[1]
if MODE == "lstm"
    using CUDA, cuDNN
end
include("dmd_tuning.jl")                       # score, fit_template, period_samples, stabilise, build_hankel_multi
include("heterogeneity.jl")
include("lstm_model.jl")
using LinearAlgebra, Random, Printf
BLAS.set_num_threads(1)

const OUT = cache_path("heterogeneity"); mkpath(OUT)
const EPS_TUBE = 0.5
const MAXVAL, MAXLOO = 30, 6

data_all, speed, group, subject = load_gait_data()
const NT = size(data_all, 1)
trials = map(1:NT) do k
    X = Float64.(data_all[k, 1:2:end, :]); P = period_samples(X)
    n_tr = 500; μ = mean(X[1:n_tr, :], dims = 1); s = std(X[1:n_tr, :], dims = 1)
    Y = (X .- μ) ./ s
    (; k, X, P, μ, s, Ytr = Y[1:n_tr, :], Yte = Y[n_tr+1:end, :], n_run = round(Int, N_LONG_CYC * P))
end
ind = deserialize(joinpath(cache_path("gait_group"), "dmd.jls"))["individual"]
@assert all(ind[i].k == i for i in 1:NT)

nearest(x, R) = sqrt(minimum(sum(abs2, R .- x', dims = 2)) / size(R, 2))
function tube(Fc, t)
    n8 = min(size(Fc, 1), round(Int, 8t.P))
    i = findfirst(j -> !all(isfinite, Fc[j, :]) || nearest(Fc[j, :], t.Ytr) > EPS_TUBE, 1:n8)
    i === nothing ? Inf : (i - 1) / t.P
end
function evaluate(Fc, t)
    sc = try
        any(!isfinite, Fc[1:min(end, round(Int, t.P)), :]) ? (; e1 = Inf, fate = "blow-up") : score(Fc, t.Yte, t.Ytr, t.P)
    catch
        (; e1 = Inf, fate = "blow-up")
    end
    (; k = t.k, skill1 = 1 - sc.e1 / ind[t.k].e1_template, fate = sc.fate, tube = tube(Fc, t))
end
amp_ratio(fcl, Ytr, P) = mean(std(fcl[end-round(Int, 3P)+1:end, :], dims = 1)) / mean(std(Ytr, dims = 1))
verr(fc, Yv) = any(!isfinite, fc) ? Inf : sqrt(mean(abs2, fc .- Yv)) / sqrt(mean(abs2, Yv))

# ── multi-trajectory Hankel-DMD, sufficient statistics accumulated per trial (= fit_dmd_group of gait_group_fits.jl) ──
struct HankelPool
    τ::Int; Hs::Vector{Matrix{Float64}}; U::Matrix{Float64}
end
function HankelPool(Ys, τ)
    Hs = [build_hankel_multi(Y, τ) for Y in Ys]
    C = sum(H * H' for H in Hs); E = eigen(Symmetric(C))
    HankelPool(τ, Hs, E.vectors[:, sortperm(E.values, rev = true)])
end
function fit_pool(hp::HankelPool; r, variant, P, ridge = 1e-8)
    r = min(r, size(hp.U, 1) - 1); U = hp.U[:, 1:r]
    XX = zeros(r, r); ZX = zeros(r, r); ZZ = zeros(r, r)
    for H in hp.Hs
        a = U' * H; X, Z = a[:, 1:end-1], a[:, 2:end]
        XX .+= X * X'; ZX .+= Z * X'; ZZ .+= Z * Z'
    end
    reg(M) = M + ridge * tr(M) / r * I
    A = ZX / reg(XX)
    variant == "fb" && (A = real(sqrt(complex(A / (ZX' / reg(ZZ))))))
    A = stabilise(A, P; mode = variant == "unit" ? :unit : :clip)
    τ = hp.τ; d = size(hp.Hs[1], 1) ÷ (τ + 1)
    (Yh, n) -> begin
        α = U' * vec(permutedims(Yh[end-τ:end, :])); out = zeros(n, d)
        for k in 1:n; α = A * α; out[k, :] = (U * α)[τ*d+1:end]; end
        out
    end
end

pool_stats(idx) = (μ = mean(reduce(vcat, [trials[i].X[1:500, :] for i in idx]), dims = 1),
                   s = std(reduce(vcat, [trials[i].X[1:500, :] for i in idx]), dims = 1))
zp(t, st) = (t.X .- st.μ) ./ st.s
to_trial(F, t, st) = ((F .* st.s .+ st.μ) .- t.μ) ./ t.s

function run_pool(kind, label, idx; rng = MersenneTwister(hash(label)))
    N = length(idx); st = pool_stats(idx); Yp = Dict(i => zp(trials[i], st) for i in idx)
    Pm = median([trials[i].P for i in idx])
    het = heterogeneity([(Yp[i][1:250, :], Yp[i][251:500, :]) for i in idx]; P = Pm)
    # choose delay window, rank, variant on the held-back last two training cycles
    fitset = [Yp[i][1:500-round(Int, 2trials[i].P), :] for i in idx]
    vidx = N <= MAXVAL ? collect(1:N) : sort(randperm(rng, N)[1:MAXVAL])
    t0 = time(); best = (Inf, nothing)
    for τ in unique(round.(Int, Pm .* (0.1, 0.25, 0.5, 1.0)))
        hp = HankelPool(fitset, τ)
        for r in (16, 30, 50, 80, 120, 200), variant in ("ls", "fb", "unit")
            v = try
                f = fit_pool(hp; r, variant, P = Pm)
                mean(map(vidx) do j
                    t = trials[idx[j]]; nv = round(Int, 2t.P)
                    fcl = to_trial(f(fitset[j], round(Int, 10t.P)), t, st)
                    any(!isfinite, fcl) ? 10.0 : min(10.0, verr(fcl[1:nv, :], t.Ytr[end-nv+1:end, :]) +
                                                         0.5 * abs(log(max(amp_ratio(fcl, t.Ytr, t.P), 1e-3))))
                end)
            catch; Inf end
            v < best[1] && (best = (v, (; τ, r, variant)))
        end
    end
    g = best[2]; tune_secs = time() - t0
    t1 = time(); f = fit_pool(HankelPool([Yp[i][1:500, :] for i in idx], g.τ); g.r, g.variant, P = Pm); fit_secs = time() - t1
    rows = map(idx) do i
        t = trials[i]; e = evaluate(to_trial(f(Yp[i][1:500, :], t.n_run), t, st), t)
        (; e..., skill1_ind = ind[i].skill1, fate_ind = ind[i].fate)
    end
    # leave-one-out: a member the group model has not seen
    loo = map(N <= MAXLOO ? idx : idx[sort(randperm(rng, N)[1:MAXLOO])]) do i
        rest = setdiff(idx, [i]); st2 = pool_stats(rest)
        f2 = fit_pool(HankelPool([zp(trials[j], st2)[1:500, :] for j in rest], g.τ); g.r, g.variant,
                      P = median([trials[j].P for j in rest]))
        t = trials[i]
        (; evaluate(to_trial(f2(zp(t, st2)[1:500, :], t.n_run), t, st2), t)..., skill1_ind = ind[i].skill1)
    end
    (; kind, label, idx, N, H = het.H, k = het.k, chosen = g, rows, loo, tune_secs, fit_secs)
end

# ── LSTM on a subset of the DMD pools ──────────────────────────────────────────────────────────────────────────────
if MODE == "lstm"
    D = deserialize(joinpath(OUT, "gait_pools_dmd.jls"))
    indl = deserialize(joinpath(cache_path("gait_group"), "lstm_ind.jls"))["individual"]
    @assert all(indl[i].k == i for i in 1:NT)
    nseeds = unique([parse(Int, split(p.label)[2]) for p in D.pools if p.kind == "nearest"])[1:3]
    subj = sort(filter(p -> p.kind == "subject", D.pools), by = p -> p.H)
    sel = vcat(filter(p -> p.kind == "nearest" && parse(Int, split(p.label)[2]) in nseeds && p.N in (3, 6, 25, 100), D.pools),
               filter(p -> p.kind == "random" && p.N in (3, 6, 25, 100) && endswith(p.label, r"#[12]"), D.pools),
               subj[1:4], subj[end-1:end], filter(p -> p.kind == "group" && p.label != "all", D.pools))
    println("$(length(sel)) pools for the LSTM"); flush(stdout)
    dev = gpu_device(); res = Any[]
    for p in sel
        rng = MersenneTwister(hash(p.label)); out = p.idx[rand(rng, 1:p.N)]; rest = setdiff(p.idx, [out])
        st = pool_stats(rest)
        t1 = time()
        m = fit_lstm_winner([zp(trials[i], st)[1:500, :] for i in rest]; H = 512, lookback = 124, trainseg = 3, valseg = 1,
                            seed = 1, device = dev)
        fit_secs = time() - t1
        ev(i) = (t = trials[i]; evaluate(to_trial(forecast_lstm(m, zp(t, st)[1:500, :], t.n_run), t, st), t))
        rows = Vector{Any}(undef, length(rest))
        Threads.@threads for j in eachindex(rest)
            i = rest[j]; rows[j] = (; ev(i)..., skill1_ind = indl[i].skill1, fate_ind = indl[i].fate)
        end
        loo = [(; ev(out)..., skill1_ind = indl[out].skill1)]
        r = (; p.kind, p.label, idx = p.idx, held_out = out, p.N, p.H, rows, loo, fit_secs, best_epoch = m.best_epoch, val = m.val)
        push!(res, r)
        @printf("%-22s N %3d  H %5.2f  Δskill1 %+6.2f (group better %3.0f%%, ok %3.0f%%)  unseen Δ %+6.2f  epoch %4d val %.4f  %.1f min\n",
                p.label, p.N, p.H, median([x.skill1 - x.skill1_ind for x in rows]), 100mean([x.skill1 > x.skill1_ind for x in rows]),
                100mean([x.fate == "ok" for x in rows]), loo[1].skill1 - loo[1].skill1_ind, m.best_epoch, m.val, fit_secs / 60)
        flush(stdout)
        serialize(joinpath(OUT, "gait_pools_lstm.jls"), (; pools = res))
    end
    println("→ cache/heterogeneity/gait_pools_lstm.jls")
    exit()
end

# ── pools ───────────────────────────────────────────────────────────────────
rng = MersenneTwister(2026)
st_all = pool_stats(1:NT)
het_all = let Y = [zp(t, st_all) for t in trials]
    halves = [(y[1:250, :], y[251:500, :]) for y in Y]
    Pm = median([t.P for t in trials])
    clouds = [(delay_cloud(A, Pm), delay_cloud(B, Pm)) for (A, B) in halves]
    allX = reduce(vcat, [vcat(A, B) for (A, B) in clouds]); μ = mean(allX, dims = 1); F = svd(allX .- μ)
    kk = findfirst(>=(0.95), cumsum(F.S .^ 2) ./ sum(F.S .^ 2)); V = F.V[:, 1:kk]
    proj = [((A .- μ) * V, (B .- μ) * V) for (A, B) in clouds]
    w = [nn_dist(A, B) for (A, B) in proj]
    M = zeros(NT, NT)
    Threads.@threads for u in 1:NT
        for v in 1:NT
            M[u, v] = u == v ? 0.0 : nn_dist(proj[u][1], proj[v][2]) / sqrt(w[u] * w[v])
        end
    end
    (M .+ M') ./ 2
end
pools = Any[]
for (s, seed) in enumerate(randperm(rng, NT)[1:12]), N in (2, 3, 6, 12, 25, 50, 100)
    push!(pools, ("nearest", "nearest $seed N=$N", sortperm(het_all[seed, :])[1:N]))   # the seed itself is first
end
for N in (2, 3, 6, 12, 25, 50, 100), rep in 1:6
    push!(pools, ("random", "random N=$N #$rep", sort(randperm(rng, NT)[1:N])))
end
for u in unique(subject)
    push!(pools, ("subject", "subject $u", findall(==(u), subject)))
end
for gname in unique(group)
    idx = findall(==(gname), group); q = quantile(speed[idx], 0:0.2:1)
    for b in 1:5
        ib = filter(i -> q[b] <= speed[i] <= q[b+1], idx)
        ib = [ib[findfirst(==(u), subject[ib])] for u in unique(subject[ib])]
        length(ib) >= 3 && push!(pools, ("speed", "speed $gname q$b", ib))
    end
    push!(pools, ("group", "group $gname", idx))
end
push!(pools, ("group", "all", collect(1:NT)))
println("$(length(pools)) pools, $(sum(length(p[3]) for p in pools)) trial memberships"); flush(stdout)

res = Vector{Any}(undef, length(pools))
order = sortperm([length(p[3]) for p in pools], rev = true)              # big pools first
Threads.@threads :dynamic for j in order
    kind, label, idx = pools[j]
    res[j] = run_pool(kind, label, idx)
    r = res[j]
    @printf("%-22s N %3d  H %5.2f  Δskill1 %+6.2f (group better %3.0f%%)  LOO Δ %+6.2f  %s  %.0f s\n", label, r.N, r.H,
            median([x.skill1 - x.skill1_ind for x in r.rows]), 100mean([x.skill1 > x.skill1_ind for x in r.rows]),
            median([x.skill1 - x.skill1_ind for x in r.loo]), string(r.chosen), r.tune_secs)
    flush(stdout)
end
serialize(joinpath(OUT, "gait_pools_dmd.jls"), (; pools = res, het_all))
println("→ cache/heterogeneity/gait_pools_dmd.jls")
