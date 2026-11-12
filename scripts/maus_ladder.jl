#!/usr/bin/env julia
# What does modelling the whole system cost, against modelling the deviation from the cycle? Three representations of
# the same running trials (Maus et al. 2015; maus_learning_all.jl), one target and one score:
#   rt   real time, whole signal: angles resampled uniformly in time (the mean stride = NPH samples); a model must
#        produce the cycle, its timing and the deviation
#   sn   stride-normalised, whole signal: every stride resampled to NPH samples between its actual boundaries (timing
#        given), the cycle not removed
#   dev  stride-normalised deviation from the 61-stride moving-average stride (maus_learning_all.jl, read from its cache)
# Score (all rungs): R² of the deviation at horizon h, 1 − Σ‖ŷ(t+h) − y(t+h)‖² / Σ‖d(t+h)‖², where d is the deviation
# from the moving-average stride at each sample's actual phase (rt: interpolated within the stride that contains it),
# i.e. skill against that average stride. Training: the first NTR strides; test: the second half; settings on an inner
# split (maus_learning.jl). Models:
#   dmd     Hankel-DMD on the whole signal (opfit), delay window 16–100 samples (up to a stride), rank 16–600, ridge
#           α ∈ {1e-4, 1e-6, 1e-8} × S₁² (opfit's default 1e-4 shrinks every direction below ~1% of the leading singular
#           value, which on the whole signal is the cycle's: it removes the deviation's dynamics, whatever the rank)
#   esn     ESN on the whole signal, no phase input (as the self-driving benchmark), 300 units, 8 settings
#   ltp     phase-varying affine linear model z_{t+1} = A(θ_t) z_t + b(θ_t) on the leading components of the last 8
#           samples (A, b Fourier series in θ); sn: θ = position in the stride; rt: θ causal, the time since the last
#           stride boundary over the previous stride's duration, advanced at that rate during a forecast
# Usage: julia -t 16 --project=. scripts/maus_ladder.jl       → cache/maus/ladder.jls
#        julia -t 16 --project=. scripts/maus_ladder.jl dmd   refit DMD only, replacing its entries in ladder.jls
include("maus_learning_all.jl")
const HISTW = 110

function cut_times(P)
    b = Float64[]
    for k in ceil(Int, P.ϕ[1] / 2π)+1:floor(Int, P.ϕ[end] / 2π)
        i = findfirst(>=(2π * k), P.ϕ); (i === nothing || i == 1) && continue
        push!(b, P.tc[i-1] + (P.tc[i] - P.tc[i-1]) * (2π * k - P.ϕ[i-1]) / (P.ϕ[i] - P.ϕ[i-1]))
    end
    b
end
lerp(X, τ) = (i = clamp(floor(Int, τ), 1, size(X, 1) - 1); a = τ - i; (1 - a) .* X[i, :] .+ a .* X[i+1, :])

"Both whole-signal representations of a trial, with deviations and phase indices (0-based, per sample)."
function rung_data(X)
    P = pair_phase(X); b = cut_times(P); S = cut_strides(X, P.ϕ, P.tc); n, _, p = size(S)
    sd = std(flat(S), dims = 1); Sn = S ./ reshape(sd, 1, 1, p); Bs = movmean(Sn, MA)
    Y = flat(Sn); D = Y .- flat(Bs)
    ph_sn = [mod(t - 1, NPH) for t in 1:size(Y, 1)]; bidx_sn = [(s - 1) * NPH + 1 for s in 1:n+1]
    T̄ = (b[n+1] - b[1]) / n; Δ = T̄ / NPH; τ = collect(b[1]:Δ:b[n+1]-1e-9)
    Yrt = reduce(vcat, [(lerp(X, x) ./ vec(sd))' for x in τ])
    Drt = similar(Yrt); ph_rt = zeros(Int, length(τ))
    for (i, x) in enumerate(τ)
        s = clamp(searchsortedlast(b, x), 1, n); f = (x - b[s]) / (b[s+1] - b[s]); j = f * NPH
        j0 = clamp(floor(Int, j), 0, NPH - 1); a = j - j0; j1 = j0 + 1
        Bv = j1 <= NPH - 1 ? (1 - a) .* Bs[s, j0+1, :] .+ a .* Bs[s, j1+1, :] :
             (s < n ? (1 - a) .* Bs[s, NPH, :] .+ a .* Bs[s+1, 1, :] : Bs[s, NPH, :])
        Drt[i, :] = Yrt[i, :] .- Bv
        prev = s >= 2 ? b[s] - b[s-1] : b[s+1] - b[s]                       # causal phase: previous stride's duration
        ph_rt[i] = clamp(floor(Int, NPH * (x - b[s]) / prev), 0, NPH - 1)
    end
    rate_rt = [(s = clamp(searchsortedlast(b, x), 1, n); prev = s >= 2 ? b[s] - b[s-1] : b[s+1] - b[s]; Δ / prev) for x in τ]
    bidx_rt = [searchsortedfirst(τ, b[s]) for s in 1:n+1]
    (; n, Y, D, ph_sn, bidx_sn, Yrt, Drt, ph_rt, rate_rt, bidx_rt)
end

score_full(pred, Y, D, ts, hidx) = (E = zeros(length(hidx)); R = zeros(length(hidx));
    for t in ts; P = pred(t); for (j, k) in enumerate(hidx); E[j] += sum(abs2, P[HZ[k], :] .- Y[t+HZ[k], :]); R[j] += sum(abs2, D[t+HZ[k], :]); end; end;
    1 .- E ./ R)

"Phase-varying affine linear model on rows; phase index ph (0-based) per sample; returns A(k), b(k), basis, mean."
function fit_ltp_aff(Yv, rows, ph, d, r, Hh, λ)
    Y = Yv[rows, :]; μ = mean(Y, dims = 1); H = hankel(Y .- μ, d); F = svd(H); r = min(r, size(F.U, 2))
    U = F.U[:, 1:r]; Z = U' * H; nc = size(Z, 2)
    phc = [ph[first(rows)+c+d-2] for c in 1:nc]
    Φ = reduce(hcat, [(f = fphase(phc[c], Hh); vcat(kron(f, Z[:, c]), f)) for c in 1:nc-1])
    G = Φ * Φ'; Bm = Z[:, 2:end] * Φ'; W = Bm / (G + λ * mean(diag(G)) * I); nf = 2Hh + 1
    Aph = [sum(fphase(k, Hh)[i] .* W[:, (i-1)*r+1:i*r] for i in 1:nf) for k in 0:NPH-1]
    bph = [W[:, nf*r+1:end] * fphase(k, Hh) for k in 0:NPH-1]
    (; Aph, bph, U, μ, d)
end
"Forecast hmax samples from t; phase of sample u: phfun(t, s) for s = 0, 1, … (s = 0 is t itself)."
function ltp_aff_pred(Yv, M, phfun)
    p = size(Yv, 2)
    t -> begin
        z = M.U' * vec(permutedims(Yv[t-M.d+1:t, :] .- M.μ)); out = zeros(HZ[end], p)
        for s in 1:HZ[end]
            k = phfun(t, s - 1) + 1; z = M.Aph[k] * z .+ M.bph[k]; out[s, :] = (M.U*z)[end-p+1:end] .+ vec(M.μ)
        end
        out
    end
end
function esn_whole(Yv)
    n, p = size(Yv); cache = Dict{Tuple{Float64, Float64}, Any}()
    settings = [(; N = 300, ρ, α, λ) for ρ in (0.5, 0.9) for α in (0.3, 1.0) for λ in (1e-4, 1e-2)]
    states(s) = get!(cache, (s.ρ, s.α)) do
        rng = MersenneTwister(1); N = s.N
        W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= s.ρ / maximum(abs, eigvals(W))
        Win = 0.5 .* (2 .* rand(rng, N, p) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
        R = zeros(N, n); r = zeros(N)
        for t in 1:n; r = (1 - s.α) .* r .+ s.α .* tanh.(W * r .+ Win * Yv[t, :] .+ b); R[:, t] = r; end
        (; W, Win, b, R)
    end
    fit = (rows, s) -> begin
        E = states(s); ix = collect(first(rows)+50:last(rows)-1)
        F = vcat(E.R[:, ix], Yv[ix, :]', ones(1, length(ix))); T = Yv[ix.+1, :]'
        G = F * F'; Wout = T * F' / (G + s.λ * tr(G) / size(G, 1) * I)
        t -> begin
            r = E.R[:, t]; u = Yv[t, :]; out = zeros(HZ[end], p)
            for st in 1:HZ[end]
                y = Wout * vcat(r, u, 1.0); out[st, :] = y
                (any(!isfinite, y) || maximum(abs, y) > 1e3) && (out[st:end, :] .= NaN; break)
                r = (1 - s.α) .* r .+ s.α .* tanh.(E.W * r .+ E.Win * y .+ E.b); u = y
            end
            out
        end
    end
    (settings, fit)
end
function ladder_families(Yv, rung, R)
    F = Dict{Symbol, Any}()
    F[:dmd] = ([(d, r, α) for d in (16, 33, 50, 100) for r in (16, 32, 64, 128, 256, 450, 600) for α in (1e-4, 1e-6, 1e-8) if r <= 6d],
               (rows, s) -> begin A, U = opfit(Yv[rows, :]; d = s[1], r = s[2], α = s[3]); μ = mean(Yv[rows, :], dims = 1); t -> roll(A, U, μ, Yv, t, s[1], HZ[end]) end)
    F[:esn] = esn_whole(Yv)
    ph = rung == :rt ? R.ph_rt : R.ph_sn
    phfun = rung == :rt ? ((t, s) -> mod(floor(Int, R.ph_rt[t] + NPH * R.rate_rt[t] * s), NPH)) : ((t, s) -> mod(t - 1 + s, NPH))
    F[:ltp] = ([(8, r, h, λ) for r in (16, 32, 64) for h in (2, 4, 8) for λ in (1e-3, 1e-2)],
               (rows, s) -> ltp_aff_pred(Yv, fit_ltp_aff(Yv, rows, ph, s...), phfun))
    F
end
"Forecasts clipped to the training range ± half its width (a diverging run counts as a large, bounded error)."
safe(pred, lo, hi) = t -> (P = pred(t); P[.!isfinite.(P)] .= 0.0; clamp.(P, lo, hi))
diverged(pred, ts, lo, hi) = mean(t -> (P = pred(t); any(!isfinite, P) || any(P .< lo) || any(P .> hi)), ts)

function trial_ladder(X; only = nothing)
    R = rung_data(X); out = Dict{Tuple{Symbol, Symbol, Int}, Any}(); nh = R.n ÷ 2
    for rung in (:rt, :sn)
        Yv, Dv, bidx = rung == :rt ? (R.Yrt, R.Drt, R.bidx_rt) : (R.Y, R.D, R.bidx_sn)
        ts = (bidx[nh+1] + HISTW):STEP:(size(Yv, 1) - HZ[end])
        for (name, (settings, fit)) in ladder_families(Yv, rung, R), ntr in SIZES_ALL
            (ntr > nh || (only !== nothing && name ∉ only)) && continue
            rows = 1:bidx[ntr+1]-1; inner = 1:bidx[ntr÷2+1]-1; va = bidx[ntr÷2+1]:bidx[ntr+1]-1
            vst = (first(va) + HISTW):STEP:(last(va) - HZ[HV[end]])
            isempty(vst) && continue
            w = maximum(Yv[rows, :], dims = 1) .- minimum(Yv[rows, :], dims = 1)
            lo = minimum(Yv[rows, :], dims = 1) .- 0.5w; hi = maximum(Yv[rows, :], dims = 1) .+ 0.5w
            vs = map(settings) do s
                f = try fit(inner, s) catch; nothing end
                f === nothing ? -Inf : mean(score_full(safe(f, lo, hi), Yv, Dv, vst, HV))
            end
            all(isinf, vs) && continue
            s = settings[argmax(vs)]; f = fit(rows, s)
            out[(rung, name, ntr)] = (; setting = s, r2 = score_full(safe(f, lo, hi), Yv, Dv, ts, 1:length(HZ)),
                                      diverged = diverged(f, ts, lo, hi))
        end
    end
    out
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    keep = [qc_run(pair_strides(t.X)) < 5 for t in A.trials]; T = A.trials[keep]
    if get(ARGS, 1, "") == "dmd"                                   # refit DMD only; keep the other models' entries
        Rs = deserialize(joinpath(ROOT, "cache", "maus", "ladder.jls")).R
        Rd = tmap(t -> trial_ladder(t.X; only = (:dmd,)), T)
        for (r, rd) in zip(Rs, Rd); (r === nothing || rd === nothing) || merge!(r, rd); end
    else
        Rs = tmap(t -> trial_ladder(t.X), T)
    end
    bad = count(isnothing, Rs)
    dev = deserialize(joinpath(ROOT, "cache", "maus", "learning_all_cpu_0.0.jls"))
    @printf("%d trials (%d failed). Median R² of the deviation at 1/8 / 1/4 stride, by training strides\n", length(Rs) - bad, bad)
    @printf("%-14s", "strides"); for n in SIZES_ALL; @printf("%14d", n); end; println()
    for rung in (:rt, :sn), name in (:dmd, :esn, :ltp)
        @printf("%-14s", "$rung $name")
        for n in SIZES_ALL
            v = [r[(rung, name, n)].r2 for r in Rs if r !== nothing && haskey(r, (rung, name, n))]
            isempty(v) ? @printf("%14s", "—") : @printf("%14s", @sprintf("%.2f / %.2f", median(x[2] for x in v), median(x[4] for x in v)))
        end
        println()
    end
    for name in (:dmd, :esn, :ltp)
        @printf("%-14s", "dev $name")
        for n in SIZES_ALL
            v = [r[(name, n)].r2 for r in dev.R if r !== nothing && haskey(r, (name, n))]
            isempty(v) ? @printf("%14s", "—") : @printf("%14s", @sprintf("%.2f / %.2f", median(x[2] for x in v), median(x[4] for x in v)))
        end
        println()
    end
    serialize(joinpath(ROOT, "cache", "maus", "ladder.jls"), (; R = Rs, SIZES = SIZES_ALL, HZ))
end
