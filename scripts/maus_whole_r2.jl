#!/usr/bin/env julia
# R2 for the whole-signal models of the running trials (Figure 5 B; maus_ladder.jl): run freely for 30 strides from the
# end of the 150 training strides, with the settings maus_ladder.jl chose. Over the last 10 strides: amplitude (mean
# channel SD relative to the training data's), period (autocorrelation of the first channel, relative to the mean
# stride, 100 samples), and the waveform error of the last stride against the training average stride (best circular
# shift; RMS over channels relative to the signal's SD). Fates as in the benchmark (self_driving_benchmark.jl): ok (amplitude 0.75–1.33, period within
# 10%), decays (0.25–0.75), collapse (< 0.25), blow-up (non-finite or > 4), wrong (period off or amplitude > 1.33).
# Phase-varying linear (affine), real time: the phase advances at the last training stride's rate during the free run.
# SSMLearnPy: maus_ssm_whole_r2.py runs it (its settings from maus_ssm.py); classified here the same way.
# Usage: julia -t 16 --project=. scripts/maus_whole_r2.jl   → cache/maus/whole_r2.jls
empty!(ARGS)
include("maus_ladder.jl")
using FFTW, NPZ
const NRUN = 30NPH

function period_ac(x; lo = 20)
    x = x .- mean(x); n = length(x)
    ac = real.(ifft(abs2.(fft(vcat(x, zeros(n))))))[1:n]; ac ./= ac[1]
    pk = [i for i in lo:n÷2 if ac[i] > ac[i-1] && ac[i] >= ac[i+1] && ac[i] > 0.3]
    isempty(pk) && return NaN
    i = pk[1]; a, b, c = ac[i-1], ac[i], ac[i+1]; (i - 1) + 0.5 * (a - c) / (a - 2b + c)
end
function fate(P, Ytr, avg)
    any(!isfinite, P) && return (; fate = "blow-up", amp = Inf, per = NaN, wave = NaN)
    tail = P[end-10NPH+1:end, :]; amp = mean(std(tail, dims = 1)) / mean(std(Ytr, dims = 1))
    per = amp > 0.05 ? period_ac(tail[:, 1]) / NPH : NaN
    last = tail[end-NPH+1:end, :]
    wave = minimum(sqrt(mean(abs2, circshift(last, (k, 0)) .- avg)) for k in 0:NPH-1) / mean(std(Ytr, dims = 1))
    rhythm = isfinite(per) && abs(per - 1) < 0.1
    f = amp > 4 ? "blow-up" : amp < 0.25 ? "collapse" : (!rhythm || amp > 1.33) ? "wrong" : amp >= 0.75 ? "ok" : "decays"
    (; fate = f, amp, per, wave)
end
"Long free run of the ESN of esn_whole (same reservoir and readout) from sample t."
function esn_long(Yv, rows, s, t)
    n, p = size(Yv); rng = MersenneTwister(1); N = s.N
    W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= s.ρ / maximum(abs, eigvals(W))
    Win = 0.5 .* (2 .* rand(rng, N, p) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    R = zeros(N, n); r = zeros(N)
    for τ in 1:n; r = (1 - s.α) .* r .+ s.α .* tanh.(W * r .+ Win * Yv[τ, :] .+ b); R[:, τ] = r; end
    ix = collect(first(rows)+50:last(rows)-1); F = vcat(R[:, ix], Yv[ix, :]', ones(1, length(ix)))
    G = F * F'; Wout = Yv[ix.+1, :]' * F' / (G + s.λ * tr(G) / size(G, 1) * I)
    r = R[:, t]; u = Yv[t, :]; out = fill(NaN, NRUN, p)
    for k in 1:NRUN
        y = Wout * vcat(r, u, 1.0); (any(!isfinite, y) || maximum(abs, y) > 1e6) && break
        out[k, :] = y; r = (1 - s.α) .* r .+ s.α .* tanh.(W * r .+ Win * y .+ b); u = y
    end
    out
end
function trial_whole_r2(X, LR)
    R = rung_data(X); out = Dict{Tuple{Symbol, Symbol}, Any}()
    for rung in (:rt, :sn)
        Yv, bidx = rung == :rt ? (R.Yrt, R.bidx_rt) : (R.Y, R.bidx_sn)
        rows = 1:bidx[151]-1; t = last(rows); Ytr = Yv[rows, :]
        avg = reduce(vcat, [mean(reduce(vcat, [Yv[(s-1)*NPH+k:(s-1)*NPH+k, :] for s in 1:150]), dims = 1) for k in 1:NPH])   # sn: the average stride
        for name in (:ltp, :dmd, :esn)
            haskey(LR, (rung, name, 150)) || continue
            s = LR[(rung, name, 150)].setting; mult = NaN
            P = if name == :dmd
                A, U = opfit(Ytr; d = s[1], r = s[2], α = s[3]); roll(A, U, mean(Ytr, dims = 1), Yv, t, s[1], NRUN)
            elseif name == :esn
                esn_long(Yv, rows, s, t)
            else
                ph = rung == :rt ? R.ph_rt : R.ph_sn
                M = fit_ltp_aff(Yv, rows, ph, s...); p = size(Yv, 2)
                mult = maximum(abs, eigvals(foldl((Pm, A) -> A * Pm, M.Aph; init = Matrix(1.0I, size(M.Aph[1])...))))
                z = M.U' * vec(permutedims(Yv[t-M.d+1:t, :] .- M.μ)); Q = zeros(NRUN, p)
                for k in 1:NRUN
                    j = rung == :rt ? mod(floor(Int, R.ph_rt[t] + NPH * R.rate_rt[t] * (k - 1)), NPH) : mod(t - 1 + k - 1, NPH)
                    z = M.Aph[j+1] * z .+ M.bph[j+1]; Q[k, :] = (M.U * z)[end-p+1:end] .+ vec(M.μ)
                end
                Q
            end
            out[(rung, name)] = merge(fate(P, Ytr, avg), (; mult))
        end
    end
    out
end
if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    keep = [qc_run(pair_strides(t.X)) < 5 for t in A.trials]; T = A.trials[keep]
    LAD = deserialize(joinpath(ROOT, "cache", "maus", "ladder.jls")).R
    Rs = tmap(i -> trial_whole_r2(T[i].X, LAD[i]), eachindex(T))
    # SSMLearnPy's free runs (maus_ssm_whole_r2.py), classified the same way
    fs = joinpath(ROOT, "cache", "maus", "ssm_whole_runs.npz")
    if isfile(fs)
        RUNS = npzread(fs)["runs"]
        for i in eachindex(T)
            Rs[i] === nothing && continue
            R = rung_data(T[i].X)
            for (j, rung) in enumerate((:rt, :sn))
                Yv, bidx = rung == :rt ? (R.Yrt, R.bidx_rt) : (R.Y, R.bidx_sn); rows = 1:bidx[151]-1; Ytr = Yv[rows, :]
                avg = reduce(vcat, [mean(reduce(vcat, [Yv[(s-1)*NPH+k:(s-1)*NPH+k, :] for s in 1:150]), dims = 1) for k in 1:NPH])
                P = RUNS[i, j, :, :]; P[isnan.(P)] .= Inf
                Rs[i][(rung, :ssm)] = merge(fate(P, Ytr, avg), (; mult = NaN))
            end
        end
    end
    @printf("%d trials (%d failed)\n", length(Rs), count(isnothing, Rs))
    for rung in (:rt, :sn), name in (:ltp, :dmd, :esn, :ssm)
        v = [r[(rung, name)] for r in Rs if r !== nothing && haskey(r, (rung, name))]
        @printf("%s %-4s fates %s | amplitude %.2f, waveform error %.3f (medians)%s\n", rung, name,
                join(["$f $(count(x -> x.fate == f, v))" for f in ("ok", "decays", "collapse", "blow-up", "wrong")], ", "),
                median(x.amp for x in v), median(filter(isfinite, [x.wave for x in v])),
                name == :ltp ? @sprintf(", largest multiplier %.2f (max %.2f)", median(x.mult for x in v), maximum(x.mult for x in v)) : "")
    end
    serialize(joinpath(ROOT, "cache", "maus", "whole_r2.jls"), Rs)
end
