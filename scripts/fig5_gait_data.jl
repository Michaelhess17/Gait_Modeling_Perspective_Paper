#!/usr/bin/env julia
# Human gait: how much of a recording, and of a model fitted to it, is the mean cycle? (Fig 5; OUTLINE.md §5)
#
# Data: the companion paper's superset (../Gait_NonNormality_Paper3/data/human_gait_superset.npy): 299 treadmill
# trials × 1500 samples (100 Hz, 15 s) × 6 sagittal angles (hip, knee, ankle of limb 1 then limb 2; stroke trials
# paretic limb first), 42 people (17 + 5 able-bodied, 11 higher- and 9 lower-functioning stroke survivors).
#
# Strides: cut where the unwrapped Hilbert phase of the limb-1 hip angle crosses 2πk + π/2 (the hip's
# descending zero crossing: a consistent origin in every trial, and steep, so the cut is precise); each stride
# resampled onto 100 points of its own duration. Mean cycle = mean over strides; deviations =
# strides minus the mean cycle (stride-duration variability is thus removed; within-stride timing is not).
#
# Operator measures are the companion paper's, reimplemented and checked against its per-trial cache
# (figure6_metrics_d32_fixedrank70_all_stride100_ridge0.0001.csv): trial uniformly rescaled to 100 samples per
# mean stride, channels centred, delay-coordinate DMD (window d, rank 70, ridge α = 1e-4 relative to s₁², Schur
# stabilisation), transient growth TG = max_{k ≤ 100} ‖Ãᵏ‖, joint response map at the stride frequency.
# Mean-cycle-only surrogate: the trial's own mean cycle tiled to the trial's length (no deviations at all).
#
# Identification: the companion paper's split-half protocol (07_individual_routing_signatures.jl): each person's
# trials split at random into halves, features averaged per half, z-scored against the able-bodied half-A
# distribution; a person is identified when their other half is the nearest (cosine) match. 200 splits.
#
# Usage: julia --project=. scripts/fig5_gait_data.jl      (writes cache/fig5/gait.jls)

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using NPZ, CSV, DataFrames, Statistics, LinearAlgebra, FFTW, Printf, Random, Serialization

const ROOT = dirname(@__DIR__)
const P3 = get(ENV, "GAIT_P3", expanduser("~/Documents/Gait_NonNormality_Paper3"))
const NPH = 100
const WINDOWS = [16, 32, 50, 100]

# ── strides and the mean cycle ───────────────────────────────────────────────
function analytic(x)
    n = length(x); h = zeros(n); h[1] = 1
    n % 2 == 0 ? (h[2:n÷2] .= 2; h[n÷2+1] = 1) : (h[2:(n+1)÷2] .= 2)
    ifft(fft(x) .* h)
end
unwrap(p) = (q = copy(p); for i in 2:length(q); q[i] -= 2π * round((q[i] - q[i-1]) / 2π); end; q)

"Strides (n × NPH × channels) cut at 2πk + π/2 of the limb-1 hip angle's unwrapped Hilbert phase."
function strides(X; nph = NPH)
    ϕ = unwrap(angle.(analytic(X[:, 1] .- mean(X[:, 1])))) .- π / 2
    b = Float64[]
    for k in ceil(Int, ϕ[1] / 2π)+1:floor(Int, ϕ[end] / 2π)
        i = findfirst(>=(2π * k), ϕ); (i === nothing || i == 1) && continue
        push!(b, i - 1 + (2π * k - ϕ[i-1]) / (ϕ[i] - ϕ[i-1]))
    end
    S = zeros(length(b) - 1, nph, size(X, 2))
    for s in 1:length(b)-1, (j, t) in enumerate(range(b[s], b[s+1]; length = nph + 1)[1:nph])
        i = clamp(floor(Int, t), 1, size(X, 1) - 1); a = t - i
        S[s, j, :] = (1 - a) .* X[i, :] .+ a .* X[i+1, :]
    end
    S
end

# ── the companion paper's operator pipeline ──────────────────────────────────
"Uniform rescale so one mean stride (total Hilbert-phase advance of PC1) spans `steps` samples."
function stride_rescale(X; steps = NPH)
    Xc = X .- mean(X, dims = 1); sig = Xc * svd(Xc).V[:, 1]
    ϕ = unwrap(angle.(analytic(sig .- mean(sig)))); c = abs(ϕ[end] - ϕ[1]) / 2π / (length(ϕ) - 1)
    T = size(X, 1); n = round(Int, (T - 1) * c * steps) + 1; out = zeros(n, size(X, 2))
    for (k, t) in enumerate(range(1, T; length = n))
        j = clamp(floor(Int, t), 1, T - 1); a = t - j; out[k, :] = (1 - a) .* X[j, :] .+ a .* X[j+1, :]
    end
    out
end
function hankel(Y, d)
    n, p = size(Y); H = zeros(p * d, n - d + 1)
    for c in 1:n-d+1, l in 0:d-1; H[l*p+1:(l+1)*p, c] = Y[c+l, :]; end
    H
end
"Pin eigenvalues outside the unit circle onto it, block by block in the real Schur form."
function stabilize(A)
    F = schur(A); T = copy(F.T); n = size(T, 1); i = 1
    while i <= n
        if i < n && !iszero(T[i+1, i])
            m = maximum(abs, eigvals(T[i:i+1, i:i+1])); m > 1 && (T[i:i+1, i:i+1] ./= m); i += 2
        else
            abs(T[i, i]) > 1 && (T[i, i] /= abs(T[i, i])); i += 1
        end
    end
    F.Z * T * F.Z'
end
function opfit(Y; d = 32, r = 70, α = 1e-4)
    H = hankel(Y .- mean(Y, dims = 1), d); F = svd(H[:, 1:end-1]); r = min(r, length(F.S))
    U = F.U[:, 1:r]; S = F.S[1:r]
    stabilize(U' * H[:, 2:end] * F.V[:, 1:r] * Diagonal(S ./ (S .^ 2 .+ α * F.S[1]^2))), U
end
tg(A; K = NPH) = (B = Matrix(1.0I, size(A)...); m = 1.0; for _ in 1:K; B = A * B; m = max(m, opnorm(B)); end; m)
"Joint response map at the stride frequency (6 × 6 block gains of the resolvent, normalised to sum 1)."
function opmap(Y; d = 32, p = size(Y, 2))
    A, U = opfit(Y; d); R = inv(exp(im * 2π / NPH) * I - A)
    rows(j) = [j + l * p for l in 0:d-1]
    G = [opnorm(U[rows(o), :] * R * U[rows(i), :]') for o in 1:p, i in 1:p]
    vec(G ./ sum(G))
end
function meancycle_only(Y)
    m = dropdims(mean(strides(Y), dims = 1), dims = 1)
    reduce(vcat, fill(m, cld(size(Y, 1), NPH)))[1:size(Y, 1), :]
end

# ── identification ───────────────────────────────────────────────────────────
function identify(F, subs, grp; nrep = 200, seed = 11)
    rng = MersenneTwister(seed); us = [s for s in unique(subs) if count(subs .== s) >= 2]
    isab = [grp[findfirst(subs .== s)] == "AB" for s in us]; hits = Float64[]
    for _ in 1:nrep
        A = zeros(length(us), length(F[1])); B = similar(A)
        for (i, s) in enumerate(us)
            idx = shuffle(rng, findall(subs .== s)); h = div(length(idx), 2)
            A[i, :] = mean(F[idx[1:h]]); B[i, :] = mean(F[idx[h+1:end]])
        end
        μ = vec(mean(A[isab, :], dims = 1)); σ = max.(vec(std(A[isab, :], dims = 1)), eps())
        DA = (A .- μ') ./ σ'; DB = (B .- μ') ./ σ'
        DA ./= sqrt.(sum(abs2, DA, dims = 2)); DB ./= sqrt.(sum(abs2, DB, dims = 2))
        C = DA * DB'; push!(hits, mean([argmax(C[i, :]) == i for i in axes(C, 1)]))
    end
    (; rate = mean(hits), lo = quantile(hits, 0.025), hi = quantile(hits, 0.975), n = length(us))
end

if abspath(PROGRAM_FILE) == @__FILE__
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    subs = String.(meta.subject); grp = String.(meta.lf_or_hf); n = size(data, 1)
    nstr = zeros(Int, n); devshare = zeros(n)
    TG = Dict((k, d) => zeros(n) for k in (:real, :mean), d in WINDOWS)
    F = Dict(k => Vector{Float64}[] for k in (:rom, :shape, :map_real, :map_mean, :dev_corr, :dev_sd))
    for i in 1:n
        X = data[i, :, :]; S = strides(X); m = dropdims(mean(S, dims = 1), dims = 1); D = S .- reshape(m, 1, NPH, 6)
        nstr[i] = size(S, 1); devshare[i] = sum(abs2, D) / sum(abs2, S .- mean(S, dims = (1, 2)))
        rom = vec(maximum(m, dims = 1) .- minimum(m, dims = 1)); push!(F[:rom], rom ./ sum(rom))
        push!(F[:shape], vec((m .- mean(m, dims = 1))[1:2:end, :]))
        Dn = D ./ reshape(std(D, dims = 1), 1, NPH, 6); C = mean(cor(Dn[:, j, :]) for j in 1:NPH)
        push!(F[:dev_corr], [C[a, b] for a in 1:6 for b in a+1:6])
        push!(F[:dev_sd], vec(dropdims(std(D, dims = 1), dims = 1)[1:2:end, :]))
        Y = stride_rescale(X); M = meancycle_only(Y)
        for d in WINDOWS
            TG[(:real, d)][i] = log10(tg(opfit(Y; d)[1])); TG[(:mean, d)][i] = log10(tg(opfit(M; d)[1]))
        end
        push!(F[:map_real], opmap(Y)); push!(F[:map_mean], opmap(M))
        i % 50 == 0 && println("trial $i / $n")
    end
    ident = Dict(k => identify(F[k], subs, grp) for k in keys(F))
    # example: an able-bodied HYA trial near 120 cm/s with many strides
    cand = findall((meta.source .== "GaitSignatures_HealthyYoungAdultStudy") .& (abs.(meta.speed .- 120) .<= 15))
    ex = cand[argmax(nstr[cand])]
    example = (; i = ex, subject = subs[ex], speed = meta.speed[ex], S = strides(data[ex, :, :]), raw = data[ex, :, :])
    # cross-check against the companion paper's per-trial transient growth (d = 32)
    c6 = CSV.read(joinpath(P3, "results", "figure6_metrics_d32_fixedrank70_all_stride100_ridge0.0001.csv"), DataFrame)
    j = [findfirst(==(i), c6.trial_index) for i in 1:n]; ok = j .!== nothing
    check = cor(TG[(:real, 32)][ok], c6.log_tgmax[j[ok]])
    mkpath(joinpath(ROOT, "cache", "fig5"))
    serialize(joinpath(ROOT, "cache", "fig5", "gait.jls"),
              (; subs, grp, speed = Float64.(meta.speed), source = String.(meta.source), nstr, devshare, TG, ident, example,
                 check, windows = WINDOWS))
    @printf("check vs companion log TG (d=32): r = %.4f over %d trials\n", check, count(ok))
    @printf("strides per trial %d–%d (median %d); deviation share median %.2f%%\n", extrema(nstr)..., median(nstr), 100median(devshare))
    for g in ("AB", "HF", "LF"), d in WINDOWS
        m = grp .== g
        @printf("%s d=%3d  log TG real %.3f  mean-cycle-only %.3f\n", g, d, median(TG[(:real, d)][m]), median(TG[(:mean, d)][m]))
    end
    for d in WINDOWS[1:3]; @printf("d=%d: r(real, mean-only) = %.3f\n", d, cor(TG[(:real, d)], TG[(:mean, d)])); end
    for (k, v) in sort(collect(ident); by = x -> -x[2].rate)
        @printf("identify %-9s %.3f [%.3f–%.3f] of %d (chance %.3f)\n", k, v.rate, v.lo, v.hi, v.n, 1 / v.n)
    end
end
