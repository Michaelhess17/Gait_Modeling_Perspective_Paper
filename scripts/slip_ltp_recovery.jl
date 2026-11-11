#!/usr/bin/env julia
# Can the Fourier-in-phase operator recover known Floquet multipliers, and how many strides, what rank and how many
# harmonics does it need? Kicked SLIP walks (slip_truth.jl): stride-level kicks to (y, θ) at VLO, noise-free steps in
# between; observed (y, ẋ, ẏ), each step resampled onto 50 points of its own duration (phase 0 = VLO, just after the
# kick). Truth: eigenvalues of the exact step-to-step Jacobian — open loop κ = 0: 0.628 ± 0.713i (ρ 0.950); speed
# feedback κ = 0.1: 0.758 ± 0.431i (ρ 0.872). (κ = 0.2, real multipliers 0.916 and 0.651, falls within 30–200 steps
# under these kicks.) Energy is conserved, so deviations live in a 2-D plane: the operator's one-step monodromy should
# carry the two true multipliers and nothing else of size.
#
# Pipeline as for the human data (fit_ltp in maus_floquet_models.jl): n consecutive steps, channels standardised by
# the chunk's SD, deviation from the chunk's mean step, Hankel of delay d, POD to rank r, z_{t+1} = A(θ_t) z_t with A a
# Fourier series in phase to Hh harmonics, ridge λ relative to the mean diagonal of the Gram matrix (here assembled
# from per-phase moments, which is the same least squares). Measurement noise: i.i.d., SD = ratio × the
# stride-to-stride SD of each variable (as fig3_slip_maps.jl).
#   error      the two largest monodromy eigenvalues matched to the two true ones, mean distance in the complex plane
#   selection  what one would do without truth: each setting fitted on the first half of the chunk and scored on the
#              second (R² of the deviation 6 samples = 12% of a step ahead); the best setting refitted on the chunk
#   section    reference: affine least-squares map of the phase-0 state in its top-2 PCs (fig3_slip_maps.jl)
#
# Usage: julia --project=. -t 16 scripts/slip_ltp_recovery.jl      (writes cache/slip_ltp/recovery.jls)

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using Serialization, Statistics, LinearAlgebra, Random, Printf

const ROOT = dirname(@__DIR__)
const NP = 50
const NS = [8, 16, 32, 64, 128, 256, 512]
const NOISE = [0.0, 0.1, 0.25, 0.5]
const DR = [(1, 2), (1, 3), (4, 2), (4, 3), (4, 4), (4, 6), (4, 12), (8, 2), (8, 3), (8, 4), (8, 6), (8, 12), (8, 24)]
const HS = [0, 1, 2, 4, 8, 16]
const LS = [1e-6, 1e-3, 1e-2, 1e-1]
const SETS = [(d, r, h, λ) for (d, r) in DR for h in HS for λ in LS]
const HSEL = 6

hankel(Y, d) = (n = size(Y, 1); reduce(vcat, [permutedims(Y[l+1:n-d+1+l, :]) for l in 0:d-1]))
fph(k, Hh) = (θ = 2π * k / NP; vcat(1.0, [f(h * θ) for h in 1:Hh for f in (cos, sin)]))
const FPH = Dict(Hh => reduce(hcat, [fph(k, Hh) for k in 0:NP-1]) for Hh in HS)

"Deviation series (rows = samples) of steps S (n × NP × p): standardised channels minus the mean step."
function devseries(S)
    n, _, p = size(S); sd = std(reduce(vcat, [S[k, :, :] for k in 1:n]), dims = 1)
    Sn = S ./ reshape(sd, 1, 1, p); M = mean(Sn, dims = 1)[1, :, :]
    reduce(vcat, [Sn[k, :, :] .- M for k in 1:n])
end

"Fourier-in-phase operators fitted to rows `rows` of D (rows[1] at phase 0), one per ridge in λs."
function ltp_fit(D, rows, d, r, Hh, λs)
    Y = D[rows, :]; μ = vec(mean(Y, dims = 1)); H = hankel(Y .- μ', d); F = svd(H); r = min(r, size(F.U, 2))
    U = F.U[:, 1:r]; Z = U' * H; nc = size(Z, 2); ph = [mod(c + d - 2, NP) for c in 1:nc-1]
    nf = 2Hh + 1; Fm = FPH[Hh]; G = zeros(nf * r, nf * r); B = zeros(r, nf * r)
    for k in 0:NP-1
        idx = findall(==(k), ph); isempty(idx) && continue
        Zk = Z[:, idx]; f = Fm[:, k+1]
        G .+= kron(f * f', Zk * Zk'); B .+= kron(f', Z[:, idx .+ 1] * Zk')
    end
    md = mean(diag(G))
    map(λs) do λ
        W = B / (G + λ * md * I)
        (; Aph = [sum(Fm[i, k+1] .* W[:, (i-1)*r+1:i*r] for i in 1:nf) for k in 0:NP-1], U, μ, d)
    end
end
function mono_ev(M)
    P = Matrix(1.0I, size(M.Aph[1])...); for A in M.Aph; P = A * P; end
    ev = eigvals(P); ev = ComplexF64.(ev[sortperm(abs.(ev), rev = true)])
    vcat(ev, fill(NaN + 0im, 4))[1:4]
end
"R² of the deviation h samples ahead from starts ts."
function skill(M, D, ts, h)
    p = size(D, 2); C = M.U[end-p+1:end, :]; Ut = Matrix(M.U'); d = M.d; r = size(M.U, 2)
    e = 0.0; s0 = 0.0; z = zeros(r); w = zeros(r); hv = zeros(p * d); y = zeros(p)
    for t in ts
        for l in 0:d-1, j in 1:p; hv[l*p+j] = D[t-d+1+l, j] - M.μ[j]; end
        mul!(z, Ut, hv)
        for s in 1:h; mul!(w, M.Aph[mod(t + s - 2, NP)+1], z); z, w = w, z; end
        mul!(y, C, z)
        for j in 1:p; e += (y[j] + M.μ[j] - D[t+h, j])^2; s0 += D[t+h, j]^2; end
    end
    1 - e / s0
end
function section_ev(S)
    X = S[:, 1, :]; Xc = X .- mean(X, dims = 1); Xr = Xc * svd(Xc).V[:, 1:2]
    Z = hcat(Xr[1:end-1, :], ones(size(Xr, 1) - 1)); B = (Z \ Xr[2:end, :])'
    ev = eigvals(B[:, 1:2]); ComplexF64.(ev[sortperm(abs.(ev), rev = true)])
end
"Top two estimated eigenvalues matched to the two true ones: mean distance in the complex plane."
pair_err(ev, tr) = min(abs(ev[1] - tr[1]) + abs(ev[2] - tr[2]), abs(ev[1] - tr[2]) + abs(ev[2] - tr[1])) / 2

function job(w, n, c, ratio)
    rng = MersenneTwister(hash((w.kappa, w.seed, n, c, ratio)))
    sdw = vec(mean(std(w.S, dims = 1), dims = 2))
    S = w.S[(c-1)*n+1:c*n, :, :]; S = S .+ ratio .* reshape(sdw, 1, 1, :) .* randn(rng, size(S))
    D = devseries(S); nh = n ÷ 2; tr = 1:nh*NP; ts = nh*NP+10:2:n*NP-HSEL
    ev = fill(NaN + 0im, 4, length(SETS)); sk = fill(NaN, length(SETS)); k = 0
    for (d, r) in DR, Hh in HS
        Ma = ltp_fit(D, 1:n*NP, d, r, Hh, LS); Mh = ltp_fit(D, tr, d, r, Hh, LS)
        for li in eachindex(LS)
            k += 1; ev[:, k] = mono_ev(Ma[li]); sk[k] = skill(Mh[li], D, ts, HSEL)
        end
    end
    (; kappa = w.kappa, seed = w.seed, n, c, noise = ratio, sec = section_ev(S), ev, sk)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    wd = joinpath(ROOT, "cache", "truth", "slip_walks"); files = [joinpath(wd, f) for f in readdir(wd)]
    walks = [w for w in deserialize.(files) if !w.fallen && w.steps >= 1536]
    truth = Dict(w.kappa => (ev = eigvals(w.J); ComplexF64.(ev[sortperm(abs.(ev), rev = true)])) for w in walks)
    @printf("%d walks; truth: %s\n", length(walks), join(["κ=$κ $(round.(truth[κ], digits = 3))" for κ in sort(collect(keys(truth)))], "  "))
    jobs = [(w, n, c, ratio) for w in walks for n in NS for c in 1:(n <= 256 ? 4 : 3) for ratio in NOISE]
    out = Vector{Any}(undef, length(jobs)); done = Threads.Atomic{Int}(0)
    Threads.@threads :dynamic for i in eachindex(jobs)
        out[i] = job(jobs[i]...)
        Threads.atomic_add!(done, 1) % 500 == 0 && (println("  $(done[]) / $(length(jobs)) jobs"); flush(stdout))
    end
    mkpath(joinpath(ROOT, "cache", "slip_ltp"))
    serialize(joinpath(ROOT, "cache", "slip_ltp", "recovery.jls"), (; rows = out, SETS, NS, NOISE, truth, HSEL))
    println("wrote cache/slip_ltp/recovery.jls")
end
