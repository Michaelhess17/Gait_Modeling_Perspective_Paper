#!/usr/bin/env julia
# Data for Fig 1 (walking lives in the transient): ground truth only, no fitting.
#
#   cFHN   coupled FitzHugh–Nagumo of systems_truth.jl (ε = 0.08, coupling 0.02). Disturbances arrive
#          continuously (white noise of equal size in every observed coordinate, i.e. each variable scaled by
#          its SD along the cycle). For a disturbance at phase s and age a, the transverse deviation observed
#          at s + a is Q'Π D⁻¹ Φ(s + a, s) D b (phase direction projected out, as in systems_truth.jl).
#          G[i, m] = ‖Q'Π D⁻¹ Φ D‖²_F for start phase i and age m (grid steps of T/NGRID).
#            typical response to one disturbance   sqrt(mean_i G[i, m] / mean_i G[i, 0])
#            share of the observed (stationary) transverse variance from disturbances younger than a
#                                                  Σ_i Σ_{m < a} G / Σ_i Σ_m G
#   SLIP   single-step VLO map J of slip_truth.jl, open loop and with speed feedback (κ = 0.1); one kick per
#          step (at VLO) of the size used for the kicked walks; deviations scaled by their stationary SD.
#          Share of the stationary variance from kicks younger than k steps (k/2 strides). Also one cached
#          kicked walk (κ = 0.1): observed VLO deviations and the linear response to each kick.
#   human  multipliers reported for human walking, ρ ≈ 0.3–0.8 per stride, one-dimensional: 1 − ρ^(2n).
#
# → cache/fig1/transient.jls

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("systems_truth.jl")              # cfhn!, propagators_from, complement, field (main block not run)
using Serialization, LinearAlgebra, Statistics, Printf

const ROOT = dirname(@__DIR__)
const NCYC = 10                          # ages up to 10 cycles (ρ¹⁰ ≈ 0.02)
const STEP = 2                           # every 2nd grid phase as a start

# ── coupled FHN ─────────────────────────────────────────────────────────────
r = deserialize(joinpath(ROOT, "cache", "truth", "systems_truth.jls"))["cfhn"]
T, U, D, ρ = r.T, r.U, r.D, r.ρ
nph = size(U, 1) - 1; F = field(cfhn!, nothing)
PQ = map(1:nph) do k
    fsc = D \ F(U[k, :]); Π = I - fsc * fsc' / dot(fsc, fsc); Q = complement(fsc); Q' * Π
end
starts = 1:STEP:nph
G = zeros(length(starts), NCYC * nph + 1)
Threads.@threads for (ii, i) in collect(enumerate(starts))
    P = propagators_from(cfhn!, nothing, U[i, :], T); M = P[nph+1]
    Mk = Matrix(1.0I, 4, 4)
    for m in 0:NCYC*nph
        k, q = divrem(m, nph)
        q == 0 && m > 0 && (Mk = M * Mk)
        Φ = P[q+1] * Mk
        G[ii, m+1] = sum(abs2, PQ[(i - 1 + m) % nph + 1] * (D \ Φ * D))
    end
end
age_c = (0:NCYC*nph) ./ nph
g = vec(mean(G; dims = 1))
typical = sqrt.(g ./ g[1])
share_c = cumsum(g) ./ sum(g)
@printf("cFHN ρ = %.3f; typical response peak %.2f at %.2f cycles; at 1, 2 cycles %.2f, %.2f (ρ: %.2f, %.2f)\n",
        ρ, maximum(typical), age_c[argmax(typical)], typical[nph+1], typical[2nph+1], ρ, ρ^2)
for a in (0.5, 1, 2, 3, 5)
    @printf("  share of observed transverse variance from disturbances < %.1f cycles old: %.2f\n", a,
            share_c[round(Int, a * nph)])
end
# worst case over phase and direction (Fig 4A), first two cycles
worst = r.curve; worst_τ = collect(r.τ)

# ── SLIP ────────────────────────────────────────────────────────────────────
sl = deserialize(joinpath(ROOT, "cache", "truth", "slip_truth.jls"))
const KICK = [0.0005, 0.001]
function slip_age(J; nmax = 200)
    Q = Diagonal(KICK .^ 2)
    Σ = reshape((I - kron(J, J)) \ vec(Matrix(Q)), 2, 2)
    Ds = Diagonal(sqrt.(diag(Σ))); Js = Ds \ J * Ds; Qs = Ds \ Q / Ds
    # deviation observed just before the kick at step k: Σ_{j ≥ 1} J^j Q J^j' (age j steps)
    w = [tr(Js^j * Qs * (Js^j)') for j in 1:nmax]
    (; Js, Ds, ρ = maximum(abs, eigvals(J)), age_steps = collect(1:nmax), share = cumsum(w) ./ sum(w),
       growth = [opnorm(Js^j) for j in 0:40])
end
slip0 = slip_age(sl["truth_0.0"].J); slip1 = slip_age(sl["truth_0.1"].J)
for (lab, s) in (("open loop", slip0), ("speed feedback", slip1))
    @printf("SLIP %s ρ = %.3f per step (%.3f per stride): share from kicks ≤ 1, 2, 4, 10 strides old: %s\n", lab, s.ρ,
            s.ρ^2, join([@sprintf("%.2f", s.share[2n]) for n in (1, 2, 4, 10)], ", "))
end

# one cached kicked walk with speed feedback: observed VLO deviations and each kick's linear response
w = deserialize(joinpath(ROOT, "cache", "truth", "slip_walks", "walk_k0.1_s1.jls"))
NW = 24
Dy = slip1.Ds; dev_obs = [(Dy \ (w.Z[k, :] .- w.zstar))[1] for k in 1:NW+1]     # Z[k]: arriving at VLO k
resp = fill(NaN, NW, NW + 1)                                                    # resp[j, k]: kick j → VLO k
for j in 1:NW, k in j+1:NW+1
    resp[j, k] = (Dy \ (w.J^(k - j) * w.K[j, :]))[1]
end
dev_lin = [sum(filter(!isnan, resp[:, k]); init = 0.0) for k in 1:NW+1]
@printf("kicked walk: max |observed − sum of linear responses| = %.3f SD (observed SD %.2f)\n",
        maximum(abs.(dev_obs .- dev_lin)), std(dev_obs))
raw = w.raw                                                                     # [t y ẋ ẏ], first 20 steps
# step boundaries (VLO, where each step's segment was stitched on): a short time gap that is isolated; the
# solver also saves extra points at contact events, which appear as clusters of short gaps
dt = median(diff(raw[:, 1])); gaps = findall(diff(raw[:, 1]) .< dt - 1e-7); tg = raw[gaps, 1]
isolated = [all(abs(tg[i] - tg[j]) > 0.003 for j in eachindex(tg) if j != i) for i in eachindex(tg)]
vlo_idx = vcat(1, gaps[isolated] .+ 1)
@printf("raw record: %d step boundaries, mean step %.3f s\n", length(vlo_idx), mean(diff(raw[vlo_idx, 1])))

# human multiplier range, one-dimensional
age_h = range(0, 10; length = 201)
human = (; ρ = (0.3, 0.8), lo = 1 .- 0.8 .^ (2 .* age_h), hi = 1 .- 0.3 .^ (2 .* age_h), age = collect(age_h))

mkpath(joinpath(ROOT, "cache", "fig1"))
serialize(joinpath(ROOT, "cache", "fig1", "transient.jls"),
          (; cfhn = (; ρ, age = collect(age_c), typical, share = share_c, worst, worst_τ),
             slip = (; open = (; ρ = slip0.ρ, age = slip0.age_steps ./ 2, share = slip0.share),
                       fb = (; ρ = slip1.ρ, age = slip1.age_steps ./ 2, share = slip1.share)),
             walk = (; dev_obs, dev_lin, resp, raw, vlo_idx, kappa = w.kappa, ρ = slip1.ρ),
             human))
println("→ cache/fig1/transient.jls")
