#!/usr/bin/env julia
# Running deviations forecast at matched horizons (Rosenberg et al. 2020, J R Soc Interface 17:20200487, Fig 5:
# Δ = 6.25 … 100% of a stride, from every initial phase). Long running trials (maus_extract.jl), 6 sagittal joint
# angles, stride-normalised (100 samples per stride, hip phase), standardised; deviation from the 61-stride moving-
# average stride (maus_lattice.jl :ma) or from the trial's mean stride (:mean). Fit on the first half of the strides,
# scored on the second, from every 5th sample (≥ 150 samples of history). R² = 1 − Σ error² / Σ deviation² per horizon
# (0 = the baseline stride, the analogue of Rosenberg's PV model on their own condition).
#
# Models
#   DMD d=…   delay DMD on the deviations (opfit: ridge 1e-4, Schur-stabilised), free rollout from the window ending
#             at the initial sample; delay window d ∈ DELAYS samples (0.16–1.5 strides); rank chosen on an inner split
#             of the training half (fit on its first half, scored on its second, mean over horizons)
#   LPV vel   Rosenberg's phase-varying linear model, kinematic inputs only: deviation and its derivative at the
#             initial phase (+ constant) → deviation at +Δ, one least-squares map per Δ and per phase bin (NBIN bins;
#             their Gaussian phase weighting is replaced by bins). No exoskeleton torque: there is no input here.
#   LPV delay the same with a delay window as input: deviations at t, t−4, …, t−28 (⅓ stride)
#   LTI delay the same delay input, one map per Δ for all phases (no phase dependence)
#   lattice   one Floquet mode, μ chosen on the same inner split (ρ 0.05:0.1:0.95 × rotation 0:30:180°, error summed
#             over these horizons), shape b(θ) refitted on the training half, ψ fitted to the 50 samples ending at the
#             initial sample, run forward
# Per-joint R² (left and right averaged) is kept for comparison with Rosenberg's hip / knee / ankle.
#
# Usage: julia --project=. -t 16 scripts/maus_horizon.jl      (writes cache/maus/horizon.jls)

include("maus_lattice.jl")

const HZ = [6, 12, 19, 25, 38, 50, 75, 100]
const DELAYS = [16, 32, 50, 100, 150]
const DRANKS = [8, 16, 32, 64, 128, 256]
const NBIN = 20
const LAGS = 0:4:28
const STEP = 5
const HIST = 150

"Free rollout of a delay DMD from the window ending at row t (hmax rows)."
function roll(A, U, μD, D, t, d, hmax)
    p = size(D, 2); α = U' * vec(permutedims(D[t-d+1:t, :] .- μD)); out = zeros(hmax, p)
    for k in 1:hmax; α = A * α; out[k, :] = (U * α)[end-p+1:end] .+ vec(μD); end
    out
end
"Squared errors (horizon × channel) of `pred(t)` (hmax × p) over starts ts, and the reference Σ deviation²."
function score(pred, D, ts)
    E = zeros(length(HZ), size(D, 2)); R = zero(E)
    for t in ts
        P = pred(t)
        for (k, h) in enumerate(HZ); E[k, :] .+= (P[h, :] .- D[t+h, :]) .^ 2; R[k, :] .+= D[t+h, :] .^ 2; end
    end
    E, R
end
starts(r0, r1) = (r0 + HIST):STEP:(r1 - maximum(HZ))

function dmd_scores(D, tr, va, te)
    out = Dict{Int, Any}()
    for d in DELAYS
        ranks = [r for r in DRANKS if r < 6d]
        F = Dict(r => opfit(D[tr, :]; d, r) for r in ranks); μ = mean(D[tr, :], dims = 1)
        vs = [(E = score(t -> roll(F[r]..., μ, D, t, d, HZ[end]), D, starts(first(va), last(va))); sum(E[1]) / sum(E[2])) for r in ranks]
        rb = ranks[argmin(vs)]; trf = first(tr):last(va)
        Ff = opfit(D[trf, :]; d, r = rb); μf = mean(D[trf, :], dims = 1)
        out[d] = (; rank = rb, sc = score(t -> roll(Ff..., μf, D, t, d, HZ[end]), D, starts(first(te), last(te))))
    end
    out
end

phasebin(t) = 1 + (mod(t - 1, NPH) * NBIN) ÷ NPH
"Direct least-squares maps (one per horizon, per phase bin unless pooled) from inputs X(t) to D(t+Δ)."
function direct_scores(D, X, tr, te; pooled = false, λ = 1e-3)
    p = size(D, 2); ts_tr = starts(first(tr), last(tr)); ts_te = starts(first(te), last(te))
    bins = pooled ? fill(1, NPH) : [phasebin(t) for t in 1:NPH]; key(t) = bins[mod1(t, NPH)]
    W = Dict{Tuple{Int, Int}, Matrix{Float64}}()
    for (k, h) in enumerate(HZ), b in unique(bins)
        T = [t for t in first(tr)+maximum(LAGS):last(tr)-h if key(t) == b]
        Z = hcat(ones(length(T)), reduce(vcat, [X(t)' for t in T])); Yt = reduce(vcat, [D[t+h, :]' for t in T])
        W[(k, b)] = (Z' * Z + λ * size(Z, 1) * Diagonal([0; ones(size(Z, 2) - 1)])) \ (Z' * Yt)
    end
    E = zeros(length(HZ), p); R = zero(E)
    for t in ts_te, (k, h) in enumerate(HZ)
        z = vcat(1.0, X(t)); E[k, :] .+= (W[(k, key(t))]' * z .- D[t+h, :]) .^ 2; R[k, :] .+= D[t+h, :] .^ 2
    end
    E, R
end

function lattice_scores(D, k1, μ, ts; W = 50)
    M = fit_mode_fast(segments(D, 0, k1), μ); bθ = M.b; Ks = (size(bθ, 1) - 1) ÷ 2
    Bt = [[sum(bθ[k+Ks+1, c] * exp(im * k * 2π * (t - 1) / NPH) for k in -Ks:Ks) for c in axes(bθ, 2)] for t in 1:NPH]
    shape(t) = Bt[mod1(t, NPH)]
    pred = function (t)
        Q = [exp(μ * (s - t)) .* shape(s) for s in t-W+1:t+HZ[end]]
        Qw = reduce(vcat, [permutedims(q) for q in Q[1:W]])
        c = hcat(vec(real(Qw)), vec(-imag(Qw))) \ vec(D[t-W+1:t, :])
        reduce(vcat, [permutedims(real.((c[1] + im * c[2]) .* q)) for q in Q[W+1:end]])
    end
    score(pred, D, ts)
end
function lattice_choose(D, nh, va, te)
    grid = [μof(ρ, rot) for ρ in 0.05:0.1:0.95 for rot in 0:30:180]
    v = [(E = lattice_scores(D, nh ÷ 2, μ, starts(first(va), last(va))); sum(E[1]) / sum(E[2])) for μ in grid]
    μ = grid[argmin(v)]
    (; μ, sc = lattice_scores(D, nh, μ, starts(first(te), last(te))))
end

function trial_horizon(S, how)
    n = size(S, 1); nh = n ÷ 2; _, _, D = split_baseline(S, how)
    tr = 1:nh*NPH; va = (nh ÷ 2)*NPH+1:nh*NPH; te = nh*NPH+1:n*NPH
    Dv = similar(D); Dv[2:end-1, :] = (D[3:end, :] .- D[1:end-2, :]) ./ 2; Dv[1, :] = Dv[2, :]; Dv[end, :] = Dv[end-1, :]
    vel(t) = vcat(D[t, :], Dv[t, :]); dly(t) = reduce(vcat, [D[t-l, :] for l in LAGS])
    (; dmd = dmd_scores(D, 1:(nh÷2)*NPH, va, te),
       lpv_vel = direct_scores(D, vel, tr, te), lpv_delay = direct_scores(D, dly, tr, te),
       lti_delay = direct_scores(D, dly, tr, te; pooled = true),
       lattice = lattice_choose(D, nh, va, te))
end

r2(E, R; chans = :) = 1 .- vec(sum(E[:, chans], dims = 2)) ./ vec(sum(R[:, chans], dims = 2))

const CHAN = Dict(:angles => ["hip R", "knee R", "ankle R", "hip L", "knee L", "ankle L"],
                  :com => ["CoM lateral", "CoM height", "v lateral", "v forward", "v vertical"])

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    keep = qc_run.([strides(t.X) for t in A.trials]) .< 5; T = A.trials[keep]
    out = Dict{Tuple{Symbol, Symbol}, Any}()
    for (st, how) in ((:angles, :ma), (:angles, :mean), (:com, :ma))
        Ss = [state_strides(t, st) for t in T]
        R = tmap(i -> trial_horizon(Ss[i], how), collect(eachindex(Ss)))
        @printf("\n== %s, baseline %s: R² by horizon (%% stride), median over %d trials (%d failed) ==\n", st, how, count(!isnothing, R), count(isnothing, R))
        R = filter(!isnothing, R); out[(st, how)] = R
        @printf("%-22s", "horizon"); for h in HZ; @printf("%7.1f", h); end; println()
        show(lab, f) = (@printf("%-22s", lab); for k in eachindex(HZ); @printf("%7.2f", median(f(r)[k] for r in R)); end; println())
        for d in DELAYS; show(@sprintf("DMD d=%d (rank %s)", d, join(sort(unique(r.dmd[d].rank for r in R)), "/")), r -> r2(r.dmd[d].sc...)); end
        for (lab, k) in (("LPV vel (Rosenberg)", :lpv_vel), ("LPV delay", :lpv_delay), ("LTI delay", :lti_delay))
            show(lab, r -> r2(getfield(r, k)...))
        end
        show("lattice", r -> r2(r.lattice.sc...))
        @printf("  lattice μ chosen: ρ %s\n", join([@sprintf("%.2f", exp(real(r.lattice.μ) * NPH)) for r in R], " "))
        groups = st == :angles ? [("hip", [1, 4]), ("knee", [2, 5]), ("ankle", [3, 6])] : [(c, [k]) for (k, c) in enumerate(CHAN[:com])]
        println("  per channel, horizons 12.5%, 25%, 50%, 100% (DMD d=50 | LPV vel | LPV delay):")
        for (nm, ch) in groups
            @printf("    %-12s", nm)
            for f in (r -> r.dmd[50].sc, r -> r.lpv_vel, r -> r.lpv_delay)
                for k in (2, 4, 6, 8); @printf("%6.2f", median(r2(f(r)...; chans = ch)[k] for r in R)); end; print("  |")
            end
            println()
        end
        flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "maus", "horizon.jls"), (; out, HZ, DELAYS, CHAN))
    println("wrote cache/maus/horizon.jls")
end
