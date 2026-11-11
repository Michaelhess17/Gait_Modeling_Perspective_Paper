#!/usr/bin/env julia
# Lattice-constrained Floquet model (gait_lattice.jl) on long treadmill running trials (Maus et al. 2015; joint angles
# from maus_extract.jl): 4 people × 5–6 trials × ~320 strides at one self-selected speed each. The walking trials of
# the companion paper have 10–15 strides at several speeds, too few to tell a flat forecast-error profile from a
# missing one; here a trial has ~160 strides to fit on and ~160 to score, and a person's other trials are at the
# same speed, so μ chosen on them is not confounded by speed (cf. gait_lattice_speed.jl).
#
# Strides: hip (thigh) Hilbert phase of the right leg (fig5_gait_data.jl strides()), 100 points each; channels
# standardised by their SD. Two baselines, removed before modelling and scoring:
#   :mean  the trial's mean stride (as gait_lattice.jl)
#   :ma    a 61-stride centred moving average of the strides (Maus et al.'s de-trending; 6–22% of the deviation
#          variance is slow trend, which a near-neutral mode would otherwise 'predict'). Each scored stride is 1/61
#          of its own baseline.
# Model, segments, half-stride window, blocks (rest of stride, +1, +2 strides), μ grid and baselines (deviation DMD,
# rank chosen like μ; companion DMD on the full signal) are gait_lattice.jl's. Fit on the first half of the strides,
# scored on the second. Choice of μ (and deviation-DMD rank):
#   person   the person's other trials (same speed), summed normalised error over strides +1 and +2
#   inner    within the trial, without the scored half: fit on strides [0, n/4), choose on [n/4, n/2)
#   on-test  on the scored strides (optimistic), with NSH stride-shuffled copies as its null
# DDFA return map (Revzen & Guckenheimer; Maus et al. 2015): section state = deviation at phase φ₀ and φ₀ + 5
# (12-D), least-squares map from stride k to k+1 on the first half, and its instrumental-variable version
# (instrument: stride k−1); leading multiplier and R² of the section deviation 1 and 2 strides ahead on the second half.
# Mode shape b(θ) refitted on all strides of the trial at the person's μ.
# Two state spaces, cut at the same hip phase: :angles (6 sagittal joint angles) and :com, the centre of mass
# [lateral position, height, lateral, forward and vertical velocity] (velocities by central differences of the
# 250 Hz CoM, which Maus et al. computed from kinematics and forces), the variables Maus et al. predicted.
# QC: a trial is dropped when any channel has ≥ 5 consecutive strides whose deviation RMS exceeds 4× its median
# (subject 2 r5: left knee and ankle, strides 68–98, a left-leg marker artefact with no reconstruction flag).
#
# Usage: julia --project=. -t 16 scripts/maus_lattice.jl      (writes cache/maus/lattice.jls; ~40 min)

include("gait_lattice.jl")
using Random

const MA = 61
const NSH = 3
const SECTIONS = (1, 26, 51, 76)

"""gait_lattice.jl's fit_mode with a cheaper b-step: segment s's design block is H·K_s, H = [Gr Gi] (T_SEG × 50),
K_s = [aI −cI; −cI −aI] (ψ_s = a + ic). With H = Q_h R_h (thin QR), ‖H K_s x − S_s‖² = ‖R_h K_s x − Q_hᵀ S_s‖² + const,
so the stacked problem is solved exactly on 50-row blocks. Same start and alternating scheme; predictions agree with
fit_mode to ~1e-15 at rotations 15–165°. At rotation 0° and 180° the multiplier is real, the k and −k (resp. −k−1)
columns are conjugate and the real design is rank-deficient (cond ~1e32): fit_mode's training error then rises and
falls between iterations (NaN by 40), while this one decreases monotonically (running trial, ρ 0.05: 1915 vs 2060)."""
function fit_mode_fast(segs, μ; Ks = KS, iters = 15)
    G = basis(μ, Ks); nb = size(G, 2); p = size(segs[1], 2)
    F_ = qr(hcat(real(G), imag(G))); Qh = Matrix(F_.Q); Rh = F_.R; HS = [Qh' * S for S in segs]
    Rp = pinv(hcat(real(G), -imag(G)))
    Z = reduce(hcat, [(C = Rp * S; vec(C[1:nb, :] .- im .* C[nb+1:end, :])) for S in segs])
    F = svd(Z); b = reshape(F.U[:, 1] .* F.S[1], nb, p); ψ = conj.(F.V[:, 1])
    Ib = Matrix{Float64}(I, nb, nb)
    for _ in 1:iters
        Qb = G * b; Pq = pinv(hcat(vec(real(Qb)), vec(-imag(Qb))))          # a real mode leaves the 2nd column zero
        ψ = [(c = Pq * vec(S); c[1] + im * c[2]) for S in segs]
        M = reduce(vcat, [Rh * [real(ψ[s])*Ib -imag(ψ[s])*Ib; -imag(ψ[s])*Ib -real(ψ[s])*Ib] for s in eachindex(segs)])
        Bc = M \ reduce(vcat, HS); b = Bc[1:nb, :] .+ im .* Bc[nb+1:end, :]
    end
    (; G, b)
end

movmean(S, w) =(n = size(S, 1); h = w ÷ 2; cat([mean(S[max(1, k - h):min(n, k + h), :, :], dims = 1) for k in 1:n]..., dims = 1))
flat(S) = reduce(vcat, [S[k, :, :] for k in 1:size(S, 1)])

"Standardised signal Y, baseline B and deviation D (rows = samples), from strides S (n × NPH × p)."
function split_baseline(S, how)
    n, _, p = size(S); sd = std(flat(S), dims = 1); Sn = S ./ reshape(sd, 1, 1, p)
    Bs = how == :mean ? repeat(mean(Sn, dims = 1), n) : movmean(Sn, MA)
    Y = flat(Sn); B = flat(Bs); Y, B, Y .- B
end

"Longest run of consecutive strides with deviation RMS > 4× median, per channel."
function qc_run(S)
    D = S .- mean(S, dims = 1); worst = 0
    for c in axes(S, 3)
        a = vec(sqrt.(mean(abs2, D[:, :, c], dims = 2))); bad = a .> 4median(a); r = 0
        for b in bad; r = b ? r + 1 : 0; worst = max(worst, r); end
    end
    worst
end

"Free run of a delay DMD from the window ending at row t0 + W of Y (T_SEG − W rows, not de-meaned)."
function dmd_free(A, U, μY, Y, t0; d = 32, W = W_WIN)
    tw = t0 + W; h = vec(permutedims(Y[tw-d+1:tw, :] .- μY)); α = U' * h; p = size(Y, 2); out = zeros(T_SEG - W, p)
    for k in 1:T_SEG-W; α = A * α; out[k, :] = (U * α)[end-p+1:end] .+ vec(μY); end
    out
end

"Block errors on test strides [te) for every grid μ, deviation-DMD rank and the companion DMD; models fitted on [tr)."
function profiles(Y, B, D, tr, te; W = W_WIN, Ks = KS, dmds = true)
    trs = segments(D, tr...); tes = segments(D, te...); p = size(D, 2)
    (length(trs) < 2 || isempty(tes)) && return nothing
    ref = sum(blocks(zeros(T_SEG - W, p), S_) for S_ in tes)
    prof = zeros(length(RHOS), length(ROTS), 3)
    for (i, ρ) in enumerate(RHOS), (j, rot) in enumerate(ROTS)
        Mm = fit_mode_fast(trs, μof(ρ, rot); Ks)
        prof[i, j, :] = sum(blocks(predict_mode(Mm, S_; W)[W+1:end, :], S_) for S_ in tes)
    end
    dmds || return (; prof, ref)
    starts = [k * NPH for k in te[1]:te[2]-1 if (k + 3) * NPH <= size(D, 1) && k + 3 <= te[2]]
    rows = tr[1]*NPH+1:tr[2]*NPH
    A, U = opfit(Y[rows, :]); μY = mean(Y[rows, :], dims = 1)
    cdmd = sum(blocks(dmd_free(A, U, μY, Y, t0) .- B[t0+W+1:t0+T_SEG, :], S_) for (t0, S_) in zip(starts, tes))
    μD = mean(D[rows, :], dims = 1)
    derr(r) = (F = opfit(D[rows, :]; r); sum(blocks(dmd_free(F[1], F[2], μD, D, t0), S_) for (t0, S_) in zip(starts, tes)))
    ddmd = reduce(vcat, [derr(r)' for r in RANKS])
    (; prof, ref, cdmd, ddmd)
end

"DDFA return map at section φ₀ (LS and IV): leading multiplier and section R² 1 and 2 strides ahead on the test half."
function ddfa(D, n, φ0)
    Ds = permutedims(reshape(D, NPH, n, :), (2, 1, 3))
    Z = hcat(Ds[:, φ0, :], Ds[:, φ0+5, :])'; p = size(Ds, 3); nh = n ÷ 2
    X0, X1 = Z[:, 1:nh-1], Z[:, 2:nh]; Xm, Xc, Xp = Z[:, 1:nh-2], Z[:, 2:nh-1], Z[:, 3:nh]
    maps = (ls = X1 * pinv(X0), iv = (Xp * Xm') / (Xc * Xm'))
    map(maps) do A
        λ = eigvals(A); l = λ[argmax(abs.(λ))]
        r2(h) = (P = (A^h * Z[:, nh:n-h])[1:p, :]; T = Z[1:p, nh+h:n]; 1 - sum(abs2, P .- T) / sum(abs2, T))
        (; ρ = abs(l), rot = abs(rad2deg(angle(l))), r1 = r2(1), r2 = r2(2), λ)
    end
end

"Everything for one trial and baseline."
function trial_all(S, how; seed = 1)
    n = size(S, 1); nh = n ÷ 2; Y, B, D = split_baseline(S, how)
    main = profiles(Y, B, D, (0, nh), (nh, n))
    inner = profiles(Y, B, D, (0, nh ÷ 2), (nh ÷ 2, nh))
    shuf = map(1:NSH) do s
        Sp = S[randperm(MersenneTwister(1000seed + s), n), :, :]; Yp, Bp, Dp = split_baseline(Sp, how)
        profiles(Yp, Bp, Dp, (0, nh), (nh, n); dmds = false)
    end
    dd = Dict(φ0 => ddfa(D, n, φ0) for φ0 in SECTIONS)
    (; main, inner, shuf, ddfa = dd, n, D)
end

crit2(P, ref) = (P[:, :, 2] .+ P[:, :, 3]) ./ (ref[2] + ref[3])
skill(r, ij) = [1 - r.prof[ij, b] / r.ref[b] for b in 1:3]
dskill(r, k) = [1 - r.ddmd[k, b] / r.ref[b] for b in 1:3]

"Per-trial skills under the three selection rules."
function choose(R, subs)
    out = Vector{Any}(nothing, length(R))
    for i in eachindex(R)
        r = R[i]; r === nothing && continue; m = r.main
        js = [j for j in eachindex(R) if j != i && R[j] !== nothing && subs[j] == subs[i]]
        ijp = argmin(sum(crit2(R[j].main.prof, R[j].main.ref) for j in js))
        kp = argmin(sum(vec(sum(R[j].main.ddmd[:, 2:3], dims = 2)) ./ sum(R[j].main.ref[2:3]) for j in js))
        iji = argmin(crit2(r.inner.prof, r.inner.ref)); ki = argmin(vec(sum(r.inner.ddmd[:, 2:3], dims = 2)))
        ijo = argmin(crit2(m.prof, m.ref))
        null = mean([skill(s, argmin(crit2(s.prof, s.ref))) for s in r.shuf])
        out[i] = (; person = (ρ = RHOS[ijp[1]], rot = ROTS[ijp[2]], lat = skill(m, ijp), ddmd = dskill(m, kp), rank = RANKS[kp]),
                  inner = (ρ = RHOS[iji[1]], rot = ROTS[iji[2]], lat = skill(m, iji), ddmd = dskill(m, ki), rank = RANKS[ki]),
                  ontest = (ρ = RHOS[ijo[1]], rot = ROTS[ijo[2]], lat = skill(m, ijo), null),
                  cdmd = [1 - m.cdmd[b] / m.ref[b] for b in 1:3])
    end
    out
end

"Mode shape over the stride (NPH × p complex) at μ, fitted on all strides of the trial."
function shape(D, n, μ; Ks = KS)
    b = fit_mode_fast(segments(D, 0, n), μ; Ks).b; θ = range(0, 2π; length = NPH + 1)[1:NPH]
    permutedims(reduce(hcat, [sum(b[k+Ks+1, :] .* exp(im * k * t) for k in -Ks:Ks) for t in θ]))
end

mi(v) = @sprintf("%.2f [%.2f, %.2f]", median(v), quantile(v, 0.25), quantile(v, 0.75))

"CoM state [lateral position, height, lateral, forward, vertical velocity] (central differences, 250 Hz)."
comstate(t) = (c = t.com; v = similar(c); v[2:end-1, :] = (c[3:end, :] .- c[1:end-2, :]) .* (t.fs / 2);
               v[1, :] = v[2, :]; v[end, :] = v[end-1, :]; hcat(c[:, 1], c[:, 3], v))
"Strides of a trial in a state space (:angles or :com), cut at the right hip phase."
state_strides(t, st) = st == :angles ? strides(t.X) : strides(hcat(t.X[:, 1], comstate(t)))[:, :, 2:end]

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    Sa = [strides(t.X) for t in A.trials]
    Sc = [strides(hcat(t.X[:, 1], comstate(t)))[:, :, 2:end] for t in A.trials]
    qc = qc_run.(Sa); keep = qc .< 5
    for (t, q) in zip(A.trials, qc)
        q >= 5 && @printf("dropped subj%d r%d: %d consecutive outlying strides\n", t.subj, t.rep, q)
    end
    T = A.trials[keep]; subs = [t.subj for t in T]; states = Dict(:angles => Sa[keep], :com => Sc[keep])
    @printf("%d trials, %s strides each\n", length(T), mi(Float64.(size.(states[:angles], 1))))
    res = Dict{Tuple{Symbol, Symbol}, Any}()
    for st in (:angles, :com), how in (:ma, :mean)
        Ss = states[st]; t0 = time()
        R = tmap(i -> trial_all(Ss[i], how; seed = i), collect(eachindex(Ss)))
        @printf("\n== %s, baseline %s (%.0f s; %d failed) ==\n", st, how, time() - t0, count(isnothing, R))
        H = choose(R, subs)
        μp = Dict(s => (ij = argmin(sum(crit2(R[j].main.prof, R[j].main.ref) for j in findall(subs .== s))); (RHOS[ij[1]], ROTS[ij[2]])) for s in unique(subs))
        for rule in (:person, :inner, :ontest)
            for (b, nm) in ((1, "rest of stride"), (2, "+1 stride"), (3, "+2 strides"))
                lat = [getfield(h, rule).lat[b] for h in H]
                extra = rule == :ontest ? @sprintf("  stride-shuffled %s", mi([h.ontest.null[b] for h in H])) :
                        @sprintf("  deviation DMD %s  companion DMD %s", mi([getfield(h, rule).ddmd[b] for h in H]), mi([h.cdmd[b] for h in H]))
                @printf("  %-7s R² %-15s lattice %s%s\n", rule, nm, mi(lat), extra)
            end
        end
        for s in unique(subs)
            I = findall(subs .== s)
            @printf("  subj%d: person μ %.2f ∠ %3d° | inner-split ρ per trial %s | on-test ρ %s\n", s, μp[s]...,
                    join([@sprintf("%.2f", H[i].inner.ρ) for i in I], " "), join([@sprintf("%.2f", H[i].ontest.ρ) for i in I], " "))
            for φ0 in SECTIONS
                @printf("      DDFA φ₀ %2d: LS |λ| %s ∠ %s°, R² +1 %s  |  IV |λ| %s, R² +1 %s\n", φ0,
                        join([@sprintf("%.2f", R[i].ddfa[φ0].ls.ρ) for i in I], " "), join([@sprintf("%.0f", R[i].ddfa[φ0].ls.rot) for i in I], " "),
                        join([@sprintf("%.2f", R[i].ddfa[φ0].ls.r1) for i in I], " "),
                        join([@sprintf("%.2f", R[i].ddfa[φ0].iv.ρ) for i in I], " "), join([@sprintf("%.2f", R[i].ddfa[φ0].iv.r1) for i in I], " "))
            end
        end
        shapes = [shape(R[i].D, R[i].n, μof(μp[subs[i]]...)) for i in eachindex(R)]
        res[(st, how)] = (; R = [(; main = r.main, inner = r.inner, shuf = r.shuf, ddfa = r.ddfa, n = r.n) for r in R], H, μp, shapes)
        flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "maus", "lattice.jls"),
              (; res, subs, reps = [t.rep for t in T], speed = [t.speed for t in T], nstrides = size.(states[:angles], 1),
                 mean_cycles = Dict(k => [mean(S, dims = 1)[1, :, :] for S in v] for (k, v) in states),
                 channels = Dict(:angles => A.channels, :com => ["CoM lateral", "CoM height", "v lateral", "v forward", "v vertical"])))
    println("wrote cache/maus/lattice.jls")
end
