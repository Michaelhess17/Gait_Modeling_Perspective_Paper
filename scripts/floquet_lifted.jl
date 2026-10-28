#!/usr/bin/env julia
# Representing a limit cycle's Floquet structure in linear models (OUTLINE.md §6 ¶6–7). SLIP, centre-of-mass state.
#
# Koopman theory: a stable limit cycle (frequency ω, Floquet exponent μ; SLIP truth 0.769 ∠ 13.6° per step) has
# Koopman eigenvalues on the lattice {ikω + mμ}, eigenfunctions e^{ikθ(x)} ψ(x)^m. A linear model of the state
# should hold exact harmonics on the unit circle and the same sideband next to every harmonic. Hankel-DMD fitted
# to 3-step decays blends each harmonic with its sideband (they are 0.038 cycles per step apart) and leaves the
# cycle amplitude neutral, so its off-cycle deviation plateaus. Remedies tested here, same data and scoring:
#
#   C  data design: Hankel-DMD on 10 decays of 30 steps instead of 3 (slip_long_decays.jl), with and without
#      stride kicks; τ ≤ 165 (the test warm-up is 170 samples)
#   D  certification: ResDMD residuals (Colbrook & Townsend) of each eigenpair on held-out decays
#   A  phase-lifted Floquet model: phase supplied nonlinearly (nearest point on a fitted cycle), deviation from
#      the cycle evolved by a linear time-periodic map δx_{t+1} = Σ_h A_h e^{ihθ_t} z_t (z_t: lagged deviations)
#   B  lattice-constrained fit: x_r(t) = a(ωt+φ_r) + Re[ψ_r e^{μt} b(ωt+φ_r)], a and b Fourier series shared by
#      all decays, (ω, μ) the only spectral parameters
#
# Scoring (all methods): free run of 30 steps after the W_WARM warm-up of the 4 off-cycle test starts; rate over
# the first 4 steps against the model's own on-cycle run (as Fig 6B), deviation after 25 steps / initial (plateau),
# and the multipliers each model implies.
#
# Usage: julia --project=. scripts/floquet_lifted.jl <C|D|A|B>      (writes cache/floquet/lifted_<part>.jls)

include("offcycle_v2.jl")
using Printf

const P = SPC_F
const NH = 30P
const LD = deserialize(cache_path("truth/slip_long_decays.jls"))
const LSR = deserialize(cache_path("truth/slip_long_starts.jls"))
const YS, MU, SIG = raw_training("slip", "decays"; seed = 1)        # 10 × 3 steps, offcycle_v2's standardisation
std_(X) = (X .- MU) ./ SIG
const YL = std_.(LD.decays[1])                                        # 10 × 30 steps, stride kicks
const YLC = std_.(LD.decays[11])                                      # 10 × 30 steps, kick-free
const HOLD_S = std_.(slipd().decays[2]); const HOLD_L = std_.(LD.decays[2]); const HOLD_LC = std_.(LD.decays[12])
const CTRUE = std_(cycle_states("slip"))
const STARTS = true_starts("slip"); const ONS = true_oncycle("slip")
const TRUTH_L = [std_(X) for X in LSR.runs[1.0]]
const EXACT = slipd().λ

cycenv(d) = [sqrt(mean(abs2, d[k*P+1:(k+1)*P])) for k in 0:div(length(d), P)-1]
wnear(X, R) = [minimum(sum(abs2, R[max(1, i - 60):min(end, i + 60), :] .- X[i:i, :], dims = 2))^0.5 for i in axes(X, 1)]

"Score a predictor warm-up window ↦ NH × 3 free run, on the 4 test starts."
function score(pred)
    rows = NamedTuple[]
    for (T, T0, TL) in zip(STARTS, ONS, TRUTH_L)
        et = cycenv(nearest(TL[W_WARM+1:end, :], CTRUE))
        fc = pred(std_(T)[1:W_WARM, :]); f0 = pred(std_(T0)[1:W_WARM, :])
        if fc === nothing || f0 === nothing || !all(isfinite, fc) || maximum(abs, fc) > 50
            push!(rows, (; truth4 = (et[5] / et[1])^0.25, own4 = NaN, plateau = NaN, vs_true25 = NaN)); continue
        end
        eo = cycenv(wnear(fc, f0)); ev = cycenv(nearest(fc, CTRUE))
        push!(rows, (; truth4 = (et[5] / et[1])^0.25, own4 = (eo[5] / eo[1])^0.25, plateau = eo[26] / eo[1],
                       vs_true25 = ev[26] / ev[1]))
    end
    rows
end
fmt(v) = join([@sprintf("%.2f", x) for x in v], " ")
show_score(lab, s) = @printf("%-34s 4-step rate %s (truth %s) | plateau %s | vs true cycle %s\n", lab,
                             fmt([r.own4 for r in s]), fmt([r.truth4 for r in s]), fmt([r.plateau for r in s]), fmt([r.vs_true25 for r in s]))

"DMD eigenvalues relative to its own cycle: nearest harmonic n, rotation per cycle δ (°), |λ| per cycle."
function dmd_lattice(A)
    λ = eigvals(A); f = angle.(λ) .* P ./ 2π
    fc = f[argmin(abs.(abs.(λ) .- 1) .+ 10 .* (f .< 0.5))]; Tc = P / fc
    [(; n = round(Int, fj / fc), δ = 360 * (fj / fc - round(fj / fc)), ρ = abs(l)^Tc, λ = l) for (fj, l) in zip(f, λ) if fj >= -1e-9]
end
function show_lattice(A; nshow = 12)
    L = sort(filter(x -> x.ρ > 0.5, dmd_lattice(A)), by = x -> -x.ρ)
    println("    ", join([@sprintf("(%d, %+.1f°, %.3f)", x.n, x.δ, x.ρ) for x in L[1:min(end, nshow)]], " "))
end

# ═════════════════════════════════════════════════════════════════════════════
# C: Hankel-DMD on long decays
# ═════════════════════════════════════════════════════════════════════════════
function part_C()
    out = NamedTuple[]
    for (lab, Ys) in (("3-step decays", YS), ("30-step decays", YL), ("30-step kick-free", YLC)), τ in (82, 110, 165), r in (60, 120, 200)
        lab == "3-step decays" && !(τ == 82 && r == 120) && continue
        m = try dmd_model_multi(Ys; τ, r, variant = "ls", ridge = 1e-8) catch; continue end
        s = score(W -> dmd_free(m, W, NH))
        show_score(@sprintf("C %s τ=%d r=%d", lab, τ, r), s); show_lattice(m.A)
        push!(out, (; data = lab, τ, r, score = s, lattice = dmd_lattice(m.A)))
    end
    serialize(cache_path("floquet/lifted_C.jls"), out)
end

# ═════════════════════════════════════════════════════════════════════════════
# D: ResDMD residuals on held-out decays
# ═════════════════════════════════════════════════════════════════════════════
"Residual ‖Ψ_Y g − λ Ψ_X g‖ / ‖Ψ_X g‖ of each eigenpair of K = Aᵀ (Koopman matrix on the reduced delay basis)."
function resdmd(m, Yh)
    Hs = [build_hankel_multi(Y, m.τ) for Y in Yh]
    X = reduce(hcat, [m.U' * H[:, 1:end-1] for H in Hs]); Z = reduce(hcat, [m.U' * H[:, 2:end] for H in Hs])
    G = X * X'; Am = X * Z'; L = Z * Z'
    F = eigen(Matrix(m.A')); res = similar(F.values, Float64)
    for (i, (λ, g)) in enumerate(zip(F.values, eachcol(F.vectors)))
        num = real(g' * (L - λ * Am' - conj(λ) * Am + abs2(λ) * G) * g); den = real(g' * G * g)
        res[i] = sqrt(max(num, 0) / den)
    end
    F.values, res
end
"DMD restricted to the eigenpairs in `keep` (by index into eigen(A))."
function dmd_restrict(m, keep)
    F = eigen(m.A); W = inv(F.vectors)
    A = real(F.vectors[:, keep] * Diagonal(F.values[keep]) * W[keep, :])
    (; m..., A)
end
function part_D()
    out = NamedTuple[]
    for (lab, Ys, Yh, τ, r) in (("3-step", YS, HOLD_S, 82, 120), ("30-step", YL, HOLD_L, 110, 120))
        m = dmd_model_multi(Ys; τ, r, variant = "ls", ridge = 1e-8)
        λ, res = resdmd(m, Yh)
        f = angle.(λ) .* P ./ 2π; fc = f[argmin(abs.(abs.(λ) .- 1) .+ 10 .* (f .< 0.5))]; Tc = P / fc
        println("D $lab decays (τ=$τ r=$r): eigenpairs with |λ| per cycle > 0.5, sorted by residual")
        idx = [i for i in eachindex(λ) if abs(λ[i])^Tc > 0.5 && f[i] >= -1e-9]
        for i in sort(idx, by = i -> res[i])
            @printf("    n %2d  δ %+6.1f°  |λ| %.3f  residual %.4f\n", round(Int, f[i] / fc), 360 * (f[i] / fc - round(f[i] / fc)),
                    abs(λ[i])^Tc, res[i])
        end
        for q in (0.25, 0.5)
            ε = quantile(res, q); keep = findall(res .<= ε)
            mk = dmd_restrict(m, keep)
            show_score(@sprintf("D %s keep residual ≤ q%.2f (%d/%d)", lab, q, length(keep), length(res)), score(W -> dmd_free(mk, W, NH)))
        end
        push!(out, (; data = lab, τ, r, λ, res))
    end
    serialize(cache_path("floquet/lifted_D.jls"), out)
end

# ═════════════════════════════════════════════════════════════════════════════
# A: phase-lifted Floquet model (linear time-periodic map on the deviation from a fitted cycle)
# ═════════════════════════════════════════════════════════════════════════════
# Phase θ = ωt − φ_r: for a decay the asymptotic phase advances exactly uniformly, so one offset per decay (by
# alignment) and the period (autocorrelation) fix it. Cycle: Fourier series in θ fitted to all samples (biased by
# the decays' mean deviation; the affine term below absorbs that). Deviation δ = x − x*(θ), lag vector
# z_t = (δ_t, δ_{t−ℓ}, …). One period ahead, so nothing is composed (a per-sample map multiplied over a period
# compounds its errors, as the piecewise maps of Fig 3D; tried, multipliers up to 10²):
#   z_{t+P} = W [f(θ_t) ⊗ z_t ; f(θ_t)]          f = (1, cos hθ, sin hθ), h ≤ H
# M(θ) = Σ_h f_h(θ) W_h is the monodromy at phase θ; its eigenvalues are the Floquet multipliers at every phase
# (reported: median over 22 phases). The phase offset left in δ is the neutral direction (multiplier ≈ 1).
fbasis(θ, K) = vcat(1.0, reduce(vcat, [[cos(k * θ), sin(k * θ)] for k in 1:K]; init = Float64[]))
cyc(Cf, K, θ) = vec(fbasis(θ, K)' * Cf)
cycmat(Cf, K, θs) = reduce(hcat, [cyc(Cf, K, t) for t in θs])'

"Offset φ minimising ‖Y − x*(ω t − φ)‖ over the window (cycle tabulated on a fine grid)."
function align_phase(Y, Cf, K, ω; ngrid = 1100)
    θg = range(0, 2π; length = ngrid + 1)[1:end-1]; G = cycmat(Cf, K, θg); t = 0:size(Y, 1)-1
    e = map(0:ngrid-1) do s
        idx = mod.(round.(Int, (ω .* t) ./ step(θg)) .- s, ngrid) .+ 1; sum(abs2, Y .- G[idx, :])
    end
    θg[argmin(e)]
end

"Period (autocorrelation, parabolic peak, unless ω is given), per-decay offsets and Fourier cycle."
function fit_phase_cycle(Ys; K = 20, iters = 4, ω = nothing)
    if ω === nothing
        ac(l) = mean(sum(Y[1:end-l, :] .* Y[1+l:end, :]) / (size(Y, 1) - l) for Y in Ys)
        ls = 60:160; a = ac.(ls); j = argmax(a[21:end]) + 20
        Pc = ls[j] + (a[j-1] - a[j+1]) / (2 * (a[j-1] - 2a[j] + a[j+1])); ω = 2π / Pc
    end
    Pc = 2π / ω; ref = Ys[1]; φs = zeros(length(Ys))
    Cf = reduce(hcat, fbasis.(ω .* (0:size(ref, 1)-1), K))' \ ref
    for _ in 1:iters
        φs = [align_phase(Y, Cf, K, ω) for Y in Ys]
        F = reduce(vcat, [reduce(hcat, fbasis.(ω .* (0:size(Y, 1)-1) .- φ, K))' for (Y, φ) in zip(Ys, φs)])
        Cf = F \ reduce(vcat, Ys)
    end
    (; Cf, K, ω, Pc, φs)
end

lagvec(D, t, L, ℓ) = reduce(vcat, [D[t-j*ℓ, :] for j in 0:L-1])

function fit_lifted(Ys; K = 20, H = 8, L = 3, ℓ = 8, ridge = 1e-6)
    C = fit_phase_cycle(Ys; K); Pn = round(Int, C.Pc)
    Φ = Vector{Float64}[]; T = Vector{Float64}[]
    for (Y, φ) in zip(Ys, C.φs)
        θ = C.ω .* (0:size(Y, 1)-1) .- φ; D = Y .- cycmat(C.Cf, K, θ)
        for t in 1+(L-1)*ℓ:size(Y, 1)-Pn
            f = fbasis(θ[t], H); push!(Φ, vcat(kron(f, lagvec(D, t, L, ℓ)), f)); push!(T, lagvec(D, t + Pn, L, ℓ))
        end
    end
    F = reduce(hcat, Φ); Tm = reduce(hcat, T)
    Wd = Tm * F' / (F * F' + ridge * tr(F * F') / size(F, 1) * I)
    (; C..., Pn, H, L, ℓ, Wd, d = size(Ys[1], 2), nsamp = size(F, 2))
end
"Monodromy at phase θ (on the lag vector) and the affine term."
function monodromy(M, θ)
    nz = M.L * M.d; f = fbasis(θ, M.H)
    sum(f[h] .* M.Wd[:, (h-1)*nz+1:h*nz] for h in 1:2M.H+1), M.Wd[:, (2M.H+1)*nz+1:end] * f
end

"Free run of n samples after the warm-up window W: each sample from the one a period earlier."
function lifted_free(M, W, n)
    φ = align_phase(W, M.Cf, M.K, M.ω); nw = size(W, 1)
    θ = M.ω .* (0:nw+n-1) .- φ; D = zeros(nw + n, M.d); D[1:nw, :] = W .- cycmat(M.Cf, M.K, θ[1:nw])
    for t in nw+1:nw+n
        s = t - M.Pn; A, b = monodromy(M, θ[s]); D[t, :] = (A * lagvec(D, s, M.L, M.ℓ) .+ b)[1:M.d]
    end
    cycmat(M.Cf, M.K, θ[nw+1:end]) .+ D[nw+1:end, :]
end

"Floquet multipliers: eigenvalues of M(θ) at 22 phases; neutral (closest to 1) and the two largest others."
function lifted_floquet(M)
    evs = [sort(eigvals(monodromy(M, θ)[1]), by = x -> -abs(x)) for θ in range(0, 2π; length = 23)[1:end-1]]
    neu = [e[argmin(abs.(e .- 1))] for e in evs]
    tr2 = [(r = e[setdiff(eachindex(e), argmin(abs.(e .- 1)))]; r[1:2]) for e in evs]
    lead = [t[argmax(abs.(t) .+ 1e-9 .* imag.(t))] for t in tr2]
    (; neutral = median(abs.(neu)), mod = median(abs.(lead)), ang = median(rad2deg.(abs.(angle.(lead)))),
       complex = count(t -> abs(imag(t[1])) > 1e-9, tr2), second = median(abs.(last.(tr2))), evs)
end

function part_A()
    out = NamedTuple[]
    for (lab, Ys) in (("3-step decays", YS), ("30-step kicked", YL), ("30-step kick-free", YLC)), H in (4, 8), (L, ℓ) in ((1, 1), (2, 10))
        M = fit_lifted(Ys; H, L, ℓ); fq = lifted_floquet(M)
        s = score(W -> lifted_free(M, W, NH))
        show_score(@sprintf("A %s H=%d L=%d ℓ=%d", lab, H, L, ℓ), s)
        @printf("    period %.2f; %d samples; Floquet per cycle: neutral %.3f | leading %.3f ∠ %.1f° (complex at %d/22 phases) | second %.3f\n",
                M.Pc, M.nsamp, fq.neutral, fq.mod, fq.ang, fq.complex, fq.second)
        push!(out, (; data = lab, H, L, ℓ, score = s, floquet = fq, Pc = M.Pc))
    end
    serialize(cache_path("floquet/lifted_A.jls"), out)
end

# ═════════════════════════════════════════════════════════════════════════════
# B: lattice-constrained fit (Koopman spectrum of a limit cycle: harmonics kω, sidebands kω + μ)
# ═════════════════════════════════════════════════════════════════════════════
# x_r(t) = a(ϑ) + Re[ψ_r e^{μt} b(ϑ)],  ϑ = ωt − φ_r,  a = Σ_{k≤K} real Fourier,  b = Σ_{|k|≤Ks} b_k e^{ikϑ} (complex)
# Stage 1: free sideband amplitudes per decay; (σ, ν) of μ = σ + iν by variable projection (grid over per-cycle
# modulus and rotation, Nelder–Mead); offsets φ_r by alignment; the period from the autocorrelation, or for decays of
# ≥ 10 cycles by golden section on the cycle-only fit of their second halves (a period error of 0.03 samples builds
# a secular phase drift over 30 cycles that the fit otherwise absorbs as a near-neutral sideband).
# Stage 2, at the stage-1 μ: one mode shape b shared by all decays and one complex amplitude ψ_r per decay
# (alternating least squares; refining μ here drifted). A forecast from a new window fits only its offset φ and
# amplitude ψ, so it returns to the shared cycle a.
sideb(t, ϑ, σ, ν, Ks) = reduce(hcat, [[exp(σ * t) * cos(ν * t + k * ϑ), exp(σ * t) * sin(ν * t + k * ϑ)] for k in -Ks:Ks])

"Stage 1 design and residual for (σ, ν): shared cycle columns plus per-decay sideband columns."
function lattice_stage1(Ys, C, σ, ν; Ks = 10)
    nr = [size(Y, 1) for Y in Ys]; ns = 2(2Ks + 1); ncyc = 2C.K + 1
    F = zeros(sum(nr), ncyc + length(Ys) * ns); row = 0
    for (r, (Y, φ)) in enumerate(zip(Ys, C.φs))
        for t in 0:nr[r]-1
            ϑ = C.ω * t - φ; row += 1
            F[row, 1:ncyc] = fbasis(ϑ, C.K); F[row, ncyc+(r-1)*ns+1:ncyc+r*ns] = vec(sideb(t, ϑ, σ, ν, Ks))
        end
    end
    X = reduce(vcat, Ys); B = F \ X
    sum(abs2, X - F * B), B
end

"Minimal Nelder–Mead on a 2-vector."
function nm2(f, x0, h; iters = 60)
    S = [x0, x0 .+ [h[1], 0.0], x0 .+ [0.0, h[2]]]; v = f.(S)
    for _ in 1:iters
        o = sortperm(v); S, v = S[o], v[o]; c = (S[1] .+ S[2]) ./ 2
        xr = c .+ (c .- S[3]); fr = f(xr)
        if fr < v[1]
            xe = c .+ 2 .* (c .- S[3]); fe = f(xe); (S[3], v[3]) = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < v[2]
            S[3], v[3] = xr, fr
        else
            xc = c .+ 0.5 .* (S[3] .- c); fc = f(xc)
            if fc < v[3]; S[3], v[3] = xc, fc
            else; S[2] = S[1] .+ 0.5 .* (S[2] .- S[1]); S[3] = S[1] .+ 0.5 .* (S[3] .- S[1]); v[2], v[3] = f(S[2]), f(S[3]); end
        end
    end
    S[argmin(v)], minimum(v)
end
percycle(C, σ, ν) = (exp(σ * C.Pc), mod(rad2deg(ν * C.Pc) + 180, 360) - 180)

"Stage 2: shared cycle a and mode shape b, per-decay complex amplitude ψ_r, for fixed (σ, ν)."
function lattice_stage2(Ys, C, σ, ν, ψ0; Ks = 10, iters = 25)
    ψ = copy(ψ0); X = reduce(vcat, Ys); ncyc = 2C.K + 1; nb = 2Ks + 1; Ab = nothing
    E(t, ϑ) = [exp((σ + im * ν) * t + im * k * ϑ) for k in -Ks:Ks]
    rows(r, Y, φ) = [(t, C.ω * t - φ) for t in 0:size(Y, 1)-1]
    for _ in 1:iters
        F = reduce(vcat, [reduce(vcat, [hcat(fbasis(ϑ, C.K)', real.(ψ[r] .* E(t, ϑ))', real.(im .* ψ[r] .* E(t, ϑ))')
                                        for (t, ϑ) in rows(r, Y, φ)]) for (r, (Y, φ)) in enumerate(zip(Ys, C.φs))])
        Ab = F \ X
        a = Ab[1:ncyc, :]; b = Ab[ncyc+1:ncyc+nb, :] .+ im .* Ab[ncyc+nb+1:end, :]      # (2Ks+1) × d complex
        for (r, (Y, φ)) in enumerate(zip(Ys, C.φs))
            R = Y .- reduce(hcat, [cyc(a, C.K, ϑ) for (_, ϑ) in rows(r, Y, φ)])'
            g = reduce(hcat, [vec(transpose(E(t, ϑ)) * b) for (t, ϑ) in rows(r, Y, φ)])'   # n × d complex
            G = hcat(vec(real.(g)), vec(-imag.(g))); c = G \ vec(R); ψ[r] = c[1] + im * c[2]
        end
    end
    a = Ab[1:ncyc, :]; b = Ab[ncyc+1:ncyc+nb, :] .+ im .* Ab[ncyc+nb+1:end, :]
    res = sum(sum(abs2, Y .- reduce(hcat, [cyc(a, C.K, ϑ) .+ real.(ψ[r] .* vec(transpose(E(t, ϑ)) * b)) for (t, ϑ) in rows(r, Y, φ)])')
              for (r, (Y, φ)) in enumerate(zip(Ys, C.φs)))
    (; a, b, ψ, res)
end

function fit_lattice(Ys; K = 20, Ks = 10)
    C = fit_phase_cycle(Ys; K)
    obj(C, p) = lattice_stage1(Ys, C, log(clamp(p[1], 0.05, 0.999)) / C.Pc, deg2rad(p[2]) / C.Pc; Ks)[1]
    grid = [(ρ, r) for ρ in 0.5:0.05:0.95, r in 0:4:60]; vals = [obj(C, [g...]) for g in grid]
    p1, _ = nm2(p -> obj(C, p), [grid[argmin(vals)]...], [0.03, 2.0])
    if minimum(size(Y, 1) for Y in Ys) >= 10C.Pc                           # long decays: period from the second
        late = [Y[size(Y, 1)÷2+1:end, :] for Y in Ys]                         # halves, which sit on the cycle
        function fP(Pc)
            Cl = fit_phase_cycle(late; K, ω = 2π / Pc, iters = 2)
            sum(sum(abs2, Y .- cycmat(Cl.Cf, K, Cl.ω .* (0:size(Y, 1)-1) .- φ)) for (Y, φ) in zip(late, Cl.φs))
        end
        lo, hi = C.Pc - 0.3, C.Pc + 0.3; g = (sqrt(5) - 1) / 2
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo); f1, f2 = fP(x1), fP(x2)
        for _ in 1:16
            f1 < f2 ? ((hi, x2, f2) = (x2, x1, f1); x1 = hi - g * (hi - lo); f1 = fP(x1)) :
                      ((lo, x1, f1) = (x1, x2, f2); x2 = lo + g * (hi - lo); f2 = fP(x2))
        end
        C = fit_phase_cycle(Ys; K, ω = 2π / ((lo + hi) / 2))
    end
    p1, _ = nm2(p -> obj(C, p), p1, [0.02, 1.0])
    σ, ν = log(p1[1]) / C.Pc, deg2rad(p1[2]) / C.Pc
    _, B1 = lattice_stage1(Ys, C, σ, ν; Ks)
    ncyc = 2K + 1; ns = 2(2Ks + 1)                                     # rank-1 start for ψ_r from stage-1 amplitudes
    Cr = [B1[ncyc+(r-1)*ns+1:ncyc+r*ns, :] for r in eachindex(Ys)]
    Z = reduce(hcat, [vec(Cr[r][1:2:end, :] .- im .* Cr[r][2:2:end, :]) for r in eachindex(Ys)])
    U = svd(Z); ψ0 = conj.(U.V[:, 1]) .* U.S[1]
    S2 = lattice_stage2(Ys, C, σ, ν, ψ0; Ks)                          # shared mode shape at the stage-1 μ
    (; C..., Ks, σ, ν, stage1 = percycle(C, σ, ν), rank1 = U.S[1]^2 / sum(abs2, U.S), S2..., d = size(Ys[1], 2))
end

"Forecast: fit the window's offset φ (local search) and amplitude ψ, then continue the lattice model."
function lattice_free(M, W, n)
    nw = size(W, 1); t = 0:nw-1; E(t, ϑ) = [exp((M.σ + im * M.ν) * t + im * k * ϑ) for k in -M.Ks:M.Ks]
    function fitψ(φ)
        ϑ = M.ω .* t .- φ; R = W .- reduce(hcat, [cyc(M.a, M.K, x) for x in ϑ])'
        g = reduce(hcat, [vec(transpose(E(tt, x)) * M.b) for (tt, x) in zip(t, ϑ)])'
        G = hcat(vec(real.(g)), vec(-imag.(g))); c = G \ vec(R); ψ = c[1] + im * c[2]
        sum(abs2, vec(R) .- G * c), ψ
    end
    φ0 = align_phase(W, M.a, M.K, M.ω); φs = φ0 .+ range(-0.3, 0.3; length = 61)
    best = argmin([fitψ(φ)[1] for φ in φs]); φ = φs[best]; ψ = fitψ(φ)[2]
    tf = nw:nw+n-1
    reduce(hcat, [cyc(M.a, M.K, M.ω * tt - φ) .+ real.(ψ .* vec(transpose(E(tt, M.ω * tt - φ)) * M.b)) for tt in tf])'
end

function part_B()
    out = NamedTuple[]
    for (lab, Ys) in (("3-step decays", YS), ("30-step kick-free", YLC), ("30-step kicked", YL)), Ks in (6,)
        M = fit_lattice(Ys; Ks); ρ, rot = percycle(M, M.σ, M.ν)
        s = score(W -> lattice_free(M, W, NH))
        show_score(@sprintf("B %s Ks=%d", lab, Ks), s); flush(stdout)
        @printf("    period %.3f samples; multiplier per cycle %.3f ∠ %.1f°; rank-1 share of decay amplitudes %.3f\n", M.Pc, ρ, rot, M.rank1)
        push!(out, (; data = lab, Ks, score = s, mult = (ρ, rot), Pc = M.Pc, rank1 = M.rank1))
    end
    serialize(cache_path("floquet/lifted_B.jls"), out)
end

# ═════════════════════════════════════════════════════════════════════════════
# E: neural phase–isostable model (linear dynamics in Koopman eigenfunction coordinates, neural decoder)
# ═════════════════════════════════════════════════════════════════════════════
# Latent phase θ = ωt − φ_r and isostable ψ_r(t) = ψ_r e^{μt} evolve exactly linearly; the decoder is B's
# first-order model plus an MLP correction that vanishes on the cycle:
#   x̂ = a(θ) + Re[ψ(t) b(θ)] + N(θ, ψ(t)) − N(θ, 0)
# so the cycle is isolated and the Floquet multiplier e^{μP} is a parameter. Initialised from B; trained by Adam
# on all samples; μ, a, b, φ_r, ψ_r and the MLP jointly, or with μ frozen at B's estimate. A forecast fits the
# window's φ and ψ.
const NN_K = 3
e_net(hidden) = Lux.Chain(Lux.Dense(2NN_K + 2, hidden, tanh), Lux.Dense(hidden, hidden, tanh), Lux.Dense(hidden, 3))

function e_decode(p, net, st, K, Ks, ω, Pc, t, φ, ψr, ψi; μ = (p.sc, p.nc))
    θ = ω .* t .- φ; σ = μ[1] / Pc; ν = μ[2] / Pc
    Cc = cos.(θ .* (1:K)'); Ss = sin.(θ .* (1:K)')
    A = p.a[1:1, :] .+ Cc * p.a[2:2:end, :] .+ Ss * p.a[3:2:end, :]
    Ck = cos.(θ .* (-Ks:Ks)'); Sk = sin.(θ .* (-Ks:Ks)')
    Bre = Ck * p.br .- Sk * p.bi; Bim = Sk * p.br .+ Ck * p.bi
    e = exp.(σ .* t); c = e .* (ψr .* cos.(ν .* t) .- ψi .* sin.(ν .* t)); s = e .* (ψr .* sin.(ν .* t) .+ ψi .* cos.(ν .* t))
    Fθ = vcat(cos.((1:NN_K) .* θ'), sin.((1:NN_K) .* θ'))
    N1 = first(net(vcat(Fθ, c', s'), p.nn, st)); N0 = first(net(vcat(Fθ, zero(c'), zero(s')), p.nn, st))
    A .+ c .* Bre .- s .* Bim .+ (N1 .- N0)'
end

function fit_neural(Ys, B; seed = 1, hidden = 32, iters = 3000, lr = 2e-3, freeze_mu = false)
    net = e_net(hidden); ps, st = Lux.setup(MersenneTwister(seed), net)
    ps = Lux.fmap(x -> x isa AbstractArray ? 0.1 .* x : x, ps)                       # start near B
    p0 = ComponentArray(; a = B.a, br = real.(B.b), bi = imag.(B.b), sc = B.σ * B.Pc, nc = B.ν * B.Pc,
                        ψr = real.(B.ψ), ψi = imag.(B.ψ), φ = copy(B.φs), nn = ComponentArray(ps))
    ts = [collect(0.0:size(Y, 1)-1) for Y in Ys]; ntot = sum(length, Ys)
    μf = (B.σ * B.Pc, B.ν * B.Pc)
    dec(p, r) = freeze_mu ? e_decode(p, net, st, B.K, B.Ks, B.ω, B.Pc, ts[r], p.φ[r], p.ψr[r], p.ψi[r]; μ = μf) :
                            e_decode(p, net, st, B.K, B.Ks, B.ω, B.Pc, ts[r], p.φ[r], p.ψr[r], p.ψi[r])
    loss(p, _) = sum(sum(abs2, dec(p, r) .- Ys[r]) for r in eachindex(Ys)) / ntot
    sol = solve(OptimizationProblem(OptimizationFunction(loss, Optimization.AutoZygote()), p0),
                OptimizationOptimisers.Adam(lr); maxiters = iters)
    pf = freeze_mu ? (q = copy(sol.u); q.sc = μf[1]; q.nc = μf[2]; q) : sol.u
    (; p = pf, net, st, K = B.K, Ks = B.Ks, ω = B.ω, Pc = B.Pc, loss0 = loss(p0, nothing), loss = loss(sol.u, nothing))
end

"Forecast: grid over the window's offset with ψ by least squares on the linear part, then refine (φ, ψ) by Adam."
function neural_free(E, W, n)
    nw = size(W, 1); t = collect(0.0:nw-1); p = E.p
    dec(φ, ψr, ψi, tt) = e_decode(p, E.net, E.st, E.K, E.Ks, E.ω, E.Pc, tt, φ, ψr, ψi)
    function lin(φ)
        A0 = dec(φ, 0.0, 0.0, t); Gr = dec(φ, 1e-3, 0.0, t) .- A0; Gi = dec(φ, 0.0, 1e-3, t) .- A0
        G = hcat(vec(Gr), vec(Gi)) ./ 1e-3; c = G \ vec(W .- A0); sum(abs2, vec(W .- A0) .- G * c), c
    end
    φ0 = align_phase(W, p.a, E.K, E.ω); φs = φ0 .+ range(-0.3, 0.3; length = 31)
    j = argmin([lin(φ)[1] for φ in φs]); c = lin(φs[j])[2]
    q0 = [φs[j], c[1], c[2]]
    f(q, _) = sum(abs2, dec(q[1], q[2], q[3], t) .- W)
    q = solve(OptimizationProblem(OptimizationFunction(f, Optimization.AutoForwardDiff()), q0),
              OptimizationOptimisers.Adam(1e-3); maxiters = 300).u
    dec(q[1], q[2], q[3], collect(Float64, nw:nw+n-1))
end

function part_E()
    out = NamedTuple[]
    for (lab, Ys) in (("3-step decays", YS),)
        B = fit_lattice(Ys; Ks = 6)
        for freeze in (false, true), seed in 1:3
            E = fit_neural(Ys, B; seed, iters = 1500, freeze_mu = freeze); ρ, rot = exp(E.p.sc), rad2deg(E.p.nc)
            s = score(W -> neural_free(E, W, NH))
            show_score(@sprintf("E %s%s seed %d", lab, freeze ? " μ frozen at B" : "", seed), s); flush(stdout)
            @printf("    loss %.5f → %.5f; multiplier per cycle %.3f ∠ %.1f° (B start %.3f ∠ %.1f°)\n", E.loss0, E.loss, ρ, rot, percycle(B, B.σ, B.ν)...)
            push!(out, (; data = lab, freeze, seed, score = s, mult = (ρ, rot), loss = E.loss))
        end
    end
    serialize(cache_path("floquet/lifted_E.jls"), out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    part = ARGS[1]
    @printf("exact multiplier %.3f ∠ %.1f° per step\n", maximum(abs, EXACT), rad2deg(maximum(angle.(EXACT))))
    part == "C" && part_C()
    part == "D" && part_D()
    part == "A" && part_A()
    part == "B" && part_B()
    part == "E" && part_E()
end
