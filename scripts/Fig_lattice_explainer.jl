#!/usr/bin/env julia
# How a multi-mode lattice model represents dynamics near a limit cycle, and what it can tell us — on a system whose
# Floquet modes are known, then on running.
#
# System (period 1, 100 samples per cycle): a limit-cycle oscillator (x, y) whose radial correction rate depends on
# phase, λ(θ) = λ₀(1 + κ cos(θ − θc)), strongest around θc (like corrections concentrated in one part of a stride),
# drives a damped oscillator (z, v):
#     ẋ = λ(θ)(1 − r²)x − ωy,  ẏ = λ(θ)(1 − r²)y + ωx,  ż = v,  v̇ = −Ω²z − 2ζΩv + βx,   θ = atan(y, x), θ̇ = ω.
# Isochrons are radial, so a radial kick leaves the phase unchanged and the deviation from the cycle at the same time
# is the deviation at the same phase (the stride-normalised setting of the gait data). Observed: x and z only.
# Floquet modes of the deviation (exact, linear theory): mode 1, μ₁ = −2λ₀ (ρ₁ = 0.15 per cycle, real): the radial
# deviation, shape ∝ G(θ)cos θ in x (G(θ) = exp(−2λ₀κ sin(θ − θc)/ω): it shrinks mostly around θc), plus the response it
# forces in z; mode 2, μ₂ = −ζΩ + iΩ_d (ρ₂ = 0.7, turning 108° per cycle from the nearest harmonic): free ringing of z.
# The lattice model δ(t) = Σⱼ Re[ψⱼ e^{μⱼt} bⱼ(θ)] (maus_floquet_models.jl fit_modes; |k| ≤ 12) is fitted to 40
# noisy kicked decays (3 cycles from the first cycle origin after the kick; kicks at random phases, radial and in z;
# noise SD 0.01), with μ's chosen greedily on a grid by held-out reconstruction error over 20 more decays, then refined.
#
#   (A) a kicked run: x and z against the cycle; (B) its deviation as the sum of the two modes (fitted: solid colour;
#   exact linear theory: dashed), and the forecast from the first half cycle (dotted); (C) anatomy of each fitted mode:
#   the multiplier as a phasor shrinking by ρ and turning by its rotation every cycle, and the shape over one cycle;
#   (D) the lattice spectrum: each mode's exponent repeats next to every harmonic; Hankel-DMD on the same data fits every
#   replica separately; (E) excitation: each mode's amplitude after a kick, against the phase of the kick, radial kicks
#   and kicks in z; (F) running (maus_floquet_models.jl): held-out forecast skill of the lattice as modes are added,
#   against the Fourier-in-phase operator.
#
# Usage: julia --project=. scripts/Fig_lattice_explainer.jl      (writes figures/Fig_lattice_explainer.*)

include("maus_floquet_models.jl")
include("paper_style.jl")

plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const ω = 2π
const RHO1, RHO2 = 0.15, 0.7
const λ0 = -log(RHO1) / 2; const κ = 0.9; const θc = π
const Ωz = 2π * 1.3; const ζΩ = -log(RHO2); const Ωd = sqrt(Ωz^2 - ζΩ^2); const βz = 20.0
const ROT2 = 360 * (Ωd / 2π - round(Ωd / 2π))
const SIG = 0.01
P(s) = s^2 + 2ζΩ * s + Ωz^2
lam(θ) = λ0 * (1 + κ * cos(θ - θc))
f(u) = (r2 = u[1]^2 + u[2]^2; l = lam(atan(u[2], u[1]));
        [l * (1 - r2) * u[1] - ω * u[2], l * (1 - r2) * u[2] + ω * u[1], u[4], -Ωz^2 * u[3] - 2ζΩ * u[4] + βz * u[1]])
cycle(t) = [cos(ω * t), sin(ω * t), real(βz * cis(ω * t) / P(im * ω)), real(im * ω * βz * cis(ω * t) / P(im * ω))]

"Deviation from the cycle (samples k/NPH, k ≥ ⌈t0·NPH⌉, up to 4 cycles or n samples) after a kick at t0: radial a, z b, v c."
function kicked(t0, a, b, c; sub = 10, n = nothing)
    u = cycle(t0) .+ [a * cos(ω * t0), a * sin(ω * t0), b, c]; t = t0; h = 1 / (NPH * sub)
    k0 = ceil(Int, t0 * NPH); ks = n === nothing ? (k0:4NPH-1) : (k0:k0+n-1); out = zeros(length(ks), 4)
    for (i, k) in enumerate(ks)
        while t < k / NPH - 1e-12
            dt = min(h, k / NPH - t)
            k1 = f(u); k2 = f(u .+ dt / 2 .* k1); k3 = f(u .+ dt / 2 .* k2); k4 = f(u .+ dt .* k3)
            u = u .+ dt / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4); t += dt
        end
        out[i, :] = u .- cycle(k / NPH)
    end
    collect(ks) ./ NPH, out
end
"Observed segment: x and z deviations over cycles 1–3 (300 samples from the cycle origin at t = 1), plus noise."
segment(t, δ; rng) = (i = findfirst(≈(1.0), t); δ[i:i+T_SEG-1, [1, 3]] .+ SIG .* randn(rng, T_SEG, 2))

# exact linear decomposition of a kick into the two Floquet modes
Gθ(θ) = exp(-2λ0 * κ * sin(θ - θc) / ω)
const HK = (n = 512; θs = 2π .* (0:n-1) ./ n; [mean(Gθ.(θs) .* cos.(θs) .* cis.(-k .* θs)) for k in -15:15])
function exact_modes(t0, a, b, c, t)
    μ1 = -2λ0; C = a * exp(2λ0 * t0) / Gθ(ω * t0)
    x1 = C .* exp.(μ1 .* t) .* Gθ.(ω .* t) .* cos.(ω .* t)
    zf(τ) = C * sum(HK[k+16] * βz * exp((μ1 + im * k * ω) * τ) / P(μ1 + im * k * ω) for k in -15:15)
    vf(τ) = C * sum(HK[k+16] * βz * (μ1 + im * k * ω) * exp((μ1 + im * k * ω) * τ) / P(μ1 + im * k * ω) for k in -15:15)
    z1 = real.(zf.(t)); q = -ζΩ + im * Ωd
    z20, v20 = b - real(zf(t0)), c - real(vf(t0)); di = (real(q) * z20 - v20) / imag(q); D = z20 + im * di
    z2 = real.(D .* exp.(q .* (t .- t0)))
    (; x1, z1, z2)
end

# ── data ─────────────────────────────────────────────────────────────────────
rng = MersenneTwister(7)
randkick(rng) = (t0 = rand(rng) * 0.95; w = rand(rng); (t0, 0.25 * w * (2rand(rng) - 1), 0.3 * (1 - w) * (2rand(rng) - 1), 0.0))
mk(kk, rng) = begin t, δ = kicked(kk...); segment(t, δ; rng) end
TR = [randkick(rng) for _ in 1:40]; HO = [randkick(rng) for _ in 1:20]
segs_tr = [mk(k, rng) for k in TR]; segs_ho = [mk(k, rng) for k in HO]

"Held-out reconstruction error: shapes fitted on the training decays, ψ's fitted to each whole held-out segment."
function holdout(μs)
    bs = fit_modes(segs_tr, μs); Gs = [basis(μ, KS) for μ in μs]
    sum(segs_ho) do S
        Dq = reduce(hcat, [(q = G * b; hcat(vec(real(q)), vec(-imag(q)))) for (G, b) in zip(Gs, bs)])
        sum(abs2, vec(S) .- Dq * (Dq \ vec(S)))
    end
end
grid = [(ρ, rot) for ρ in 0.05:0.05:0.95 for rot in 0:10:180]
μs = ComplexF64[]; prs = Tuple{Float64, Float64}[]; errs = [sum(sum(abs2, S) for S in segs_ho)]     # K = 0: no model
for K in 1:3
    e = [holdout(vcat(μs, μof(g...))) for g in grid]; i = argmin(e)
    push!(errs, e[i]); push!(prs, grid[i]); push!(μs, μof(grid[i]...))
end
@printf("held-out error by number of modes: %s\n", join([@sprintf("%.4f", x) for x in errs], " → "))
prs = prs[1:2]
"Coordinate refinement of the two modes' (ρ, rotation) on the held-out error."
function refine(prs)
    for _ in 1:3, j in 1:2, (dρ, dr) in ((0.02, 0), (-0.02, 0), (0, 4), (0, -4), (0.005, 0), (-0.005, 0), (0, 1), (0, -1))
        cand = copy(prs); cand[j] = (clamp(cand[j][1] + dρ, 0.01, 0.99), clamp(cand[j][2] + dr, 0, 180))
        holdout([μof(p...) for p in cand]) < holdout([μof(p...) for p in prs]) && (prs = cand)
    end
    sort(prs, by = first)
end
prs = refine(prs); μfit = [μof(p...) for p in prs]
@printf("fitted multipliers: %s   (truth %.2f∠0°, %.2f∠%.0f°)\n", join([@sprintf("%.2f∠%.0f°", p...) for p in prs], ", "), RHO1, RHO2, ROT2)
bfit = fit_modes(vcat(segs_tr, segs_ho), μfit); Gfit = [basis(μ, KS) for μ in μfit]

# Hankel-DMD on the same training segments (observed x, z deviations), eigenvalues per cycle
function dmd_eigs(segs; d = 20, r = 14)
    X = reduce(hcat, [hankel(S, d)[:, 1:end-1] for S in segs]); Y = reduce(hcat, [hankel(S, d)[:, 2:end] for S in segs])
    F = svd(X); U, s, V = F.U[:, 1:r], F.S[1:r], F.V[:, 1:r]
    eigvals(U' * Y * V * Diagonal(1 ./ s))
end
λd = dmd_eigs(segs_tr)
@printf("DMD eigenvalues (freq, |λ| per cycle): %s\n", join([@sprintf("(%.2f, %.2f)", abs(angle(l)) * NPH / 2π, abs(l)^NPH) for l in λd if abs(l)^NPH > 1e-3], " "))

# ── figure ───────────────────────────────────────────────────────────────────
fig = Figure(size = (COL2, 150MM))
top = fig[1, 1] = GridLayout(); bot = fig[2, 1] = GridLayout()
CM = (CAT[1], CAT[2]); CX = C_TRUTH
ex = (0.8, 0.45, 0.25, 0.0); tx, δx = kicked(ex...); M = exact_modes(ex..., tx)

# A: a kicked run
gA = top[1, 1] = GridLayout()
for (k, (c, nm)) in enumerate(((1, "x"), (3, "z")))
    ax = Axis(gA[k, 1]; ylabel = nm, xticklabelsvisible = k == 2, xlabel = k == 2 ? "Time (cycles)" : "",
              title = k == 1 ? "A kicked limit cycle" : "", limits = ((0, 4), nothing), xticks = 0:4)
    tc = range(0, 4; length = 400); lines!(ax, tc, [cycle(t)[c] for t in tc]; color = C_REF, linestyle = :dash, linewidth = 0.9)
    lines!(ax, tx, δx[:, c] .+ [cycle(t)[c] for t in tx]; color = CX, linewidth = 1.0)
    vlines!(ax, ex[1]; color = CAT[4], linewidth = 1.0)
    k == 1 && text!(ax, ex[1] + 0.05, 1.3; text = "kick", fontsize = 6.5, color = C_TRUTH, align = (:left, :top))
    k == 1 && ylims!(ax, -1.45, 1.45)
end
text!(content(gA[2, 1]), 3.95, -1.0; text = "dashed: the cycle", fontsize = 6.5, color = C_REF, align = (:right, :bottom))
rowgap!(gA, 3); plabel!(gA[1, 1], "A")

# B: deviation = sum of modes
S_ex = segment(tx, δx; rng = MersenneTwister(1)); ts = 1 .+ (0:T_SEG-1) ./ NPH
Dq = reduce(hcat, [(q = G * b; hcat(vec(real(q)), vec(-imag(q)))) for (G, b) in zip(Gfit, bfit)]); co = Dq \ vec(S_ex)
contrib(j) = real.((co[2j-1] + im * co[2j]) .* (Gfit[j] * bfit[j]))
W = 50; cw = Dq[[i for c in 0:1 for i in c*T_SEG+1:c*T_SEG+W], :] \ vec(S_ex[1:W, :]); fc = reshape(Dq * cw, T_SEG, 2)
i1 = findfirst(≈(1.0), tx)
gB = top[1, 2] = GridLayout()
for (k, (c, nm, ttl)) in enumerate(((1, "δx", "The deviation is a sum of modes"), (2, "δz", "")))
    ax = Axis(gB[k, 1]; ylabel = nm, xticklabelsvisible = k == 2, xlabel = k == 2 ? "Time (cycles)" : "", title = ttl,
              limits = ((1, 4), nothing), xticks = 1:4)
    for cyc in 1:3; vspan!(ax, cyc + (θc / 2π) - 0.25, cyc + (θc / 2π) + 0.25; color = (C_SHADE, 0.9)); end
    hlines!(ax, 0; color = C_REF, linewidth = 0.4)
    lines!(ax, ts, S_ex[:, c]; color = CX, linewidth = 0.8)
    tru = c == 1 ? (M.x1[i1:i1+T_SEG-1], zeros(T_SEG)) : (M.z1[i1:i1+T_SEG-1], M.z2[i1:i1+T_SEG-1])
    for j in 1:2
        lines!(ax, ts, contrib(j)[:, c]; color = CM[j], linewidth = 1.2)
        lines!(ax, ts, tru[j]; color = CM[j], linewidth = 0.8, linestyle = :dash)
    end
    lines!(ax, ts[W+1:end], fc[W+1:end, c]; color = C_TRUTH, linewidth = 0.9, linestyle = :dot)
    vlines!(ax, 1 + W / NPH; color = C_REF, linewidth = 0.5)
end
ab = content(gB[1, 1])
text!(ab, 3.98, maximum(S_ex[:, 1]); text = "black: data   colour: modes 1, 2 (fitted; dashed exact)\ndotted: forecast from the first half cycle\ngrey: half cycle of strongest correction",
      fontsize = 6, align = (:right, :top))
rowgap!(gB, 3); plabel!(gB[1, 1], "B")

# C: anatomy of each mode
gC = top[1, 3] = GridLayout()
for j in 1:2
    ρ, rot = prs[j]
    ap = Axis(gC[1, j]; aspect = DataAspect(), title = @sprintf("mode %d\n×%.2f ∠%d°", j, ρ, round(Int, rot)),
              limits = (-1.15, 1.15, -1.15, 1.15), xticksvisible = false, yticksvisible = false, xticklabelsvisible = false,
              yticklabelsvisible = false, leftspinevisible = false, bottomspinevisible = false, titlesize = 6.5)
    lines!(ap, cos.(range(0, 2π; length = 100)), sin.(range(0, 2π; length = 100)); color = C_SHADE, linewidth = 0.6)
    z = [ρ^k * cis(deg2rad(rot) * k) for k in 0:6]; zz = [ρ^(s) * cis(deg2rad(rot) * s) for s in range(0, 6; length = 300)]
    lines!(ap, real.(zz), imag.(zz); color = (CM[j], 0.6), linewidth = 0.8)
    for (k, w) in enumerate(z[1:4])
        arrows2d!(ap, [0.0], [0.0], [real(w)], [imag(w)]; color = (CM[j], 1 - 0.2(k - 1)), shaftwidth = 0.8, tipwidth = 4, tiplength = 4)
    end
    text!(ap, 1.1, -1.12; text = "per cycle, 0 → 3", fontsize = 5.5, color = C_REF, align = (:right, :bottom))
    as = Axis(gC[2, j]; xlabel = "Phase (cycle)", ylabel = j == 1 ? "shape b(θ)" : "", xticks = 0:0.5:1, limits = ((0, 1), (-1.15, 1.15)),
              yticklabelsvisible = j == 1)
    vspan!(as, θc / 2π - 0.25, θc / 2π + 0.25; color = (C_SHADE, 0.9))
    sh = real.((co[2j-1] + im * co[2j]) .* (Gfit[j] * bfit[j]))[1:NPH, :] ./ exp.(real(μfit[j]) .* (0:NPH-1))
    sh ./= maximum(abs, sh)
    lines!(as, (0:NPH-1) ./ NPH, sh[:, 1]; color = CM[j], linewidth = 1.2)
    lines!(as, (0:NPH-1) ./ NPH, sh[:, 2]; color = CM[j], linewidth = 1.0, linestyle = :dash)
    j == 1 && text!(as, 0.02, -1.1; text = "solid x, dashed z", fontsize = 5.5, align = (:left, :bottom))
end
rowsize!(gC, 1, Relative(0.5)); rowgap!(gC, 4); colgap!(gC, 6); plabel!(gC[1, 1], "C")
colsize!(top, 1, Relative(0.27)); colsize!(top, 3, Relative(0.3)); colgap!(top, 14)

# D: the lattice spectrum
aD = Axis(bot[1, 1]; xlabel = "Frequency (cycles per cycle)", ylabel = "|λ| per cycle", title = "One exponent per mode, at every harmonic",
          limits = (-0.1, 3.6, 0, 1.08), xticks = 0:1:3)
for k in 0:3
    scatter!(aD, [k], [1.0]; marker = :circle, color = :white, strokecolor = C_REF, strokewidth = 0.8, markersize = 6)
end
text!(aD, 3.55, 0.93; text = "harmonics (the cycle)", fontsize = 6, color = C_REF, align = (:right, :top))
for j in 1:2
    ρ, rot = prs[j]; xs = sort(unique(round.([abs(k + s * rot / 360) for k in -4:4 for s in (-1, 1)], digits = 4)))
    xs = filter(x -> x <= 3.55, xs)
    lines!(aD, [-0.1, 3.6], [ρ, ρ]; color = (CM[j], 0.25), linewidth = 3)
    scatter!(aD, xs, fill(ρ, length(xs)); marker = :diamond, color = CM[j], markersize = 6)
end
fr = abs.(angle.(λd)) .* NPH ./ 2π; md = abs.(λd) .^ NPH; keep = md .> 0.02
scatter!(aD, fr[keep], md[keep]; color = C_TRUTH, marker = :xcross, markersize = 6)
scatter!(aD, [abs(Ωd / 2π)], [RHO2]; marker = :star5, color = :transparent, strokecolor = C_TRUTH, strokewidth = 0.9, markersize = 11)
scatter!(aD, [1.0], [RHO1]; marker = :star5, color = :transparent, strokecolor = C_TRUTH, strokewidth = 0.9, markersize = 11)
text!(aD, 0.0, 0.47; text = "◆ lattice fit: 2 exponents, all positions\n× Hankel-DMD, eigenvalues free: finds\n   only the slow mode in these noisy decays\n☆ exact", fontsize = 6, align = (:left, :center))
plabel!(bot[1, 1], "D")

# E: what excites which mode. ψⱼ fitted to the 3 cycles from the kick with the fitted exponents and shapes at absolute
# phase; each mode's size = RMS of its contribution (x and z) over the cycle that follows the kick.
phs = 0.025:0.05:0.975
const TABS = [(B = [sum(b[k+KS+1, c] * cis(k * 2π * (s - 1) / NPH) for k in -KS:KS) for s in 1:NPH, c in 1:2]; B ./ sqrt(mean(abs2, B))) for b in bfit]
function mode_amp(kk)
    t, δ = kicked(kk...; n = T_SEG); S = δ[:, [1, 3]] .+ SIG .* randn(MersenneTwister(2), T_SEG, 2); k0 = ceil(Int, kk[1] * NPH)
    Q = [[exp(μ * (i - 1)) * tb[mod(k0 + i - 1, NPH)+1, c] for i in 1:T_SEG, c in 1:2] for (μ, tb) in zip(μfit, TABS)]
    c = reduce(hcat, [hcat(vec(real(q)), vec(-imag(q))) for q in Q]) \ vec(S)
    [sqrt(mean(abs2, real.((c[2j-1] + im * c[2j]) .* Q[j][1:NPH, :]))) for j in 1:2]     # RMS over the cycle after the kick
end
aE = Axis(bot[1, 2]; xlabel = "Phase of the kick (cycle)", ylabel = "Mode's deviation in the cycle\nafter the kick (RMS)",
          title = "Which disturbance excites which mode", limits = ((0, 1), (0, nothing)), xticks = 0:0.25:1)
vspan!(aE, θc / 2π - 0.25, θc / 2π + 0.25; color = (C_SHADE, 0.9))
for (kind, kk, ls) in (("radial kick", t0 -> (t0, 0.2, 0.0, 0.0), :solid), ("kick in z", t0 -> (t0, 0.0, 0.2, 0.0), :dash))
    A = reduce(hcat, [mode_amp(kk(t0)) for t0 in phs])
    for j in 1:2; lines!(aE, phs, A[j, :]; color = CM[j], linestyle = ls, linewidth = 1.2); end
end
axislegend(aE, [LineElement(color = CM[1]), LineElement(color = CM[2]), LineElement(color = C_REF), LineElement(color = C_REF, linestyle = :dash)],
           ["mode 1 (fast)", "mode 2 (slow, ringing)", "radial kick", "kick in z"]; position = :lt, labelsize = 6)
plabel!(bot[1, 2], "E")

# F: running
FM = deserialize(joinpath(ROOT, "cache", "maus", "floquet_models.jls")).out[:ma]
aF = Axis(bot[1, 3]; xlabel = "Lattice modes K", ylabel = "Held-out skill (1 − error / deviation)", title = "Running: no small set of modes",
          xticks = 1:6, limits = ((0.7, 6.3), (0, 0.6)))
for r in FM; lines!(aF, 1:length(r.lattice.path), 1 .- r.lattice.path; color = (CM[1], 0.25), linewidth = 0.6); end
Km = maximum(length(r.lattice.path) for r in FM)
med = [median(1 - r.lattice.path[k] for r in FM if length(r.lattice.path) >= k) for k in 1:Km]
lines!(aF, 1:Km, med; color = CM[1], linewidth = 1.6); scatter!(aF, 1:Km, med; color = CM[1], markersize = 4)
lt = median(1 - r.ltp.cfg.e for r in FM)
hlines!(aF, lt; color = C_TRUTH, linewidth = 1.0, linestyle = :dash)
text!(aF, 6.25, lt + 0.01; text = "Fourier-in-phase operator\n(64-dim state)", fontsize = 6, align = (:right, :bottom))
mm = median(abs(r.ltp.mult[1]) for r in FM)
text!(aF, 0.8, 0.58; text = @sprintf("leading multiplier per stride:\n≈ %.2f (operator), 0.25–0.40 (1-mode lattice)", mm),
      fontsize = 6, align = (:left, :top))
plabel!(bot[1, 3], "F")
colsize!(bot, 1, Relative(0.34)); colgap!(bot, 16); rowgap!(fig.layout, 12); rowsize!(fig.layout, 1, Relative(0.52))
save_paper(fig, "Fig_lattice_explainer")
