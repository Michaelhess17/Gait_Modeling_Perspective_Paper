#!/usr/bin/env julia
# Two ways to give a linear model of running deviations the limit cycle's phase structure, scored exactly as
# maus_horizon.jl (same trials, deviations from the 61-stride moving-average stride, fit on the first half of the
# strides, scored on the second from every 5th sample, R² by horizon 6–100% of a stride; hyperparameters chosen on an
# inner split of the training half: fit on its first half, scored on its second, error summed over the horizons).
#
#   lattice K   multi-mode lattice: δ(t) = Σⱼ Re[ψⱼ e^{μⱼ t} bⱼ(θ_t)], j = 1…K. Each mode has its own exponent and its
#               own Fourier shape bⱼ(θ) (|k| ≤ 12) shared by all segments; ψⱼ per segment (3-stride segments from
#               stride origins). Fit by alternating least squares: ψ given the shapes, then all shapes jointly given ψ,
#               whose normal equations reduce to moment sums over segments (K_s = a J₁ + c J₂, see fit_modes).
#               μ's chosen greedily: add the grid μ (ρ 0.001–0.95 per stride × rotation 0:30:180°) that most lowers
#               the validation error, refitting all shapes; stop at < 0.5% improvement or K = KMAX. Forecast: ψ's
#               fitted to the 50 samples ending at the initial sample, modes run forward.
#   LTP         Fourier-series-in-phase operator on a POD-reduced delay state: z_{t+1} = A(θ_t) z_t,
#               A(θ) = A₀ + Σ_{h≤Hh} (A_h cos hθ + B_h sin hθ), ridge least squares on the lifted regressors
#               [z; z cos θ; z sin θ; …]. Grid: delay d ∈ LTP_D, rank r ∈ LTP_R, harmonics Hh ∈ LTP_H, ridge ∈ LTP_L.
#               Hh = 0 is a time-invariant operator in the same machinery (reported as "LTP, no phase").
#               Floquet multipliers = eigenvalues of the one-stride monodromy Π A(θ_k).
#   lattice(LTP μ)  the multi-mode lattice with its μ's taken from the chosen LTP's monodromy (largest |λ| first, one
#               per conjugate pair), K chosen on the inner split: do the two descriptions of the structure agree?
#
# Usage: julia --project=. -t 16 scripts/maus_floquet_models.jl      (writes cache/maus/floquet_models.jls)
#        julia --project=. -t 16 scripts/maus_floquet_models.jl ltp  (LTP only, wider grid LTPX_*, joint angles and CoM;
#                                                                     writes cache/maus/ltp_wide.jls)

include("maus_horizon.jl")

const KMAX = 6
const PSI_RIDGE = 1e-2
const LGRID = [μof(ρ, rot) for ρ in (0.001, 0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95) for rot in 0:30:180]
const LTP_D = [16, 32]
const LTP_R = [16, 32, 64]
const LTP_H = [0, 1, 2, 4, 8]
const LTP_L = [1e-4, 1e-2]
const LTPX_D, LTPX_R, LTPX_H = [8, 16, 32], [32, 64, 128], [0, 4, 8, 12, 16]   # the first grid chose its edge (16, 64, 8)

# ── multi-mode lattice ───────────────────────────────────────────────────────
"Rank-1 start for one mode on (residual) segments, as fit_mode_fast."
function init_mode(segs, G, nb, p)
    Rp = pinv(hcat(real(G), -imag(G)))
    Z = reduce(hcat, [(C = Rp * S; vec(C[1:nb, :] .- im .* C[nb+1:end, :])) for S in segs])
    F = svd(Z); reshape(F.U[:, 1] .* F.S[1], nb, p)
end

"""Multi-mode lattice on segments (T × p each) for exponents μs. Segment s: S_s ≈ Σⱼ Hⱼ K_sⱼ xⱼ, Hⱼ = [Re Gⱼ Im Gⱼ],
xⱼ = [Re bⱼ; Im bⱼ], K_sⱼ = aⱼJ₁ + cⱼJ₂ (ψ_sⱼ = aⱼ + icⱼ, J₁ = [I 0; 0 −I], J₂ = [0 −I; −I 0]). Shape step:
N_jl = Σ_s K_sⱼ Hⱼ'H_l K_sl expands into four moment sums Σ_s αⱼβ_l times Jα Qⱼₗ Jβ; tiny ridge (real μ makes the
design rank-deficient). `init`: shapes of the first modes (greedy continuation)."""
function fit_modes(segs, μs; Ks = KS, iters = 20, init = nothing, λ = 1e-9)
    K = length(μs); T, p = size(segs[1]); ns = length(segs)
    Gs = [basis(μ, Ks; T) for μ in μs]; nb = size(Gs[1], 2); m = 2nb
    Hs = [hcat(real(G), imag(G)) for G in Gs]
    J1 = Matrix(Diagonal(vcat(ones(nb), -ones(nb)))); J2 = [zeros(nb, nb) -I(nb); -I(nb) zeros(nb, nb)]
    QJ = [(Q = Hs[j]' * Hs[l]; (J1 * Q * J1, J1 * Q * J2, J2 * Q * J1, J2 * Q * J2)) for j in 1:K, l in 1:K]
    psi(bs) = (k = length(bs); Dq = reduce(hcat, [(q = Gs[j] * bs[j]; hcat(vec(real(q)), vec(-imag(q)))) for j in 1:k]);
               Pq = pinv(Dq); [Pq * vec(S) for S in segs])
    bs = init === nothing ? Matrix{ComplexF64}[] : collect(init)
    while length(bs) < K
        j = length(bs) + 1
        res = isempty(bs) ? segs : (ψs = psi(bs); [S .- sum(real.((ψ[2i-1] + im * ψ[2i]) .* (Gs[i] * bs[i])) for i in eachindex(bs)) for (S, ψ) in zip(segs, ψs)])
        push!(bs, init_mode(res, Gs[j], nb, p))
    end
    for _ in 1:iters
        ψs = psi(bs); a = [ψ[2j-1] for ψ in ψs, j in 1:K]; c = [ψ[2j] for ψ in ψs, j in 1:K]
        N = zeros(m * K, m * K); r = zeros(m * K, p)
        for j in 1:K, l in 1:K
            q = QJ[j, l]
            N[(j-1)*m+1:j*m, (l-1)*m+1:l*m] = dot(a[:, j], a[:, l]) .* q[1] .+ dot(a[:, j], c[:, l]) .* q[2] .+
                                                dot(c[:, j], a[:, l]) .* q[3] .+ dot(c[:, j], c[:, l]) .* q[4]
        end
        for j in 1:K
            Sa = sum(a[s, j] .* segs[s] for s in 1:ns); Sc = sum(c[s, j] .* segs[s] for s in 1:ns)
            r[(j-1)*m+1:j*m, :] = J1 * (Hs[j]' * Sa) .+ J2 * (Hs[j]' * Sc)
        end
        x = (N + λ * tr(N) / size(N, 1) * I) \ r
        bs = [x[(j-1)*m+1:(j-1)*m+nb, :] .+ im .* x[(j-1)*m+nb+1:j*m, :] for j in 1:K]
    end
    bs
end

"""Forecast function for a multi-mode lattice: ψ's fitted to the W samples ending at t (ridge λψ relative to the mean
diagonal: K free amplitudes on 50 samples otherwise let near-collinear modes cancel in the window and diverge after
it), modes run forward."""
function modes_predictor(D, bs, μs; W = 50, Ks = KS, λψ = PSI_RIDGE)
    tabs = [[sum(b[k+Ks+1, c] * exp(im * k * 2π * (t - 1) / NPH) for k in -Ks:Ks) for t in 1:NPH, c in axes(b, 2)] for b in bs]
    hmax = HZ[end]; p = size(D, 2)
    function (t)
        Qs = [[exp(μ * (s - t)) * tb[mod1(s, NPH), c] for s in t-W+1:t+hmax, c in 1:p] for (μ, tb) in zip(μs, tabs)]
        Dq = reduce(hcat, [hcat(vec(real(Q[1:W, :])), vec(-imag(Q[1:W, :]))) for Q in Qs])
        G = Dq' * Dq; co = (G + λψ * mean(diag(G)) * I) \ (Dq' * vec(D[t-W+1:t, :]))
        sum(real.((co[2j-1] + im * co[2j]) .* Qs[j][W+1:end, :]) for j in eachindex(Qs))
    end
end
verr(D, bs, μs, ts; kw...) = (E = score(modes_predictor(D, bs, μs; kw...), D, ts); sum(E[1]) / sum(E[2]))

function lattice_greedy(D, nh, va, te)
    trs = segments(D, 0, nh ÷ 2); vts = starts(first(va), last(va))[1:2:end]      # every 10th sample while searching
    μs = ComplexF64[]; bprev = nothing; best = Inf; path = Float64[]
    for K in 1:KMAX
        cands = [(μ, (b = fit_modes(trs, vcat(μs, μ); init = bprev); (b, verr(D, b, vcat(μs, μ), vts)))) for μ in LGRID]
        i = argmin([c[2][2] for c in cands]); e = cands[i][2][2]; push!(path, e)
        e > best * (1 - 0.005) && break
        best = e; push!(μs, cands[i][1]); bprev = cands[i][2][1]
    end
    b = fit_modes(segments(D, 0, nh), μs)
    (; μs, path, sc = score(modes_predictor(D, b, μs), D, starts(first(te), last(te))))
end

# ── Fourier-series-in-phase operator ─────────────────────────────────────────
fphase(k, Hh) = (θ = 2π * k / NPH; vcat(1.0, [f(h * θ) for h in 1:Hh for f in (cos, sin)]))

"Fit z_{t+1} = A(θ_t) z_t on rows; returns per-phase operators, the POD basis and the mean."
function fit_ltp(D, rows, d, r, Hh, λs)
    Y = D[rows, :]; μ = mean(Y, dims = 1); H = hankel(Y .- μ, d); F = svd(H); r = min(r, size(F.U, 2))
    U = F.U[:, 1:r]; Z = U' * H; nc = size(Z, 2)
    ph = [mod(rows[1] + c + d - 3, NPH) for c in 1:nc]                 # 0-based phase of column c's last sample
    Φ = reduce(hcat, [kron(fphase(ph[c], Hh), Z[:, c]) for c in 1:nc-1]); G = Φ * Φ'; Bm = Z[:, 2:end] * Φ'
    md = mean(diag(G)); nf = 2Hh + 1
    map(λs) do λ
        W = Bm / (G + λ * md * I)
        Aph = [sum(fphase(k, Hh)[i] .* W[:, (i-1)*r+1:i*r] for i in 1:nf) for k in 0:NPH-1]
        (; Aph, U, μ, d)
    end
end
function ltp_predictor(D, M)
    p = size(D, 2); hmax = HZ[end]
    function (t)
        z = M.U' * vec(permutedims(D[t-M.d+1:t, :] .- M.μ)); out = zeros(hmax, p)
        for s in 1:hmax; z = M.Aph[mod(t + s - 2, NPH)+1] * z; out[s, :] = (M.U * z)[end-p+1:end] .+ vec(M.μ); end
        out
    end
end
monodromy(M) = foldl((P, A) -> A * P, M.Aph; init = Matrix(1.0I, size(M.Aph[1])...))

function ltp_select(D, nh, va, te; Ds = LTP_D, Rs = LTP_R, Hs = LTP_H, Ls = LTP_L)
    rows_in = 1:(nh÷2)*NPH; vts = starts(first(va), last(va)); best = Dict{Bool, Any}()
    for d in Ds, r in Rs, Hh in Hs
        for (λ, M) in zip(Ls, fit_ltp(D, rows_in, d, r, Hh, Ls))
            E = score(ltp_predictor(D, M), D, vts); e = sum(E[1]) / sum(E[2]); k = Hh > 0
            (!haskey(best, k) || e < best[k].e) && (best[k] = (; e, d, r, Hh, λ))
        end
    end
    map((true, false)) do k
        c = best[k]; M = fit_ltp(D, 1:nh*NPH, c.d, c.r, c.Hh, [c.λ])[1]
        λm = eigvals(monodromy(M)); λm = λm[sortperm(abs.(λm), rev = true)]
        (; cfg = c, sc = score(ltp_predictor(D, M), D, starts(first(te), last(te))), mult = λm, M)
    end
end

"Lattice with the LTP's monodromy multipliers as exponents (one per conjugate pair, largest first), K on the inner split."
function lattice_from_ltp(D, nh, va, te, mult)
    λs = ComplexF64[]; for λ in mult; (imag(λ) < -1e-9 || any(x -> abs(x - λ) < 1e-9, λs)) && continue; push!(λs, λ); length(λs) == KMAX && break; end
    μall = [log(complex(λ)) / NPH for λ in λs]
    trs = segments(D, 0, nh ÷ 2); vts = starts(first(va), last(va))
    errs = [verr(D, fit_modes(trs, μall[1:K]), μall[1:K], vts) for K in eachindex(μall)]; K = argmin(errs)
    b = fit_modes(segments(D, 0, nh), μall[1:K])
    (; K, μs = μall[1:K], sc = score(modes_predictor(D, b, μall[1:K]), D, starts(first(te), last(te))))
end

function trial_floquet(S, how)
    n = size(S, 1); nh = n ÷ 2; _, _, D = split_baseline(S, how)
    va = (nh ÷ 2)*NPH+1:nh*NPH; te = nh*NPH+1:n*NPH
    ltp, ltp0 = ltp_select(D, nh, va, te)
    (; lattice = lattice_greedy(D, nh, va, te), ltp = (; ltp.cfg, ltp.sc, ltp.mult), ltp0 = (; ltp0.cfg, ltp0.sc, ltp0.mult),
       lattice_ltp = lattice_from_ltp(D, nh, va, te, ltp.mult))
end

if abspath(PROGRAM_FILE) == (@__FILE__) && get(ARGS, 1, "") == "ltp"
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    T = A.trials[qc_run.([strides(t.X) for t in A.trials]) .< 5]
    out = Dict{Symbol, Any}()
    for st in (:angles, :com)
        Ss = [state_strides(t, st) for t in T]
        R = tmap(collect(eachindex(Ss))) do i
            n = size(Ss[i], 1); nh = n ÷ 2; _, _, D = split_baseline(Ss[i], :ma)
            ltp, ltp0 = ltp_select(D, nh, (nh ÷ 2)*NPH+1:nh*NPH, nh*NPH+1:n*NPH; Ds = LTPX_D, Rs = LTPX_R, Hs = LTPX_H)
            (; ltp = (; ltp.cfg, ltp.sc, ltp.mult), ltp0 = (; ltp0.cfg, ltp0.sc, ltp0.mult))
        end
        @printf("\n== %s, baseline ma, wide LTP grid: R² by horizon, median over %d trials (%d failed) ==\n", st, count(!isnothing, R), count(isnothing, R))
        R = filter(!isnothing, R); out[st] = R
        @printf("%-28s", "horizon"); for h in HZ; @printf("%7.1f", h); end; println()
        for (lab, k) in (("LTP (Fourier in phase)", :ltp), ("LTP, no phase (Hh = 0)", :ltp0))
            @printf("%-28s", lab); for j in eachindex(HZ); @printf("%7.2f", median(r2(getfield(r, k).sc...)[j] for r in R)); end; println()
        end
        @printf("  chosen (d, r, Hh, ridge): %s\n", join([@sprintf("(%d,%d,%d,%g)", r.ltp.cfg.d, r.ltp.cfg.r, r.ltp.cfg.Hh, r.ltp.cfg.λ) for r in R], " "))
        @printf("  monodromy |λ| top 3: %s\n", join([join([@sprintf("%.2f", abs(x)) for x in r.ltp.mult[1:3]], "/") for r in R], "  "))
        if st == :com
            println("  per channel, horizons 12.5 / 25 / 50 / 100%:")
            for (k, c) in enumerate(["CoM lateral", "CoM height", "v lateral", "v forward", "v vertical"])
                @printf("    %-12s %s\n", c, join([@sprintf("%.2f", median(r2(r.ltp.sc...; chans = [k])[j] for r in R)) for j in (2, 4, 6, 8)], " / "))
            end
        end
        flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "maus", "ltp_wide.jls"), (; out, HZ))
    println("wrote cache/maus/ltp_wide.jls")
elseif abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    Ss = [strides(t.X) for t in A.trials]; keep = qc_run.(Ss) .< 5; T = A.trials[keep]; Ss = Ss[keep]; subs = [t.subj for t in T]
    HF = joinpath(ROOT, "cache", "maus", "horizon.jls"); Hz = isfile(HF) ? deserialize(HF) : nothing
    out = Dict{Symbol, Any}()
    for how in (:ma,)
        R = tmap(i -> trial_floquet(Ss[i], how), collect(eachindex(Ss)))
        bad = count(isnothing, R); R = filter(!isnothing, R); out[how] = R
        @printf("\n== joint angles, baseline %s: R² by horizon (%% stride), median over %d trials (%d failed) ==\n", how, length(R), bad)
        @printf("%-28s", "horizon"); for h in HZ; @printf("%7.1f", h); end; println()
        show(lab, v) = (@printf("%-28s", lab); for k in eachindex(HZ); @printf("%7.2f", median(x[k] for x in v)); end; println())
        if Hz !== nothing && haskey(Hz.out, (:angles, how))
            H0 = Hz.out[(:angles, how)]
            show("ref: DMD d=50", [r2(r.dmd[50].sc...) for r in H0]); show("ref: LPV delay (binned)", [r2(r.lpv_delay...) for r in H0])
            show("ref: lattice, 1 mode", [r2(r.lattice.sc...) for r in H0])
        end
        show("lattice, K modes (greedy)", [r2(r.lattice.sc...) for r in R])
        show("LTP (Fourier in phase)", [r2(r.ltp.sc...) for r in R])
        show("LTP, no phase (Hh = 0)", [r2(r.ltp0.sc...) for r in R])
        show("lattice, LTP multipliers", [r2(r.lattice_ltp.sc...) for r in R])
        @printf("  greedy lattice: K chosen %s\n", join([length(r.lattice.μs) for r in R], " "))
        @printf("  LTP chosen (d, r, Hh): %s\n", join([@sprintf("(%d,%d,%d)", r.ltp.cfg.d, r.ltp.cfg.r, r.ltp.cfg.Hh) for r in R], " "))
        @printf("  LTP monodromy |λ|, top 3 per trial: %s\n", join([join([@sprintf("%.2f", abs(x)) for x in r.ltp.mult[1:3]], "/") for r in R], "  "))
        for s in unique(subs)
            I = findall(subs .== s)
            @printf("  subj%d lattice μ (ρ∠rot per stride): %s\n", s, join([join([@sprintf("%.2f∠%.0f", exp(real(μ) * NPH), rad2deg(imag(μ) * NPH)) for μ in R[i].lattice.μs], ",") for i in I], " | "))
        end
        flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "maus", "floquet_models.jls"), (; out, subs, HZ))
    println("wrote cache/maus/floquet_models.jls")
end
