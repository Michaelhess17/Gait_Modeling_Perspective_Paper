#!/usr/bin/env julia
# Why does Hankel-DMD get v₁, w₁, w₂ of the coupled FHN nearly right but not their difference w₁ − w₂ (small kicks,
# cfhn_smooth_models.jl)? The model's own cycle is not synchronized (w₁ − w₂ RMS 0.09 on an unkicked run, truth 0.005)
# and its harmonics are inharmonic (frequencies 1.004, 2.007, 3.011 × the cycle's). Candidate causes:
#   resolution  110 samples per cycle resolve the sharp phases of the waveform, and the small lag between the units,
#               poorly; denser sampling adds delay coordinates within the same window
#   noise       process noise in the 30 training cycles biases the fit
# Design: the small-kick experiment of cfhn_smooth_data.jl regenerated at 110, 220 and 440 samples per cycle, with the
# same process noise or none; delay window fixed in cycles (0.25–1.5), rank 12–200, ridge 1e-6 or 1e-4, chosen by the
# same validation (forecasts from 1.55 cycles of independent runs). Reported for the chosen fit: frequency error of
# the fundamental, per-cycle modulus of harmonics 1–4, w₁ − w₂ RMS of the unkicked run, and the kicked forecast's
# errors in v₁ and w₁ − w₂ (cycles 1 and 4–5), for the four test kicks (0.3 SD on unit 2).
# Usage: julia -t 6 --project=. scripts/cfhn_density_probe.jl [variants]   (variants: least-squares, forward-backward and
#        total-least-squares DMD on the noisy data at 110 samples per cycle) [data] (30–900 noisy training cycles)
include("fig6_relaxation_data.jl")
include("cfhn_smooth_system.jl")
using Printf
BLAS.set_num_threads(1)
R0 = deserialize(joinpath(SMOOTH_DIR, "regimes.jls")); c = R0.selected
p = (c.ε, c.k, c.drive); noise0 = 0.02sqrt(R0.rows[1].T / c.T)
sdc = vec(std(c.C[1:end-1, :], dims = 1))

function dataset(seed, count, n, noise)
    rng = MersenneTwister(seed)
    map(1:count) do k
        u = c.C[rand(rng, 1:size(c.C, 1)-1), :] .+ 0.2sdc .* randn(rng, 4)
        if noise > 0
            g!(du, u, p_, t) = (du .= 0.0; du[1] = noise; du[3] = noise)
            sol = solve(SDEProblem(smooth_fhn!, g!, u, (0.0, (3n - 1) * c.T / n), p; seed = seed * 1000 + k), EM();
                        dt = c.T / 2200, saveat = c.T / n)          # same integration step at every sampling rate
        else
            sol = solve(ODEProblem(smooth_fhn!, u, (0.0, (3n - 1) * c.T / n), p), Vern9(); SMOOTH_OPT..., saveat = c.T / n)
        end
        permutedims(Array(sol))
    end
end
function clipA(A)
    F = schur(complex(A)); T = copy(F.T)
    for i in axes(T, 1); abs(T[i, i]) > 1 && (T[i, i] /= abs(T[i, i])); end
    real(F.Z * T * F.Z')
end
function free(m, Yh, nn)
    α = m.U' * vec(permutedims(Yh[end-m.τ:end, :])); out = zeros(nn, 4)
    for k in 1:nn; α = m.A * α; out[k, :] = (m.U * α)[m.τ*4+1:end]; end
    out
end
function run_condition(n, noise; variant = "ls", count = 10)
    train = dataset(1, count, n, noise); valid = dataset(101, 10, n, noise)
    A_ = reduce(vcat, train); μ = mean(A_, dims = 1); σ = std(A_, dims = 1); st(X) = (X .- μ) ./ σ
    Ys = st.(train); V = st.(valid); W = round(Int, 170 / 110 * n)
    tests = map((1, 221, 441, 661)) do ph
        u = c.C[ph, :]; kk = copy(u); kk[3:4] .+= 0.3sdc[3:4]
        (; truth = st(smooth_run(kk, p, c.T, 8; n)), oncycle = st(smooth_run(u, p, c.T, 8; n)))
    end
    best = (Inf, nothing, nothing)
    for wc in (0.25, 0.5, 1.0, 1.5)
        τ = round(Int, wc * n); τ + 1 <= W || continue
        Hs = [build_hankel_multi(Y, τ) for Y in Ys]; Hc = reduce(hcat, Hs)
        E = eigen(Symmetric(Hc * Hc')); Uall = E.vectors[:, end:-1:1]
        X0 = reduce(hcat, [h[:, 1:end-1] for h in Hs]); Z0 = reduce(hcat, [h[:, 2:end] for h in Hs])
        for r in (12, 30, 60, 120, 200), ridge in (1e-6, 1e-4)
            U = Uall[:, 1:min(r, size(Uall, 2))]; X = U' * X0; Z = U' * Z0
            reg(M) = M + ridge * tr(M) / size(M, 1) * I
            Af = if variant == "tls"                                  # total least squares (Hemati et al. 2017)
                Vq = svd(vcat(X, Z)).V[:, 1:size(X, 1)]; Xp = X * Vq * Vq'; Zp = Z * Vq * Vq'
                Zp * Xp' / reg(Xp * Xp')
            else
                Z * X' / reg(X * X')
            end
            variant == "fb" && (Af = real(sqrt(complex(Af / (X * Z' / reg(Z * Z'))))))   # forward-backward (Dawson et al. 2016)
            m = (; A = clipA(Af), U, τ)
            v = mean(V) do Y
                F = free(m, Y[1:W, :], size(Y, 1) - W); any(!isfinite, F) ? Inf : mean(abs2, F .- Y[W+1:end, :])
            end
            v < best[1] && (best = (v, m, (; window = wc, τ, r, ridge)))
        end
    end
    v, m, g = best
    λ = eigvals(m.A); fr = angle.(λ) .* n ./ (2π); per = abs.(λ) .^ n
    harm = map(1:4) do k
        cnd = [i for i in eachindex(λ) if abs(fr[i] - k) < 0.05]
        isempty(cnd) ? (NaN, NaN) : (i = cnd[argmax(per[cnd])]; (fr[i] - k, per[i]))
    end
    dd(X, a, b) = sqrt(mean(abs2, X[a:b, 2] .- X[a:b, 4]))
    rows = map(tests) do t
        F = free(m, t.truth[1:W, :], 5n); F0 = free(m, t.oncycle[1:W, :], 5n)
        T = t.truth[W+1:W+5n, :]
        (; v1_c1 = sqrt(mean(abs2, F[1:n, 1] .- T[1:n, 1])), v1_c5 = sqrt(mean(abs2, F[4n+1:5n, 1] .- T[4n+1:5n, 1])),
           dd_err = sqrt(mean(abs2, (F[3n+1:5n, 2] .- F[3n+1:5n, 4]) .- (T[3n+1:5n, 2] .- T[3n+1:5n, 4]))),
           dd_own = dd(F0, 3n + 1, 5n), dd_true = dd(T, 3n + 1, 5n))
    end
    md(f) = median(f.(rows))
    (; n, noise, g, val = v, harm, v1_c1 = md(r -> r.v1_c1), v1_c5 = md(r -> r.v1_c5), dd_err = md(r -> r.dd_err),
       dd_own = md(r -> r.dd_own), dd_true = md(r -> r.dd_true))
end

conds = !isempty(ARGS) && ARGS[1] == "variants" ? [(110, noise0, v, 10) for v in ("ls", "fb", "tls")] :
        !isempty(ARGS) && ARGS[1] == "data" ? [(110, noise0, "ls", k) for k in (10, 30, 100, 300)] :
        [(n, nz, "ls", 10) for nz in (noise0, 0.0) for n in (110, 220, 440)]
res = Vector{Any}(undef, length(conds))
Threads.@threads :dynamic for i in eachindex(conds)
    res[i] = (; run_condition(conds[i][1], conds[i][2]; variant = conds[i][3], count = conds[i][4])..., variant = conds[i][3],
              cycles = 3conds[i][4])
end
println("standardized units; medians over the four test kicks; w₁ − w₂ in cycles 4–5 of the forecast")
for r in res
    @printf("%-4s %-6s %4d training cycles %3d/cycle | window %.2f cyc, r %3d, ridge %.0e | fundamental freq err %+.4f | harmonic moduli/cycle %s | v₁ err c1 %.3f c5 %.3f | w₁−w₂: forecast err %.3f, own unkicked %.3f (truth kicked %.3f)\n",
            r.variant, r.noise > 0 ? "noisy" : "clean", r.cycles, r.n, r.g.window, r.g.r, r.g.ridge, r.harm[1][1],
            join([@sprintf("%.3f", h[2]) for h in r.harm], "/"), r.v1_c1, r.v1_c5, r.dd_err, r.dd_own, r.dd_true)
end
mkpath(joinpath(SMOOTH_DIR, "density_probe"))
serialize(joinpath(SMOOTH_DIR, "density_probe", isempty(ARGS) ? "probe.jls" : "probe_$(ARGS[1]).jls"), res)
