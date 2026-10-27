#!/usr/bin/env julia
# Does total-least-squares or forward–backward DMD help under measurement noise?
# Decay-trained Hankel-DMD (n decays × 3 cycles, process noise as elsewhere) plus i.i.d. measurement noise
# σm (SD units) on the training data. Fits: ls (ridge least squares), fb (forward–backward, Dawson et al.
# 2016), tls (total least squares, Hemati et al. 2017: project X, Z onto the top-r right singular vectors
# of [X; Z]). All clip |λ| > 1 onto the unit circle. Scored as in dmd_delay_sweep.jl (clean warm-up):
#   rate = per-cycle off-cycle relaxation (truth ≈ |μ|), amp = on-cycle free-run amplitude after 6 cycles
#   relative to the true cycle (1 = sustains the cycle).
# Usage: julia --project=. scripts/dmd_noise_variants.jl <sys> <σm> <ntraj>
include("dmd_delay_sweep.jl")

function dmd_fit(Ys; τ, r, variant, ridge = 1e-6)
    Hs = [build_hankel_multi(Y, τ) for Y in Ys]
    r = min(r, size(Hs[1], 1) - 1); U = svd(reduce(hcat, Hs)).U[:, 1:r]
    X = reduce(hcat, [U' * H[:, 1:end-1] for H in Hs]); Z = reduce(hcat, [U' * H[:, 2:end] for H in Hs])
    reg(M) = M + ridge * tr(M) / r * I
    if variant == "tls"
        V = svd([X; Z]).V[:, 1:r]; X, Z = (X * V) * V', (Z * V) * V'
    end
    A = Z * X' / reg(X * X')
    variant == "fb" && (A = real(sqrt(complex(A / (X * Z' / reg(Z * Z'))))))
    (; A = stabilise_unit_clip(A), U, τ, d = size(Ys[1], 2))
end

if abspath(PROGRAM_FILE) == @__FILE__
    sys, σm, ntraj = ARGS[1], parse(Float64, ARGS[2]), parse(Int, ARGS[3])
    Ys, μ, σ = decays(sys, ntraj, 3)
    rng = MersenneTwister(77); Ys = [Y .+ σm .* randn(rng, size(Y)) for Y in Ys]
    std_(X) = (X .- μ) ./ σ; Ctrue = std_(cycle_states(sys)); ampT = mean(std(Ctrue, dims = 1))
    for τ in round.(Int, SPC_F .* (1.0, 1.5, 2.0)), r in (20, 40, 80, 160), variant in ("ls", "fb", "tls")
        W = τ + 10; runs = truth_runs(sys, W, OFFSETS.cfhn)
        m = try dmd_fit(Ys; τ, r, variant) catch; continue end
        et = Vector{Float64}[]; em = Vector{Float64}[]; amps = Float64[]
        for (T, T0) in runs
            Ts = std_(T); e = env(nearest(Ts[W+1:end, :], Ctrue)); push!(et, e ./ e[1])
            fc = dmd_free(m, Ts[1:W, :], 6SPC_F); f0 = dmd_free(m, std_(T0)[1:W, :], 6SPC_F)
            (all(isfinite, fc) && all(isfinite, f0)) || (push!(em, fill(NaN, 5)); continue)
            a = env(nearest(fc, f0)); push!(em, a ./ a[1])
            push!(amps, mean(std(f0[end-SPC_F+1:end, :], dims = 1)) / ampT)
        end
        med(v) = [median(filter(isfinite, [x[k] for x in v])) for k in 1:5]
        tr, md = med(et), med(em)
        @printf("%s σm=%.2f n=%3d τ=%.1fcyc r=%3d %-3s | rate %.2f (truth %.2f) | amp %.2f\n",
                sys, σm, ntraj, τ / SPC_F, r, variant, md[5]^(1/4), tr[5]^(1/4), isempty(amps) ? NaN : median(amps))
    end
end
