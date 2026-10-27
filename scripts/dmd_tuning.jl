#!/usr/bin/env julia
# Tune Hankel-DMD properly for the self-driving benchmark.
#
# The benchmark's DMD used one short delay window (P/10), ranks 8/12/20 and
# only clipped eigenvalues ABOVE the unit circle, so least-squares noise bias
# (eigenvalues pulled inside the circle) made every free run decay. Here we
# sweep
#   delay depth  τ ∈ {P/10, P/4, P/2, P}
#   rank         r ∈ {8, 16, 30, 50, 80}  (capped at the Hankel row count)
#   variant      ls    ridge least squares, clip |λ| > 1 onto the circle
#                fb    forward–backward DMD (cancels the leading noise bias)
#                unit  ls, then modes that persist (|λ|^P > 0.5 per cycle) are
#                      placed ON the unit circle — the periodic/unitary
#                      constraint used for oscillators (cf. physics-informed DMD)
# and choose per cell on the TRAINING data only: fit on all but the last two
# training cycles, free-run, and score the error over those two cycles (ties
# broken toward sustained amplitude: score = val error + 0.5·|log(free-run
# amplitude ratio)|, with the amplitude measured over 10 free-running cycles from
# the fit window, so settings whose free run drifts in size are penalised using
# training data only). The chosen setting is refitted on the
# full training window and scored exactly like the other models. The best
# setting chosen on the test data is reported too, as an upper bound.
#
# Usage: julia --project=. scripts/dmd_tuning.jl <system> <regime>

include("self_driving_benchmark.jl")

"Modify eigenvalues in the complex Schur basis (keeps the rest of the operator)."
function stabilise(A, P; mode = :clip)
    F = schur(complex(A)); T = copy(F.T)
    for i in axes(T, 1)
        λ = T[i, i]; m = abs(λ)
        if m > 1 || (mode == :unit && m^P > 0.5)
            T[i, i] = λ / m
        end
    end
    return real(F.Z * T * F.Z')
end

function fit_dmd_variant(Y; τ, r, variant, P, ridge = 1e-8)
    d = size(Y, 2); H = build_hankel_multi(Y, τ)
    r = min(r, size(H, 1) - 1)
    U = svd(H).U[:, 1:r]
    a = U' * H; X, Z = a[:, 1:end-1], a[:, 2:end]
    reg(M) = M + ridge * tr(M) / r * I
    Af = Z * X' / reg(X * X')
    A = if variant == "fb"
        Ab = X * Z' / reg(Z * Z')
        real(sqrt(complex(Af / Ab)))
    else
        Af
    end
    A = stabilise(A, P; mode = variant == "unit" ? :unit : :clip)
    function forecast(Ytr, n)
        h = vec(permutedims(Ytr[end-τ:end, :])); α = U' * h
        out = zeros(n, d)
        for k in 1:n
            α = A * α; out[k, :] = (U * α)[τ*d+1:end]
        end
        out
    end
    return forecast
end

function tune_cell(system, regime)
    D = npzread(cache_path("selfdrive/data_$(replace(system, ":" => ""))_$(regime).npz"))
    Ytr, Yte, P, n_run = D["Ytr"], D["Yte"], D["P"], Int(D["n_run"])
    nv = round(Int, 2P)
    Yfit, Yval = Ytr[1:end-nv, :], Ytr[end-nv+1:end, :]
    grid = [(; τ, r, variant) for τ in unique(max.(4, round.(Int, P .* (0.1, 0.25, 0.5, 1.0)))),
                                  r in (8, 16, 30, 50, 80), variant in ("ls", "fb", "unit")]
    res = NamedTuple[]
    amp_ref = mean(std(Ytr, dims = 1))
    for g in grid
        val = try
            fcl = fit_dmd_variant(Yfit; g..., P)(Yfit, round(Int, 10P))
            if any(!isfinite, fcl)
                Inf
            else
                fc = fcl[1:nv, :]
                amp = mean(std(fcl[end-round(Int, 3P)+1:end, :], dims = 1)) / amp_ref
                sqrt(mean(abs2, fc .- Yval)) / sqrt(mean(abs2, Yval)) + 0.5 * abs(log(max(amp, 1e-3)))
            end
        catch
            Inf
        end
        test = try
            score(fit_dmd_variant(Ytr; g..., P)(Ytr, n_run), Yte, Ytr, P)
        catch
            (; e1 = Inf, e3 = Inf, e5 = Inf, amp = Inf, per = NaN, fate = "blow-up")
        end
        push!(res, (; g..., val, test...))
    end
    chosen = res[argmin([x.val for x in res])]
    oracle = res[argmin([isfinite(x.e3) ? x.e3 : Inf for x in res])]
    # free-run traces of the chosen setting for the figure
    Fc = fit_dmd_variant(Ytr; τ = chosen.τ, r = chosen.r, variant = chosen.variant, P)(Ytr, n_run)
    return (; system, regime, res, chosen, oracle,
              Fc = Float32.(Fc[1:round(Int, 8P), :]), P)
end

if abspath(PROGRAM_FILE) == @__FILE__
    system, regime = ARGS[1], ARGS[2]
    out = tune_cell(system, regime)
    serialize(cache_path("selfdrive/dmdtuned_$(replace(system, ":" => ""))_$(regime).jls"), out)
    for (lab, x) in (("chosen (train-only)", out.chosen), ("oracle (test)", out.oracle))
        @printf("%-22s %-9s %-20s τ=%3d r=%2d %-4s | val %.2f | e1 %.2f e3 %.2f e5 %.2f | amp %.2f per %.2f %s\n",
                system, regime, lab, x.τ, x.r, x.variant, x.val, x.e1, x.e3, x.e5, x.amp, x.per, x.fate)
    end
end
