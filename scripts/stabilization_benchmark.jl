#!/usr/bin/env julia
# What the spectral constraint contributes across the self-driving benchmark (replaces the two-trial
# Fig_stabilization example in the main text).
#
# For every benchmark cell (13 system–regime cells with a dmdtuned_*.jls):
#   paired     the train-selected setting (τ, r, variant) refitted with the constraint changed:
#              :none (raw least squares / forward–backward), :clip (|λ| > 1 onto the circle),
#              :unit (clip, and persistent modes, |λ|^P > 0.5, onto the circle)
#   retuned    unconstrained DMD (variants ls and fb, stabilization :none) selected on the training data by the
#              same criterion as dmd_tuning.jl, so that it is not penalised for settings chosen with a constraint
# Every fit is scored by self_driving_benchmark.jl's score() (first-cycle error, 30-cycle fate).
#
# Usage: julia -t 8 --project=. scripts/stabilization_benchmark.jl   → cache/selfdrive/stabilization_benchmark.jls

include("dmd_tuning.jl")

const FAIL = (; e1 = Inf, e3 = Inf, e5 = Inf, amp = Inf, per = NaN, fate = "blow-up")
const FATES = ["ok", "decays", "wrong", "collapse", "blow-up"]

function val_error(Yfit, Yval, P, nv, amp_ref, g)
    try
        fcl = fit_dmd_variant(Yfit; g..., P, stabilization = :none)(Yfit, round(Int, 10P))
        any(!isfinite, fcl) && return Inf
        amp = mean(std(fcl[end-round(Int, 3P)+1:end, :], dims = 1)) / amp_ref
        sqrt(mean(abs2, fcl[1:nv, :] .- Yval)) / sqrt(mean(abs2, Yval)) + 0.5 * abs(log(max(amp, 1e-3)))
    catch
        Inf
    end
end

function cell(system, regime)
    D = npzread(cache_path("selfdrive/data_$(system)_$(regime).npz"))
    R = deserialize(cache_path("selfdrive/dmdtuned_$(system)_$(regime).jls"))
    Ytr, Yte, P, n_run = D["Ytr"], D["Yte"], D["P"], Int(D["n_run"])
    sc(g, mode) = try
        Fc = fit_dmd_variant(Ytr; g..., P, stabilization = mode)(Ytr, n_run)
        any(!isfinite, Fc) ? FAIL : score(Fc, Yte, Ytr, P)
    catch
        FAIL
    end
    c = R.chosen; g0 = (; τ = c.τ, r = c.r, variant = c.variant)
    paired = Dict(m => sc(g0, m) for m in (:none, :clip, :unit))
    # retuned without any constraint
    nv = round(Int, 2P); Yfit, Yval = Ytr[1:end-nv, :], Ytr[end-nv+1:end, :]
    amp_ref = mean(std(Ytr, dims = 1))
    grid = [(; τ, r, variant) for τ in unique(max.(4, round.(Int, P .* (0.1, 0.25, 0.5, 1.0)))),
                                  r in (8, 16, 30, 50, 80), variant in ("ls", "fb")]
    vals = [val_error(Yfit, Yval, P, nv, amp_ref, g) for g in grid]
    gr = grid[argmin(vals)]
    retuned = (; gr..., val = minimum(vals), sc(gr, :none)...)
    (; system, regime, chosen = g0, chosen_fate = c.fate, chosen_e1 = c.e1, paired, retuned)
end

cells = [(m[1], m[2]) for f in readdir(cache_path("selfdrive"))
         for m in (match(r"^dmdtuned_(.+)_(realistic|long)\.jls$", f),) if m !== nothing]
out = Vector{Any}(undef, length(cells))
Threads.@threads for i in eachindex(cells)
    out[i] = cell(cells[i]...)
end
serialize(cache_path("selfdrive/stabilization_benchmark.jls"), out)

@printf("%-9s %-9s | %-26s | %-18s %-18s %-18s | %s\n", "system", "regime", "chosen", "paired: none",
        "clip", "unit", "retuned unconstrained")
for o in out
    p = o.paired
    f(x) = @sprintf("%-8s e1 %5.2f", x.fate, min(x.e1, 99))
    @printf("%-9s %-9s | τ=%3d r=%2d %-4s %-8s | %s %s %s | τ=%3d r=%2d %-3s %s\n", o.system, o.regime, o.chosen.τ,
            o.chosen.r, o.chosen.variant, o.chosen_fate, f(p[:none]), f(p[:clip]), f(p[:unit]), o.retuned.τ,
            o.retuned.r, o.retuned.variant, f(o.retuned))
end
for (lab, get) in (("paired none", o -> o.paired[:none]), ("paired clip", o -> o.paired[:clip]),
                   ("paired unit", o -> o.paired[:unit]), ("retuned none", o -> o.retuned),
                   ("chosen (published)", o -> (; fate = o.chosen_fate)))
    fs = [get(o).fate for o in out]
    @printf("%-20s %s\n", lab, join(["$(f) $(count(==(f), fs))" for f in FATES], "  "))
end
