#!/usr/bin/env julia
# Data for Figure 5 (file Fig6_linear_affordances.jl; OUTLINE.md §4 ¶2–¶4): off-cycle relaxation of the decay-trained
# models (the free runs cached by offcycle_v2.jl, for every model) and the multipliers of each model's own cycle on
# SLIP (DMD and ESN refitted).
#
# Scoring, changed from offcycle_v2.jl (2026-11-03): relaxation to the TRUE cycle from the real perturbed state.
#   * Every model is warmed up on the true perturbed trajectory (offcycle_v2.jl's free runs, cached for all models,
#     neural ODE included) and its deviation is the distance to the true cycle, as for the truth, so model and truth are
#     scored identically and nothing model-specific enters the reference.
#   * Distance is measured to the curve through the cycle's samples (nearest point on the polyline), not to the nearest
#     sample: on coupled FHN consecutive cycle samples are up to 0.53 apart during the jumps, against deviations of
#     0.12–0.28, so nearest-sample distance has a floor that made the truth's rate look start-dependent (0.70–0.84);
#     measured to the curve it is 0.69 from every start, the multiplier. The data the models see are unchanged.
#   * The perturbation is the original one (coupled FHN: unit 2 offset by 0.3 × its cycle SD; SLIP: as slip_decays.jl).
#   * Also recorded: each run's end offset, the distance of its last 3 of 30 cycles from the true cycle. A model's
#     distance to the true cycle cannot fall below it, and on coupled FHN it is as large as the perturbation.
#   (Measuring a model from its own unperturbed run instead isolates the response to the perturbation but depends on
#   which reference is taken, and does not ask whether the model predicts the real return.)
# Multipliers of each model's own cycle (SLIP):
#   DMD  eigenvalues of its operator in the cycle frame (nearest harmonic, rotation per step, |λ| per step), as Fig S6B
#   ESN  eigenvalues of the monodromy (closed-loop Jacobians chained over one period of its own settled cycle), the one
#        nearest 1 (phase) removed
#   NODE needs the fitted vector field (not cached): pending a refit
#
# Usage: julia --project=. scripts/fig6_relaxation_data.jl      (writes cache/fig6/relaxation.jls)

include("offcycle_v2.jl")
using Printf

"Distance from each row of X to the curve through the rows of C (closed: also the segment back to the start)."
function dist_curve(X, C; closed)
    n = size(C, 1); nxt = closed ? [2:n; 1] : collect(2:n); P0 = C[closed ? (1:n) : (1:n-1), :]
    Dv = C[nxt, :] .- P0; L2 = max.(vec(sum(abs2, Dv, dims = 2)), eps())
    map(axes(X, 1)) do i
        x = X[i:i, :]; t = clamp.(vec(sum((x .- P0) .* Dv, dims = 2)) ./ L2, 0, 1)
        sqrt(minimum(vec(sum(abs2, x .- (P0 .+ t .* Dv), dims = 2))))
    end
end
rate(e) = (e[5] / e[1])^(1 / 4)
settled(F) = F[end-3SPC_F+1:end, :]

"DMD eigenvalues in the cycle frame: nearest harmonic n, rotation per cycle δ (°), |λ| per cycle."
function cycle_frame(A)
    λ = eigvals(A); f = angle.(λ) ./ (2π)                  # cycles per sample
    fund = minimum(f[(f .> 1e-4) .& (abs.(λ) .> 0.999)])     # the fundamental sits on the unit circle
    [(; n = round(Int, fj / fund), δ = 360 * (fj / fund - round(fj / fund)), ρ = abs(l)^(1 / fund), λ = l)
     for (fj, l) in zip(f, λ) if fj >= -1e-9]
end

"Monodromy eigenvalues of an ESN's own settled cycle (phase eigenvalue removed), sorted by modulus."
function esn_multipliers(m, s0)
    step = esn_step(m); jac = esn_jac(m); s = copy(s0)
    S = run_steps(step, s, 40SPC_F); S === nothing && return nothing
    tail = S[:, end-6SPC_F+1:end]; c = cross_idx(tail[m.N+1, :]); length(c) < 3 && return nothing
    n = round(Int, mean(diff(c))); i0 = size(S, 2) - 6SPC_F + round(Int, c[1])
    M, clos = chain(step, jac, S[:, i0], n, m.d)
    ev = eigvals(M); i = argmin(abs.(ev .- 1)); rest = ev[setdiff(eachindex(ev), i)]
    (; ev = rest[sortperm(abs.(rest), rev = true)][1:min(end, 12)], period = n, closure = clos)
end

function relax(sys)
    Ys, μ, σ = raw_training(sys, "decays"; seed = 1); std_(X) = (X .- μ) ./ σ
    Ct = std_(cycle_states(sys)); starts = true_starts(sys)
    Et = [env(dist_curve(std_(T)[W_WARM+1:end, :], Ct; closed = true)) for T in starts]
    rows = NamedTuple[]
    for model in ("dmd", "esn", "node")
        for f in sort(filter(f -> occursin(Regex("^offv2_$(sys)_$(model)_\\d+\\.jls\$"), f), readdir(cache_path("floquet"))))
            o = deserialize(cache_path("floquet/" * f))
            ok(fc) = fc !== nothing && maximum(abs, fc) <= 50
            E = [ok(fc) ? env(dist_curve(fc[1:5SPC_F, :], Ct; closed = true)) : fill(NaN, 5) for fc in o.fcs]
            off = [ok(fc) ? sqrt(mean(abs2, dist_curve(settled(fc), Ct; closed = true))) : NaN for fc in o.fcs]
            push!(rows, (; model, seed = o.seed, rates = rate.(E), dev0 = first.(E), dev4 = last.(E), offset = off, envs = E))
        end
    end
    (; sys, truth = rate.(Et), dev0 = first.(Et), dev4 = last.(Et), truth_envs = Et, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fm(v) = join([@sprintf("%5.2f", x) for x in v], " ")
    R = Dict{String, Any}()
    for sys in ("cfhn", "slip")
        r = relax(sys); R[sys] = r
        println("\n$sys  truth rate per start: ", fm(r.truth), " | deviation cycle 0 ", fm(r.dev0), ", cycle 4 ", fm(r.dev4))
        for x in r.rows
            @printf("%-4s seed %d | rate to the true cycle %s  |err| %.3f | deviation cycle 0 %s, cycle 4 %s | end offset %s\n",
                    x.model, x.seed, fm(x.rates), mean(filter(isfinite, abs.(x.rates .- r.truth))), fm(x.dev0), fm(x.dev4), fm(x.offset))
        end
    end
    # multipliers of each model's own cycle, SLIP
    sys = "slip"; dt = sys_dt(sys); Ys, μ, σ = raw_training(sys, "decays"; seed = 1); std_(X) = (X .- μ) ./ σ
    md = dmd_model_multi(Ys; chosen_hparams(sys, "decays", "dmd")...); cf = cycle_frame(md.A)
    hp = chosen_hparams(sys, "decays", "esn"); ons = true_oncycle(sys)
    esn = map(1:4) do seed
        m = fit_esn_multi(Ys; hp..., seed); esn_multipliers(m, esn_tf(m, std_(ons[1]), W_WARM))
    end
    exact = slipd().λ
    println("\nSLIP exact multiplier: ", exact)
    near = sort([x for x in cf if abs(x.δ) > 2 && x.ρ > 0.3], by = x -> -x.ρ)
    println("DMD sidebands (n, rotation per step, |λ| per step): ",
            join([@sprintf("(%d, %+.1f°, %.3f)", x.n, x.δ, x.ρ) for x in near[1:min(end, 8)]], " "))
    for (k, e) in enumerate(esn)
        e === nothing && (println("ESN seed $k: no settled cycle"); continue)
        println(@sprintf("ESN seed %d (period %d samples, closure %.1e): ", k, e.period, e.closure),
                join([@sprintf("%.3f∠%+.1f°", abs(z), rad2deg(angle(z))) for z in e.ev[1:6]], "  "))
    end
    mkpath(cache_path("fig6"))
    serialize(cache_path("fig6/relaxation.jls"), (; relax = R, dmd_cycle_frame = cf, esn, exact,
                                                   lattice = deserialize(cache_path("figS6/slip.jls")).mult))
end
