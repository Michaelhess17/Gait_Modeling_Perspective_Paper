#!/usr/bin/env julia
# Data for Figure 5 (file Fig6_linear_affordances.jl; OUTLINE.md §4 ¶2–¶4): off-cycle relaxation of the decay-trained
# models (the fits of offcycle_v2.jl, refitted here for DMD and ESN; the neural ODE from its cached runs) and the
# multipliers of each model's own cycle on SLIP.
#
# Scoring, changed from offcycle_v2.jl (2026-11-03):
#   * Distance to a reference curve is measured to the curve through its samples (nearest point on the polyline),
#     not to the nearest sample. On coupled FHN consecutive cycle samples are up to 0.53 apart during the jumps,
#     against deviations of 0.12–0.28, so nearest-sample distance has a floor that made the truth's rate look
#     start-dependent (0.70–0.84); measured to the curve it is 0.69 from every start, the multiplier. The data the
#     models see are unchanged (same sampling).
#   * A model's deviation is the distance from its perturbed run to its own UNPERTURBED run (from the true on-cycle
#     warm-up at the same phase) within ±½ cycle of the same time, so phase shifts do not count (as for the truth) and
#     whatever the model does without the perturbation cancels. That matters: each model's cycle sits 0.13–0.16 from the
#     true cycle, as far as the perturbation, so both runs keep moving onto the model's own cycle, and DMD's waveform
#     drifts slowly (higher harmonics ≈ 0.97–0.98 per cycle). Two alternatives are kept as diagnostics: the model's
#     settled attractor (last 3 of 30 cycles; charges DMD for its drift: coupled FHN 0.18) and any point of its first 6
#     unperturbed cycles (the earlier scoring; matches states from other times).
#   * The perturbation is the original one (coupled FHN: unit 2 offset by 0.3 × its cycle SD; SLIP: as slip_decays.jl).
#   The neural ODE is not refitted (mock-up): its unperturbed runs are not cached, so its cached free runs are scored
#   against their own last 3 cycles, a different and only provisional reference.
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
"Distance from fc[t] to the unperturbed run f0 within ±half a cycle of the same time."
dist_local(fc, f0) = (h = SPC_F ÷ 2; [dist_curve(fc[t:t, :], f0[max(1, t - h):min(size(f0, 1), t + h), :]; closed = false)[1]
                                      for t in axes(fc, 1)])

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
    dt = sys_dt(sys); Ys, μ, σ = raw_training(sys, "decays"; seed = 1); std_(X) = (X .- μ) ./ σ
    Ct = std_(cycle_states(sys)); starts = true_starts(sys); ons = true_oncycle(sys)
    truth = [rate(env(dist_curve(std_(T)[W_WARM+1:end, :], Ct; closed = true))) for T in starts]
    rows = NamedTuple[]
    for (model, seeds) in (("dmd", 1:1), ("esn", 1:4))
        hp = chosen_hparams(sys, "decays", model)
        for seed in seeds
            pred = fit_predictor(sys, model, Ys, hp, seed, dt)
            loc = Float64[]; setl = Float64[]; any6 = Float64[]
            for (T, T0) in zip(starts, ons)
                fc = pred(std_(T)); f0 = pred(std_(T0))
                if fc === nothing || f0 === nothing || maximum(abs, fc) > 50
                    push!(loc, NaN); push!(setl, NaN); push!(any6, NaN); continue
                end
                push!(loc, rate(env(dist_local(fc[1:5SPC_F, :], f0[1:6SPC_F, :]))))
                push!(setl, rate(env(dist_curve(fc[1:5SPC_F, :], settled(f0); closed = false))))
                push!(any6, rate(env(dist_curve(fc[1:5SPC_F, :], f0[1:6SPC_F, :]; closed = false))))
            end
            push!(rows, (; model, seed, rates = loc, rates_settled = setl, rates_any6 = any6, mock = false))
        end
    end
    for f in sort(filter(f -> occursin(Regex("^offv2_$(sys)_node_\\d+\\.jls\$"), f), readdir(cache_path("floquet"))))
        o = deserialize(cache_path("floquet/" * f))
        self = [fc === nothing || maximum(abs, fc) > 50 ? NaN :
                rate(env(dist_curve(fc[1:5SPC_F, :], settled(fc); closed = false))) for fc in o.fcs]
        push!(rows, (; model = "node", seed = o.seed, rates = self, rates_settled = self, rates_any6 = fill(NaN, 4), mock = true))
    end
    (; sys, truth, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fm(v) = join([@sprintf("%5.2f", x) for x in v], " ")
    R = Dict{String, Any}()
    for sys in ("cfhn", "slip")
        r = relax(sys); R[sys] = r
        println("\n$sys  truth rate per start: ", fm(r.truth))
        for x in r.rows
            e(v) = mean(filter(isfinite, abs.(v .- r.truth)))
            @printf("%-4s seed %d%s | same-time unperturbed run %s  |err| %.3f | settled %s (%.3f) | any of first 6 cycles %s (%.3f)\n",
                    x.model, x.seed, x.mock ? " (cached runs, own last cycles: mock)" : "", fm(x.rates), e(x.rates),
                    fm(x.rates_settled), e(x.rates_settled), fm(x.rates_any6), e(x.rates_any6))
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
