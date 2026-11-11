#!/usr/bin/env julia
# Matched-data comparison for Fig 3C (OUTLINE.md §3 ¶7, §4 ¶2, ¶6): do DMD and the ESN predict the return to the true
# cycle better with more training decays? Coupled FHN (new decays from dmd_delay_sweep.jl's generator) and SLIP
# (10, 30, 100 decays = 1, 3, 10 of slip_decays.jl's independent 10-decay sets, pooled). Same scoring as fig6_relaxation_data.jl (distance to the
# curve through the true cycle's samples, from the 4 real perturbed starts; with no perturbation, the drift of the
# model's run from the true cycle by cycle 4). History: coupled FHN, 2 true cycles before the kick + the 170-sample
# warm-up (lets 1.5-cycle windows run); SLIP, the 170-sample warm-up (its windows are shorter).
# Each model is reported at its Fig 5 setting (tuned on 10 decays; ESN median over seeds 1–4) and as the best of a
# small grid, the latter chosen on the test score itself (an optimistic bound, not a fair selection).
#
# Usage: julia --project=. -t 16 scripts/matched_data_return.jl     (writes cache/fig6/matched_data.jls)
include("fig6_relaxation_data.jl")
include("dmd_delay_sweep.jl")
using Printf
BLAS.set_num_threads(1)

const n = SPC_F; const PH = (1, 28, 55, 83)
esn_pred(m) = H -> (S = run_steps(esn_step(m), esn_tf(m, H, size(H, 1)), 30n); S === nothing ? nothing : permutedims(S[m.N+1:end, :]))
dmd_pred(m) = H -> dmd_free(m, H, 30n)

function score(pred, Hk, Ho, Ct, truth)
    Fk = [pred(h) for h in Hk]; Fo = [pred(h) for h in Ho]
    bad(F) = F === nothing || any(!isfinite, F) || maximum(abs, F) > 50
    any(bad, vcat(Fk, Fo)) && return nothing
    ek = [env(dist_curve(F[1:5n, :], Ct; closed = true)) for F in Fk]
    (; k4 = median(last.(ek)), rerr = mean(abs.(rate.(ek) .- rate.(truth))),
       drift4 = median([env(dist_curve(F[1:5n, :], Ct; closed = true))[5] for F in Fo]), envs = ek)
end

function training(sys, ntraj)
    sys == "cfhn" && return ntraj == 10 ? raw_training(sys, "decays"; seed = 1) : decays(sys, ntraj, 3; seed = 1)
    raw = reduce(vcat, [slipd().decays[k] for k in 1:ntraj ÷ 10]); allX = reduce(vcat, raw)
    μ = mean(allX, dims = 1); σ = std(allX, dims = 1); [(X .- μ) ./ σ for X in raw], μ, σ
end
OUT = Dict{String, Any}()
for SYS in ("cfhn", "slip")
hp_d = chosen_hparams(SYS, "decays", "dmd"); hp_e = chosen_hparams(SYS, "decays", "esn")
dgrid = vec([(; τ, r, variant = "ls", ridge = 1e-6) for τ in (SYS == "cfhn" ? (82, 110, 165) : (55, 82, 110)), r in (60, 120, 200, 300)])
egrid = vec([merge(hp_e, (; N, ρ, λ)) for N in (400, 800), ρ in (0.9, 1.2), λ in (1e-6, 1e-4)])
cs = cycle_states(SYS); starts = true_starts(SYS); ons = true_oncycle(SYS)
out = Dict{Int, Any}()
println("\n######## $SYS   ESN Fig 5 setting: ", hp_e, "   DMD: ", hp_d)
for ntraj in (10, 30, 100)
    Ys, μ, σ = training(SYS, ntraj)
    std_(X) = (X .- μ) ./ σ; Ct = std_(cs)
    pre(ph) = SYS == "cfhn" ? std_(cs[[mod1(ph - k, n) for k in 2n:-1:1], :]) : zeros(0, size(cs, 2))
    Hk = [vcat(pre(ph), std_(T)[1:W_WARM, :]) for (ph, T) in zip(PH, starts)]
    Ho = [vcat(pre(ph), std_(T)[1:W_WARM, :]) for (ph, T) in zip(PH, ons)]
    truth = [env(dist_curve(std_(T)[W_WARM+1:end, :], Ct; closed = true)) for T in starts]
    jobs = vcat([(:dmd, hp_d, 0, true)], [(:dmd, g, 0, false) for g in dgrid],
                [(:esn, hp_e, s, true) for s in 1:4], [(:esn, g, 1, false) for g in egrid])
    res = Vector{Any}(undef, length(jobs))
    Threads.@threads for i in eachindex(jobs)
        kind, g, s, fixed = jobs[i]
        res[i] = try
            pred = kind == :dmd ? dmd_pred(dmd_model_multi(Ys; g...)) : esn_pred(fit_esn_multi(Ys; g..., seed = s))
            r = score(pred, Hk, Ho, Ct, truth); r === nothing ? nothing : (; kind, g, s, fixed, r...)
        catch
            nothing
        end
    end
    R = [r for r in res if r !== nothing]; out[ntraj] = (; R, truth)
    @printf("\n%3d decays (truth's cycle-4 distance %.3f)\n", ntraj, median(last.(truth)))
    for kind in (:dmd, :esn)
        F = [r for r in R if r.kind == kind && r.fixed]; G = [r for r in R if r.kind == kind]
        isempty(F) ? println("  $kind Fig 5 setting: diverged") :
            @printf("  %-3s Fig 5 setting: cycle-4 distance %.3f, rate |err| %.3f, no-kick drift %.3f  (%d of %d runs ok)\n", kind,
                    median([r.k4 for r in F]), median([r.rerr for r in F]), median([r.drift4 for r in F]), length(F), kind == :dmd ? 1 : 4)
        isempty(G) || (b = G[argmin([r.rerr for r in G])];
            @printf("  %-3s best of grid (chosen on test): cycle-4 distance %.3f, rate |err| %.3f, no-kick drift %.3f  %s\n", kind, b.k4, b.rerr, b.drift4, b.g))
    end
end
OUT[SYS] = out
end
mkpath(cache_path("fig6")); serialize(cache_path("fig6/matched_data.jls"), OUT)
