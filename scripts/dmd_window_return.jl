#!/usr/bin/env julia
# Can Hankel-DMD predict the return to the true cycle on coupled FHN (Fig 5A), given a longer delay window or more
# data? (OUTLINE.md §4 ¶2, ¶6.) Scoring as fig6_relaxation_data.jl: distance to the curve through the true cycle's
# samples, from the 4 real perturbed starts; also, with no perturbation at all, how far the model's run from the true
# cycle drifts by cycle 4 and where it settles. Delay windows longer than the 1.7-cycle warm-up use the true on-cycle
# data before the kick as history (shorter windows read only their last τ + 1 samples, so they are unaffected).
#   Part 1: 10 decays × 3 cycles (Fig 5's data), delay window ⅛–2 cycles × rank 16–300 × variant (ls, fb, unit) × ridge.
#   Part 2: 10, 30, 100, 300 decays × 3 cycles (dmd_delay_sweep.jl's generator), window ¾–1.5 cycles × rank 60–300.
# Result (2026-11-03): at 10 decays no setting beats Fig 5's (1-cycle window, rank 120): cycle-4 distance 0.107 against
# the truth's 0.030, because even unperturbed its run drifts 0.07 from the true cycle within 4 cycles. More data shrink
# the drift and favour a 1.5-cycle window: 30 decays 0.057 (rate |error| 0.08), 100 decays 0.046 (0.07), 300 decays
# 0.044 (0.06). A floor remains from DMD's own cycle (drift 0.03 by cycle 4, settled offset 0.11 even at 300 decays).
#
# Usage: julia --project=. -t 16 scripts/dmd_window_return.jl

include("fig6_relaxation_data.jl")
include("dmd_delay_sweep.jl")
using Printf
BLAS.set_num_threads(1)

println("== Part 1: delay window, 10 decays")
sys = "cfhn"
Ys, μ, σ = raw_training(sys, "decays"; seed = 1); std_(X) = (X .- μ) ./ σ
cs = cycle_states(sys); n = SPC_F; Ct = std_(cs)
PH = (1, 28, 55, 83)
pre(ph) = std_(cs[[mod1(ph - k, n) for k in 2n:-1:1], :])              # two true cycles before the kick
starts = true_starts(sys); ons = true_oncycle(sys)
Hk = [vcat(pre(ph), std_(T)[1:W_WARM, :]) for (ph, T) in zip(PH, starts)]
Ho = [vcat(pre(ph), std_(T)[1:W_WARM, :]) for (ph, T) in zip(PH, ons)]
truth = [env(dist_curve(std_(T)[W_WARM+1:end, :], Ct; closed = true)) for T in starts]
tmed = [median([e[k] for e in truth]) for k in 1:5]
grid = vec([(; τ, r, variant, ridge) for τ in round.(Int, n .* (1/8, 1/4, 1/2, 3/4, 1, 1.5, 2)),
                                         r in (16, 30, 60, 120, 200, 300), variant in ("ls", "fb", "unit"), ridge in (1e-8, 1e-6)])
grid = [g for g in grid if g.r < 4 * (g.τ + 1)]
res = Vector{Any}(undef, length(grid))
Threads.@threads for i in eachindex(grid)
    g = grid[i]
    res[i] = try
        m = dmd_model_multi(Ys; g...)
        runs(H) = [dmd_free(m, h, 30n) for h in H]
        Fo = runs(Ho); Fk = runs(Hk)
        any(F -> any(!isfinite, F) || maximum(abs, F) > 50, vcat(Fo, Fk)) && error("diverged")
        drift4 = median([env(dist_curve(F[1:5n, :], Ct; closed = true))[5] for F in Fo])     # no perturbation, cycle 4
        offset = median([sqrt(mean(abs2, dist_curve(F[end-3n+1:end, :], Ct; closed = true))) for F in Fo])
        ek = [env(dist_curve(F[1:5n, :], Ct; closed = true)) for F in Fk]
        kmed = [median([e[k] for e in ek]) for k in 1:5]
        rerr = mean(abs.(rate.(ek) .- rate.(truth)))
        (; g..., drift4, offset, k4 = kmed[5], rerr, kmed)
    catch
        nothing
    end
end
R = [r for r in res if r !== nothing]
@printf("%d of %d settings ran. Truth, median distance to the true cycle, cycles 0–4: %s\n", length(R), length(grid),
        join([@sprintf("%.3f", x) for x in tmed], " "))
showrow(r) = @printf("τ %3d (%.2f cyc) r %3d %-4s ridge %.0e | no kick: drift by cycle 4 %.3f, settled offset %.3f | kicked: cycle-4 distance %.3f (truth %.3f), rate |err| %.3f\n",
                  r.τ, r.τ / n, r.r, r.variant, r.ridge, r.drift4, r.offset, r.k4, tmed[5], r.rerr)
println("\nFig 5's setting:"); foreach(showrow, [r for r in R if r.τ == 110 && r.r == 120 && r.variant == "ls" && r.ridge == 1e-6])
println("\nBest by cycle-4 distance after the kick (closest to the truth's):"); foreach(showrow, sort(R, by = r -> abs(r.k4 - tmed[5]))[1:8])
println("\nBest per delay window (lowest settled offset):")
for τ in sort(unique(r.τ for r in R)); showrow(sort([r for r in R if r.τ == τ], by = r -> r.offset)[1]); end

println("\n== Part 2: amount of data")
sys = "cfhn"; n = SPC_F; cs = cycle_states(sys); PH = (1, 28, 55, 83)
starts = true_starts(sys); ons = true_oncycle(sys)
grid = vec([(; τ, r, variant = "ls", ridge = 1e-6) for τ in (82, 110, 165), r in (60, 120, 200, 300)])
for ntraj in (10, 30, 100, 300)
    Ys, μ, σ = ntraj == 10 ? raw_training(sys, "decays"; seed = 1) : decays(sys, ntraj, 3; seed = 1)
    std_(X) = (X .- μ) ./ σ; Ct = std_(cs)
    pre(ph) = std_(cs[[mod1(ph - k, n) for k in 2n:-1:1], :])
    Hk = [vcat(pre(ph), std_(T)[1:W_WARM, :]) for (ph, T) in zip(PH, starts)]
    Ho = [vcat(pre(ph), std_(T)[1:W_WARM, :]) for (ph, T) in zip(PH, ons)]
    truth = [env(dist_curve(std_(T)[W_WARM+1:end, :], Ct; closed = true)) for T in starts]
    t4 = median(last.(truth))
    res = Vector{Any}(undef, length(grid))
    Threads.@threads for i in eachindex(grid)
        g = grid[i]
        res[i] = try
            m = dmd_model_multi(Ys; g...)
            Fo = [dmd_free(m, h, 30n) for h in Ho]; Fk = [dmd_free(m, h, 5n) for h in Hk]
            drift4 = median([env(dist_curve(F[1:5n, :], Ct; closed = true))[5] for F in Fo])
            off = median([sqrt(mean(abs2, dist_curve(F[end-3n+1:end, :], Ct; closed = true))) for F in Fo])
            ek = [env(dist_curve(F[1:5n, :], Ct; closed = true)) for F in Fk]
            (; g..., drift4, off, k4 = median(last.(ek)), rerr = mean(abs.(rate.(ek) .- rate.(truth))))
        catch
            nothing
        end
    end
    R = sort([r for r in res if r !== nothing], by = r -> r.k4)
    b = R[1]
    @printf("%3d decays | best: τ %3d r %3d | no kick: drift by cycle 4 %.3f, settled offset %.3f | kicked: cycle-4 distance %.3f (truth %.3f), rate |err| %.3f\n",
            ntraj, b.τ, b.r, b.drift4, b.off, b.k4, t4, b.rerr)
end
