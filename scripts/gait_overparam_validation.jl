#!/usr/bin/env julia
# Training loss against held-out autonomous-rollout error for the gait neural ODEs (gait_overparam.jl caches;
# OUTLINE.md §4 ¶3, to-do "validation, not training loss"). Each fit's forecast runs freely from the last training
# sample over the held-out 5 s; error = cumulative RMS up to the lead time, in units of the held-out RMS.
# Printed per fit: final training loss, held-out rollout error at 1 s and 3 s, learned period / stride period and
# multiplier of the attractor; then which fit each criterion would pick per trial.
# Finding (2026-11-03): rollout error flags some wrong-rhythm fits that training loss does not (trial 231 seed 3:
# 1.05 at 3 s against 0.24–0.40), but on trial 91 the lowest 3-s rollout error picks a fit that settles onto a cycle
# 15× the stride period: short-horizon skill does not guarantee the right long-run attractor.
# One-step (teacher-forced) validation error needs the fitted weights, which the caches do not store.
#
# Usage: julia --project=. scripts/gait_overparam_validation.jl

using Serialization, Statistics, Printf

const CACHE = joinpath(dirname(@__DIR__), "cache")
fs = sort(filter(x -> occursin(r"^gait_overparam_v2_t\d+_s\d+\.jls$", x), readdir(CACHE)))
rows = sort([deserialize(joinpath(CACHE, f)) for f in fs], by = r -> (r.trial, r.seed))
at(r, t) = (n = round(Int, t / r.dt); length(r.err_t) >= n ? r.err_t[n] : NaN)

@printf("%-6s %-4s %4s  %10s  %9s %9s  %8s %7s  %s\n", "trial", "grp", "seed", "train loss", "held 1 s", "held 3 s",
        "P/Pstr", "|μ|", "attractor")
for r in rows
    @printf("%-6d %-4s %4d  %10.4f  %9.3f %9.3f  %8.2f %7.3f  %s\n", r.trial, r.group, r.seed, r.loss, at(r, 1.0),
            at(r, 3.0), r.period_ratio, r.μ, r.kind)
end
println()
for t in unique(r.trial for r in rows)
    R = [r for r in rows if r.trial == t]
    a = R[argmin([r.loss for r in R])]; b = R[argmin(replace([at(r, 3.0) for r in R], NaN => Inf))]
    @printf("trial %d: lowest training loss → P/Pstr %.2f, |μ| %.2f;  lowest held-out 3-s error → P/Pstr %.2f, |μ| %.2f\n",
            t, a.period_ratio, a.μ, b.period_ratio, b.μ)
end
