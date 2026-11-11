#!/usr/bin/env julia
# Summary of human_ltp_lowdim.jl: per-trial Fourier-in-phase operators at low rank, two phase estimates.
#   running  8-stride chunks: skill on held-out strides of the same trial (R² 12.5% of a stride ahead), and agreement of
#            the chunk's structure with the same setting fitted on 240 strides (leading multiplier; correlation of the
#            log quarter-stride growth profiles); share of the chunk metrics' variance between trials
#   walking  each trial's operator on the same person's other trials (own) and other people's (other); phase gain =
#            the same setting without harmonics; structure agreement between two trials of the same person vs of two
#            people of the same group, at speeds within ×/÷ 1.25
# Usage: julia --project=. scripts/human_ltp_lowdim_summary.jl
using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
using Serialization, Statistics, Printf, Random
const ROOT = dirname(@__DIR__)
const L = deserialize(joinpath(ROOT, "cache", "fig5", "human_ltp_lowdim.jls"))
SETL = L.SETL; ns = length(SETL); k12 = 2
q(v) = (v = filter(isfinite, v); @sprintf("%.3f [%.3f, %.3f]", median(v), quantile(v, 0.25), quantile(v, 0.75)))
share(v, lab) = (u = unique(lab); wt = mean(var(v[lab .== x]) for x in u); bt = var([mean(v[lab .== x]) for x in u]); bt / (bt + wt))
noH(k) = findfirst(==((SETL[k][1], SETL[k][2], 0, SETL[k][4])), SETL)

for how in (:hip, :pair)
    RA = [x for x in L.RA if x.how == how]
    println("\n══ phase: $how ══  running, $(length(RA)) trials, 10 chunks of 8 strides each")
    chs = [median(vcat([x.ch_sk[k12, :, k] for x in RA]...)) for k in 1:ns]
    println("  best (d, r, H, λ) per rank by chunk skill: chunk R² 12.5% | 240-stride fit R² | multiplier chunk vs 240 | profile r chunk–240 | between-trial share: multiplier, mean growth")
    for r in unique(getindex.(SETL, 2))
        ks = [k for k in 1:ns if SETL[k][2] == r]; k = ks[argmax(chs[ks])]
        pr = [cor(x.ch_g[:, c, k], x.full_g[:, k]) for x in RA for c in 1:10]
        mc = vcat([x.ch_m[:, k] for x in RA]...); lab = vcat([fill("$(x.subj)_$(x.rep)", 10) for x in RA]...)
        mg = vcat([vec(mean(x.ch_g[:, :, k], dims = 1)) for x in RA]...)
        @printf("  r %2d %-20s %s | %.3f | %.2f vs %.2f | %.2f | %.2f, %.2f\n", r, string(SETL[k]), q(vcat([x.ch_sk[k12, :, k] for x in RA]...)),
                median(x.full_sk[k12, k] for x in RA), median(mc), median(x.full_m[k] for x in RA), median(filter(isfinite, pr)), share(mc, lab), share(mg, lab))
    end
    kf = argmax([median(x.full_sk[k12, k] for x in RA) for k in 1:ns])
    @printf("  best with 240 strides: %s R² %.3f\n", string(SETL[kf]), median(x.full_sk[k12, kf] for x in RA))

    RB = L.RB[how]
    ok = [x for x in RB if x.nown > 0]
    println("  walking, $(length(ok)) trials with another trial of the same person at similar speed")
    mo = [median(x.own_sk[k12, k] for x in ok) for k in 1:ns]
    println("  best (d, r, H, λ) per rank by own-trial skill: own R² 12.5% | without phase | other people")
    for r in unique(getindex.(SETL, 2))
        ks = [k for k in 1:ns if SETL[k][2] == r]; k = ks[argmax(mo[ks])]
        @printf("  r %2d %-20s %s | %.3f | %.3f\n", r, string(SETL[k]), q([x.own_sk[k12, k] for x in ok]),
                median(x.own_sk[k12, noH(k)] for x in ok), median(x.oth_sk[k12, k] for x in ok if x.noth > 0))
    end
    k = argmax(mo); k0 = noH(k)
    @printf("  best overall %s; by group (person means): own with / without phase | other people with phase | own − other [95%%]\n", string(SETL[k]))
    for g in ("AB", "HF", "LF")
        x = [y for y in ok if y.group == g && y.noth > 0]; ps = unique(getfield.(x, :subject))
        pm(f) = [mean(f(y) for y in x if y.subject == p) for p in ps]
        w1, w0, b1 = pm(y -> y.own_sk[k12, k]), pm(y -> y.own_sk[k12, k0]), pm(y -> y.oth_sk[k12, k])
        rng = MersenneTwister(4); bs = [median(rand(rng, w1 .- b1, length(ps))) for _ in 1:4000]
        @printf("    %s (%2d people, %3d trials): %.3f / %.3f | %.3f | %+.3f [%+.3f, %+.3f]\n", g, length(ps), length(x), median(w1), median(w0),
                median(b1), median(w1 .- b1), quantile(bs, 0.025), quantile(bs, 0.975))
    end
    near(a, b) = max(a.speed, b.speed) / min(a.speed, b.speed) <= 1.25
    pairs_same = [(a, b) for (ia, a) in enumerate(RB) for b in RB[ia+1:end] if a.subject == b.subject && near(a, b)]
    pairs_diff = [(a, b) for (ia, a) in enumerate(RB) for b in RB[ia+1:end] if a.subject != b.subject && a.group == b.group && near(a, b)]
    for (lab, kk) in (("best", k), ("best rank ≤ 4", argmax([SETL[j][2] <= 4 ? mo[j] : -Inf for j in 1:ns])))
        @printf("  structure agreement at %s %s: profile r same person %s vs different people %s | |Δ multiplier| same %.3f vs different %.3f | multiplier %s\n",
                lab, string(SETL[kk]), q([cor(a.g[:, kk], b.g[:, kk]) for (a, b) in pairs_same]), q([cor(a.g[:, kk], b.g[:, kk]) for (a, b) in pairs_diff]),
                median(abs(a.m[kk] - b.m[kk]) for (a, b) in pairs_same), median(abs(a.m[kk] - b.m[kk]) for (a, b) in pairs_diff), q([x.m[kk] for x in RB]))
    end
end
