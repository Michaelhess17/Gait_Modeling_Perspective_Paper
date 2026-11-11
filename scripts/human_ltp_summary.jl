#!/usr/bin/env julia
# Summary of human_ltp.jl: reliability of the Fourier-in-phase operator's structure at 8 strides (running chunks), and
# group / speed effects in the walking superset.
#   reliability  running: share of a metric's variance that lies between trials (chunks of the same trial agree) and
#                between people; growth profiles: correlation of two chunks of a trial against two different people
#   groups       medians at matched speed (trials 40–80 cm/s, person means), person-bootstrap 95% intervals of the
#                group differences; OLS of each metric on group + log speed (person-cluster bootstrap)
#   speed        within-person Spearman ρ with speed (people with ≥ 4 trials)
#   profiles     log quarter-stride growth by phase, group means (person means first), and limb asymmetry
#                log(growth into limb 1 / into limb 2) averaged over phase (limb 1 = paretic in stroke)
# Usage: julia --project=. scripts/human_ltp_summary.jl

using Serialization, Statistics, Printf, Random, LinearAlgebra
const ROOT = dirname(@__DIR__)
const C = deserialize(joinpath(ROOT, "cache", "fig5", "human_ltp.jls"))
rk(v) = invperm(sortperm(v)) .* 1.0
mi(v) = @sprintf("%.3f [%.3f, %.3f]", median(v), quantile(v, 0.25), quantile(v, 0.75))
ok = findall(!isnothing, C.human); H = C.human[ok]; sub, grp, spd = C.subject[ok], C.group[ok], C.speed[ok]
@printf("setting (d, r, Hh, ridge) %s; without phase %s; %d human trials, %d people\n", string(C.setting), string(C.setting0), length(H), length(unique(sub)))
asym(h) = mean(log.(h.growth[:, 2] ./ h.growth[:, 3]))
lg(h) = log.(h.growth[:, 1])
METRICS = [("leading multiplier", h -> h.mult1), ("2nd multiplier", h -> h.mult2), ("skill 12.5%", h -> h.skill),
           ("phase gain 12.5%", h -> h.phase), ("mean log growth", h -> mean(lg(h))), ("growth range (max−min log)", h -> maximum(lg(h)) - minimum(lg(h))),
           ("limb asymmetry", asym)]

# ── reliability on running chunks ────────────────────────────────────────────
Rk = [(c, r) for (c, r) in zip(C.chunks, C.running) if r !== nothing]
println("\nRunning, $(length(Rk)) chunks of 8 strides: share of variance between trials | between people; median over trials")
for (nm, f) in METRICS
    v = [f(r) for (_, r) in Rk]; tr = ["$(c[1])_$(c[2])" for (c, _) in Rk]; pe = [c[1] for (c, _) in Rk]
    wt = mean(var(v[tr .== u]) for u in unique(tr)); bt = var([mean(v[tr .== u]) for u in unique(tr)])
    wp = mean(var(v[pe .== u]) for u in unique(pe)); bp = var([mean(v[pe .== u]) for u in unique(pe)])
    @printf("  %-28s trials %.2f | people %.2f | %s\n", nm, bt / (bt + wt), bp / (bp + wp), mi(v))
end
prof = [lg(r) for (_, r) in Rk]; tr = ["$(c[1])_$(c[2])" for (c, _) in Rk]; pe = [c[1] for (c, _) in Rk]
same = [cor(prof[i], prof[j]) for i in eachindex(prof) for j in i+1:length(prof) if tr[i] == tr[j]]
diff_ = [cor(prof[i], prof[j]) for i in eachindex(prof) for j in i+1:length(prof) if pe[i] != pe[j]]
@printf("  growth profile correlation: same trial %s | different people %s\n", mi(same), mi(diff_))

# ── groups and speed ─────────────────────────────────────────────────────────
G3 = ("AB", "HF", "LF")
println("\nWalking, trials 40–80 cm/s, person means: median by group; difference to AB with person-bootstrap 95%")
win = findall((spd .>= 40) .& (spd .<= 80))
for (nm, f) in METRICS
    v = [f(h) for h in H]
    pm = Dict(g => [mean(v[i] for i in win if sub[i] == s) for s in unique(sub[win]) if grp[findfirst(==(s), sub)] == g] for g in G3)
    rng = MersenneTwister(1)
    dd = [begin
              bs = [median(rand(rng, pm[g], length(pm[g]))) - median(rand(rng, pm["AB"], length(pm["AB"]))) for _ in 1:4000]
              @sprintf("%s−AB %+.3f [%+.3f, %+.3f]", g, median(pm[g]) - median(pm["AB"]), quantile(bs, 0.025), quantile(bs, 0.975))
          end for g in ("HF", "LF")]
    @printf("  %-28s AB %.3f (n%d) HF %.3f (n%d) LF %.3f (n%d) | %s\n", nm, median(pm["AB"]), length(pm["AB"]), median(pm["HF"]), length(pm["HF"]),
            median(pm["LF"]), length(pm["LF"]), join(dd, "  "))
end
println("\nWithin-person Spearman ρ with speed (median; share > 0), by group; OLS slope on log speed and group effects (person bootstrap)")
for (nm, f) in METRICS
    v = [f(h) for h in H]
    line = String[]
    for g in G3
        ρ = [cor(rk(spd[(sub .== s)]), rk(v[(sub .== s)])) for s in unique(sub[grp .== g]) if count(sub .== s) >= 4]
        push!(line, @sprintf("%s %+.2f (%2.0f%% > 0)", g, median(ρ), 100mean(ρ .> 0)))
    end
    X = hcat(ones(length(v)), log.(spd), grp .== "HF", grp .== "LF"); β = X \ v
    us = unique(sub); rng = MersenneTwister(2)
    bs = map(1:2000) do _
        pick = reduce(vcat, [findall(sub .== u) for u in rand(rng, us, length(us))]); X[pick, :] \ v[pick]
    end
    ci(k) = @sprintf("%+.3f [%+.3f, %+.3f]", β[k], quantile([b[k] for b in bs], 0.025), quantile([b[k] for b in bs], 0.975))
    @printf("  %-28s %s | log speed %s  HF %s  LF %s\n", nm, join(line, "  "), ci(2), ci(3), ci(4))
end

# ── profiles ─────────────────────────────────────────────────────────────────
println("\nLog quarter-stride growth by phase (every 10%), group means of person means, trials 40–80 cm/s; phase 0 = limb-1 hip descending zero")
for g in G3
    P = [mean(lg(H[i]) for i in win if sub[i] == s) for s in unique(sub[win]) if grp[findfirst(==(s), sub)] == g]
    m = mean(P); @printf("  %-3s %s\n", g, join([@sprintf("%+.2f", m[k]) for k in 1:10:100], " "))
end
hipmax(i) = argmax(C.mean_cycles[ok[i]][:, 1]) - 1
@printf("  limb-1 hip flexion peak (≈ initial contact) at phase %s %% (median by group: %s)\n", "",
        join([@sprintf("%s %d", g, round(Int, median(hipmax(i) for i in eachindex(H) if grp[i] == g))) for g in G3], ", "))
