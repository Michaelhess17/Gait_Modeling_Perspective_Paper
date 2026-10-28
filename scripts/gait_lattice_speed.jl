#!/usr/bin/env julia
# Does choosing μ from trials at other speeds hide a speed-dependent multiplier? (OUTLINE.md §6 ¶6; reads
# cache/fig5/gait_lattice.jls, seconds.) A person's trials are each at a different speed (AB span about 2×), so the
# out-of-trial rule in gait_lattice.jl assumes the multiplier does not change with speed. Rules compared, R² of the
# deviation 1 and 2 strides ahead, median [IQR] over trials:
#   person, other speeds         gait_lattice.jl's rule (Fig S6D)
#   person, nearest speed        the person's one trial closest in speed
#   group, 10 nearest speeds     other people of the same group, the 10 trials closest in speed
#   on the scored strides        the ceiling for any rule at the trial's own speed (selection included; see
#                                gait_lattice_null.jl for the same with the strides shuffled)
# Also: within each person, the rank correlation of each trial's own best modulus with its speed; and, with
# cache/fig5/gait_lattice_null.jls, the on-test skill paired with the same on that trial's stride-shuffled copies
# (people bootstrap), split by whether the trial's own best modulus is near-neutral (≥ 0.8).
#
# Usage: julia --project=. scripts/gait_lattice_speed.jl

using Serialization, Statistics, Printf, Random

const RHOS = 0.05:0.05:0.95
crit(r) = (r.prof[:, :, 2] .+ r.prof[:, :, 3]) ./ (r.ref[2] + r.ref[3])
skill(r, ij) = [1 - r.prof[ij, b] / r.ref[b] for b in 2:3]
choose(R, js) = argmin(sum(crit(R[j]) for j in js))
rk(v) = invperm(sortperm(v)) .* 1.0
mi(v) = @sprintf("%.2f [%.2f, %.2f]", median(v), quantile(v, 0.25), quantile(v, 0.75))

C = deserialize(joinpath(dirname(@__DIR__), "cache", "fig5", "gait_lattice.jls"))
NF = joinpath(dirname(@__DIR__), "cache", "fig5", "gait_lattice_null.jls"); N = isfile(NF) ? deserialize(NF) : nothing
R, subs, grp, sp = C.gait, C.subs, C.grp, C.speed
ok = findall(!isnothing, R); dsp(j, i) = abs(log(sp[j] / sp[i]))
for G in ("AB", "HF", "LF")
    I = [i for i in ok if grp[i] == G]; out = Dict{String, Vector{Vector{Float64}}}()
    for i in I
        own = [j for j in ok if j != i && subs[j] == subs[i]]; oth = [j for j in ok if subs[j] != subs[i] && grp[j] == G]
        rules = ["person, other speeds" => own, "person, nearest speed" => (isempty(own) ? own : [own[argmin(dsp.(own, i))]]),
                 "group, 10 nearest speeds" => oth[sortperm(dsp.(oth, i))[1:min(10, end)]], "on the scored strides" => [i]]
        for (k, js) in rules; isempty(js) || push!(get!(out, k, Vector{Float64}[]), skill(R[i], choose(R, js))); end
    end
    @printf("%s (%d trials)\n", G, length(I))
    for k in ("person, other speeds", "person, nearest speed", "group, 10 nearest speeds", "on the scored strides")
        @printf("   %-26s R² +1 %s   +2 %s\n", k, mi(first.(out[k])), mi(last.(out[k])))
    end
    ρ = [RHOS[choose(R, [i])[1]] for i in I]
    cs = [cor(rk(sp[I][J]), rk(ρ[J] .+ 1e-9 .* (1:length(J)))) for s in unique(subs[I]) for J in (findall(subs[I] .== s),) if length(J) >= 3]
    @printf("   own best modulus vs speed within a person: rank correlation %s over %d people\n", mi(cs), length(cs))
    N === nothing && continue
    sh = N.gait[G]; @assert length(sh) == N.nsh * length(I)
    d = [C.held[i].insample[2] - mean(v[2] for v in sh[N.nsh*(k-1)+1:N.nsh*k]) for (k, i) in enumerate(I)]
    s = subs[I]; us = unique(s); rng = MersenneTwister(1)
    bs = [median(reduce(vcat, [d[s .== u] for u in rand(rng, us, length(us))])) for _ in 1:2000]; hi = ρ .>= 0.8
    @printf("   on-test skill, real − stride-shuffled, R² +1: %.3f (people bootstrap 95%% %.3f to %.3f); own best ρ ≥ 0.8: %.3f (n %d), < 0.8: %.3f (n %d)\n",
            median(d), quantile(bs, 0.025), quantile(bs, 0.975), median(d[hi]), count(hi), median(d[.!hi]), count(.!hi))
end
