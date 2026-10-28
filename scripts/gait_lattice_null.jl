#!/usr/bin/env julia
# Stride-shuffle null for the lattice model's per-trial skill (Fig S6F; OUTLINE.md §6 ¶6).
#
# Out-of-trial selection takes μ from the person's other trials, which are at other speeds. The null needs no such
# choice: each trial's strides are put in random order (NSH permutations) and μ is chosen on the scored strides
# themselves, the most favourable rule there is. Shuffling keeps the mean stride and the spread of the deviations and
# removes any carry-over from one stride to the next, so whatever skill survives is selection over the μ grid.
# Compared with the same on-test choice on the real stride order (gait_lattice.jl, `insample`).
# Result (2026-10-28): kicked SLIP walks 0.57 → 0.05 one stride ahead; gait 0.07 / 0.16 / 0.15 → 0.04 / 0.08 / 0.08
# (AB / HF / LF), so about half of the on-test skill is selection (paired test: gait_lattice_speed.jl).
#
# Usage: julia --project=. -t 12 scripts/gait_lattice_null.jl      (writes cache/fig5/gait_lattice_null.jls)

include("gait_lattice.jl")
using Random

const NSH = 3

"Block errors of every grid μ on one trial, lattice model only."
function lattice_profiles(S; Ks = KS, W = W_WIN)
    n, _, p = size(S); _, _, D = deviations(S); nh = n ÷ 2
    tr = segments(D, 0, nh); te = segments(D, nh, n)
    (length(tr) < 2 || length(te) < 1) && return nothing
    ref = sum(blocks(zeros(T_SEG - W, p), S_) for S_ in te)
    prof = zeros(length(RHOS), length(ROTS), 3)
    for (i, ρ) in enumerate(RHOS), (j, rot) in enumerate(ROTS)
        Mm = fit_mode(tr, μof(ρ, rot); Ks)
        prof[i, j, :] = sum(blocks(predict_mode(Mm, S_; W)[W+1:end, :], S_) for S_ in te)
    end
    (; prof, ref)
end
ontest(r) = (io = argmin(crit(r)); [1 - r.prof[io, b] / r.ref[b] for b in 1:3])

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    G = deserialize(joinpath(ROOT, "cache", "fig5", "gait_lattice.jls"))
    K = deserialize(joinpath(ROOT, "cache", "truth", "slip_kicked_walks.jls"))
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    idx = findall(!isnothing, G.gait)
    jobs = vcat([(:slip, w, s) for w in eachindex(K.walks) for s in 1:NSH], [(:gait, i, s) for i in idx for s in 1:NSH])
    res = tmap(jobs) do (kind, i, s)
        S = kind == :slip ? K.walks[i] : strides(data[i, :, :])
        r = lattice_profiles(S[randperm(MersenneTwister(1000i + s), size(S, 1)), :, :]); r === nothing ? nothing : ontest(r)
    end
    pick(kind, keep) = [res[k] for k in eachindex(jobs) if jobs[k][1] == kind && keep(jobs[k][2]) && res[k] !== nothing]
    slip = pick(:slip, _ -> true); gait = Dict(g => pick(:gait, i -> G.grp[i] == g) for g in ("AB", "HF", "LF"))
    real = Dict("SLIP" => [h.insample for h in G.slip_held if h !== nothing],
                (g => [h.insample for h in G.held[G.grp .== g] if h !== nothing] for g in ("AB", "HF", "LF"))...)
    for (g, sh) in (("SLIP", slip), ("AB", gait["AB"]), ("HF", gait["HF"]), ("LF", gait["LF"]))
        @printf("%-5s μ on scored strides, R² rest / +1 / +2:  real %s   shuffled %s\n", g,
                join((@sprintf("%.2f", median(v[b] for v in real[g])) for b in 1:3), " / "),
                join((@sprintf("%.2f", median(v[b] for v in sh)) for b in 1:3), " / "))
    end
    serialize(joinpath(ROOT, "cache", "fig5", "gait_lattice_null.jls"), (; slip, gait, nsh = NSH))
end
