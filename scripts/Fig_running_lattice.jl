#!/usr/bin/env julia
# The lattice-constrained Floquet model on long treadmill running trials (Maus et al. 2015 data; maus_lattice.jl).
#
#   (A) One runner's right-leg joint angles over the stride: mean cycle and ± 2 SD of the ~320 strides (stride-
#       normalised, hip phase). Deviations are ~1% of the signal variance.
#   (B) Is a multiplier identified? Normalised forecast error over strides +1 and +2 against the assumed modulus
#       (best rotation at each), median and IQR over trials: running joint angles and CoM (61-stride moving-average
#       baseline), human walking (companion superset) and kicked SLIP walks (gait_lattice.jl; truth 0.769).
#   (C) R² of the deviation by horizon, model fitted on the first half of each trial and scored on the second:
#       lattice (μ from the person's other trials), deviation DMD (rank likewise), DDFA least-squares return map
#       (section deviation, mean over 4 sections; strides only), and the lattice with μ chosen on the scored strides,
#       real order (dashed) and stride-shuffled (dotted). Median and IQR over trials.
#   (D) Multiplier modulus per stride, per trial: lattice μ chosen within the trial (inner split of the training
#       half) and the DDFA least-squares eigenvalue (mean over sections); bar: the person's μ (other trials).
#
# Usage: julia --project=. scripts/Fig_running_lattice.jl      (writes figures/Fig_running_lattice.*)

include("fig5_gait_data.jl")
include("paper_style.jl")

plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const L = deserialize(joinpath(ROOT, "cache", "maus", "lattice.jls"))
const G = deserialize(joinpath(ROOT, "cache", "fig5", "gait_lattice.jls"))
const A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
const RH = 0.05:0.05:0.95
const SECS = (1, 26, 51, 76)
crit(r) = (r.prof[:, :, 2] .+ r.prof[:, :, 3]) ./ (r.ref[2] + r.ref[3])
q3(v) = quantile(v, (0.25, 0.5, 0.75))

fig = Figure(size = (COL2, 118MM))
top = fig[1, 1] = GridLayout(); bot = fig[2, 1] = GridLayout()

# ── A: mean cycle and stride spread ──────────────────────────────────────────
t = A.trials[1]; S = strides(t.X); ph = range(0, 100; length = NPH)
gA = top[1, 1] = GridLayout()
for (k, (c, nm)) in enumerate(((1, "hip (thigh)"), (2, "knee"), (3, "ankle")))
    ax = Axis(gA[k, 1]; ylabel = nm * " (°)", xticklabelsvisible = k == 3, xlabel = k == 3 ? "Stride (%)" : "",
              title = k == 1 ? @sprintf("Running, subject %d: %d strides", t.subj, size(S, 1)) : "", limits = ((0, 100), nothing))
    m = vec(mean(S[:, :, c], dims = 1)); s = vec(std(S[:, :, c], dims = 1))
    band!(ax, ph, m .- 2s, m .+ 2s; color = (CAT[1], 0.25)); lines!(ax, ph, m; color = C_TRUTH, linewidth = 1.0)
end
rowgap!(gA, 2); plabel!(gA[1, 1], "A")

# ── B: forecast error against the assumed modulus ────────────────────────────
aB = Axis(top[1, 2]; xlabel = "Assumed multiplier modulus per stride", ylabel = "Forecast error, strides +1, +2\n(1 = mean stride; median, IQR)",
          title = "Only SLIP walks pick a multiplier", limits = (0, 1, 0.45, 1.3), xticks = 0:0.25:1)
hlines!(aB, 1.0; color = C_REF, linewidth = 0.6); vlines!(aB, 0.769; color = C_TRUTH, linewidth = 0.6, linestyle = :dot)
text!(aB, 0.76, 0.47; text = "SLIP truth", fontsize = 6.5, align = (:right, :bottom))
LINES = [("SLIP kicked walks", [crit(r) for r in G.slip if r !== nothing], C_TRUTH, :solid),
         ("walking (AB, 15 s)", [crit(r) for (r, g) in zip(G.gait, G.grp) if r !== nothing && g == "AB"], C_REF, :solid),
         ("running, joint angles", [crit(r.main) for r in L.res[(:angles, :ma)].R], CAT[1], :solid),
         ("running, CoM", [crit(r.main) for r in L.res[(:com, :ma)].R], CAT[2], :dash)]
for (k, (lab, Cs, col, ls)) in enumerate(LINES)
    P = reduce(hcat, [vec(minimum(C, dims = 2)) for C in Cs]); Q = reduce(vcat, [collect(q3(r))' for r in eachrow(P)])
    band!(aB, RH, clamp.(Q[:, 1], 0.45, 1.3), clamp.(Q[:, 3], 0.45, 1.3); color = (col, 0.12))
    lines!(aB, RH, clamp.(Q[:, 2], 0.45, 1.3); color = col, linewidth = 1.2, linestyle = ls)
    lines!(aB, [0.04, 0.1], fill(0.66 - 0.05 * (k - 1), 2); color = col, linewidth = 1.2, linestyle = ls)
    text!(aB, 0.12, 0.66 - 0.05 * (k - 1); text = lab, fontsize = 6.5, align = (:left, :center))
end
plabel!(top[1, 2], "B")
colsize!(top, 1, Relative(0.4)); colgap!(top, 18)

# ── C: skill by horizon ──────────────────────────────────────────────────────
gC = bot[1, 1] = GridLayout()
for (k, (st, ttl)) in enumerate(((:angles, "Running, joint angles"), (:com, "Running, centre of mass")))
    Rs = L.res[(st, :ma)]; H = Rs.H
    ax = Axis(gC[1, k]; title = ttl, ylabel = k == 1 ? "R², deviation (0 = moving-average stride)" : "",
              xticks = (1:3, ["rest of\nstride", "+1\nstride", "+2\nstrides"]), limits = (0.5, 3.5, -0.12, 0.5))
    hlines!(ax, 0.0; color = C_REF, linewidth = 0.6)
    series = [("lattice", [h.person.lat for h in H], CAT[5], :diamond, -0.15, :solid),
              ("deviation DMD", [h.person.ddmd for h in H], CAT[1], :circle, 0.0, :solid),
              ("DDFA return map", [[NaN, mean(r.ddfa[φ].ls.r1 for φ in SECS), mean(r.ddfa[φ].ls.r2 for φ in SECS)] for r in Rs.R], CAT[3], :rect, 0.15, :solid)]
    for (lab, V, col, mk, dx, _) in series
        for b in 1:3
            v = filter(isfinite, [x[b] for x in V]); isempty(v) && continue
            q = q3(v); lines!(ax, [b + dx, b + dx], clamp.([q[1], q[3]], -0.12, 0.5); color = col, linewidth = 1.1)
            scatter!(ax, [b + dx], [clamp(q[2], -0.115, 0.5)]; color = col, marker = mk, markersize = 5.5, strokecolor = :white, strokewidth = 0.4)
        end
    end
    lines!(ax, 1:3, [median(h.ontest.lat[b] for h in H) for b in 1:3]; color = CAT[5], linewidth = 0.8, linestyle = :dash)
    lines!(ax, 1:3, [median(h.ontest.null[b] for h in H) for b in 1:3]; color = CAT[5], linewidth = 0.9, linestyle = :dot)
    if k == 1
        for (j, (lab, _, col, mk, _, _)) in enumerate(series)
            scatter!(ax, [1.9], [0.47 - 0.045 * (j - 1)]; color = col, marker = mk, markersize = 5)
            text!(ax, 2.0, 0.47 - 0.045 * (j - 1); text = lab, fontsize = 6.5, align = (:left, :center))
        end
        for (j, (ls, lab)) in enumerate(((:dash, "lattice, μ on scored strides"), (:dot, "same, strides shuffled")))
            y = 0.335 - 0.045 * (j - 1); lines!(ax, [1.82, 1.97], [y, y]; color = CAT[5], linewidth = 0.9, linestyle = ls)
            text!(ax, 2.0, y; text = lab, fontsize = 6.5, align = (:left, :center))
        end
    end
end
colgap!(gC, 8); plabel!(gC[1, 1], "C")

# ── D: multipliers per trial ─────────────────────────────────────────────────
us = unique(L.subs)
aD = Axis(bot[1, 2]; xticks = (1:length(us), ["subj $u" for u in us]), ylabel = "Multiplier modulus per stride",
          title = "Per-trial estimates (joint angles)", limits = (0.4, length(us) + 0.6, 0, 1))
Rs = L.res[(:angles, :ma)]
for (i, u) in enumerate(us)
    I = findall(L.subs .== u); μ = Rs.μp[u][1]
    lines!(aD, [i - 0.3, i + 0.3], [μ, μ]; color = CAT[5], linewidth = 2.0)
    scatter!(aD, i .- 0.12 .+ 0.05 .* randn(MersenneTwister(i), length(I)), [Rs.H[j].inner.ρ for j in I]; color = CAT[5], marker = :diamond, markersize = 5)
    scatter!(aD, i .+ 0.12 .+ 0.05 .* randn(MersenneTwister(10i), length(I)), [mean(Rs.R[j].ddfa[φ].ls.ρ for φ in SECS) for j in I]; color = CAT[3], marker = :rect, markersize = 5)
end
text!(aD, 0.5, 0.97; text = "◆ lattice, within trial   ■ DDFA least squares\nbar: person's μ from other trials", fontsize = 6.5, align = (:left, :top))
plabel!(bot[1, 2], "D")
colsize!(bot, 1, Relative(0.62)); colgap!(bot, 18); rowgap!(fig.layout, 12)
save_paper(fig, "Fig_running_lattice")
