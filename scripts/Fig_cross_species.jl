#!/usr/bin/env julia
# Cross-species operator metrics against locomotor speed: rat pups, two mouse datasets, human walking, fly walking.
#
#   (A) The fly tracks hold no stride rhythm (fly_spectra.jl): walking-window spectra of fly forward velocity and
#       heading rate fall monotonically to the noise floor through the 5–20 Hz stepping band (median over 29 strains;
#       grey lines: single strains); the running hip angle (maus_extract.jl) shows its stride and harmonics.
#   (B) Speed effects at each dataset's own deck settings (cross_species_summary.jl): share of units lower at the
#       faster speed, 95% bootstrap interval; filled = stride time (25 samples per stride), open = real time.
#   (C) Henrici's stride-time speed effect across DMD rank and ridge (cross_species_robustness.jl), humans
#       (slowest vs fastest trial) and Cntnap2 / L7-Tsc1 mice (slower vs faster half); the mean-cycle surrogate at
#       the deck setting is given beside each grid. Box: the deck setting used in B.
#   (D) Fly strains: Henrici of one fit per fly (all walking), residual after a linear fit on walking speed,
#       strain mean ± SE, by species group.
#
# Usage: julia --project=. scripts/Fig_cross_species.jl      (writes figures/Fig_cross_species.*)

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using CSV, DataFrames, Serialization, Statistics, Printf

const ROOT = dirname(@__DIR__)
const CS = joinpath(ROOT, "cache", "cross_species")
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const SP = deserialize(joinpath(ROOT, "cache", "fly", "spectra.jls"))
const SU = deserialize(joinpath(CS, "summary.jls"))
const RH = CSV.read(joinpath(CS, "robust_human.csv"), DataFrame)
const RC = CSV.read(joinpath(CS, "robust_cohort.csv"), DataFrame)

fig = Figure(size = (COL2, 132MM))
top = fig[1, 1] = GridLayout(); bot = fig[2, 1] = GridLayout()

# ── A: no stride rhythm in the fly tracks ────────────────────────────────────
db(P) = 10 .* log10.(max.(P, 1e-14))
aA = Axis(top[1, 1]; xscale = log10, xlabel = "Frequency (Hz)", ylabel = "Power (dB re total)",
          title = "Fly tracks: no stepping peak", limits = (0.5, 50, -105, 0), xticks = [0.5, 1, 2, 5, 10, 20, 50],
          xtickformat = v -> [x < 1 ? string(x) : string(round(Int, x)) for x in v])
vspan!(aA, 5, 20; color = C_SHADE)
text!(aA, 9.5, -2; text = "fly stepping\nband", fontsize = 6.5, color = C_REF, align = (:center, :top))
f = SP.f; sel = f .>= 0.5
for (s, v) in SP.flies; lines!(aA, f[sel], db(v.P[:forward])[sel]; color = (C_REF, 0.25), linewidth = 0.4); end
med(k) = [median(db(v.P[k])[i] for v in values(SP.flies)) for i in eachindex(f)]
lines!(aA, f[sel], med(:forward)[sel]; color = CAT[5], linewidth = 1.4)
lines!(aA, f[sel], med(:heading_rate)[sel]; color = CAT[5], linewidth = 1.1, linestyle = :dash)
fr = SP.running.f; sr = (fr .>= 0.5) .& (fr .<= 50)
lines!(aA, fr[sr], db(SP.running.P)[sr]; color = C_TRUTH, linewidth = 1.0)
text!(aA, 13, -39; text = "running\nhip angle", fontsize = 6.5, align = (:left, :bottom))
text!(aA, 0.55, -66; text = "fly forward velocity\n(dashed: heading rate)", fontsize = 6.5, color = C_TRUTH, align = (:left, :top))
plabel!(top[1, 1], "A")

# ── B: speed effects at the deck settings ────────────────────────────────────
DS = ["rat pups (development)", "mice, Cntnap2/L7-Tsc1", "mice, wheel", "human walking", "fly walking (body velocity)"]
DSL = ["rat pups", "mice, open field", "mice, wheel", "human", "fly"]
MB = [(:participation_ratio, "Participation ratio"), (:henrici, "Henrici departure"), (:log_tg_stride, "Transient growth"),
      (:diagonal_dominance, "Self-dominance")]
gB = top[1, 2] = GridLayout()
for (k, (m, ttl)) in enumerate(MB)
    ax = Axis(gB[1, k]; title = ttl, xticks = (1:5, DSL), limits = (0.4, 5.6, -0.03, 1.03), yticks = 0:0.25:1,
              ylabel = k == 1 ? "Share of units lower\nat the faster speed" : "", yticklabelsvisible = k == 1, xticklabelsize = 6.5, xticklabelrotation = π / 3)
    hlines!(ax, 0.5; color = C_REF, linewidth = 0.5, linestyle = :dash)
    for (time, dx, filled) in ((:stride, -0.13, true), (:real, 0.13, false))
        for (i, d) in enumerate(DS)
            r = [t for t in SU.S[time].tab if t.dataset == d && t.metric == m]; isempty(r) && continue; r = r[1]
            lines!(ax, [i + dx, i + dx], [r.lo, r.hi]; color = filled ? C_TRUTH : C_REF, linewidth = 0.9)
            scatter!(ax, [i + dx], [r.share]; color = filled ? C_TRUTH : :white, strokecolor = filled ? C_TRUTH : C_REF,
                     strokewidth = 0.9, markersize = 5)
        end
    end
    k == 1 && (scatter!(ax, [0.75], [0.12]; color = C_TRUTH, markersize = 5); text!(ax, 0.9, 0.12; text = "stride time", fontsize = 6.5, align = (:left, :center));
               scatter!(ax, [0.75], [0.04]; color = :white, strokecolor = C_REF, strokewidth = 0.9, markersize = 5);
               text!(ax, 0.9, 0.04; text = "real time", fontsize = 6.5, align = (:left, :center)))
end
colgap!(gB, 6); plabel!(gB[1, 1], "B")
colsize!(top, 1, Relative(0.3)); colgap!(top, 14)

# ── C: Henrici's stride-time speed effect across rank and ridge ──────────────
DIV = cgrad([CAT[2], colorant"#d9d8d4", CAT[1]], [0.0, 0.5, 1.0])     # orange = higher when faster, blue = lower
ridgelab(a) = a == 0 ? "0" : @sprintf("10^{%d}", round(Int, log10(a)))
rlabels = ["0", "1e-6", "1e-4", "1e-2"]
function grid_share(df, ranks, unitlevels)
    G = fill(NaN, length(ranks), 4)
    for (i, r) in enumerate(ranks), (j, a) in enumerate([0.0, 1e-6, 1e-4, 1e-2])
        x = df[(df.kind .== "real") .& (df.rank .== r) .& (df.ridge .== a), :]; G[i, j] = unitlevels(x)
    end
    G
end
hum_low(x) = (g = combine(groupby(x, [:subject, :speed]), :henrici => mean => :y);
              mean((y = g.y[g.subject .== s]; sp = g.speed[g.subject .== s]; y[argmax(sp)] < y[argmin(sp)]) for s in unique(g.subject)))
coh_low(x) = (g = combine(groupby(x, [:subject, :level]), :henrici => mean => :y);
              mean(g.y[(g.subject .== s) .& (g.level .== "fast")][1] < g.y[(g.subject .== s) .& (g.level .== "slow")][1]
                   for s in unique(g.subject) if count(g.subject .== s) == 2))
gC = bot[1, 1] = GridLayout()
for (k, (lab, df, ranks, f, deck)) in enumerate((("Human walking", RH, [10, 20, 40, 80, 120], hum_low, (5, 3)),
                                                    ("Mice, Cntnap2/L7-Tsc1", RC, [20, 40, 160], coh_low, (3, 3))))
    G = grid_share(df, ranks, f)
    surr = f(df[df.kind .== "meancycle", :])
    ax = Axis(gC[1, k]; title = lab, xticks = (1:4, rlabels), yticks = (1:length(ranks), string.(ranks)),
              xlabel = "Ridge (× s₁²)", ylabel = k == 1 ? "DMD rank" : "", limits = (0.5, 4.5, 0.5, length(ranks) + 0.5))
    heatmap!(ax, 1:4, 1:length(ranks), permutedims(G); colormap = DIV, colorrange = (0, 1))
    for i in eachindex(ranks), j in 1:4
        text!(ax, j, i; text = @sprintf("%.0f", 100G[i, j]), fontsize = 6.5, align = (:center, :center), color = C_TRUTH)
    end
    poly!(ax, Rect(deck[2] - 0.5, deck[1] - 0.5, 1, 1); color = :transparent, strokecolor = C_TRUTH, strokewidth = 1.2)
    Label(gC[2, k], @sprintf("mean cycle only (boxed setting): %.0f%%", 100surr); fontsize = 6.5, color = C_TRUTH, tellwidth = false)
end
Colorbar(gC[1, 3]; colormap = DIV, colorrange = (0, 100), label = "% lower when faster", ticks = 0:25:100, height = Relative(0.85))
rowgap!(gC, 2); colgap!(gC, 8); plabel!(gC[1, 1], "C")

# ── D: fly strains, speed-adjusted Henrici ───────────────────────────────────
fl = SU.fly; X = hcat(ones(nrow(fl)), fl.speed); res = fl.henrici .- X * (X \ fl.henrici)
MELSUB = Set(["melanogaster", "simulans", "sechellia", "mauritiana", "yakuba", "santomea", "teissieri", "erecta"])
OTHER = Set(["pseudoobscura", "persimilis", "willistoni", "virilis", "arizonae"])
grp(s) = s in MELSUB ? 2 : s in OTHER ? 3 : 1
GL = ["lines la, z, zh", "mel. subgroup", "other groups"]
st = [(s, mean(res[fl.strain .== s]), count(fl.strain .== s) > 1 ? std(res[fl.strain .== s]) / sqrt(count(fl.strain .== s)) : NaN, count(fl.strain .== s))
      for s in unique(fl.strain)]
st = sort(filter(x -> x[4] >= 3, st), by = x -> x[2])
aD = Axis(bot[1, 2]; yticks = (1:length(st), [x[1] for x in st]), yticklabelsize = 5.5, xlabel = "Henrici, residual on walking speed",
          title = "Fly strains (body-velocity fits)", limits = (-0.03, 0.03, 0.3, length(st) + 0.7), xticks = -0.02:0.01:0.02)
vlines!(aD, 0.0; color = C_REF, linewidth = 0.5)
for (i, x) in enumerate(st)
    c = CAT[grp(x[1])]
    lines!(aD, [x[2] - x[3], x[2] + x[3]], [i, i]; color = c, linewidth = 0.9)
    scatter!(aD, [x[2]], [i]; color = c, markersize = 4.5, marker = (:circle, :rect, :diamond)[grp(x[1])])
end
for (k, l) in enumerate(GL)
    scatter!(aD, [0.009], [16 - 1.4k]; color = CAT[k], marker = (:circle, :rect, :diamond)[k], markersize = 4.5)
    text!(aD, 0.0105, 16 - 1.4k; text = l, fontsize = 6.5, align = (:left, :center))
end
plabel!(bot[1, 2], "D")
colsize!(bot, 1, Relative(0.62)); colgap!(bot, 18); rowgap!(fig.layout, 12); rowsize!(fig.layout, 1, Relative(0.47))
save_paper(fig, "Fig_cross_species")
