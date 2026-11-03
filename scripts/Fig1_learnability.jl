#!/usr/bin/env julia
# Figure 1 — Learnability: on-cycle data identify the cycle, not the dynamics around it (OUTLINE.md §2).
#
#   (A) What the data cover. Coupled FHN (two synchronised units), the same 30-cycle budget recorded as one
#       long trajectory (process noise only) or as 10 decays × 3 cycles from perturbed starts. The slow
#       desynchrony w₁ − w₂ at a Poincaré section decays by |μ| per cycle; on-cycle data barely sample it, so
#       even this one-dimensional regression gets the slope wrong.
#       Data: scripts/fig1_coverage_data.jl → cache/fig1/coverage_cfhn.jls.
#   (B) Held-out forecasts do not pin the dynamics. Over-parameterised neural ODEs fitted to 10 s of real gait
#       (scripts/gait_overparam.jl, 6 seeds × 3 trials): the period of the cycle each learned field settles onto,
#       and the transverse multiplier of that cycle, against the validation error of its autonomous forecast over
#       the held-out 5 s (cumulative RMS / held-out RMS; 1 = no better than forecasting the mean). Trials 36 and 90
#       of the earlier version have no held-out forecast cached and are left out until refitted.
#   (C) With transient data the relaxation becomes identifiable — given enough of it. Per-cycle off-cycle
#       relaxation rate of Hankel-DMD fitted to n decays (scripts/dmd_delay_sweep.jl; 16 settings of delay
#       window × rank per data size, least squares). Grey band: truth.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics, Printf, CSV   # CSV: InlineStrings for the v1 gait caches

const ROOT = dirname(@__DIR__)
cachef(p...) = joinpath(ROOT, "cache", p...)
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0),
                        halign = :right)

fig = Figure(size = (COL2, 172MM))
gA = fig[1, 1] = GridLayout(); gB = fig[2, 1] = GridLayout(); gC = fig[3, 1] = GridLayout()

# ── A: data coverage ─────────────────────────────────────────────────────────
# Desynchrony of the slow variables, w₁ − w₂, read once per cycle at a Poincaré section (upward v₁ crossings):
# its cycle-to-cycle map has slope |μ|. On-cycle data barely move along it, so the slope is poorly determined.
cov = deserialize(cachef("fig1", "coverage_cfhn.jls"))
se, sd = cov.sec_excited, cov.sec_decays
slope(seqs) = (x = reduce(vcat, [q[1:end-1] for q in seqs]); y = reduce(vcat, [q[2:end] for q in seqs]);
               (x' * y) / (x' * x))
axA1 = Axis(gA[1, 1]; xlabel = "Cycle", ylabel = "Desynchrony w₁ − w₂\n(at section)",
            title = "What 30 cycles of data cover")
refline!(axA1, 0.0; linestyle = :solid)
scatterlines!(axA1, 1:length(se), se; color = (C_REF, 0.9), markersize = 3.5, linewidth = 0.8,
              label = "one long trajectory")
for (k, q) in enumerate(sd)
    x0 = 3 * (k - 1)                                         # lay the 3-cycle decays end to end
    scatterlines!(axA1, x0 .+ (1:length(q)), q; color = CAT[1], markersize = 3.5, linewidth = 0.9,
                  label = k == 1 ? "10 decays × 3 cycles" : nothing)
end
axislegend(axA1; position = (:right, :bottom), framevisible = false, labelsize = 7, unique = true)
axA2 = Axis(gA[1, 2]; xlabel = "Desynchrony, cycle k", ylabel = "Desynchrony, cycle k + 1",
            title = "Cycle-to-cycle map", aspect = 1)
lo = minimum(vcat(se, reduce(vcat, sd))) * 1.1; hi = 0.08
xs = [lo, hi]
lines!(axA2, xs, cov.μ_true .* xs; color = C_TRUTH, linewidth = 1.4, label = @sprintf("truth, |μ| = %.2f", cov.μ_true))
scatter!(axA2, se[1:end-1], se[2:end]; color = (C_REF, 0.9), markersize = 4)
lines!(axA2, xs, slope([se]) .* xs; color = C_REF, linestyle = :dash, linewidth = 1.2,
       label = @sprintf("fit, one trajectory: %.2f", slope([se])))
scatter!(axA2, reduce(vcat, [q[1:end-1] for q in sd]), reduce(vcat, [q[2:end] for q in sd]); color = CAT[1], markersize = 4)
lines!(axA2, xs, slope(sd) .* xs; color = CAT[1], linestyle = :dash, linewidth = 1.2,
       label = @sprintf("fit, decays: %.2f", slope(sd)))
limits!(axA2, lo, hi, lo, hi)
axislegend(axA2; position = (:right, :bottom), framevisible = false, labelsize = 7)
plabel!(gA[1, 1], "A")
colsize!(gA, 2, Auto(0.55))
colgap!(gA, 14)

# ── B: held-out forecast error against the learned dynamics (gait neural ODEs) ──
const SEEDS = 1:6; const STRIDE_LIKE = 3.0
runs = NamedTuple[]
for t in (91, 243, 231), s in SEEDS
    f = cachef("gait_overparam_v2_t$(t)_s$(s).jls"); isfile(f) && push!(runs, deserialize(f))
end
heldout(r) = isempty(r.err_t) ? NaN : r.err_t[end]      # validation error over the held-out 5 s
cyc = [r for r in runs if r.kind == :cycle && isfinite(r.period_ratio)]
markers = Dict(91 => :circle, 243 => :diamond, 231 => :utriangle)
axB1 = Axis(gB[1, 1]; xlabel = "Held-out forecast error (5 s)", ylabel = "Learned period / stride period",
            title = "Learned rhythm", yscale = log2,
            yticks = ([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"]))
axB2 = Axis(gB[1, 2]; xlabel = "Held-out forecast error (5 s)", ylabel = "Transverse multiplier |μ|",
            title = "Learned stability")
vlines!(axB1, 1.0; color = C_REF, linewidth = 0.6, linestyle = :dot)
vlines!(axB2, 1.0; color = C_REF, linewidth = 0.6, linestyle = :dot)
text!(axB1, 0.98, 1.5; text = "forecast\nthe mean", fontsize = 6.5, color = C_REF, align = (:right, :bottom))
hspan!(axB1, STRIDE_LIKE, 32; color = C_SHADE)
text!(axB1, 0.02, 0.98; space = :relative, align = (:left, :top), fontsize = 7,
      text = "shaded: not a stride-like cycle")
refline!(axB1, 1.0; linestyle = :solid); refline!(axB2, 1.0)
trials = [(91, "AB"), (243, "HF"), (231, "LF")]
for (t, g) in trials
    c = [r for r in cyc if r.trial == t]; isempty(c) && continue
    kw = (; color = GROUP_COL[g], marker = markers[t], markersize = 6)
    scatter!(axB1, heldout.(c), [r.period_ratio for r in c]; kw...)
    cs = [r for r in c if r.period_ratio < STRIDE_LIKE]
    isempty(cs) || scatter!(axB2, heldout.(cs), [r.μ for r in cs]; kw...)
end
ylims!(axB1, 0.4, 32)
text!(axB2, 0.02, 0.02; space = :relative, align = (:left, :bottom), fontsize = 7,
      text = "stride-like cycles only")
Legend(gB[1, 3], [MarkerElement(color = GROUP_COL[g], marker = markers[t], markersize = 6) for (t, g) in trials],
       ["$g $t" for (t, g) in trials]; title = "gait trial", titlesize = 7, framevisible = false)
n_no = count(r -> r.kind != :cycle, runs); n_long = count(r -> r.period_ratio >= STRIDE_LIKE, cyc)
@printf("B: %d fits, %d settle on no cycle, %d on a cycle ≥ %d strides long\n", length(runs), n_no, n_long,
        round(Int, STRIDE_LIKE))
plabel!(gB[1, 1], "B")
colgap!(gB, 12)

# ── C: DMD relaxation vs transient data ──────────────────────────────────────
rows = deserialize(cachef("floquet", "dmd_datasize_rows.jls"))
NS = [10, 30, 100, 300]
for (c, (sys, title)) in enumerate((("cfhn", "Coupled FHN"), ("sl", "Stuart–Landau")))
    ax = Axis(gC[1, c]; xscale = log10, xticks = (NS, string.(NS)), xlabel = "Training decays (3 cycles each)",
              ylabel = c == 1 ? "Per-cycle relaxation rate" : "", title, limits = (8, 380, 0.45, 1.1))
    tr = [r.truth for r in rows if r.sys == sys]
    band!(ax, [8, 380], fill(minimum(tr), 2), fill(maximum(tr), 2); color = (C_TRUTH, 0.12))
    groups = sys == "cfhn" ? [("all", w -> true, CAT[1])] :
             [("1 cycle", w -> w < 1.2, CAT[4]), ("≥ 1.5 cycles", w -> w > 1.2, CAT[1])]
    for (lab, sel, col) in groups
        S = [[r.rate for r in rows if r.sys == sys && r.n == n && sel(r.win)] for n in NS]
        for (n, s) in zip(NS, S)
            jit = exp.(0.07 .* range(-1, 1; length = max(length(s), 2)))[1:length(s)]   # spread, log x
            scatter!(ax, n .* jit, min.(s, 1.09); color = (col, 0.45), markersize = 4)
        end
        lines!(ax, NS, median.(S); color = col, linewidth = 1.6)
    end
    refline!(ax, 1.0; linestyle = :dot)
    text!(ax, 370, 1.01; text = "no relaxation", fontsize = 7, color = C_REF, align = (:right, :bottom))
    c == 1 && plabel!(gC[1, 1], "C")
end
Legend(gC[2, 1:2],
       [[MarkerElement(color = (CAT[1], 0.45), marker = :circle, markersize = 4), LineElement(color = CAT[1], linewidth = 1.6)],
        [MarkerElement(color = (CAT[4], 0.45), marker = :circle, markersize = 4), LineElement(color = CAT[4], linewidth = 1.6)],
        PolyElement(color = (C_TRUTH, 0.12))],
       ["DMD settings and median (Stuart–Landau: delay window ≥ 1.5 cycles)", "Stuart–Landau, window 1 cycle", "truth"];
       orientation = :horizontal, framevisible = false, labelsize = 7, tellwidth = false)
rowgap!(gC, 4)
colgap!(gC, 14)
rowgap!(fig.layout, 10)
save_paper(fig, "Fig1_learnability")
