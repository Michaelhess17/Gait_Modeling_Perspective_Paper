#!/usr/bin/env julia
# Neural ODEs on real gait: a good fit, an unconstrained dynamics
#
# A deliberately over-parameterised neural ODE (3 × 128, ~35k parameters vs
# 3,000 training values), trained with multiple shooting on 10 s of six z-scored
# joint angles from one trial per group, six random seeds each
# (gait_overparam.jl → cache/gait_overparam_v2_t<trial>_s<seed>.jls).
#
#   (A–C) Hip–knee angle–angle diagrams (limb 1). Grey: training strides; black:
#         the held-out 5 s; colour: each seed's autonomous forecast from the last
#         training sample. Axes are framed on the data; forecasts that leave the
#         frame are cut at the edge.
#   (D)   Forecast error against lead time for every seed, with the
#         phase-aligned mean-cycle template (same information) for reference.
#   (E)   Period of the cycle each learned field settles onto, relative to the
#         subject's measured stride period, against the validation error of the
#         autonomous forecast over the held-out 5 s (cumulative RMS / held-out RMS).
#   (F)   Transverse Floquet multiplier of that learned cycle against the same
#         validation error: held-out forecasts do not pin the dynamics down.
#
# Set NODE_FIG_SUFFIX=_smoke3 to build from smoke-test caches while developing.

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using Serialization, Statistics

const SUFFIX = get(ENV, "NODE_FIG_SUFFIX", "")
const TRIALS = [91, 243, 231]                 # AB, HF, LF (as in gait_node_multishoot.jl)
const SEEDS  = 1:6
const HIP, KNEE = 1, 2

function load_runs(trial)
    out = NamedTuple[]
    for s in SEEDS
        f = joinpath(dirname(@__DIR__), "cache", "gait_overparam_v2_t$(trial)_s$(s)$(SUFFIX).jls")
        isfile(f) && push!(out, deserialize(f))
    end
    return out
end
runs = Dict(t => load_runs(t) for t in TRIALS)
all_runs = reduce(vcat, values(runs))
isempty(all_runs) && error("no gait_overparam_v2 caches found (suffix '$SUFFIX')")
# (the earlier summary-only runs on AB trials 36 and 90 have no held-out forecast, so they are not shown)
heldout(r) = isempty(r.err_t) ? NaN : r.err_t[end]      # validation error over the held-out 5 s

"""
Phase-aligned mean-cycle template forecast with the same information as the
NODE: training window `Y` (n × d), stride period `P` (s), sample step `dt`.
Returns predictions for n_ahead samples after the last training sample.
"""
function template_forecast(Y, P, dt, n_ahead; nbins = min(100, floor(Int, P / dt)))
    n, d = size(Y)   # nbins ≤ samples per stride, so no phase bin is empty
    ph = mod.((0:n-1) .* dt ./ P, 1.0)
    b = min.(floor.(Int, ph .* nbins), nbins - 1) .+ 1
    T = zeros(nbins, d)
    for j in 1:d, k in 1:nbins
        sel = b .== k
        T[k, j] = any(sel) ? mean(Y[sel, j]) : NaN
    end
    T = (circshift(T, 1) .+ T .+ circshift(T, -1)) ./ 3
    at(φ) = begin
        x = mod(φ, 1.0) * nbins - 0.5; i0 = floor(Int, x); w = x - i0
        (1 - w) .* T[mod(i0, nbins) + 1, :] .+ w .* T[mod(i0 + 1, nbins) + 1, :]
    end
    L = round(Int, P / dt); rel = ((0:L-1) .- (L - 1)) .* dt ./ P
    seg = Y[end-L+1:end, :]
    shifts = range(0, 1, length = 401)[1:end-1]
    errs = [mean(abs2, reduce(hcat, [at(s + r) for r in rel])' .- seg) for s in shifts]
    φ0 = shifts[argmin(errs)]
    return reduce(hcat, [at(φ0 + k * dt / P) for k in 1:n_ahead])'
end

fig = Figure(size = (COL2, 120MM))
gtop = fig[1, 1] = GridLayout()
gbot = fig[2, 1] = GridLayout()
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold,
                        padding = (0, 6, 2, 0), halign = :right)

# ── (A–C) hip–knee diagrams ──────────────────────────────────────────────────
for (k, t) in enumerate(TRIALS)
    rs = runs[t]; isempty(rs) && continue
    r0 = rs[1]; g = r0.group
    ax = Axis(gtop[1, k]; xlabel = "Hip angle (z)", ylabel = k == 1 ? "Knee angle (z)" : "",
              title = @sprintf("%s trial %d, %.0f cm/s", g, t, r0.speed))
    lims = framed_limits(vcat(r0.Y_tr[:, HIP], r0.Y_te[:, HIP]),
                         vcat(r0.Y_tr[:, KNEE], r0.Y_te[:, KNEE]); pad = 0.12)
    lines!(ax, r0.Y_tr[:, HIP], r0.Y_tr[:, KNEE]; color = (C_REF, 0.35), linewidth = 0.6)
    lines!(ax, r0.Y_te[:, HIP], r0.Y_te[:, KNEE]; color = C_TRUTH, linewidth = 1.4)
    for r in rs
        r.forecast_ok || continue
        x, y, _ = clip_to_view(r.F[:, HIP], r.F[:, KNEE], lims)
        lines!(ax, x, y; color = (GROUP_COL[g], 0.75), linewidth = 0.9)
    end
    scatter!(ax, [r0.Y_tr[end, HIP]], [r0.Y_tr[end, KNEE]]; color = C_TRUTH,
             markersize = 6, marker = :circle, strokecolor = :white, strokewidth = 0.8)
    limits!(ax, lims...)
    plabel!(gtop[1, k], ("A", "B", "C")[k])
end
Legend(gtop[0, 1:3],
       [LineElement(color = (C_REF, 0.5)), LineElement(color = C_TRUTH, linewidth = 1.4),
        [LineElement(color = GROUP_COL[g]) for g in GROUP_ORDER_FIG]...,
        MarkerElement(color = C_TRUTH, marker = :circle, markersize = 6)],
       ["training strides", "held-out 5 s",
        ["$g forecasts" for g in GROUP_ORDER_FIG]..., "forecast start"]; orientation = :horizontal, halign = :left)

# ── (D) forecast error vs lead ───────────────────────────────────────────────
axD = Axis(gbot[1, 1]; xlabel = "Lead time (s)", ylabel = "Forecast error (SD units)",
           title = "Forecast error")
refline!(axD, 1.0)
for t in TRIALS
    rs = runs[t]; isempty(rs) && continue
    g = rs[1].group
    for r in rs
        isempty(r.err_t) && continue
        lines!(axD, (1:length(r.err_t)) .* r.dt, min.(r.err_t, 2.4);
               color = (GROUP_COL[g], 0.6), linewidth = 0.8)
    end
    r0 = rs[1]; n = size(r0.Y_te, 1)
    Tp = template_forecast(r0.Y_tr, r0.P_stride, r0.dt, n)
    rms = sqrt(mean(abs2, r0.Y_te))
    e = [sqrt(mean(abs2, Tp[1:i, :] .- r0.Y_te[1:i, :])) / rms for i in 1:n]
    lines!(axD, (1:n) .* r0.dt, e; color = GROUP_COL[g], linewidth = 1.6, linestyle = :dash)
end
text!(axD, 4.9, 1.0; text = "predict the mean", color = C_REF, fontsize = 7,
      align = (:right, :bottom))
text!(axD, 4.9, 0.05; text = "dashed: mean-cycle template", color = C_TRUTH,
      fontsize = 7, align = (:right, :bottom))
limits!(axD, 0, 5, 0, 2.5)
plabel!(gbot[1, 1], "D")

# ── (E, F) learned cycle vs held-out forecast error ──────────────────────────
cyc(rs) = [r for r in rs if r.kind == :cycle && isfinite(r.period_ratio)]
axE = Axis(gbot[1, 2]; xlabel = "Held-out forecast error (5 s)", ylabel = "Learned period / stride period",
           title = "Learned rhythm", yscale = log2,
           yticks = ([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"]))
axF = Axis(gbot[1, 3]; xlabel = "Held-out forecast error (5 s)", ylabel = "Transverse multiplier |μ|",
           title = "Learned stability")
vlines!(axE, 1.0; color = C_REF, linewidth = 0.6, linestyle = :dot)
vlines!(axF, 1.0; color = C_REF, linewidth = 0.6, linestyle = :dot)
# a learned "cycle" many strides long is not a stride-like (period-1) orbit, and a
# Floquet multiplier read off it at that period is meaningless: show it in E,
# leave it out of F, and count it
const STRIDE_LIKE = 3.0
hspan!(axE, STRIDE_LIKE, 32; color = C_SHADE)
text!(axE, 0.02, 0.98; space = :relative, align = (:left, :top), fontsize = 7,
      text = "shaded: not a stride-like cycle")
refline!(axE, 1.0; linestyle = :solid)
refline!(axF, 1.0)
markers = Dict(91 => :circle, 243 => :diamond, 231 => :utriangle)
for (t, rs) in [(t, runs[t]) for t in TRIALS]
    c = cyc(rs); isempty(c) && continue
    g = String(c[1].group)
    kw = (; color = GROUP_COL[g], marker = markers[t], markersize = 6)
    scatter!(axE, heldout.(c), [r.period_ratio for r in c]; kw...)
    cs = [r for r in c if r.period_ratio < STRIDE_LIKE]
    isempty(cs) || scatter!(axF, heldout.(cs), [r.μ for r in cs]; kw...)
end
pool = all_runs
n_nocycle = count(r -> r.kind != :cycle, pool)
n_long = count(r -> r.kind == :cycle && r.period_ratio >= STRIDE_LIKE, pool)
text!(axF, 0.02, 0.02; space = :relative, align = (:left, :bottom), fontsize = 7,
      text = "stride-like cycles only")
@printf("fits: %d total, %d no cycle, %d cycle ≥ %d strides long\n",
        length(pool), n_nocycle, n_long, round(Int, STRIDE_LIKE))
ylims!(axE, 0.4, 32)
Legend(gbot[1, 4],
       [MarkerElement(color = GROUP_COL[String(rs[1].group)], marker = markers[t], markersize = 6)
        for (t, rs) in [(t, runs[t]) for t in TRIALS if !isempty(runs[t])]],
       [@sprintf("%s %d", runs[t][1].group, t) for t in TRIALS if !isempty(runs[t])];
       title = "trial", titlesize = 7)
plabel!(gbot[1, 2], "E"); plabel!(gbot[1, 3], "F")

colgap!(gtop, 14); colgap!(gbot, 12); rowgap!(gtop, 2)
rowsize!(fig.layout, 1, Relative(0.55))
save_paper(fig, "FigS2_node_gait" * SUFFIX)

for t in TRIALS, r in runs[t]
    @printf("trial %3d seed %d | loss %.4f | %-11s P/Pstride %.2f |μ| %.3f | err@1s %.2f @3s %.2f\n",
            t, r.seed, r.loss, String(r.kind), r.period_ratio, r.μ,
            length(r.err_t) >= 50 ? r.err_t[50] : NaN,
            length(r.err_t) >= 150 ? r.err_t[150] : NaN)
end
