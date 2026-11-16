#!/usr/bin/env julia
# Figure_01.jl — Why model the dynamics of walking, and what such a model must provide (manuscript: @fig-overview).
# Run: julia --project=. scripts/figures/Figure_01.jl
# A schematic overview; no data.
#   A  a recording of walking is a repeating average stride plus departures from it, the responses to disturbances
#   B  the average stride misses the dynamics: two systems with the same cycle, traversed at the same rate (so the same
#      average stride), respond differently to the same push. Exact solutions of  ṙ = −k (r − 1),  θ̇ = 2π  (time in
#      strides; state r(cos θ, −sin θ)): 1, k = 5 (returns within half a stride); 2, k = 0.7 (over several strides)
#   C  stability numbers miss the response: the size of one response to a disturbance (exact, ẋ = A x with A upper
#      triangular, time in strides; size = RMS over disturbance directions), the early rate a short-term Lyapunov
#      exponent reads (over the first stride), the late rate a Floquet multiplier reads (the slowest decay), and the
#      whole transient structure; blue, a second system with the same early rate and multiplier but a different
#      response; the next disturbance arrives within about a stride
#   D  what a model of walking dynamics must provide: the requirements R1–R6
# Usage: julia --project=. scripts/figures/Figure_01.jl   → figures/Figure_01.{pdf,svg,png}

include(joinpath(@__DIR__, "..", "paper_style.jl"))
using LinearAlgebra, Printf, Random

const C_DEP = CAT[2]                       # departures and responses, and the disturbances that cause them
const C_1, C_2 = C_TRUTH, CAT[1]           # the two systems in B, black and blue as the pairs of Figure 2
const C_LE, C_FM = CAT[5], CAT[3]          # Lyapunov exponent, Floquet multiplier (as Figure 2)
const C_LOOP = (C_REF, 0.85)
header!(g, letter, txt) = Label(g[0, :], rich(rich(letter * "  ", font = :bold, fontsize = 10), rich(txt, fontsize = 7.5));
                                halign = :left, tellwidth = false, padding = (0, 0, 2, 0))
blank(g; kw...) = (ax = Axis(g; kw...); hidedecorations!(ax); hidespines!(ax); ax)
arrow!(ax, x, y, dx, dy; color = C_TRUTH, w = 0.8, tip = 4) =
    arrows2d!(ax, [x], [y], [dx], [dy]; color, shaftwidth = w, tipwidth = tip, tiplength = tip)

fig = Figure(size = (COL2, 104MM))
top = fig[1, 1] = GridLayout(); bot = fig[2, 1] = GridLayout()
gA = top[1, 1] = GridLayout(); gB = top[1, 2] = GridLayout(); gC = top[1, 3] = GridLayout()
colgap!(top, 16); rowgap!(fig.layout, 8)

# ── A: a recording is the average stride plus departures ──────────────────────
wave(t) = 0.55 * (cos(2π * t) + 0.35cos(4π * t + 1.0))
TK = [0.35, 1.2, 1.75, 2.6]; AK = [0.9, -1.1, 0.7, -0.8]
resp(t) = sum(a * (t > tk ? 2.2(t - tk) * exp(-4.5(t - tk)) * cos(9(t - tk)) : 0.0) for (tk, a) in zip(TK, AK))
tA = range(0, 3; length = 600)
axA = blank(gA[1, 1]; limits = (-0.05, 3.05, -3.3, 3.0))
y1, y2, y3 = 1.3, -0.5, -2.2
lines!(axA, tA, y1 .+ wave.(tA) .+ 0.2 .* resp.(tA); color = C_TRUTH, linewidth = 1.0)
lines!(axA, tA, y2 .+ wave.(tA); color = C_LOOP, linewidth = 2.0)
band!(axA, tA, fill(y3, length(tA)), y3 .+ 0.55 .* resp.(tA); color = (C_DEP, 0.25))
lines!(axA, tA, y3 .+ 0.55 .* resp.(tA); color = C_DEP, linewidth = 0.9)
scatter!(axA, TK, fill(y1 + 1.0, length(TK)); marker = :dtriangle, color = C_DEP, markersize = 5)
text!(axA, 0, y1 + 1.3; text = "recording", fontsize = 6.5, align = (:left, :bottom))
text!(axA, 0.62, y1 + 1.3; text = "▼ disturbances", fontsize = 6.5, color = C_DEP, align = (:left, :bottom))
text!(axA, 0, y2 + 0.85; text = "= average stride: walking when nothing happens", fontsize = 6.5, color = C_REF, align = (:left, :bottom))
text!(axA, 0, y3 + 0.62; text = "+ departures (enlarged): its responses", fontsize = 6.5, color = C_DEP, align = (:left, :bottom))
text!(axA, 0, y3 - 0.62; text = "small, but spread over many directions", fontsize = 6, color = C_DEP, align = (:left, :top))
header!(gA, "A", "Walking: a cycle plus responses")

# ── B: the average stride misses the dynamics ──────────────────────────────────
SYS = [(5.0, 0.0, C_1, :solid), (0.7, 0.0, C_2, :dash)]
θ0 = -π / 2; DV = 0.6; R0 = 1 + DV
st(t, k, c) = (r = 1 + (R0 - 1) * exp(-k * t); θ = θ0 + 2π * t + 2π * c * (R0 - 1) * (1 - exp(-k * t)) / k; (r * cos(θ), -r * sin(θ)))
gBl = gB[1, 1] = GridLayout()
φc = range(0, 2π; length = 200)
for (j, (k, c, col, ls)) in enumerate(SYS)
    ax = blank(gBl[1, j]; aspect = DataAspect(), limits = (-1.75, 1.75, -1.75, 1.75))
    lines!(ax, cos.(φc), sin.(φc); color = C_LOOP, linewidth = 2.0)
    Q = [st(t, k, c) for t in range(0, j == 1 ? 0.6 : 3.0; length = 600)]
    lines!(ax, first.(Q), last.(Q); color = col, linewidth = 1.0, linestyle = ls)
    arrow!(ax, 0.0, 1.0, 0.0, DV - 0.05; color = C_DEP, w = 1.2, tip = 4.5)
    text!(ax, 0, 0; text = "$j", fontsize = 8, color = col, align = (:center, :center), font = :bold)
end
Label(gBl[2, 1:2], "same cycle, same push"; fontsize = 6.5, color = C_REF, tellwidth = false)
rowgap!(gBl, 0); colgap!(gBl, 2)
axB = Axis(gB[2, 1]; limits = (-0.25, 3.1, -1.5, 1.5), xlabel = "strides after the push", ylabel = "departure",
           xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false,
           xlabelsize = 6.5, ylabelsize = 6.5)
tB = range(-0.25, 3.1; length = 700)
hlines!(axB, 0; color = C_REF, linewidth = 0.5)
for (k, c, col, ls) in SYS
    lines!(axB, tB, [t < 0 ? 0.0 : st(t, k, c)[1] - cos(θ0 + 2π * t) for t in tB]; color = col, linewidth = 1.1, linestyle = ls)
end
text!(axB, 0.3, -1.45; text = "1 returns within half a stride", fontsize = 6, color = C_1, align = (:left, :bottom))
text!(axB, 3.05, 1.47; text = "2 returns over several strides", fontsize = 6, color = C_2, align = (:right, :top))
Label(gB[3, 1], "The average stride cannot tell them apart;"; fontsize = 6.5, halign = :left, tellwidth = false)
Label(gB[4, 1], "only a model of the dynamics predicts each."; fontsize = 6.5, font = :bold, halign = :left, tellwidth = false)
rowsize!(gB, 1, Relative(0.42)); rowgap!(gB, 2); rowgap!(gB, 3, 0)
header!(gB, "B", "The average stride misses the dynamics")

# ── C: stability numbers miss most of the response ─────────────────────────────
tri(a, b, c) = [a c; 0.0 b]; tC = range(0, 6; length = 601)
szM(M, t) = norm(exp(M * t)) / sqrt(2)
"Coupling c that gives tri(a, b, c) the response size `target` at 1 stride (the same early rate)."
function match_c(a, b, target)
    lo, hi = 0.0, 60.0
    for _ in 1:100; m = (lo + hi) / 2; szM(tri(a, b, m), 1.0) < target ? (lo = m) : (hi = m); end
    (lo + hi) / 2
end
A = tri(-0.4, -2.5, 9.0); sz(t) = szM(A, t)
A2 = tri(-0.4, -0.75, match_c(-0.4, -0.75, sz(1.0)))                 # same early rate and multiplier, different response
s = maximum(real, eigvals(A)); r = sz.(tC)
axC = Axis(gC[1, 1]; yscale = log10, limits = (0, 6, 0.07, 9), xlabel = "strides since a disturbance", ylabel = "size of the response",
           xticks = ([0, 1, 2, 3, 4, 5, 6], ["0", "1", "2", "3", "4", "5", "6"]), yticksvisible = false, yticklabelsvisible = false,
           xlabelsize = 6.5, ylabelsize = 6.5, xticklabelsize = 6)
vspan!(axC, 0, 1; color = (C_LE, 0.12)); vspan!(axC, 3, 6; color = (C_FM, 0.12))
hlines!(axC, 1.0; color = C_REF, linewidth = 0.5, linestyle = :dot)
lines!(axC, [0, 1], [1, sz(1.0)]; color = C_LE, linestyle = :dash, linewidth = 1.0)
c0 = r[findfirst(>=(3.0), tC)] / exp(s * 3.0); tf = tC[tC .>= 3.2]
lines!(axC, tf, 0.45 .* c0 .* exp.(s .* tf); color = C_FM, linestyle = :dash, linewidth = 1.0)
lines!(axC, tC, szM.(Ref(A2), tC); color = C_2, linewidth = 1.2, linestyle = :dash)
lines!(axC, tC, r; color = C_TRUTH, linewidth = 1.5)
text!(axC, 3.1, 3.3; text = "same early and late\nrates, different response", fontsize = 6, color = C_2, align = (:left, :top))
text!(axC, 0.06, 0.42; text = "Lyapunov\nexponent:\nearly rate", fontsize = 6, color = C_LE, align = (:left, :top))
text!(axC, 4.5, 8.0; text = "Floquet multiplier:\nlate rate", fontsize = 6, color = C_FM, align = (:center, :top))
text!(axC, 1.15, 8.0; text = "transient\nstructure: the\nwhole response", fontsize = 6, align = (:left, :top))
scatter!(axC, [1.0], [0.085]; marker = :utriangle, color = C_DEP, markersize = 5)
text!(axC, 1.08, 0.085; text = "next disturbance", fontsize = 6, color = C_DEP, align = (:left, :center), offset = (3, 0))
Label(gC[2, 1], "It also has a structure no single number gives:\nwhich movements carry it, when in the stride it\narrives, and whether the timing shifts.";
      fontsize = 6.5, halign = :left, justification = :left, tellwidth = false)
rowgap!(gC, 4)
header!(gC, "C", "Stability numbers miss the response")

# ── D: what a model must provide ───────────────────────────────────────────────
REQ = [("R1", "Forecast the response", "beat the average stride\nat predicting departures"),
       ("R2", "Stay credible", "keep walking\nwhen run on its own"),
       ("R3", "Reproducible", "the same data give\nthe same answer"),
       ("R4", "Real recordings", "short, noisy, partly observed;\npractical to fit"),
       ("R5", "Interpretable structure", "how disturbances grow,\nspread and decay"),
       ("R6", "Comparable", "the same quantities across\npeople and conditions")]
function icon!(ax, id)
    if id == "R1"                                  # the forecast follows what happens; the average stride is the zero line
        f(t) = (0.55sin(5t) + 0.3sin(9t + 1)) * exp(-1.3max(t, 0))
        new(t) = t <= 0 ? 0.0 : 0.3 * (1 - exp(-2.5t)) * sin(8t + 2)
        t1 = range(-1, 0; length = 60); t2 = range(0, 1.6; length = 100)
        hlines!(ax, 0; color = C_LOOP, linewidth = 1.6); vlines!(ax, 0; color = C_REF, linewidth = 0.5, linestyle = :dot)
        lines!(ax, t1, f.(t1); color = C_TRUTH, linewidth = 1.0)
        lines!(ax, t2, f.(t2) .+ new.(t2); color = C_TRUTH, linewidth = 1.0)
        lines!(ax, t2, f.(t2); color = C_DEP, linewidth = 1.1, linestyle = :dash)
        text!(ax, 1.6, -0.98; text = "forecast", fontsize = 5.5, color = C_DEP, align = (:right, :bottom))
        limits!(ax, -1, 1.6, -1, 1)
    elseif id == "R2"                              # a free run that stays on the cycle, and one that leaves it
        φ = range(0, 6π; length = 400)
        lines!(ax, cos.(φ[1:134]), sin.(φ[1:134]); color = C_LOOP, linewidth = 2.0)
        rin = 1 .+ 0.3 .* exp.(-φ ./ 4); lines!(ax, rin .* cos.(φ), rin .* sin.(φ); color = C_TRUTH, linewidth = 0.9)
        rout = 1 .+ 0.04 .* exp.(φ ./ 5.2); lines!(ax, rout .* cos.(φ), rout .* sin.(φ); color = C_REF, linewidth = 0.7, linestyle = :dot)
        scatter!(ax, [rout[end] * cos(φ[end])], [rout[end] * sin(φ[end])]; marker = :xcross, color = C_REF, markersize = 6)
        limits!(ax, -2.8, 2.8, -2.0, 2.0)
    elseif id == "R3"                              # fits from different seeds disagree; one fit, one answer
        t = range(0, 3; length = 100); rng = MersenneTwister(4)
        for _ in 1:6
            a, b = 1 + 0.5randn(rng), 1 + 0.35randn(rng)
            lines!(ax, t, 2.2 .* a .* t .* exp.(-b .* t); color = (C_REF, 0.7), linewidth = 0.6)
        end
        lines!(ax, t, 2.2 .* t .* exp.(-t); color = C_TRUTH, linewidth = 1.4)
        limits!(ax, 0, 3, -0.1, 1.6)
    elseif id == "R4"                              # a short, noisy record
        rng = MersenneTwister(2); t = range(0, 2; length = 60)
        lines!(ax, t, cos.(2π .* t); color = C_LOOP, linewidth = 0.8, linestyle = :dash)
        scatter!(ax, t, cos.(2π .* t) .+ 0.22 .* randn(rng, length(t)); color = C_TRUTH, markersize = 2.2)
        limits!(ax, -0.05, 2.05, -1.5, 1.5)
    elseif id == "R5"                              # how a disturbance spreads between joints
        Pn = [("hip", (0.0, 0.0), (:right, :center), (-6, 0)), ("knee", (1.0, 0.8), (:center, :bottom), (0, 5)),
              ("ankle", (2.0, 0.0), (:left, :center), (6, 0))]
        pos = Dict(n => p for (n, p, _, _) in Pn)
        for (a, b, w) in (("hip", "knee", 2.2), ("knee", "ankle", 1.3), ("hip", "ankle", 0.6))
            (x1, y1), (x2, y2) = pos[a], pos[b]; d = (x2 - x1, y2 - y1); u = d ./ hypot(d...)
            arrows2d!(ax, [x1 + 0.2u[1]], [y1 + 0.2u[2]], [d[1] - 0.4u[1]], [d[2] - 0.4u[2]]; color = C_TRUTH,
                      shaftwidth = w, tipwidth = 3 + 1.5w, tiplength = 4)
        end
        for (nm, (x, y), al, off) in Pn
            scatter!(ax, [x], [y]; color = :white, strokecolor = C_TRUTH, strokewidth = 0.8, markersize = 8)
            text!(ax, x, y; text = nm, fontsize = 5.5, align = al, offset = off)
        end
        limits!(ax, -0.9, 2.9, -0.4, 1.35)
    elseif id == "R6"                              # the same quantity for every person, in two conditions
        for (i, (v1, v2)) in enumerate(((1.3, 1.9), (1.7, 2.0), (2.3, 1.5)))
            y = 4 - i
            lines!(ax, [1.0, 2.6], [y, y]; color = (C_REF, 0.4), linewidth = 0.5)
            scatter!(ax, [v1], [y]; color = C_TRUTH, markersize = 4.5)
            scatter!(ax, [v2], [y]; color = :white, strokecolor = C_TRUTH, strokewidth = 0.8, markersize = 4.5)
            text!(ax, 0.95, y; text = "person $('A' + i - 1)", fontsize = 5.5, align = (:right, :center))
        end
        limits!(ax, -0.1, 2.7, 0.4, 3.6)
    end
end
for (j, (id, ttl, desc)) in enumerate(REQ)
    icon!(blank(bot[1, j]; height = 32), id)
    Label(bot[2, j], rich(rich(id * "  ", font = :bold), ttl); fontsize = 7, halign = :center, tellwidth = false)
    Label(bot[3, j], desc; fontsize = 6.2, halign = :center, justification = :center, tellwidth = false, color = (C_TRUTH, 0.8))
end
for j in 1:6; colsize!(bot, j, Relative(1 / 6)); end
colgap!(bot, 6); rowgap!(bot, 1)
header!(bot, "D", "What a model of walking dynamics must provide")
rowsize!(fig.layout, 2, Relative(0.30))

save_paper(fig, "Figure_01")
@printf("C: Lyapunov early rate %.2f per stride; Floquet multiplier %.2f per stride; peak %.2f at %.2f strides\n",
        log(sz(1.0)), exp(s), maximum(r), tC[argmax(r)])
for (j, (k, c, _, _)) in enumerate(SYS)
    @printf("B system %d: multiplier per stride %.3f\n", j, exp(-k))
end
