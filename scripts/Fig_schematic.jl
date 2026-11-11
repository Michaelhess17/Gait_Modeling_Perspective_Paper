#!/usr/bin/env julia
# Fig 1 — Overview: what Lyapunov exponents, Floquet multipliers and the transient structure each see.
# Exact computations on small linear systems (ẋ = A x, time in strides), no fitting. Measures, as used for gait:
#   short-term Lyapunov exponent λ_S   mean divergence rate over the first stride: ln of the RMS response at 1 stride
#                                      (RMS over disturbance directions), as Rosenstein-type estimates over 0–1 stride
#   Floquet multiplier ρ               asymptotic return per stride: e^(largest Re eigenvalue)
#   transient structure                the whole response: size, timing, duration (and, in more dimensions, the
#                                      movements involved)
#   (A) overview of the paper's argument
#   (B) one response, and the part each measure reads
#   (C) three pairs of systems: each measure separates some pairs; the transient structure separates all three

using Pkg; Pkg.activate(dirname(@__DIR__); io = devnull)
include("paper_style.jl")
using LinearAlgebra, Printf

plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
C_A = C_TRUTH; C_B = CAT[1]; C_LE = CAT[2]; C_FM = CAT[3]
ts = range(0, 6; length = 601)
rms(A, t) = norm(exp(A * t)) / sqrt(size(A, 1))                   # RMS over unit disturbance directions
λS(A) = log(rms(A, 1.0))                                          # mean divergence rate over the first stride
ρ(A) = exp(maximum(real, eigvals(A)))
tri(a, b, c) = [a c; 0.0 b]
"Coupling c giving an RMS response of `target` at 1 stride."
function match_c(a, b, target; lo = 0.0, hi = 60.0)
    for _ in 1:100
        m = (lo + hi) / 2; rms(tri(a, b, m), 1.0) < target ? (lo = m) : (hi = m)
    end
    (lo + hi) / 2
end
peak(A) = (r = [rms(A, t) for t in ts]; (maximum(r), ts[argmax(r)]))

Aref = tri(-0.4, -2.5, 9.0)                                       # the response in B
target = rms(Aref, 1.0)
pairs = [(tri(-0.4, -2.5, 0.0), Aref, "Same multiplier,\ndifferent early rate"),
         (Aref, tri(-0.08, -2.5, match_c(-0.08, -2.5, target)), "Same early rate,\ndifferent return"),
         (Aref, tri(-0.4, -0.75, match_c(-0.4, -0.75, target)), "Same early rate and\nmultiplier, different peak")]
for (k, (a, b, _)) in enumerate(pairs)
    pa, pb = peak(a), peak(b)
    @printf("pair %d: λ_S %.3f vs %.3f; ρ %.3f vs %.3f; peak %.2f at %.2f vs %.2f at %.2f strides\n", k, λS(a), λS(b),
            ρ(a), ρ(b), pa..., pb...)
end

fig = Figure(size = (COL2, 122MM))
gA = fig[1, 1:2] = GridLayout(); gB = fig[2, 1] = GridLayout(); gC = fig[2, 2] = GridLayout()
rowsize!(fig.layout, 1, Relative(0.18)); colsize!(fig.layout, 1, Relative(0.40)); colgap!(fig.layout, 14); rowgap!(fig.layout, 6)

# ── A: the argument ─────────────────────────────────────────────────────────
axA = Axis(gA[1, 1]; limits = (0, 100, 0, 10)); hidedecorations!(axA); hidespines!(axA)
boxes = [("Walking is disturbed\nevery stride", "Fig 2"),
         ("Stability numbers miss the\nshort-time response and\nare hard to estimate", "Figs 2–3"),
         ("What a model must do\n(R1–R6), and what gait\nrecordings offer", "§3, Fig 4"),
         ("Testing models against\neach requirement", "Figs 5–7"),
         ("Linear models: fit, check\nand analyse the response", "§5")]
w = 17.6; gap = 2.9
for (k, (txt, ref)) in enumerate(boxes)
    x0 = (k - 1) * (w + gap)
    poly!(axA, Rect(x0, 0.6, w, 9.0); color = C_SHADE, strokecolor = (C_REF, 0.6), strokewidth = 0.6)
    text!(axA, x0 + w / 2, 5.9; text = txt, align = (:center, :center), fontsize = 7)
    text!(axA, x0 + w / 2, 1.7; text = ref, align = (:center, :center), fontsize = 6, color = C_REF)
    k < length(boxes) && arrows2d!(axA, [x0 + w + 0.3], [5.0], [gap - 0.8], [0.0]; color = C_REF, shaftwidth = 0.8,
                                   tipwidth = 4, tiplength = 4)
end
plabel!(gA[1, 1], "A")

# ── B: one response and what each measure reads ────────────────────────────
r = [rms(Aref, t) for t in ts]; s = maximum(real, eigvals(Aref))
axB = Axis(gB[1, 1]; yscale = log10, xlabel = "Strides since a disturbance", ylabel = "Size of the response\n(relative to the disturbance)",
           title = "What each measure reads", limits = (0, 6, 0.05, 8),
           yticks = ([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"]))
vspan!(axB, 0, 1; color = (C_LE, 0.10))
vspan!(axB, 3, 6; color = (C_FM, 0.10))
refline!(axB, 1.0; linestyle = :dot)
c0 = r[findfirst(>=(3.0), ts)] / exp(s * 3.0)
tf = ts[ts .>= 3.2]
lines!(axB, tf, 0.45 .* c0 .* exp.(s .* tf); color = C_FM, linestyle = :dash, linewidth = 1.2)
text!(axB, 4.3, 0.45 * c0 * exp(s * 4.3); text = "same slope:\nln ρ per stride", fontsize = 6, color = C_FM,
      align = (:right, :top), offset = (-2, -2))
lines!(axB, [0, 1], [1, rms(Aref, 1.0)]; color = C_LE, linestyle = :dash, linewidth = 1.0)
lines!(axB, ts, r; color = C_TRUTH, linewidth = 1.6)
text!(axB, 0.5, 0.42; text = "Lyapunov\nexponent:\nearly rate", fontsize = 6.5, color = C_LE, align = (:center, :top))
text!(axB, 4.5, 7.2; text = "Floquet multiplier:\nlate rate per stride", fontsize = 6.5, color = C_FM, align = (:center, :top))
text!(axB, 1.1, 7.2; text = "transient structure:\nthe whole response", fontsize = 6.5, color = C_TRUTH, align = (:left, :top))
scatter!(axB, [1.0], [0.062]; marker = :utriangle, color = C_REF, markersize = 6)
text!(axB, 1.05, 0.062; text = "the next disturbance\narrives about here (Fig 2)", fontsize = 6, color = C_REF,
      align = (:left, :center), offset = (4, 0))
plabel!(gB[1, 1], "B")

# ── C: three pairs ──────────────────────────────────────────────────────────
Label(gC[0, 1:3], rich(rich("C   ", font = :bold, fontsize = 10), rich("Pairs of systems, black and blue: which measure tells them apart?",
      fontsize = 7, color = C_REF)); halign = :left, tellwidth = false)
for (k, (a, b, title)) in enumerate(pairs)
    ax = Axis(gC[1, k]; yscale = log10, title, titlesize = 7, xlabel = "Strides", ylabel = k == 1 ? "Size of the response" : "",
              limits = (0, 6, 0.05, 8), yticks = ([0.1, 1], ["0.1", "1"]), yticklabelsvisible = k == 1, titlealign = :center)
    refline!(ax, 1.0; linestyle = :dot)
    lines!(ax, ts, [rms(a, t) for t in ts]; color = C_A, linewidth = 1.4)
    lines!(ax, ts, [rms(b, t) for t in ts]; color = C_B, linewidth = 1.4, linestyle = :dash)
    same = (abs(λS(a) - λS(b)) < 0.01, abs(ρ(a) - ρ(b)) < 0.005, false)
    tb = GridLayout(gC[2, k])
    for (row, (lab, eq)) in enumerate(zip(("Lyapunov", "Floquet", "Transient"), same))
        Label(tb[row, 1], lab; fontsize = 6.5, halign = :left, tellwidth = false)
        Label(tb[row, 2], eq ? "same" : "differs"; fontsize = 6.5, halign = :right, tellwidth = false,
              color = eq ? C_REF : C_TRUTH, font = eq ? :regular : :bold)
    end
    rowgap!(tb, 1)
end
rowsize!(gC, 2, Auto(0.30)); colgap!(gC, 8); rowgap!(gC, 4)

save_paper(fig, "Fig_schematic")
