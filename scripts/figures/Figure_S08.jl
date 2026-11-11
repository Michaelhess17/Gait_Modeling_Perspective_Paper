#!/usr/bin/env julia
# Figure_S08.jl — Imposing the limit cycle's structure on a linear model (manuscript: @suppfig-lattice). Run: julia --project=. scripts/figures/Figure_S08.jl
# Figure S6 — Imposing the limit cycle's structure on a linear model (OUTLINE.md §6 ¶6).
#
#   (A) The lattice-constrained model on one SLIP decay (scripts/figS6_slip_data.jl): state = fitted cycle a(θ) plus
#   one Floquet mode Re[ψ e^{μt} b(θ)]; the phase advances uniformly, the mode's amplitude shrinks by ρ and turns by
#   the rotation every step, and its shape b(θ) repeats every step. μ is the only spectral parameter.
#   (B) Why: a stable limit cycle's Koopman spectrum is a lattice. Read relative to the nearest harmonic, every
#   harmonic sits at (0°, 1) and every Floquet sideband at the star. Hankel-DMD (same 10 three-step decays) fits each
#   eigenvalue on its own: harmonics detuned and decaying, sidebands blended with them (labels: nearest harmonic).
#   The lattice fit returns one value; the phase–isostable network (linear latent dynamics, neural decoder; μ free)
#   drifts; on noise-driven walks of 15 steps (scripts/gait_lattice.jl) the modulus is close, the rotation is not.
#   (C) SLIP off-cycle metrics, 4 starts: |error| of the 4-step relaxation rate (Fig 6B's metric) and distance to
#   the true cycle after 25 steps / initial (median over starts). Dots: fits (seeds).
#   (D) Human gait and kicked SLIP walks: R² of the deviation from the mean stride 1 stride ahead (filled; 2 strides
#   ahead, open) from half a stride of data, median and IQR over trials. The multiplier and the DMD rank are chosen on
#   the person's other trials. Deviation DMD: delay DMD on the deviations; companion DMD: the companion paper's
#   operator on the full signal (cycle included), scored on its deviation.
#   (E) Is the multiplier identified? Each trial's normalised forecast error over strides +1 and +2 against the
#   assumed modulus (best rotation at each), median and IQR over trials: kicked SLIP walks have a minimum near the
#   truth; gait profiles are flat at the mean stride's error, so the per-person multipliers (text) are not identified.
#   (F) Lattice skill by horizon with the multiplier chosen out of trial (solid) or on the scored strides themselves
#   (dashed), and the same on stride-shuffled copies of each trial (dotted; scripts/gait_lattice_null.jl): on gait
#   about half of the on-test skill is selection, and the rest does not carry over to the person's other trials.

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf

const ROOT = dirname(dirname(@__DIR__))
plabel!(pos, L) = Label(pos[1, 1, TopLeft()], L; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const S = deserialize(joinpath(ROOT, "cache", "figS6", "slip.jls"))
const G = deserialize(joinpath(ROOT, "cache", "fig5", "gait_lattice.jls"))
const NUL = deserialize(joinpath(ROOT, "cache", "fig5", "gait_lattice_null.jls"))
const C_LAT, C_NN = CAT[5], CAT[4]

fig = Figure(size = (COL2, 128MM))
top = fig[1, 1] = GridLayout(); bot = fig[2, 1] = GridLayout()

# ── A: the model on one decay ────────────────────────────────────────────────
ex = S.ex[3]; c = argmax(vec(std(ex.Y .- ex.cyc, dims = 1))); t = (0:size(ex.Y, 1)-1) ./ S.Pc
CH = ["CoM height", "forward speed", "vertical speed"]
gA = top[1, 1] = GridLayout()
a1 = Axis(gA[1, 1]; ylabel = CH[c] * "\n(SD)", xticklabelsvisible = false,
          title = rich("x(t) = a(θ) + Re[ψ e", superscript("μt"), " b(θ)]"))
lines!(a1, t, ex.cyc[:, c]; color = C_REF, linestyle = :dash, linewidth = 1.0)
lines!(a1, t, ex.Y[:, c]; color = C_TRUTH, linewidth = 1.0)
ylims!(a1, extrema(ex.Y[:, c]) .+ (-0.15, 0.55))
text!(a1, 0.02, maximum(ex.Y[:, c]) + 0.5; text = "decay; dashed: cycle a(θ)", fontsize = 6.5, align = (:left, :top))
a2 = Axis(gA[2, 1]; xlabel = "Time (steps)", ylabel = "deviation\n(SD)")
hlines!(a2, 0.0; color = C_REF, linewidth = 0.5)
lines!(a2, t, ex.Y[:, c] .- ex.cyc[:, c]; color = C_TRUTH, linewidth = 1.0)
lines!(a2, t, ex.dev[:, c]; color = C_LAT, linewidth = 1.2)
dl = extrema(ex.Y[:, c] .- ex.cyc[:, c]); ylims!(a2, dl[1] - 0.05, dl[2] + 0.45 * (dl[2] - dl[1]))
text!(a2, 2.98, dl[2] + 0.43 * (dl[2] - dl[1]); text = @sprintf("pink: one mode, b(θ) repeats,\n× %.2f, turns %.0f° per step", S.mult...),
      fontsize = 6.5, align = (:right, :top))
rowgap!(gA, 3); plabel!(gA[1, 1], "A")

# ── B: spectrum in the cycle frame ───────────────────────────────────────────
aB = Axis(top[1, 2]; xlabel = "Rotation per step, from the nearest harmonic (°)", ylabel = "|λ| per step",
          title = "Spectrum: lattice vs Hankel-DMD", limits = (-1.5, 26, 0.48, 1.06), xticks = 0:5:25)
L = filter(x -> x.ρ > 0.5, S.dmd)
scatter!(aB, abs.([x.δ for x in L]), [x.ρ for x in L]; color = CAT[1], markersize = 5)
for x in L
    dy = x.n == 2 ? -0.018 : x.n == 3 ? 0.014 : 0.0
    text!(aB, abs(x.δ) + 0.5, x.ρ + dy; text = string(x.n), fontsize = 5.5, color = CAT[1], align = (:left, :center))
end
λx = S.exact[argmax(imag.(S.exact))]; ex_rot, ex_mod = rad2deg(angle(λx)), abs(λx)
scatter!(aB, [0.0], [1.0]; marker = :circle, color = :transparent, strokecolor = C_TRUTH, strokewidth = 1.0, markersize = 11)
text!(aB, 0.6, 1.04; text = "exact: every harmonic", fontsize = 6.5, align = (:left, :center))
scatter!(aB, [ex_rot], [ex_mod]; marker = :star5, color = :transparent, strokecolor = C_TRUTH, strokewidth = 1.0, markersize = 13)
text!(aB, ex_rot + 0.4, ex_mod - 0.03; text = "exact sideband =\nFloquet multiplier", fontsize = 6.5, align = (:left, :top))
scatter!(aB, [S.mult[2]], [S.mult[1]]; marker = :diamond, color = C_LAT, markersize = 7, strokecolor = :white, strokewidth = 0.5)
text!(aB, S.mult[2] + 1.2, S.mult[1] + 0.004; text = "lattice fit", fontsize = 6.5, align = (:left, :center))
nf = [r for r in S.rows if r.model == "neural_free"]
scatter!(aB, [r.mult[2] for r in nf], [r.mult[1] for r in nf]; marker = :utriangle, color = C_NN, markersize = 6, strokecolor = :white, strokewidth = 0.4)
text!(aB, mean(r.mult[2] for r in nf), maximum(r.mult[1] for r in nf) + 0.015; text = "phase–isostable\nnet, μ free", fontsize = 6.5, align = (:center, :bottom))
scatter!(aB, [G.slip_mu[2]], [G.slip_mu[1]]; marker = :diamond, color = :transparent, strokecolor = C_LAT, strokewidth = 1.0, markersize = 7)
text!(aB, G.slip_mu[2] + 0.8, G.slip_mu[1] - 0.012; text = "lattice fit,\nnoise-driven walks", fontsize = 6.5, align = (:left, :top))
text!(aB, 25.5, 0.5; text = "dots: Hankel-DMD eigenvalues\n(number: nearest harmonic)", fontsize = 6.5, color = CAT[1], align = (:right, :bottom))
plabel!(top[1, 2], "B")

# ── C: SLIP off-cycle metrics ────────────────────────────────────────────────
MODS = [("dmd", "Hankel-DMD", CAT[1], :circle), ("esn", "ESN", CAT[2], :circle), ("node", "neural ODE", CAT[3], :circle),
        ("lattice", "lattice fit", C_LAT, :diamond), ("neural_frozen", "phase–isostable\nnet, μ from lattice", C_NN, :utriangle),
        ("neural_free", "phase–isostable\nnet, μ free", C_NN, :dtriangle)]
gC = top[1, 3] = GridLayout()
c1 = Axis(gC[1, 1]; xlabel = "4-step rate\n|error|", yticks = (1:6, [m[2] for m in MODS]), yreversed = true,
          limits = (-0.01, 0.27, 0.4, 6.6), title = "SLIP, off-cycle starts", xticks = 0:0.1:0.2)
c2 = Axis(gC[1, 2]; xlabel = "off the true cycle\nafter 25 steps", yticksvisible = false, yticklabelsvisible = false,
          yreversed = true, limits = (-0.04, 0.85, 0.4, 6.6), xticks = 0:0.4:0.8)
vlines!(c2, median(S.truth.off25); color = C_TRUTH, linewidth = 0.8)
text!(c2, median(S.truth.off25) + 0.02, 0.45; text = "truth", fontsize = 6.5, align = (:left, :top))
for (k, (m, _, col, mk)) in enumerate(MODS)
    rs = [r for r in S.rows if r.model == m]; dy = length(rs) == 1 ? [0.0] : range(-0.18, 0.18; length = length(rs))
    e4 = [mean(abs.(r.rate4 .- S.truth.rate4)) for r in rs]; o25 = [median(r.off25) for r in rs]
    scatter!(c1, clamp.(e4, 0, 0.265), k .+ dy; color = col, marker = mk, markersize = 5, strokecolor = :white, strokewidth = 0.4)
    scatter!(c2, clamp.(o25, 0, 0.84), k .+ dy; color = col, marker = mk, markersize = 5, strokecolor = :white, strokewidth = 0.4)
end
colgap!(gC, 10); plabel!(gC[1, 1], "C")

# ── D: gait forecast skill ───────────────────────────────────────────────────
GR = [("SLIP\nwalks", G.slip_held), ("AB", G.held[G.grp .== "AB"]), ("HF", G.held[G.grp .== "HF"]), ("LF", G.held[G.grp .== "LF"])]
FM = [(:lat, "lattice", C_LAT, :diamond), (:ddmd, "deviation DMD", CAT[1], :circle), (:cdmd, "companion DMD", C_REF, :rect)]
YLO = -0.7
aD = Axis(bot[1, 1]; ylabel = "R², deviation (0 = mean stride)", title = "Forecasting the next strides",
          xticks = (1:4, first.(GR)), limits = (0.45, 4.55, YLO, 0.75))
hlines!(aD, 0.0; color = C_REF, linewidth = 0.6)
for (gi, (_, H)) in enumerate(GR)
    H = filter(!isnothing, H)
    for (mi, (f, _, col, mk)) in enumerate(FM)
        x = gi + (mi - 2) * 0.24
        v1 = [getfield(h, f)[2] for h in H]; v2 = [getfield(h, f)[3] for h in H]
        q1, m1, q3 = quantile(v1, (0.25, 0.5, 0.75))
        lines!(aD, [x, x], clamp.([q1, q3], YLO + 0.01, 0.74); color = col, linewidth = 1.2)
        scatter!(aD, [x], [clamp(m1, YLO + 0.02, 0.74)]; color = col, marker = mk, markersize = 6, strokecolor = :white, strokewidth = 0.4)
        scatter!(aD, [x + 0.08], [clamp(median(v2), YLO + 0.02, 0.74)]; color = :transparent, strokecolor = col, strokewidth = 0.8, marker = mk, markersize = 4.5)
        m1 < YLO && text!(aD, x, YLO + 0.05; text = @sprintf("%.1f", m1), fontsize = 5.5, color = C_REF, align = (:center, :bottom))
    end
end
for (mi, (_, lab, col, mk)) in enumerate(FM)
    scatter!(aD, [2.85], [0.71 - 0.085 * (mi - 1)]; color = col, marker = mk, markersize = 5)
    text!(aD, 2.93, 0.71 - 0.085 * (mi - 1); text = lab, fontsize = 6.5, align = (:left, :center))
end
text!(aD, 2.85, 0.45; text = "filled: +1 stride\nopen: +2 strides", fontsize = 6.5, color = C_REF, align = (:left, :top))
plabel!(bot[1, 1], "D")

# ── E: is the multiplier identified? forecast error against the assumed modulus ──
aE = Axis(bot[1, 2]; xlabel = "Assumed multiplier modulus per stride", ylabel = "Forecast error, strides +1, +2\n(1 = mean stride; median, IQR)",
          title = "Only SLIP walks pick a multiplier", limits = (0, 1, 0.45, 1.35), xticks = 0:0.25:1)
hlines!(aE, 1.0; color = C_REF, linewidth = 0.6)
vlines!(aE, ex_mod; color = C_TRUTH, linewidth = 0.6, linestyle = :dot)
text!(aE, ex_mod - 0.01, 0.47; text = "SLIP truth", fontsize = 6.5, align = (:right, :bottom))
crit(r) = (r.prof[:, :, 2] .+ r.prof[:, :, 3]) ./ (r.ref[2] + r.ref[3])
RH = 0.05:0.05:0.95
GLINES = [("SLIP walks", G.slip, G.slip_held, C_TRUTH)
          [(g, G.gait[G.grp .== g], G.held[G.grp .== g], GROUP_COL[g]) for g in ("AB", "HF", "LF")]]
for (k, (lab, R, _, col)) in enumerate(GLINES)
    P = reduce(hcat, [crit(r)[:, 1] for r in R if r !== nothing])          # rotation 0 (rotation is not identified)
    Q = [quantile(row, q) for row in eachrow(P), q in (0.25, 0.5, 0.75)]
    band!(aE, RH, clamp.(Q[:, 1], 0.45, 1.35), clamp.(Q[:, 3], 0.45, 1.35); color = (col, 0.12))
    lines!(aE, RH, clamp.(Q[:, 2], 0.45, 1.35); color = col, linewidth = 1.2)
    lines!(aE, [0.04, 0.1], fill(1.31 - 0.055 * (k - 1), 2); color = col, linewidth = 1.2)
    text!(aE, 0.12, 1.31 - 0.055 * (k - 1); text = lab, fontsize = 6.5, align = (:left, :center))
end
plabel!(bot[1, 2], "E")

# ── F: skill by horizon, chosen out of trial vs on the scored strides ────────
aF = Axis(bot[1, 3]; ylabel = "R², lattice model", title = "Gait skill does not transfer",
          xticks = (1:3, ["rest of\nstride", "+1\nstride", "+2\nstrides"]), limits = (0.6, 3.4, -0.15, 1.0))
hlines!(aF, 0.0; color = C_REF, linewidth = 0.6)
for (lab, _, H, col) in GLINES
    H = filter(!isnothing, H); sh = lab == "SLIP walks" ? NUL.slip : NUL.gait[lab]
    hon = [median(h.lat[b] for h in H) for b in 1:3]; ins = [median(h.insample[b] for h in H) for b in 1:3]
    lines!(aF, 1:3, [median(v[b] for v in sh) for b in 1:3]; color = col, linewidth = 0.9, linestyle = :dot)
    lines!(aF, 1:3, ins; color = col, linewidth = 0.8, linestyle = :dash)
    lines!(aF, 1:3, hon; color = col, linewidth = 1.2); scatter!(aF, 1:3, hon; color = col, markersize = 4)
end
text!(aF, 2.05, 0.66; text = "SLIP walks", fontsize = 6.5, align = (:left, :bottom))
text!(aF, 0.66, 0.19; text = "gait\n(colours as E)", fontsize = 6.5, align = (:left, :bottom))
for (k, (ls, lw, lab)) in enumerate(((:solid, 1.2, "μ out of trial"), (:dash, 0.8, "μ on scored strides"),
                                     (:dot, 0.9, "same, shuffled order")))
    y = 0.97 - 0.08 * (k - 1)
    lines!(aF, [1.7, 1.85], [y, y]; color = C_REF, linewidth = lw, linestyle = ls)
    text!(aF, 1.9, y; text = lab, fontsize = 6.5, color = C_REF, align = (:left, :center))
end
plabel!(bot[1, 3], "F")

colsize!(top, 1, Relative(0.34)); colsize!(top, 3, Relative(0.3)); colgap!(top, 16)
colsize!(bot, 1, Relative(0.34)); colgap!(bot, 16); rowgap!(fig.layout, 10)
save_paper(fig, "Figure_S08")
