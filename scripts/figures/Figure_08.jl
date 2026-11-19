#!/usr/bin/env julia
# Figure_08.jl — Recovering known phase-dependent structure and Floquet multipliers (R5) (manuscript: @fig-recovery).
# Run: julia --project=. scripts/figures/Figure_08.jl
# Ground truth: z_{t+1} = A(θ_t) z_t + noise, 3 states, 100 samples per cycle, A(θ) = I + B(θ)/100 with harmonics 0–2
# only; true multipliers per cycle 0.86 and a complex pair of modulus 0.15 (ltp_recovery.jl, ltp_recovery_ssm.py).
#   A  coefficients of A(θ) − I (per cycle): truth and the Fourier-in-phase fit (4 harmonics) from 64 cycles
#   B  two entries of A(θ) − I over the cycle, 64 cycles: truth, Fourier in phase, 20-bin phase-binned one-step maps,
#      time-invariant fit; median and interquartile range of 20 simulations
#   C  error of A(θ) (RMS over phases, relative to A(θ)'s variation over the cycle) against cycles of data
#   D  Floquet multipliers read from every model class, 64 cycles (linear: eigenvalues of the product of the per-sample
#      maps over a cycle; ESN, neural ODE, LSTM: the closed loop linearised over one cycle along its own orbit, the
#      three largest; SSMLearnPy: the linear part of its autonomous reduced map)
#   E  error of the multipliers (mean distance to the truth after the best matching) against cycles of data
#   F  a structure read from a model of the whole signal: transient growth (TG = max over one stride of ‖Ãᵏ‖) of the
#      companion paper's operator (delay DMD, rank 70, ridge 1e-4, stride-rescaled time) against the delay window, for
#      the 299 15-s gait trials (solid; groups AB, HF, LF in greys) and for each trial's mean cycle repeated with no
#      deviations (dashed); medians (fig5_gait_data.jl, cache/fig5/gait.jls). Dotted: the companion paper's window
#      (32 samples ≈ ⅓ stride). A periodic signal has nothing to amplify, yet over a window shorter than the stride its
#      harmonics are non-orthogonal and TG > 1; at a full-stride window TG = 1
#   G  what the deviations add: per-trial Δ log₁₀ TG, recorded − mean cycle only; median and interquartile range
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf, LinearAlgebra, NPZ

const ROOT = dirname(dirname(@__DIR__)); const RD = joinpath(ROOT, "cache", "recovery"); const NPH = 100
L = deserialize(joinpath(RD, "linear.jls"))
const C_LTP = colorant"#1a4f99"; const C_LPV = colorant"#78b0f5"
DIV = cgrad([CAT[1], colorant"#f4f3f0", CAT[2]])
# model, label, colour, marker, line style; nonlinear models dashed (as Figure 5)
MODELS = [(:lifted, "Fourier in phase", C_LTP, :diamond, :solid), (:binned, "phase-binned", C_LPV, :rect, :solid),
          (:dmd, "time-invariant (DMD)", CAT[1], :circle, :solid), (:esn, "ESN", CAT[2], :utriangle, :dash),
          (:lstm, "LSTM", CAT[5], :dtriangle, :dash), (:node, "neural ODE", CAT[3], :hexagon, :dash),
          (:ssm, "SSMLearnPy", CAT[4], :star5, :dash)]
NL = Dict{Symbol, Any}()
for m in (:esn, :node, :lstm)
    f = joinpath(RD, "$(m).jls"); isfile(f) || continue
    R = deserialize(f).res; R = filter(!isnothing, R)
    NL[m] = (; nc = [x.nc for x in R], err = [x.err for x in R], mult = [sort(x.mult; by = abs, rev = true)[1:3] for x in R])
end
if isfile(joinpath(RD, "ssm.npz"))
    S = npzread(joinpath(RD, "ssm.npz")); NL[:ssm] = (; nc = S["nc"], err = S["err"], mult = [S["mult"][i, :] for i in axes(S["mult"], 1)])
end
r = 3; nfit = 2L.HFIT + 1
Ξ(W) = NPH .* (W .- hcat(Matrix(1.0I, r, r), zeros(r, (nfit - 1) * r)))
Wtrue_full = hcat(L.WTRUE, zeros(r, 2(L.HFIT - L.HTRUE) * r))

fig = Figure(size = (COL2, 138MM))
top = fig[1, 1] = GridLayout(); bot = fig[2, 1] = GridLayout()
# A: coefficients
gX = top[1, 1] = GridLayout(); UX = 1.9MM; lim = maximum(abs, Ξ(Wtrue_full))
for (row, (W, ttl)) in enumerate(((Wtrue_full, "true coefficients of A(θ) − I (per cycle)"), (L.fits64[1].Wl, "Fourier in phase, fitted from 64 cycles")))
    ax = Axis(gX[row, 1]; width = nfit * r * UX, height = r * UX, yreversed = true, title = ttl, titlesize = 6.5, titlegap = 1.5)
    hidedecorations!(ax); hidespines!(ax)
    heatmap!(ax, 1:size(W, 2), 1:r, permutedims(Ξ(W)); colormap = DIV, colorrange = (-lim, lim))
    for s in r .* (1:nfit-1); vlines!(ax, s + 0.5; color = C_TRUTH, linewidth = 0.6); end
end
axl = Axis(gX[3, 1]; width = nfit * r * UX, height = 5MM, limits = (0.5, nfit * r + 0.5, 0, 1)); hidedecorations!(axl); hidespines!(axl)
for (i, l) in enumerate(vcat(["1"], [f * "\n" * (h == 1 ? "θ" : "$(h)θ") for h in 1:L.HFIT for f in ("cos", "sin")]))
    text!(axl, (i - 0.5) * r + 0.5, 0.5; text = l, fontsize = 5.5, align = (:center, :center))
end
Colorbar(gX[1:2, 2]; colormap = DIV, limits = (-lim, lim), width = 4, ticklabelsize = 6, ticksize = 2, height = 2r * UX + 6MM)
Label(gX[4, 1], "harmonics 3 and 4 come out near zero without thresholding"; fontsize = 6, color = C_REF, tellwidth = false)
rowgap!(gX, 2); panel_label!(gX, "A")
# B: two entries over the cycle
gE = top[1, 2] = GridLayout(); θs = 2π .* (0:NPH-1) ./ NPH; binc = 2π .* ((0:L.NB-1) .+ 0.5) ./ L.NB
fbk(θ, H) = vcat(1.0, [f(h * θ) for h in 1:H for f in (cos, sin)])
for (k, ((i, j), lab)) in enumerate((((1, 3), "A₁₃"), ((1, 1), "A₁₁ − 1")))
    ax = Axis(gE[k, 1]; width = 30MM, height = 14MM, ylabel = lab, xticks = ([0, π, 2π], ["0", "π", "2π"]), xticklabelsvisible = k == 2,
              xlabel = k == 2 ? "phase θ" : "", title = k == 1 ? "two entries of A(θ) − I, 64 cycles" : "", titlesize = 6.5,
              xlabelsize = 6.5, ylabelsize = 6.5, xticklabelsize = 6, yticklabelsize = 6)
    tr = [NPH * (A[i, j] - (i == j)) for A in L.Atrue]
    Lf = reduce(hcat, [[NPH * (sum(fbk(θ, L.HFIT)[q] .* g.Wl[:, (q-1)*r+1:q*r] for q in 1:nfit)[i, j] - (i == j)) for θ in θs] for g in L.fits64])
    Bf = reduce(hcat, [[NPH * (g.Ab[b][i, j] - (i == j)) for b in 1:L.NB] for g in L.fits64])
    D0 = [NPH * (g.W0[i, j] - (i == j)) for g in L.fits64]
    band!(ax, θs, [quantile(row, 0.25) for row in eachrow(Lf)], [quantile(row, 0.75) for row in eachrow(Lf)]; color = (C_LTP, 0.2))
    hlines!(ax, median(D0); color = CAT[1], linewidth = 0.9)
    lines!(ax, θs, median.(eachrow(Lf)); color = C_LTP, linewidth = 1.2)
    lines!(ax, θs, tr; color = C_TRUTH, linewidth = 0.9, linestyle = (:dot, :dense))
    rangebars!(ax, binc, [quantile(row, 0.25) for row in eachrow(Bf)], [quantile(row, 0.75) for row in eachrow(Bf)]; color = C_LPV, whiskerwidth = 0, linewidth = 0.9)
    scatter!(ax, binc, median.(eachrow(Bf)); color = C_LPV, marker = :rect, markersize = 3.5)
end
rowgap!(gE, 3); panel_label!(gE, "B")
# C: error of A(θ)
axC = Axis(top[1, 3]; width = 30MM, height = 26MM, xscale = log2, yscale = log10, xticks = L.CYC, xlabel = "cycles of data",
           ylabel = "error of A(θ), relative\nto its variation over θ", title = "recovering A(θ)", titlesize = 6.5, xlabelsize = 6.5,
           ylabelsize = 6.5, xticklabelsize = 6, yticklabelsize = 6, yticks = ([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"]))
for (m, _, col, mk, ls) in MODELS[1:3]
    C = L.curves[m]
    band!(axC, L.CYC, [quantile(C[i, :], 0.25) for i in eachindex(L.CYC)], [quantile(C[i, :], 0.75) for i in eachindex(L.CYC)]; color = (col, 0.2))
    scatterlines!(axC, L.CYC, median.(eachrow(C)); color = col, marker = mk, markersize = 4, linestyle = ls, linewidth = 1.1)
end
panel_label!(top[1, 3], "C")
colgap!(top, 14)
# D: multipliers at 64 cycles
θc = range(0, 2π; length = 200); i64 = findfirst(==(64), L.CYC)
axD = Axis(bot[1, 1]; width = 34MM, height = 34MM, aspect = DataAspect(), limits = (-1.05, 1.05, -1.05, 1.05), xlabel = "Re ρ", ylabel = "Im ρ",
           title = "multipliers read from each model, 64 cycles", titlesize = 6.5, xlabelsize = 6.5, ylabelsize = 6.5, xticklabelsize = 6,
           yticklabelsize = 6, xticks = [-1, 0, 1], yticks = [-1, 0, 1])
lines!(axD, cos.(θc), sin.(θc); color = C_REF, linewidth = 0.6, linestyle = :dot)
hlines!(axD, 0; color = (C_REF, 0.4), linewidth = 0.4); vlines!(axD, 0; color = (C_REF, 0.4), linewidth = 0.4)
for (m, _, col, mk, _) in reverse(MODELS)
    e = if m in (:lifted, :binned, :dmd)
        reduce(vcat, L.EIG[m][i64, :])
    elseif haskey(NL, m)
        reduce(vcat, [x for (x, n) in zip(NL[m].mult, NL[m].nc) if n == 64])
    else
        ComplexF64[]
    end
    isempty(e) || scatter!(axD, real.(e), imag.(e); color = (col, 0.6), marker = mk, markersize = 4)
end
scatter!(axD, real.(L.ρTRUE), imag.(L.ρTRUE); marker = :star5, color = :transparent, strokecolor = C_TRUTH, strokewidth = 0.9, markersize = 11)
panel_label!(bot[1, 1], "D")
# E: error of the multipliers against data
axE = Axis(bot[1, 2]; width = 44MM, height = 30MM, xscale = log2, yscale = log10, xticks = L.CYC, xlabel = "cycles of data",
           ylabel = "distance from the\ntrue multipliers", title = "error of the multipliers", titlesize = 6.5, xlabelsize = 6.5,
           ylabelsize = 6.5, xticklabelsize = 6, yticklabelsize = 6, yticks = ([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"]))
for (m, _, col, mk, ls) in MODELS
    if m in (:lifted, :binned, :dmd)
        C = L.ERR[m]; xs = L.CYC; md = median.(eachrow(C)); q1 = [quantile(c, 0.25) for c in eachrow(C)]; q3 = [quantile(c, 0.75) for c in eachrow(C)]
    elseif haskey(NL, m)
        xs = sort(unique(NL[m].nc)); v = [NL[m].err[NL[m].nc .== n] for n in xs]
        md = median.(v); q1 = [quantile(x, 0.25) for x in v]; q3 = [quantile(x, 0.75) for x in v]
    else
        continue
    end
    band!(axE, xs, q1, q3; color = (col, 0.15))
    scatterlines!(axE, xs, md; color = col, marker = mk, markersize = 4.5, linestyle = ls, linewidth = 1.1)
end
panel_label!(bot[1, 2], "E")
leg = [[LineElement(color = col, linestyle = ls), MarkerElement(color = col, marker = mk, markersize = 5)] for (_, _, col, mk, ls) in MODELS]
Legend(bot[1, 3], vcat([MarkerElement(marker = :star5, color = :white, strokecolor = C_TRUTH, strokewidth = 0.9, markersize = 9),
                        LineElement(color = C_TRUTH, linestyle = (:dot, :dense))], leg),
       vcat(["true multipliers", "true A(θ)"], [nm for (_, nm, _, _, _) in MODELS]); framevisible = false, labelsize = 6.5,
       patchsize = (12, 5), rowgap = 1, tellheight = false)
colgap!(bot, 16)
# F, G: a structure read from a whole-signal operator inherits the mean cycle (gait; fig5_gait_data.jl)
GD = deserialize(joinpath(ROOT, "cache", "fig5", "gait.jls")); W = GD.windows
GREY_GROUP = Dict("AB" => colorant"#2b2b2b", "HF" => colorant"#77766f", "LF" => colorant"#b3b2ad")
MK_GROUP = Dict("AB" => :circle, "HF" => :rect, "LF" => :utriangle)
qf(v, p) = quantile(filter(isfinite, v), p)
row3 = fig[3, 1] = GridLayout()
TS9 = (; titlesize = 6.5, xlabelsize = 6.5, ylabelsize = 6.5, xticklabelsize = 6, yticklabelsize = 6)
axF = Axis(row3[1, 1]; width = 62MM, height = 24MM, xlabel = "delay window (fraction of a stride)", ylabel = "transient growth\n(log₁₀ max ‖Ãᵏ‖)",
           title = "gait: transient growth of a whole-signal operator", xticks = (W ./ 100, string.(W ./ 100)),
           limits = (0.1, 1.12, -0.03, 0.95), TS9...)
vlines!(axF, 0.32; color = C_REF, linestyle = :dot, linewidth = 0.8)
text!(axF, 0.335, 0.93; text = "companion\npaper", fontsize = 5.5, color = C_REF, align = (:left, :top))
for gr in GROUP_ORDER_FIG
    k = GD.grp .== gr; col = GREY_GROUP[gr]
    re = [GD.TG[(:real, d)][k] for d in W]; mc = [GD.TG[(:mean, d)][k] for d in W]
    scatterlines!(axF, W ./ 100, median.(re); color = col, marker = MK_GROUP[gr], markersize = 4, linewidth = 1.3)
    lines!(axF, W ./ 100, median.(mc); color = col, linewidth = 1.1, linestyle = :dash)
    text!(axF, 0.52, median(re[3]) + 0.015; text = gr, color = col, fontsize = 6, align = (:left, :bottom))
end
axislegend(axF, [LineElement(color = C_REF), LineElement(color = C_REF, linestyle = :dash)], ["recorded trials", "mean cycle only"];
           position = (:right, :top), framevisible = false, labelsize = 6, rowgap = 0, margin = (0, 14, 0, 0), patchsize = (12, 5))
panel_label!(row3[1, 1], "F")
axG = Axis(row3[1, 2]; width = 62MM, height = 24MM, xlabel = "delay window (fraction of a stride)", ylabel = "added by deviations\n(Δ log₁₀ TG)",
           title = "what the deviations add", xticks = (W ./ 100, string.(W ./ 100)), limits = (0.1, 1.12, 0, 0.33), TS9...)
vlines!(axG, 0.32; color = C_REF, linestyle = :dot, linewidth = 0.8)
for gr in GROUP_ORDER_FIG
    k = GD.grp .== gr; col = GREY_GROUP[gr]
    Δ = [GD.TG[(:real, d)][k] .- GD.TG[(:mean, d)][k] for d in W]
    band!(axG, W ./ 100, qf.(Δ, 0.25), qf.(Δ, 0.75); color = (col, 0.1))
    scatterlines!(axG, W ./ 100, median.(Δ); color = col, marker = MK_GROUP[gr], linewidth = 1.3, markersize = 4)
    text!(axG, 1.02, median(Δ[end]); text = gr, color = col, fontsize = 6, align = (:left, :center))
end
panel_label!(row3[1, 2], "G")
colgap!(row3, 22); rowgap!(fig.layout, 8)
save_paper(fig, "Figure_08")

for (m, _, _, _, _) in MODELS
    if m in (:lifted, :binned, :dmd)
        @printf("%-7s multiplier error by cycles: %s\n", m, join([@sprintf("%d: %.3f", c, median(L.ERR[m][i, :])) for (i, c) in enumerate(L.CYC)], ", "))
    elseif haskey(NL, m)
        @printf("%-7s multiplier error by cycles: %s\n", m, join([@sprintf("%d: %.3f", c, median(NL[m].err[NL[m].nc .== c])) for c in sort(unique(NL[m].nc))], ", "))
    end
end
KEY5 = Dict(:lifted => "FIP", :binned => "LPV", :dmd => "DMD", :esn => "ESN", :lstm => "LSTM", :node => "NODE", :ssm => "SSM")
sc5 = Dict{String, String}()
for (m, _, _, _, _) in MODELS
    haskey(KEY5, m) || continue
    v = m in (:lifted, :binned, :dmd) ? L.ERR[m][findfirst(==(64), L.CYC), :] : haskey(NL, m) ? NL[m].err[NL[m].nc .== 64] : Float64[]
    isempty(v) || (sc5[KEY5[m]] = @sprintf("%.2f", median(v)))
end
scorecard!("R5", sc5)
