#!/usr/bin/env julia
# Figure_S05.jl — How the phase-varying model of running deviations is built (manuscript: @suppfig-running-method). Run: julia --project=. scripts/figures/Figure_S05.jl
# How the phase-varying linear model of running deviations is built (maus_learning_all.jl), on one running trial.
#   A  six joint angles (standardised) and the stride boundaries
#   B  where the phase comes from: the two leading principal components of a half-stride delay window of all angles
#      trace a loop; phase = angle around it; a stride ends each time the phase completes a turn
#   C  each stride resampled to 100 samples; the slowly varying average stride (61-stride moving average) is removed
#   D  what is modelled: the deviation, its state (the last 8 samples) and a forecast 1/8 stride ahead
#   E  the model: one linear map per step whose matrix varies smoothly with phase (Fourier series in θ), fitted by
#      one regression on [z, cos θ·z, sin θ·z, …]; the same knee deviation evolves differently by where it starts
#   F  one forecast of the deviation: phase-varying linear model against time-invariant DMD
# Usage: julia --project=. scripts/figures/Figure_S05.jl   → figures/Figure_S05.{pdf,svg,png}

include(joinpath(@__DIR__, "..", "maus_learning_all.jl"))
include(joinpath(@__DIR__, "..", "paper_style.jl"))
BLAS.set_num_threads(1)

A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls")); tr_ = A.trials[1]; X = tr_.X; fs = tr_.fs
chn = ["hip R", "knee R", "ankle R", "hip L", "knee L", "ankle L"]
Xs = (X .- mean(X, dims = 1)) ./ std(X, dims = 1)
# phase as pair_phase does, keeping the principal components for panel B
h = hip_phase(X); per = 2π * (size(X, 1) - 1) / abs(h.ϕ[end] - h.ϕ[1]); Lw = max(4, round(Int, 0.5per))
Hk = hankel(Xs, Lw); Fe = eigen(Symmetric(Hk * Hk')); V = Fe.vectors[:, end:-1:end-1]; Z = V' * Hk
P = pair_phase(X)
cuts = let b = Float64[]
    for k in ceil(Int, P.ϕ[1] / 2π)+1:floor(Int, P.ϕ[end] / 2π)
        i = findfirst(>=(2π * k), P.ϕ); (i === nothing || i == 1) && continue
        push!(b, P.tc[i-1] + (P.tc[i] - P.tc[i-1]) * (2π * k - P.ϕ[i-1]) / (P.ϕ[i] - P.ϕ[i-1]))
    end
    b
end
S = cut_strides(X, P.ϕ, P.tc); Y, B, D = split_baseline(S, :ma); n = size(S, 1)

fig = Figure(size = (COL2, 150MM))
# A: raw angles, 3 s, with the stride boundaries
t0 = round(Int, cuts[100]); win = t0:t0+round(Int, 2.4fs)
axA = Axis(fig[1, 1:2]; xlabel = "time (s)", ylabel = "joint angle (SD)", title = "six joint angles; stride boundaries from the phase")
for c in 1:6
    lines!(axA, (win .- t0) ./ fs, Xs[win, c] .+ 0.0; color = c <= 3 ? (CAT[1], 0.4 + 0.2c) : (CAT[2], 0.4 + 0.2(c - 3)), linewidth = 0.9)
end
for b in cuts; first(win) <= b <= last(win) && vlines!(axA, [(b - t0) / fs]; color = C_TRUTH, linewidth = 0.8, linestyle = :dash); end
text!(axA, 0.01, 0.98; space = :relative, text = "blue: right hip, knee, ankle; orange: left", fontsize = 6, align = (:left, :top))
panel_label!(fig[1, 1:2], "A")
# B: the loop of the two leading components
axB = Axis(fig[1, 3]; aspect = 1, xlabel = "component 1", ylabel = "component 2", title = "phase θ: angle on the loop")
seg = t0:t0+round(Int, 4per)
lines!(axB, Z[1, seg], Z[2, seg]; color = (CAT[3], 0.8), linewidth = 0.8)
i0 = t0 + round(Int, 0.3per); lines!(axB, [0, Z[1, i0]], [0, Z[2, i0]]; color = C_TRUTH, linewidth = 0.8)
scatter!(axB, [Z[1, i0]], [Z[2, i0]]; color = C_TRUTH, markersize = 5)
text!(axB, 0.35Z[1, i0], 0.35Z[2, i0]; text = "θ", fontsize = 8, offset = (4, 0))
text!(axB, 0.03, 0.03; space = :relative, text = "½-stride window, 6 angles", fontsize = 6)
panel_label!(fig[1, 3], "B")
# C: strides, average stride, deviation (right knee)
ch = 2; k0 = 150
axC = Axis(fig[2, 1]; xlabel = "% of stride", ylabel = "$(chn[ch]) (SD)", title = "strides resampled to 100 samples")
for k in k0-10:k0+10; lines!(axC, 1:100, Y[(k-1)*100+1:k*100, ch]; color = (C_REF, 0.35), linewidth = 0.6); end
lines!(axC, 1:100, B[(k0-1)*100+1:k0*100, ch]; color = C_TRUTH, linewidth = 1.4, label = "average stride (moving, 61 strides)")
lines!(axC, 1:100, Y[(k0-1)*100+1:k0*100, ch]; color = CAT[2], linewidth = 1.0, label = "one stride")
axislegend(axC; position = :lt, labelsize = 6)
ylims!(axC, nothing, 4.9)
panel_label!(fig[2, 1], "C")
# D: deviation, delay window, forecast horizon
axD = Axis(fig[2, 2:3]; xlabel = "stride", ylabel = "deviation, $(chn[ch]) (SD)",
           title = "modelled: the deviation from the average stride")
rr = (k0-1)*100+1:(k0+3)*100
lines!(axD, (rr .- 1) ./ 100, D[rr, ch]; color = CAT[2], linewidth = 1)
tt = (k0 - 1) * 100 + 160
vspan!(axD, (tt - 8) / 100, (tt - 1) / 100; color = (CAT[3], 0.35))
scatter!(axD, [(tt - 1 + 12) / 100], [D[tt+12, ch]]; color = C_TRUTH, marker = :star5, markersize = 9)
text!(axD, (tt - 8) / 100, maximum(D[rr, ch]); text = "state z: last 8 samples", fontsize = 6, align = (:right, :top), offset = (-2, 0))
text!(axD, (tt + 11) / 100, D[tt+12, ch]; text = "forecast ⅛ stride ahead", fontsize = 6, align = (:left, :bottom), offset = (4, 2))
for k in k0:k0+3; vlines!(axD, [k]; color = (C_REF, 0.5), linewidth = 0.5); end
refline!(axD, 0)
panel_label!(fig[2, 2:3], "D")
# E: the model, and its per-step gain around the stride
rows = 1:150*100
M = fit_ltp(D, rows, 8, 32, 8, [1e-2])[1]
gE = fig[3, 1] = GridLayout()
axE = Axis(gE[1, 1]; xlabel = "stride after the deviation", ylabel = "knee R deviation (relative)",
           title = "same deviation, different phase")
p_ = size(D, 2); dd = M.d
for (j, ph) in enumerate((10, 35, 60, 85))
    hvec = zeros(p_ * dd); for l in 0:dd-1; hvec[l*p_+ch] = 1.0; end      # knee deviation held over the last 8 samples
    z = M.U' * hvec; y = Float64[]
    for st in 1:50; z = M.Aph[mod(ph + st - 1, NPH)+1] * z; push!(y, (M.U*z)[end-p_+ch]); end
    lines!(axE, (1:50) ./ 100, y; color = (CAT[3], 0.35 + 0.2j), linewidth = 1.1, label = "starts at $(ph)%")
end
refline!(axE, 0)
ylims!(axE, -0.4, 1.55)
axislegend(axE; position = :rt, labelsize = 5.5, rowgap = 0, patchsize = (8, 4), nbanks = 2)
Label(fig[4, 1:3], "Model (E, F): zₜ = the last 8 samples of the 6 deviations, reduced to their leading components;  θₜ = 2π × (position in the stride).\nzₜ₊₁ = A(θₜ) zₜ  with  A(θ) = W₀ + Σₖ (Wₖᶜ cos kθ + Wₖˢ sin kθ), k ≤ 8, fitted by one ridge regression of zₜ₊₁ on [zₜ, cos θₜ·zₜ, sin θₜ·zₜ, …, sin 8θₜ·zₜ].\nTime-invariant DMD is the same model with k = 0 (A constant).";
      fontsize = 6.5, tellwidth = false, halign = :left)
panel_label!(gE[1, 1], "E")
# F: one forecast, phase-varying linear against time-invariant DMD (both fitted on strides 1–150)
Fd = opfit(D[rows, :]; d = 16, r = 32); μd = mean(D[rows, :], dims = 1)
ts = (200 - 1) * 100 + 37
pl = ltp_predictor(D, M)(ts); pd = roll(Fd..., μd, D, ts, 16, 100)
axF = Axis(fig[3, 2:3]; xlabel = "stride after the forecast start", ylabel = "deviation, $(chn[ch]) (SD)",
           title = "one forecast from a held-out stride (strides 1–150 fitted)")
hz = 1:50
lines!(axF, (-30:50) ./ 100, D[ts-30:ts+50, ch]; color = C_TRUTH, linewidth = 1.1, label = "what happened")
lines!(axF, hz ./ 100, pl[hz, ch]; color = CAT[3], linewidth = 1.2, label = "phase-varying linear")
lines!(axF, hz ./ 100, pd[hz, ch]; color = CAT[1], linewidth = 1.2, linestyle = :dash, label = "time-invariant DMD")
vlines!(axF, [0]; color = C_REF, linewidth = 0.6); refline!(axF, 0)
axislegend(axF; position = :lb, labelsize = 6)
panel_label!(fig[3, 2:3], "F")
rowsize!(fig.layout, 1, Relative(0.3))
save_paper(fig, "Figure_S05")
@printf("trial %s r%d: %d strides, fs %d Hz, stride %.2f s; multipliers of the fitted monodromy: %s\n", tr_.subj, tr_.rep, n, fs, per / fs,
        join([@sprintf("%.2f", x) for x in sort(abs.(eigvals(monodromy(M))), rev = true)[1:4]], ", "))
