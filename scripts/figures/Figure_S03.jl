#!/usr/bin/env julia
# Figure_S03.jl — Large kicks to coupled FHN (manuscript: @suppfig-large-kicks). Run: julia --project=. scripts/figures/Figure_S03.jl
# Large kicks to coupled FHN (smooth regime): examples and the summary over every held-out kick (R1 at large amplitude).
#   A, B  one amplitude kick and one coordination kick (chosen from the truth before fitting): v₁ and w₁ − w₂
#   C, D  forecast error per cycle after the kick, median and interquartile range over the 8 held-out kicks of each
#         family (each kick: median over seeds), against the forecast that the kick changed nothing (the unkicked run)
#   E     coordination kicks: distance to the true cycle per cycle, truth against each model's forecast
# Data: cfhn_large_kicks_data.jl; fits: cfhn_large_kicks_models.jl (dmd, esn, lstm <seed>).
# Usage: julia --project=. scripts/figures/Figure_S03.jl   → figures/Figure_S03.{pdf,svg,png}

include(joinpath(@__DIR__, "..", "fig6_relaxation_data.jl"))                 # dist_curve
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Printf

const root = joinpath(dirname(dirname(@__DIR__)), "cache", "cfhn_smooth", "large_kicks")
D = deserialize(joinpath(root, "data.jls")); n = D.n; K = 12n
standard(Y) = (Y .- D.μ) ./ D.σ
models = Any[("DMD", deserialize(joinpath(root, "dmd.jls")).R, CAT[1]), ("ESN", deserialize(joinpath(root, "esn.jls")).R, CAT[2])]
lstm_files = filter(f -> occursin(r"^lstm_seed\d+\.jls$", f), readdir(root))
isempty(lstm_files) || push!(models, ("LSTM", reduce(vcat, [deserialize(joinpath(root, f)).R for f in lstm_files]), CAT[3]))

percycle(e) = [sqrt(mean(abs2, e[(k-1)*n+1:k*n, :])) for k in 1:12]           # RMS within each cycle
truthz(T) = standard(T.truth[2:K+1, :])
err_model(f, T) = f.ok ? percycle(f.F[1:K, :] .- truthz(T)) : fill(NaN, 12)
err_ref(T) = percycle(standard(T.control[2:K+1, :]) .- truthz(T))
dist_cyc(F) = [sqrt(mean(abs2, dist_curve(F[(k-1)*n+1:k*n, :], D.cycle; closed = true))) for k in 1:12]
medseed(v) = (ok = filter(x -> all(isfinite, x), v); isempty(ok) ? fill(NaN, 12) : vec(median(reduce(hcat, ok), dims = 2)))
q(M, p) = [quantile(filter(isfinite, M[k, :]), p) for k in axes(M, 1)]
fams = ("amplitude", "coordination")
idx = Dict(f => findall(t -> t.kind == f, D.test) for f in fams)

fig = Figure(size = (COL2, 118MM))
gtop = fig[1, 1] = GridLayout(); gbot = fig[2, 1] = GridLayout()
tt = collect(0:K) ./ n
# A, B: examples
for (j, (ix, fam, ch, lab, xmax)) in enumerate(zip(D.chosen, fams, (1, 0), ("v₁", "w₁ − w₂"), (4, 12)))
    T = D.test[ix]
    ax = CairoMakie.Axis(gtop[1, j]; xlabel = "cycles after kick", ylabel = lab,
              title = fam == "amplitude" ? "a kick along the cycle's amplitude" : "a kick to the coordination (anti-phase)")
    obs(Y) = ch == 1 ? Y[:, 1] : Y[:, 2] .- Y[:, 4]
    for (nm, R, col) in models, row in R
        f = row.tests[ix]; f.ok || continue
        F = vcat(standard(T.truth[1:1, :]), f.F[1:K, :]) .* D.σ .+ D.μ
        CairoMakie.lines!(ax, tt, obs(F); color = (col, nm == "DMD" ? 0.9 : 0.55), linewidth = 0.9)
    end
    CairoMakie.lines!(ax, tt, obs(T.truth[1:K+1, :]); color = C_TRUTH, linestyle = :dash, linewidth = 1.1)
    CairoMakie.xlims!(ax, 0, xmax)
    panel_label!(gtop[1, j], ("A", "B")[j])
end
CairoMakie.Legend(gtop[1, 3], vcat([LineElement(color = c) for (_, _, c) in models], [LineElement(color = C_TRUTH, linestyle = :dash),
                         LineElement(color = C_REF, linestyle = :dot)]),
       vcat([nm == "DMD" ? nm : "$nm (each seed)" for (nm, _, _) in models], ["truth", "no-kick forecast"]); framevisible = false)
CairoMakie.colsize!(gtop, 3, Auto(0.32))
# C, D: forecast error per cycle, all held-out kicks
summary = Dict{Tuple{String, String}, Any}()
for (j, fam) in enumerate(fams)
    ax = CairoMakie.Axis(gbot[1, j]; xlabel = "cycle after kick", ylabel = j == 1 ? "forecast error (SD)" : "",
              title = "$(fam) kicks: all $(length(idx[fam])) held out", xticks = 1:2:11)
    ref = reduce(hcat, [err_ref(D.test[i]) for i in idx[fam]])
    CairoMakie.lines!(ax, 1:12, q(ref, 0.5); color = C_REF, linestyle = :dot)
    for (nm, R, col) in models
        M = reduce(hcat, [medseed([err_model(row.tests[i], D.test[i]) for row in R]) for i in idx[fam]])
        CairoMakie.band!(ax, 1:12, q(M, 0.25), q(M, 0.75); color = (col, 0.18))
        CairoMakie.scatterlines!(ax, 1:12, q(M, 0.5); color = col, markersize = 4)
        summary[(nm, fam)] = (; M, ref)
    end
    CairoMakie.ylims!(ax, 0, nothing)
    panel_label!(gbot[1, j], ("C", "D")[j])
end
# E: recovery of the coordination kicks, distance to the true cycle
axE = CairoMakie.Axis(gbot[1, 3]; xlabel = "cycle after kick", ylabel = "distance to true cycle (SD)",
           title = "coordination kicks: recovery", xticks = 1:2:11)
tru = reduce(hcat, [dist_cyc(truthz(D.test[i])) for i in idx["coordination"]])
band!(axE, 1:12, q(tru, 0.25), q(tru, 0.75); color = (C_TRUTH, 0.12))
lines!(axE, 1:12, q(tru, 0.5); color = C_TRUTH, linestyle = :dash)
recov = Dict{String, Any}()
for (nm, R, col) in models
    M = reduce(hcat, [medseed([row.tests[i].ok ? dist_cyc(row.tests[i].F[1:K, :]) : fill(NaN, 12) for row in R]) for i in idx["coordination"]])
    CairoMakie.band!(axE, 1:12, q(M, 0.25), q(M, 0.75); color = (col, 0.18))
    CairoMakie.scatterlines!(axE, 1:12, q(M, 0.5); color = col, markersize = 4)
    recov[nm] = M
end
CairoMakie.ylims!(axE, 0, nothing)
panel_label!(gbot[1, 3], "E")
CairoMakie.rowsize!(fig.layout, 1, Relative(0.45))
save_paper(fig, "Figure_S03")

# numbers for the text
for fam in fams, (nm, _, _) in models
    s = summary[(nm, fam)]
    e13(M) = [sqrt(mean(abs2, M[1:3, i])) for i in axes(M, 2)]
    m, r = e13(s.M), e13(s.ref)
    @printf("%-5s %-12s cycles 1–3 error: median %.2f (IQR %.2f–%.2f) | no-kick forecast %.2f | better than no-kick in %d/%d kicks\n",
            nm, fam, median(m), quantile(m, 0.25), quantile(m, 0.75), median(r), count(m .< r), length(m))
end
for (nm, M) in recov
    @printf("coordination recovery %-5s: distance cycle 1 %.2f, cycle 6 %.2f, cycle 12 %.2f (truth %.2f, %.2f, %.2f)\n", nm,
            median(M[1, :]), median(M[6, :]), median(M[12, :]), median(tru[1, :]), median(tru[6, :]), median(tru[12, :]))
end
