#!/usr/bin/env julia
# Review figures for current manuscript Figure 4A: all seeds, all four kick phases.
using Pkg; Pkg.activate(dirname(@__DIR__); io=devnull)
include("paper_style.jl")
using Serialization

D = deserialize(joinpath(dirname(@__DIR__), "cache", "fig6", "cfhn_trajectory_diagnostics.jls"))
MODELS = [("dmd", "Hankel-DMD", CAT[1]), ("esn", "ESN", CAT[2]), ("node", "Neural ODE", CAT[3])]
styles = [:solid, :dash, :dot, :dashdot]
for k in 1:4
    fig = Figure(size=(COL2, 190MM))
    resp = Figure(size=(COL2, 150MM))
    state_axes = Matrix{Axis}(undef, 4, 3)
    response_axes = Matrix{Axis}(undef, 3, 3)
    Label(fig[0, 1:3], "Coupled FHN, start $k — exact trajectories underlying Figure 4A"; fontsize=10)
    Label(resp[0, 1:3], "Coupled FHN, start $k — desynchrony and distance to the true cycle"; fontsize=10)
    for (c, (model, lab, col)) in enumerate(MODELS)
        rs = filter(r -> r.model == model, D.rows)
        a = first(rs).starts[k]
        t = (1:size(a.raw, 1)) ./ D.spc
        tw = (1:D.warmup) ./ D.spc .- D.warmup/D.spc
        for j in 1:4
            ax = Axis(fig[j, c]; title=j == 1 ? lab : "",
                      ylabel=c == 1 ? ["v₁", "w₁", "v₂", "w₂"][j] * " (raw)" : "",
                      xlabel=j == 4 ? "Cycles since autonomous release" : "")
            state_axes[j, c] = ax
            vspan!(ax, first(tw), 0; color=C_SHADE)
            lines!(ax, tw, a.warmup[:, j]; color=C_TRUTH)
            for (i, r) in enumerate(rs)
                lines!(ax, t, r.starts[k].raw[:, j]; color=(col, 0.8),
                       linestyle=styles[i], linewidth=0.9, label="seed $(r.seed)")
            end
            lines!(ax, t, a.truth[:, j]; color=C_TRUTH, linewidth=1.2, label="truth")
            xlims!(ax, first(tw), 5)
            j == 1 && axislegend(ax; position=:rt, labelsize=6)
        end
        for j in 1:3
            ax = Axis(resp[j, c]; title=j == 1 ? lab : "",
                      ylabel=c == 1 ? ["v₁ − v₂ (raw)", "w₁ − w₂ (raw)", "Distance to true cycle (SD units)"][j] : "",
                      xlabel=j == 3 ? "Cycles since autonomous release" : "")
            response_axes[j, c] = ax
            if j < 3
                vspan!(ax, first(tw), 0; color=C_SHADE)
                lines!(ax, tw, a.warmup[:, j] .- a.warmup[:, j+2]; color=C_TRUTH)
                hlines!(ax, 0; color=C_REF, linestyle=:dot, linewidth=0.5)
            end
            for (i, r) in enumerate(rs)
                b = r.starts[k]
                y = j == 3 ? b.fd : b.raw[:, j] .- b.raw[:, j+2]
                lines!(ax, t, y; color=(col, 0.75), linestyle=styles[i], linewidth=0.8,
                       label="seed $(r.seed)")
                j == 3 && scatterlines!(ax, (0:4).+0.5, b.model_env; color=col,
                                        linestyle=styles[i], markersize=3, linewidth=1.2)
            end
            y = j == 3 ? a.td : a.truth[:, j] .- a.truth[:, j+2]
            lines!(ax, t, y; color=C_TRUTH, linewidth=1.3, label="truth")
            j == 3 && scatterlines!(ax, (0:4).+0.5, a.truth_env; color=C_TRUTH, markersize=4)
            xlims!(ax, j == 3 ? 0 : first(tw), 5)
            j == 1 && axislegend(ax; position=:rt, labelsize=6)
        end
    end
    for j in 1:4
        linkyaxes!(state_axes[j, :]...)
    end
    for j in 1:3
        linkyaxes!(response_axes[j, :]...)
    end
    Label(fig[5, 1:3], "Grey: observed warm-up after the kick. Black: truth. Colour/style: each fitted model/seed. No phase alignment."; fontsize=7)
    Label(resp[4, 1:3], "Distance: thin curves are instantaneous; dots are cycle RMS, plotted at bin centres. Same metric and runs as Figure 4A."; fontsize=7)
    save_paper(fig, "Diagnostic_cfhn_states_start$k")
    save_paper(resp, "Diagnostic_cfhn_response_start$k")
end
