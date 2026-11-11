#!/usr/bin/env julia
# Same fitted operator before/after Schur spectral constraints; manuscript §4 R2.
# Run stabilization_data.jl first. No model is reselected for a test outcome.
using Pkg; Pkg.activate(dirname(@__DIR__); io=devnull)
include("paper_style.jl")
using Serialization

R = deserialize(joinpath(dirname(@__DIR__), "cache", "stabilization_gait.jls"))
fig = Figure(size=(COL2, 88MM))
for (i, r) in enumerate(R)
    t = (1:size(r.unit, 1)) .* r.dt
    label = r.trial == 91 ? "AB 91" : "LF 231"
    ax = Axis(fig[i, 1]; ylabel="Hip (z)",
              xlabel=i == length(R) ? "Time after training window (s)" : "",
              title="$label: identical fit and initial history; spectral constraint varied",
              xticklabelsvisible=i == length(R))
    vspan!(ax, 0, size(r.Yte, 1)*r.dt; color=C_SHADE)
    lines!(ax, t, r.raw[:, 1]; color=CAT[2], linestyle=:dash, label="unconstrained")
    lines!(ax, t, r.clipped[:, 1]; color=C_REF, linestyle=:dot, label="clip growing modes")
    lines!(ax, t, r.unit[:, 1]; color=CAT[1], label="unit-circle constraint")
    lines!(ax, (1:size(r.Yte, 1)).*r.dt, r.Yte[:, 1]; color=C_TRUTH, label="held-out data")
    xlims!(ax, 0, maximum(t))
    i == 1 && Legend(fig[0, 1], ax; orientation=:horizontal, labelsize=6.5)
    Label(fig[i, 1, TopLeft()], string(Char('A'+i-1)); font=:bold, fontsize=10,
          padding=(0, 6, 2, 0), halign=:right)
end
save_paper(fig, "Fig_stabilization")
