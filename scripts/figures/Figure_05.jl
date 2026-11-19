#!/usr/bin/env julia
# Figure_05.jl — R2: example free runs of every model class on the benchmark (manuscript: @fig-freeruns).
# Run: julia --project=. scripts/figures/Figure_05.jl
# Free runs from the end of a 12-cycle training window (realistic regime: 1 % measurement noise, process noise on the
# simulated systems), for four simulated systems and one human gait trial. Grey: held-out data; colour: free runs,
# every seed (ESN 8, LSTM 2, neural ODE 4) or polynomial degree (SSMLearnPy 2) overlaid. Axes framed on the data; runs
# that leave the frame are cut. The label gives the outcome counts after 30 cycles (r2_common.jl).
include(joinpath(@__DIR__, "r2_common.jl"))

fig = Figure(size = (COL2, 128MM))
gA = fig[1, 1] = GridLayout()
# ── free runs, realistic regime ──────────────────────────────────────────
for (c, (sys, title, (i, j), (li, lj))) in enumerate(SYSTEMS)
    rows, Ytr, Yte, P = load_cell(sys, "realistic")
    Yte === nothing && continue
    n8 = min(size(Yte, 1), round(Int, 8P))
    lims = framed_limits(vcat(Ytr[:, i], Yte[1:n8, i]), vcat(Ytr[:, j], Yte[1:n8, j]); pad = 0.25)
    for (r, (m, nm, col, _, _)) in enumerate(MODELS)
        last = r == length(MODELS)
        ax = Axis(gA[r, c]; xticklabelsvisible = false, yticklabelsvisible = false, xlabel = last ? li : "",
                  ylabel = last ? lj : "", title = r == 1 ? title : "", titlealign = :center, xlabelsize = 7, ylabelsize = 7,
                  xticks = LinearTicks(3), yticks = LinearTicks(3))
        lines!(ax, Yte[1:n8, i], Yte[1:n8, j]; color = (C_REF, 0.5), linewidth = 1.0)
        mr = [x for x in rows if x.model == m]
        for x in mr
            xs, ys, _ = clip_to_view(x.Fc[:, i], x.Fc[:, j], lims)
            lines!(ax, xs, ys; color = (col, 0.75), linewidth = 0.8)
        end
        limits!(ax, lims...)
        if !isempty(mr)
            parts = ["$(n) $(f)" for (n, f) in ((count(x -> x.fate == f, mr), f) for f in FATES) if n > 0]
            txt = length(parts) > 2 ? join(parts[1:2], ", ") * ",\n" * join(parts[3:end], ", ") : join(parts, ", ")
            text!(ax, 0.03, 0.97; space = :relative, align = (:left, :top), fontsize = 5.5, text = txt, color = C_TRUTH)
        end
        c == length(SYSTEMS) && Label(gA[r, c+1], nm; rotation = -π / 2, fontsize = 7, tellheight = false, color = col)
    end
end
colgap!(gA, 4); rowgap!(gA, 3)

save_paper(fig, "Figure_05")
