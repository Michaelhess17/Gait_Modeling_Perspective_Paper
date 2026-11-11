#!/usr/bin/env julia
# Walking operators with a phase that pins no channel (human_ltp_lowdim.jl, human_ltp_pairphase.jl).
#   (A) Deviation SD over the stride (able-bodied, medians over trials; channels standardised) when strides are cut at
#       the limb-1 hip's own Hilbert phase (grey) or at the phase of the leading oscillatory pair of all six angles
#       (black): the hip cut forces that hip's deviation through zero at every cut and pushes timing error into the
#       other hip.
#   (B) One 15-s trial's operator predicting the same person's other trials (R² 12.5% of a stride ahead, medians over
#       trials), by POD rank (best delay and ridge per point), with phase harmonics (filled) and without (open).
#   (C) 60-stride operators (S = (8, 24, 4, 1e-2)) from the same person's other trials (own) or from other people of the
#       group at similar speed (other), with and without phase; pair cut, with the hip-cut values as grey ticks.
#       Medians of person means.
# Usage: julia --project=. scripts/Fig_walking_phase_check.jl      (writes figures/Fig_walking_phase_check.*)

include("human_ltp_lowdim.jl")
include("paper_style.jl")
plabel!(pos, Lb) = Label(pos[1, 1, TopLeft()], Lb; fontsize = 10, font = :bold, padding = (0, 6, 2, 0), halign = :right)
const LD = deserialize(joinpath(ROOT, "cache", "fig5", "human_ltp_lowdim.jls"))
const PP = deserialize(joinpath(ROOT, "cache", "fig5", "human_ltp_pairphase.jls"))
const CUTCOL = Dict(:hip => C_REF, :pair => C_TRUTH)

data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame); grp = String.(meta.lf_or_hf)
prof = Dict(how => begin
    V = filter(!isnothing, [begin
        St, _ = cut(data[i, :, :], how); n = size(St, 1)
        n < 5 ? nothing : reshape(sqrt.(mean(abs2, reshape(devs(St), NPH, n, 6), dims = 2)), NPH, 6)
    end for i in axes(data, 1) if grp[i] == "AB"])
    median(cat(V..., dims = 3), dims = 3)[:, :, 1]
end for how in (:hip, :pair))

fig = Figure(size = (COL2, 68MM)); g = fig[1, 1] = GridLayout()

aA = Axis(g[1, 1]; xlabel = "Phase (% stride from the cut)", ylabel = "Deviation SD (standardised)", title = "Cutting at the hip pins its deviation",
          limits = ((0, 99), (0, 0.2)), xticks = 0:25:100)
for how in (:hip, :pair), (c, ls) in ((1, :solid), (4, :dash))
    lines!(aA, 0:NPH-1, prof[how][:, c]; color = CUTCOL[how], linestyle = ls, linewidth = 1.2)
end
text!(aA, 13, 0.197; text = "solid: limb-1 hip   dashed: limb-2 hip\nblack: pair cut   grey: hip cut", fontsize = 6, align = (:left, :top))
plabel!(g[1, 1], "A")

aB = Axis(g[1, 2]; xscale = log2, xticks = [2, 4, 8, 16, 24], xlabel = "Rank of the operator", ylabel = "R², own other trials, 12.5% stride",
          title = "One trial's operator", limits = ((1.7, 30), (-0.05, 0.36)))
hlines!(aB, 0; color = C_REF, linewidth = 0.6)
SETL_ = LD.SETL; rks = sort(unique(getindex.(SETL_, 2)))
for how in (:hip, :pair)
    ok = [x for x in LD.RB[how] if x.nown > 0]
    for (hsel, filled) in ((h -> h > 0, true), (h -> h == 0, false))
        v = [maximum(median(x.own_sk[2, k] for x in ok) for k in eachindex(SETL_) if SETL_[k][2] == r && hsel(SETL_[k][3])) for r in rks]
        lines!(aB, rks, v; color = CUTCOL[how], linewidth = filled ? 1.3 : 0.8, linestyle = filled ? :solid : :dash)
        scatter!(aB, rks, v; color = filled ? CUTCOL[how] : :white, strokecolor = CUTCOL[how], strokewidth = 0.9, markersize = 5)
    end
end
text!(aB, 2, 0.35; text = "filled: with phase   open: without\nblack: pair cut   grey: hip cut", fontsize = 6, align = (:left, :top))
plabel!(g[1, 2], "B")

aC = Axis(g[1, 3]; ylabel = "R², held-out trial, 12.5% stride", title = "60 strides: own vs other people",
          xticks = (1:3, ["AB", "HF", "LF"]), limits = ((0.45, 3.55), (-0.05, 0.6)))
hlines!(aC, 0; color = C_REF, linewidth = 0.6)
offs = [(-0.27, :within, 1, true), (-0.09, :within, 2, false), (0.09, :between, 1, true), (0.27, :between, 2, false)]
for (gi, gr) in enumerate(("AB", "HF", "LF"))
    for (how, main) in ((:hip, false), (:pair, true))
        W = PP.out[how].W; x = [r for r in W if r.group == gr]; ps = unique(getfield.(x, :subject))
        for (dx, fld, j, filled) in offs
            v = median([mean(getfield(r, fld)[j] for r in x if r.subject == p) for p in ps])
            if main
                scatter!(aC, [gi + dx], [v]; color = filled ? GROUP_COL[gr] : :white, strokecolor = GROUP_COL[gr], strokewidth = 0.9,
                         marker = fld == :within ? :circle : :rect, markersize = 6)
            else
                lines!(aC, [gi + dx - 0.06, gi + dx + 0.06], [v, v]; color = C_REF, linewidth = 1.0)
            end
        end
    end
end
text!(aC, 0.5, 0.59; text = "● own trials   ■ other people\nfilled: with phase   open: without\ngrey ticks: hip cut", fontsize = 6, align = (:left, :top))
plabel!(g[1, 3], "C")
colgap!(g, 16)
save_paper(fig, "Fig_walking_phase_check")
