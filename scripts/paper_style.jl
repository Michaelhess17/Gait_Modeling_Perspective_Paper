# paper_style.jl — shared CairoMakie theme and helpers for manuscript figures
#
# include() this from any Fig_*.jl script. It deliberately does NOT include
# common.jl (which loads Plots.jl); scripts that need the DMD pipeline or the
# gait loader can include both, but must then qualify Makie calls where names
# clash (e.g. `CairoMakie.plot!`).
#
# Conventions
#   * Sizes are in points (1 pt = 1/72 in). Journal column widths:
#       single column  ≈  85 mm  → COL1
#       double column  ≈ 178 mm  → COL2
#   * Colour has a job, never a rank:
#       ground truth / observed data   → C_TRUTH (near-black)
#       reference lines, null levels   → C_REF (grey)
#       method / condition categories  → CAT[1], CAT[2], ... in fixed order
#       subject groups                 → GROUP_COL["AB" / "HF" / "LF"]
#     The categorical slots are the validated reference palette (CVD ΔE ≥ 9 for
#     the first three, all pairs). Anything beyond three categories must also
#     carry a marker/linestyle difference or a direct label.
#   * Every figure is written as PDF (vector, for the journal), SVG, and a
#     300-dpi PNG (for the Quarto HTML build).

using CairoMakie
using Printf

const MM   = 72 / 25.4
const COL1 = 85MM
const COL2 = 178MM

# ── colour roles ──────────────────────────────────────────────────────────────
const C_TRUTH  = colorant"#1a1a1a"
const C_REF    = colorant"#8a8986"
const C_SHADE  = colorant"#ecebe8"     # training windows, plausible ranges
const CAT = [colorant"#2a78d6",        # 1 blue
             colorant"#eb6834",        # 2 orange
             colorant"#1baf7a",        # 3 aqua
             colorant"#eda100",        # 4 yellow  (needs secondary encoding)
             colorant"#e87ba4"]        # 5 magenta (needs secondary encoding)
const GROUP_COL = Dict("AB" => CAT[1], "HF" => CAT[2], "LF" => CAT[3])
const GROUP_ORDER_FIG = ["AB", "HF", "LF"]

const JOINTS = ["hip", "knee", "ankle"]   # per limb; channels are [hip,knee,ankle]×2

# ── theme ─────────────────────────────────────────────────────────────────────
const PAPER_THEME = Theme(
    fontsize = 8,
    font = "TeX Gyre Heros Makie",
    figure_padding = (4, 8, 4, 4),
    Axis = (
        xgridvisible = false, ygridvisible = false,
        topspinevisible = false, rightspinevisible = false,
        spinewidth = 0.6,
        xtickwidth = 0.6, ytickwidth = 0.6,
        xticksize = 2.5, yticksize = 2.5,
        xminorticksize = 1.5, yminorticksize = 1.5,
        titlesize = 8, titlefont = :regular, titlealign = :left,
        titlegap = 3,
        xlabelpadding = 2, ylabelpadding = 2,
        xticklabelsize = 7, yticklabelsize = 7,
    ),
    Legend = (
        framevisible = false, labelsize = 7, padding = (2, 2, 2, 2),
        rowgap = 0, patchsize = (12, 6), patchlabelgap = 3,
    ),
    Lines = (linewidth = 1.2,),
    Scatter = (markersize = 5, strokewidth = 0),
    Colorbar = (ticklabelsize = 7, labelsize = 8, spinewidth = 0.6, size = 6),
)
set_theme!(PAPER_THEME)

# ── helpers ───────────────────────────────────────────────────────────────────
"Bold panel letter at the top-left corner of a GridLayout cell."
function panel_label!(gp, letter)
    Label(gp[1, 1, TopLeft()], letter; fontsize = 10, font = :bold,
          padding = (0, 6, 2, 0), halign = :right)
end

"Light grey band marking an interval (e.g. the training window)."
shade!(ax, x0, x1; color = C_SHADE) = vspan!(ax, x0, x1; color = color)

"Horizontal reference line in the reference colour."
refline!(ax, y; linestyle = :dash, kw...) =
    hlines!(ax, y; color = C_REF, linewidth = 0.8, linestyle = linestyle, kw...)

"""
    clip_to_view(x, y, lims)

Return copies of `x`, `y` with points outside `lims = (xlo, xhi, ylo, yhi)`
replaced by NaN, so trajectories that leave a zoomed-in axis are cut cleanly
instead of drawing long lines toward the edge. Also returns whether anything was
cut (so the caller can annotate an escaping trajectory).
"""
function clip_to_view(x, y, lims)
    xlo, xhi, ylo, yhi = lims
    xs = float.(collect(x)); ys = float.(collect(y))
    out = (xs .< xlo) .| (xs .> xhi) .| (ys .< ylo) .| (ys .> yhi)
    xs[out] .= NaN; ys[out] .= NaN
    return xs, ys, any(out)
end

"Axis limits that frame `x`,`y` with fractional padding."
function framed_limits(x, y; pad = 0.08)
    fx = filter(isfinite, x); fy = filter(isfinite, y)
    xlo, xhi = extrema(fx); ylo, yhi = extrema(fy)
    dx = (xhi - xlo) * pad; dy = (yhi - ylo) * pad
    return (xlo - dx, xhi + dx, ylo - dy, yhi + dy)
end

"Save a figure as PDF + SVG + 300-dpi PNG under figures/."
function save_paper(fig, name; dir = joinpath(dirname(@__DIR__), "figures"))
    mkpath(dir)
    for ext in ("pdf", "svg")
        save(joinpath(dir, "$name.$ext"), fig; pt_per_unit = 1)
    end
    save(joinpath(dir, "$name.png"), fig; px_per_unit = 300 / 72)
    @info "saved" name dir
end
