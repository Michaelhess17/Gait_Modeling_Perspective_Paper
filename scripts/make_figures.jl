#!/usr/bin/env julia
# Rebuild every manuscript figure from the cached results (scripts/figures/Figure_*.jl), each in its own process.
# Usage: julia --project=. scripts/make_figures.jl            (all figures)
#        julia --project=. scripts/make_figures.jl 05 S03     (selected figures)
# The analyses that produce the caches are listed in scripts/README.md; they are not rerun here.
const ROOT = dirname(@__DIR__)
const FIGDIR = joinpath(@__DIR__, "figures")
want = isempty(ARGS) ? nothing : Set(ARGS)
files = sort(filter(f -> occursin(r"^Figure_S?\d+\.jl$", f), readdir(FIGDIR)))
want === nothing || (files = filter(f -> replace(f, "Figure_" => "", ".jl" => "") in want, files))
failed = String[]
for f in files
    t0 = time(); print(rpad(f, 18)); flush(stdout)
    ok = success(pipeline(`$(Base.julia_cmd()) --project=$ROOT $(joinpath(FIGDIR, f))`; stdout = devnull, stderr = devnull))
    println(ok ? "ok" : "FAILED", "  ", round(Int, time() - t0), " s")
    ok || push!(failed, f)
end
want === nothing && run(`$(Base.julia_cmd()) --project=$ROOT $(joinpath(@__DIR__, "scorecard.jl"))`)   # the comparison table
isempty(failed) ? println("all $(length(files)) figures rebuilt") : println("failed: ", join(failed, ", "))
