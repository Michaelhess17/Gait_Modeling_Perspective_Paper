#!/usr/bin/env julia
using Pkg; Pkg.activate(dirname(@__DIR__);io=devnull)
include("cfhn_smooth_system.jl")
using Printf
rows=map((0.08,0.12,0.18,0.25,0.35)) do ε
    c=smooth_cycle(ε)
    k=ε==0.08 ? 0.02 : smooth_match_coupling(c)
    μ=smooth_coordination(c,k)
    x=c.C[1:end-1,1]; x=x.-mean(x)
    energy=abs2.(rfft(x))[2:end]
    cumulative=cumsum(energy)./sum(energy)
    h95=findfirst(>=(0.95),cumulative); h99=findfirst(>=(0.99),cumulative)
    sharpness=maximum(abs.(diff(vcat(x,x[1]))))*length(x)/(maximum(x)-minimum(x))
    @printf("epsilon %.2f coupling %.6f period %.5f mu %.6f harmonics95/99 %d/%d sharpness %.2f closure %.2g\n",
            ε,k,c.T,μ,h95,h99,sharpness,c.closure)
    (; c...,k,μ,h95,h99,sharpness,energy=energy./sum(energy))
end
# Moderate separation: epsilon=0.25 (nominal slow coefficient four times smaller).
# Choice is made from waveform regime, before any model fitting/test scores.
selected=only(filter(r->r.ε==0.25,rows))
serialize(joinpath(SMOOTH_DIR,"regimes.jls"),(;rows,selected))
