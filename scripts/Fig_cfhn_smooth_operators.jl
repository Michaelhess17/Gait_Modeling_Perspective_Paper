#!/usr/bin/env julia
using Pkg; Pkg.activate(dirname(@__DIR__);io=devnull)
include("paper_style.jl")
using Serialization,Statistics
O=deserialize(joinpath(dirname(@__DIR__),"cache","cfhn_smooth","operators.jls"))
fig=Figure(size=(COL2,135MM))
a=Axis(fig[1,1];xlabel="Cycles after an instantaneous kick",ylabel="Response gain",
       title="A  Physical current-state response: truth versus DMD")
t=(0:O.K)./O.n
lo=[minimum(g[k] for g in O.truth_gain) for k in eachindex(t)]
hi=[maximum(g[k] for g in O.truth_gain) for k in eachindex(t)]
band!(a,t,lo,hi;color=(C_TRUTH,0.13))
lines!(a,t,[median(g[k] for g in O.truth_gain) for k in eachindex(t)];color=C_TRUTH,label="truth: median / range over phase")
lines!(a,t,O.output_gain;color=CAT[1],label="DMD: current input → current output")
axislegend(a;position=:rt,labelsize=6)
b=Axis(fig[1,2];xlabel="Cycles",ylabel="Gain",yscale=log10,
       title="B  Full delay-state growth is a different quantity")
lines!(b,t,O.state_gain;color=CAT[2],label="‖Aᵏ‖ (arbitrary reduced-state input)")
lines!(b,t,O.output_gain;color=CAT[1],label="‖CAᵏB‖ (current-state kick)")
axislegend(b;position=:rt,labelsize=6)
c=Axis(fig[2,1];xlabel="Frequency (cycles per cycle)",ylabel="Damped resolvent gain",yscale=log10,
       title="C  Frequency-selective amplification")
lines!(c,O.freq,O.resolvent_full;color=CAT[2],label="full reduced-state resolvent")
lines!(c,O.freq,O.resolvent_output;color=CAT[1],label="current input/output resolvent")
axislegend(c;position=:rt,labelsize=6)
d=Axis(fig[2,2];xlabel="Re z",ylabel="Im z",title="D  Resolvent contours / pseudospectra",
       limits=(first(O.xs),last(O.xs),first(O.ys),last(O.ys)))
hm=heatmap!(d,O.xs,O.ys,O.logres;colormap=:viridis)
contour!(d,O.xs,O.ys,O.logres;levels=[1.,2.,3.,4.],color=:white,linewidth=0.6)
θ=range(-0.3,0.3;length=200)
lines!(d,cos.(θ),sin.(θ);color=:white,linestyle=:dash)
scatter!(d,real.(O.ev),imag.(O.ev);color=:red,markersize=4)
Colorbar(fig[2,3],hm;label="log₁₀ ‖(zI−A)⁻¹‖")
Label(fig[0,1:3],"Smoother FHN: what the fitted linear operator exposes, and what its physical response predicts";fontsize=9)
Label(fig[3,1:3],"Frequency evaluation: |z|=exp(0.1/110), damping 0.1 per cycle. SD-scaled observables; B changes only the newest delay block.";fontsize=7)
save_paper(fig,"Fig_cfhn_smooth_operators")
