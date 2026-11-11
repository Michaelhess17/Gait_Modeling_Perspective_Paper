#!/usr/bin/env julia
using Pkg; Pkg.activate(dirname(@__DIR__);io=devnull)
include("paper_style.jl")
using Serialization
D=deserialize(joinpath(dirname(@__DIR__),"cache","cfhn_smooth","regimes.jls"))
fig=Figure(size=(COL2,105MM))
ax1=Axis(fig[1,1];xlabel="Phase (cycles)",ylabel="v (raw)",title="Smoother cycles with moderate time-scale separation")
ax2=Axis(fig[1,2];xlabel="v",ylabel="w",title="Synchronous orbit")
ax3=Axis(fig[2,1];xlabel="Harmonic",ylabel="Fraction of v variance",yscale=log10,title="Harmonic content")
ax4=Axis(fig[2,2];xlabel="ε",ylabel="Coordination multiplier",title="Coupling adjusted to keep contraction comparable")
for (i,r) in enumerate(D.rows)
    style=r.ε==0.25 ? :solid : :dash
    width=r.ε==0.25 ? 2.1 : 0.9
    lines!(ax1,range(0,1;length=size(r.C,1)),r.C[:,1];color=CAT[i],linewidth=width,linestyle=style,label="ε=$(r.ε)")
    lines!(ax2,r.C[:,1],r.C[:,2];color=CAT[i],linewidth=width,linestyle=style)
    lines!(ax3,1:15,r.energy[1:15];color=CAT[i],linewidth=width,linestyle=style)
    scatter!(ax4,[r.ε],[r.μ];color=CAT[i])
end
hlines!(ax4,0.69;color=C_REF,linestyle=:dot)
Legend(fig[0,1:2],ax1;orientation=:horizontal)
save_paper(fig,"Diagnostic_cfhn_smooth_regimes")
