#!/usr/bin/env julia
using Pkg; Pkg.activate(dirname(@__DIR__);io=devnull)
include("paper_style.jl")
using Serialization,Statistics
root=joinpath(dirname(@__DIR__),"cache","cfhn_smooth")
D=deserialize(joinpath(root,"data.jls"))
fig=Figure(size=(COL2,190MM))
W=170; n=D.n; t=(1:5n)./n
axes_by_row=[Axis[] for _ in 1:4]
models=[("dmd","Hankel-DMD",CAT[1]),("esn","ESN",CAT[2])]
isfile(joinpath(root,"node.jls")) && push!(models,("node","Neural ODE",CAT[3]))
for (c,(kind,lab,col)) in enumerate(models)
    R=deserialize(joinpath(root,"$kind.jls"))
    labels=("v₁ (raw)","w₁ (raw)","w₂ (raw)","w₁ − w₂ (raw)")
    axs=[Axis(fig[j,c];title=j==1 ? lab : "",ylabel=c==1 ? labels[j] : "",
              xlabel=j==4 ? "Cycles after warm-up" : "") for j in 1:4]
    for j in 1:4; push!(axes_by_row[j],axs[j]); end
    trace(F,j)=j==4 ? F[:,2].-F[:,4] : F[:,(1,2,4)[j]]
    for row in R.R
        first(row.tests).ok || continue
        F=first(row.tests).F[1:5n,:].*D.σ.+D.μ
        for j in 1:4
            lines!(axs[j],t,trace(F,j);color=(col,0.7))
        end
    end
    truth=first(D.test).truth[W+1:W+5n,:].*D.σ.+D.μ
    for j in 1:4
        lines!(axs[j],t,trace(truth,j);color=C_TRUTH,linewidth=1.3,linestyle=:dash)
    end
    bad=sum(!first(row.tests).ok for row in R.R)
    bad>0 && text!(axs[4],0,0;text="$bad failed seeds",align=(:left,:bottom))
end
for axs in axes_by_row; linkyaxes!(axs...); end
linkyaxes!(axes_by_row[2]...,axes_by_row[3]...)
Label(fig[0,1:length(models)],"Smoother coupled FHN: ε=0.25, k=0.02494, coordination multiplier 0.69";fontsize=10)
Label(fig[5,1:length(models)],"Dashed black: truth. Colour: every fitted seed. All rows: start 1; 170-sample warm-up. No phase alignment.";fontsize=7)
save_paper(fig,"Fig_cfhn_smooth_response")
