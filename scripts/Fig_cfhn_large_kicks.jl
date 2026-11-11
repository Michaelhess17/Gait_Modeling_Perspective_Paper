#!/usr/bin/env julia
include("fig6_relaxation_data.jl")
include("paper_style.jl")
using Printf
root=joinpath(dirname(@__DIR__),"cache","cfhn_smooth","large_kicks")
D=deserialize(joinpath(root,"data.jls"));n=D.n;K=12n
standard(Y)=(Y.-D.μ)./D.σ
M=Dict(kind=>deserialize(joinpath(root,"$kind.jls")) for kind in ("dmd","esn"))
fig=Figure(size=(COL2,155MM))
axs=[CairoMakie.Axis(fig[i,j];title=i==1 ? (j==1 ? "Common amplitude kick" : "Differential coordination kick") : "",
           ylabel=j==1 ? ("v₁ (raw)","w₁ − w₂ (raw)","Distance to true cycle (SD norm)")[i] : "",
           xlabel=i==3 ? "Cycles after kick" : "") for i in 1:3,j in 1:2]
t=collect(0:K)./n
for (j,ix) in enumerate(D.chosen)
    T=D.test[ix];truth=T.truth[1:K+1,:]
    for (kind,col) in (("dmd",CAT[1]),("esn",CAT[2]))
        R=M[kind]
        for row in R.R
            test=row.tests[ix];test.ok || continue
            F=vcat(standard(T.truth[1:1,:]),test.F)
            raw=F.*D.σ.+D.μ
            CairoMakie.lines!(axs[1,j],t,raw[:,1];color=(col,.7))
            CairoMakie.lines!(axs[2,j],t,raw[:,2].-raw[:,4];color=(col,.7))
            distance=dist_curve(F,D.cycle;closed=true)
            CairoMakie.lines!(axs[3,j],t,distance;color=(col,.7))
        end
        bad=sum(!row.tests[ix].ok for row in R.R)
        bad>0 && CairoMakie.text!(axs[1,j],.05,.95-(kind=="dmd" ? 0 : .1);text="$kind: $bad failed seeds",space=:relative)
    end
    CairoMakie.lines!(axs[1,j],t,truth[:,1];color=C_TRUTH,linestyle=:dash)
    CairoMakie.lines!(axs[2,j],t,truth[:,2].-truth[:,4];color=C_TRUTH,linestyle=:dash)
    CairoMakie.lines!(axs[3,j],t,dist_curve(standard(truth),D.cycle;closed=true);color=C_TRUTH,linestyle=:dash)
end
for i in 1:3; linkyaxes!(axs[i,:]...); end
Legend(fig[0,1:2],[LineElement(color=CAT[1]),LineElement(color=CAT[2]),LineElement(color=C_TRUTH,linestyle=:dash)],
       ["DMD","ESN (all four seeds)","Truth"];orientation=:horizontal)
Label(fig[4,1:2],"Kick: 1.3 cycle SD RMS. Two cycles of pre-kick history; forecast starts immediately after the kick.\nExamples chosen from truth before fitting. Both models include the coordination observable.";fontsize=7)
save_paper(fig,"Fig_cfhn_large_kicks")
report=joinpath(dirname(@__DIR__),"figures","CFHN_large_kicks.md")
open(report,"w") do io
    println(io,"# Larger, covered perturbations\n\n![Recovery comparison](Fig_cfhn_large_kicks.png)\n")
    println(io,"Smoother FHN, ε=$(D.c.ε), k=$(D.c.k). This is a synthetic feasibility experiment, not a calibration to human gait. Time is in cycles. No measurement or process noise is added in this first larger-kick experiment.\n")
    println(io,"Training: 24 independent phase/amplitude draws (seed 3101), 12 common amplitude kicks and 12 differential kicks, with magnitudes uniformly sampled from 0.4–1.6 cycle SD RMS per coordinate. Each sequence includes two pre-kick cycles and eight recovery cycles. Validation: eight independent draws (seed 3102) from the same distribution, scored over the first five post-kick cycles. Test: 16 held-out combinations, four phases × two magnitudes (0.8, 1.3) × two families. Test settings lie within the sampled phase/amplitude domain; finite training coverage is not a guarantee of interpolation accuracy.\n")
    println(io,"Common kicks point radially outward from the cycle mean in standardized (v,w) coordinates for both oscillators. Differential kicks apply opposite directions to the two oscillators. These labels describe the applied displacement, not pure dynamical modes.\n")
    println(io,"Initialization uses two cycles of undisturbed history, replaces only its newest observation with the kicked state, then predicts from the next sample. There is **no post-kick teacher forcing**. Training omits the externally imposed pre-kick → kicked transition from autonomous regression targets, but includes kicked histories and subsequent recovery transitions. Training-only state normalization and a standardized w₁−w₂ channel are used for both models. DMD eigenvalues outside the unit circle are clipped.\n")
    println(io,"The two plotted examples maximize true first-three-cycle excursion within their respective held-out families, selected before fitting (both amplitude 1.3, phase index 721). All test cases enter the following numerical summary. Every fitted seed is shown, with failures counted rather than plotted as successful trajectories.\n")
    println(io,"| Model | Family | Median first-5-cycle state RMSE | Median final-cycle distance | Failed / total |\n|---|---|---:|---:|---|")
    for kind in ("dmd","esn"),family in ("amplitude","coordination")
        errors=Float64[];offsets=Float64[];failed=0;total=0
        for row in M[kind].R,(ix,T) in enumerate(D.test)
            T.kind==family || continue
            total+=1;f=row.tests[ix]
            if !f.ok;failed+=1;continue;end
            push!(errors,sqrt(mean(abs2,f.F[1:5n,:].-standard(T.truth[2:5n+1,:]))))
            push!(offsets,sqrt(mean(abs2,dist_curve(f.F[end-n+1:end,:],D.cycle;closed=true))))
        end
        @printf(io,"| %s | %s | %.4f | %.4f | %d / %d |\n",kind,family,isempty(errors) ? NaN : median(errors),isempty(offsets) ? NaN : median(offsets),failed,total)
    end
    println(io,"\nState RMSE uses training-standardized coordinates and includes timing error; cycle distance is the nearest-point Euclidean norm to the closed true-cycle polyline and is insensitive to shared cycle phase. Forecasts last 12 cycles; independent truth simulations last 30 cycles to check eventual return. No model-specific attractor is used as the distance reference.\n")
    for family in ("amplitude","coordination")
        rows=filter(x->x.kind==family,D.diagnostics)
        println(io,"True final (30th) cycle distance range for $family: ",extrema(x.final for x in rows),".\n")
    end
    for kind in ("dmd","esn")
        println(io,"Selected $kind settings: `",M[kind].best.g,"`; validation MSE = ",M[kind].best.score,".\n")
    end
end
println("Saved ",report)
