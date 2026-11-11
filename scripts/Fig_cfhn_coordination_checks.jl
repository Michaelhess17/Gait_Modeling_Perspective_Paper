#!/usr/bin/env julia
using Pkg; Pkg.activate(dirname(@__DIR__);io=devnull)
include("paper_style.jl")
using Serialization,Statistics,LinearAlgebra
root=joinpath(dirname(@__DIR__),"cache","cfhn_smooth")
D=deserialize(joinpath(root,"data.jls")); n=D.n; W=170; K=5n
truths=[t.truth[W+1:end,:] for t in D.test]
controls=[t.oncycle[W+1:end,:] for t in D.test]
threshold=vec(mean(D.cycle[1:end-1,:];dims=1))
function crossings(Y,j)
    x=Y[1:min(size(Y,1),7n),j].-threshold[j]
    [i-x[i]/(x[i+1]-x[i]) for i in 1:length(x)-1 if x[i]<=0<x[i+1]]
end
function phase_metrics(Y)
    a=crossings(Y,2);b=crossings(Y,4)
    if length(a)<3 || length(b)<3
        return (;period=[NaN,NaN],lag=fill(NaN,5))
    end
    period=[median(diff(a)),median(diff(b))]./n
    lag=map(1:5) do k
        t=a[argmin(abs.(a.-(k-.5)*n))]
        360*(b[argmin(abs.(b.-t))]-t)/(period[1]*n)
    end
    (;period,lag)
end
function diagnostics(F,F0,T,T0)
    raw=Float64[];aligned=Float64[];shifts=Float64[];paired=Float64[];paired_error=Float64[]
    for k in 1:5
        # Leave a quarter-cycle at the left edge for symmetric interpolation support.
        inds=collect(max((k-1)*n+1,ceil(Int,.25n)):k*n)
        push!(raw,sqrt(mean(abs2,F[inds,:].-T[inds,:])))
        shifts_grid=collect(-.2n:.1:.2n)
        errors=map(shifts_grid) do s
            q=inds.+s;lo=floor.(Int,q);a=q.-lo
            interp=F[lo,:].*(1 .-a).+F[lo.+1,:].*a
            sqrt(mean(abs2,interp.-T[inds,:]))
        end
        i=argmin(errors);push!(aligned,errors[i]);push!(shifts,shifts_grid[i]/n)
        ix=(k-1)*n+1:k*n
        Δ=F[ix,:].-F0[ix,:];Δt=T[ix,:].-T0[ix,:]
        push!(paired,sqrt(mean(abs2,Δ)))
        push!(paired_error,sqrt(mean(abs2,Δ.-Δt)))
    end
    p=phase_metrics(F);pt=phase_metrics(T)
    (;raw,aligned,shifts,paired,paired_error,p.period,p.lag,lag_error=abs.(p.lag.-pt.lag))
end
summary=Dict()
for kind in ("dmd","esn"), suffix in ("","_coordination")
    file=joinpath(root,"$kind$suffix.jls")
    isfile(file) || continue
    R=deserialize(file)
    rows=[(;seed=r.seed,start=j,diagnostics(t.F,t.F0,truths[j],controls[j])...)
          for r in R.R for (j,t) in enumerate(r.tests) if t.ok]
    summary[kind*suffix]=(;rows,score=R.best.score,best=R.best.g,
                          failures=sum(!t.ok for r in R.R for t in r.tests))
end
med(rows,key)=[median(getproperty(r,key)[k] for r in rows) for k in 1:5]
truth_phase=phase_metrics.(truths)
truth_lag=[median(abs(r.lag[k]) for r in truth_phase) for k in 1:5]
truth_paired=[median(sqrt(mean(abs2,(T.-T0)[(k-1)*n+1:k*n,:])) for (T,T0) in zip(truths,controls)) for k in 1:5]
fig=Figure(size=(COL2,155MM)); xx=.5:1:4.5
axs=[Axis(fig[i,j];title=i==1 ? (j==1 ? "Hankel-DMD" : "ESN") : "",
           ylabel=j==1 ? ("State RMSE (SD units)","Absolute crossing lag (degrees)","Paired response RMS (SD units)")[i] : "",
           xlabel=i==3 ? "Cycles after warm-up" : "") for i in 1:3,j in 1:2]
for (j,kind) in enumerate(("dmd","esn"))
    for (suffix,col,lab) in (("",CAT[1],"Original"),("_coordination",CAT[2],"+ coordination observable"))
        haskey(summary,kind*suffix) || continue
        rows=summary[kind*suffix].rows
        lines!(axs[1,j],xx,med(rows,:raw);color=col,label=lab)
        lines!(axs[1,j],xx,med(rows,:aligned);color=col,linestyle=:dash)
        lines!(axs[2,j],xx,[median(abs(r.lag[k]) for r in rows) for k in 1:5];color=col)
        lines!(axs[3,j],xx,med(rows,:paired);color=col)
    end
    lines!(axs[2,j],xx,truth_lag;color=C_TRUTH,linestyle=:dot)
    lines!(axs[3,j],xx,truth_paired;color=C_TRUTH,linestyle=:dot)
end
for i in 1:3; linkyaxes!(axs[i,:]...); end
Legend(fig[0,1:2],axs[1,1];orientation=:horizontal)
Label(fig[4,1:2],"Top: solid = unaligned, dashed = shared timing correction. Other rows: dotted black = truth.\nMedians over four starts and all seeds; paired response = kicked minus unperturbed forecast.";fontsize=7)
save_paper(fig,"Fig_cfhn_coordination_checks")
serialize(joinpath(root,"coordination_checks.jls"),(;summary,truth_phase,truth_paired))
report=joinpath(dirname(@__DIR__),"figures","CFHN_coordination_checks.md")
open(report,"w") do io
    println(io,"# Coordination and timing checks\n\n![Comparison](Fig_cfhn_coordination_checks.png)\n")
    println(io,"Same held-out kicks, 170-sample warm-up, all four starts and all fitted seeds. Original training data and independently generated validation trajectories are unchanged. Parameter regime: ε=0.25, k=$(D.c.k).\n")
    println(io,"The extra channel is raw w₁−w₂, centered and scaled by its **training** standard deviation. This supplies no new information; it changes the fitting metric, POD basis, regularization geometry and (for ESN) reservoir inputs. Both conditions use the same hyperparameter grid and are selected by the same four-state validation MSE. It is an ablation of this representation, not a claim about every possible weighting.\n")
    println(io,"| Fit | Validation MSE | Median w₁ / w₂ period (true-cycle units) | Final-bin raw / aligned RMSE | Final-bin crossing-lag error (degrees) | Final-bin paired-response error | Failed starts |\n|---|---:|---|---|---:|---:|---:|")
    for key in ("dmd","dmd_coordination","esn","esn_coordination")
        haskey(summary,key) || continue
        s=summary[key];r=s.rows
        vals=(s.score,median(x.period[1] for x in r),median(x.period[2] for x in r),med(r,:raw)[5],med(r,:aligned)[5],med(r,:lag_error)[5],med(r,:paired_error)[5])
        @printf(io,"| %s | %.5f | %.5f / %.5f | %.5f / %.5f | %.3f | %.5f | %d |\n",key,vals...,s.failures)
        println(key," ",vals," settings=",s.best)
    end
    println(io,"\nPeriods are median intervals between upward crossings of a fixed true-cycle mean threshold in each w coordinate, measured over seven cycles. Crossing lag uses the nearest w₂ crossing to each w₁ crossing and is an event-timing proxy, not an asymptotic phase estimate. The plot shows absolute lag; the table compares signed predicted and true lag before taking absolute error.\n")
    println(io,"Timing correction searches ±0.2 cycle at 0.1-sample resolution separately in each cycle bin, applying **one shared shift to all four coordinates** and no amplitude adjustment. This allows common timing drift; it does not independently align the two oscillators. The first quarter-cycle is excluded from the first-bin RMSE in both comparisons to provide symmetric interpolation support.\n")
    println(io,"Paired responses subtract each model's own unperturbed forecast from its kicked forecast, with identical elapsed time and no independent alignment. This removes baseline waveform mismatch but includes any persistent phase reset, so response magnitude need not decay to zero. The table reports error against the true paired response, not just its magnitude. These are conditional forecasts after observing about 1.55 cycles of post-kick history, not predictions from the instant of the kick.\n")
    println(io,"Crossing thresholds are transformed consistently into training-standardized coordinates; this gives the same crossing times as using raw coordinates. RMSE uses the four training-standardized states. No test trajectory is used for model selection.\n")
    println(io,"For this comparison, extra-observable training substantially improves ESN timing and paired-response accuracy, while DMD gains no consistent recovery benefit. Shared timing correction also removes much of the original ESN's pointwise waveform error. These results do not support treating the original difference trace as evidence that either model class is inherently unable to learn recovery.\n")
end
println("Saved ",report)
