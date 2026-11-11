#!/usr/bin/env julia
include("fig6_relaxation_data.jl")
include("cfhn_smooth_system.jl")
BLAS.set_num_threads(1)
root=joinpath(SMOOTH_DIR,"large_kicks");mkpath(root)
c=deserialize(joinpath(SMOOTH_DIR,"regimes.jls")).selected
n=110; H=2n+1; p=(c.ε,c.k,c.drive)
center=vec(mean(c.C[1:end-1,:];dims=1)); scale=vec(std(c.C[1:end-1,:];dims=1))
function example(phase,amplitude,kind;cycles=30)
    history=c.C[mod1.(phase .+8 .*collect(-(H-1):0),880),:]
    u=vec(history[end,:]);d=(u[1:2].-center[1:2])./scale[1:2];d./=sqrt(mean(abs2,d))
    direction=kind=="amplitude" ? vcat(d,d) : vcat(d,-d)
    kicked=u.+amplitude.*scale.*direction
    truth=smooth_run(kicked,p,c.T,cycles;n)
    control=smooth_run(u,p,c.T,cycles;n)
    history[end,:]=kicked
    (;phase,amplitude,kind,history,truth,control,series=vcat(history[1:end-1,:],truth))
end
function sample(seed,count)
    rng=MersenneTwister(seed)
    [example(rand(rng,1:880),.4+1.2rand(rng),isodd(j) ? "amplitude" : "coordination";cycles=8) for j in 1:count]
end
train=sample(3101,24); validation=sample(3102,8)
test=[example(phase,a,kind) for kind in ("amplitude","coordination") for a in (.8,1.3) for phase in (61,281,501,721)]
alltrain=reduce(vcat,getfield.(train,:series)); μ=mean(alltrain;dims=1);σ=std(alltrain;dims=1)
standard(Y)=(Y.-μ)./σ
cycle=standard(c.C)
diagnostics=map(test) do t
    distance=dist_curve(standard(t.truth),cycle;closed=true)
    (;t.phase,t.amplitude,t.kind,initial=distance[1],peak=maximum(distance[1:3n]),
      first3=sqrt(mean(abs2,distance[1:3n])),final=sqrt(mean(abs2,distance[end-n+1:end])))
end
# Choose examples from truth alone, before any model fit: strongest first-three-cycle excursion in each family.
chosen=[begin inds=findall(t->t.kind==kind,test);inds[argmax([diagnostics[i].first3 for i in inds])] end for kind in ("amplitude","coordination")]
serialize(joinpath(root,"data.jls"),(;c,p,n,H,μ,σ,train,validation,test,cycle,diagnostics,chosen))
for kind in ("amplitude","coordination")
    rows=filter(x->x.kind==kind,diagnostics)
    println(kind," initial range=",extrema(x.initial for x in rows)," final-cycle distance range=",extrema(x.final for x in rows))
end
println("Examples selected from truth: ",diagnostics[chosen])
