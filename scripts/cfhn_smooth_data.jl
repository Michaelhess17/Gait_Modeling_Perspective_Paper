#!/usr/bin/env julia
include("fig6_relaxation_data.jl")
include("cfhn_smooth_system.jl")
BLAS.set_num_threads(1)
c=deserialize(joinpath(SMOOTH_DIR,"regimes.jls")).selected
p=(c.ε,c.k,c.drive); n=110; dt=c.T/n
base=deserialize(joinpath(SMOOTH_DIR,"regimes.jls")).rows[1]
# Match integrated process-noise variance per cycle to the old regime.
noise=0.02sqrt(base.T/c.T)
sdc=vec(std(c.C[1:end-1,:],dims=1))
function dataset(seed,count)
    rng=MersenneTwister(seed)
    map(1:count) do k
        u=c.C[rand(rng,1:size(c.C,1)-1),:] .+0.2sdc.*randn(rng,4)
        g!(du,u,p,t)=(du.=0.;du[1]=noise;du[3]=noise)
        sol=solve(SDEProblem(smooth_fhn!,g!,u,(0.,(3n-1)*dt),p;seed=seed*1000+k),
                  EM();dt=dt/20,saveat=dt)
        permutedims(Array(sol))
    end
end
train=dataset(1,10); validation=dataset(101,10)
alltrain=reduce(vcat,train); μ=mean(alltrain,dims=1); σ=std(alltrain,dims=1)
standard(X)=(X.-μ)./σ
Ys=standard.(train); V=standard.(validation)
test=map((1,221,441,661)) do phase
    u=c.C[phase,:]; kicked=copy(u); kicked[3:4].+=0.3sdc[3:4]
    truth=smooth_run(kicked,p,c.T,32)
    oncycle=smooth_run(u,p,c.T,32)
    (;phase,truth=standard(truth),oncycle=standard(oncycle))
end
serialize(joinpath(SMOOTH_DIR,"data.jls"),(;c,p,n,dt,noise,μ,σ,Ys,V,test,cycle=standard(c.C)))
println("Saved smooth data: epsilon=",c.ε," k=",c.k," T=",c.T," noise=",noise)
