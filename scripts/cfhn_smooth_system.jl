# Isolated parameterized FHN system for the smoother benchmark.
# Historical FHN functions and caches are intentionally not reused as truth.
using OrdinaryDiffEq, LinearAlgebra, Statistics, Random, Serialization, FFTW
const SMOOTH_DIR = joinpath(dirname(@__DIR__), "cache", "cfhn_smooth")
mkpath(SMOOTH_DIR)
const SMOOTH_OPT = (; abstol=1e-10, reltol=1e-10)
function smooth_fhn!(du, u, p, t)
    ε, k, drive = p
    v1,w1,v2,w2 = u
    du[1] = v1-v1^3/3-w1+drive+k*(v2-v1)
    du[2] = ε*(v1+0.7-0.8w1)
    du[3] = v2-v2^3/3-w2+drive+k*(v1-v2)
    du[4] = ε*(v2+0.7-0.8w2)
end
function smooth_cycle(ε; drive=0.6, n=880)
    p=(ε,0.0,drive)
    u=solve(ODEProblem(smooth_fhn!,[-1.,-0.3,-1.,-0.3],(0.,1200.),p),Vern9(); SMOOTH_OPT...).u[end]
    sol=solve(ODEProblem(smooth_fhn!,u,(0.,200.),p),Vern9(); SMOOTH_OPT...,saveat=0.002)
    x=sol[1,:]; mid=mean(x)
    crossings=[sol.t[i-1]+(mid-x[i-1])/(x[i]-x[i-1])*0.002 for i in 2:length(x) if x[i-1]<mid<=x[i]]
    length(crossings)>=5 || error("no sustained cycle for epsilon=$ε")
    T=mean(diff(crossings[end-4:end]))
    u0=sol(crossings[end-1])
    cyc=solve(ODEProblem(smooth_fhn!,u0,(0.,T),p),Vern9(); SMOOTH_OPT...,saveat=T/n)
    C=permutedims(Array(cyc)) # endpoint included
    (; ε,drive,T,u0,C,closure=norm(C[end,:]-C[1,:]))
end
function smooth_coordination(c,k)
    function var!(dz,z,p,t)
        smooth_fhn!(view(dz,1:4),view(z,1:4),(c.ε,k,c.drive),t)
        J=[1-z[1]^2-2k -1.; c.ε -0.8c.ε]
        dz[5:8].=vec(J*reshape(z[5:8],2,2))
    end
    sol=solve(ODEProblem(var!,vcat(c.u0,vec(Matrix{Float64}(I,2,2))),(0.,c.T)),Vern9();SMOOTH_OPT...)
    maximum(abs,eigvals(reshape(sol.u[end][5:8],2,2)))
end
function smooth_match_coupling(c;target=0.69)
    lo,hi=0.,0.02
    while smooth_coordination(c,hi)>target && hi<1
        hi*=1.5
    end
    smooth_coordination(c,hi)<target || error("could not bracket coordination multiplier")
    for _ in 1:30
        mid=(lo+hi)/2
        smooth_coordination(c,mid)>target ? (lo=mid) : (hi=mid)
    end
    (lo+hi)/2
end
function smooth_run(u0,p,T,ncycles;n=110)
    sol=solve(ODEProblem(smooth_fhn!,u0,(0.,ncycles*T),p),Vern9();SMOOTH_OPT...,saveat=T/n)
    permutedims(Array(sol))
end
