#!/usr/bin/env julia
using Pkg; Pkg.activate(dirname(@__DIR__);io=devnull)
include("cfhn_smooth_system.jl")
using Printf
BLAS.set_num_threads(1)
D=deserialize(joinpath(SMOOTH_DIR,"data.jls"))
M=deserialize(joinpath(SMOOTH_DIR,"dmd.jls")).model
A=M.A; r=size(A,1); C=M.U[end-3:end,:]; B=C'
n=D.n; K=2n
power=Matrix{Float64}(I,r,r)
state_gain=Float64[]; output_gain=Float64[]
for k in 0:K
    push!(state_gain,opnorm(power))
    push!(output_gain,opnorm(C*power*B))
    global power=A*power
end
# A physical kick changes the current observable block, with past observations fixed.
# True finite differences use identical raw-coordinate kicks and training-derived SD units.
truth_matrices=map((1,221,441,661)) do phase
    u=D.c.C[phase,:]; eps=1e-4
    J=[zeros(4,4) for k in 0:K]
    for j in 1:4
        up=copy(u); um=copy(u)
        up[j]+=eps*D.σ[j];um[j]-=eps*D.σ[j]
        fp=smooth_run(up,D.p,D.c.T,2;n)
        fm=smooth_run(um,D.p,D.c.T,2;n)
        diff=(fp.-fm)./D.σ./(2eps)
        for k in 0:K; J[k+1][:,j]=diff[k+1,:]; end
    end
    J
end
truth_gain=[opnorm.(J) for J in truth_matrices]
power=Matrix{Float64}(I,r,r)
relative_error=map(0:K) do k
    learned=C*power*B
    err=mean(norm(learned-J[k+1])/max(norm(J[k+1]),eps()) for J in truth_matrices)
    global power=A*power
    err
end
freq=collect(range(0,4;length=401)); damping=0.1
# Evaluate outside the unit circle; unit-circle modes would make an undamped resolvent singular.
z=exp.(damping/n .+2π*im.*freq./n)
resolvent_full=[1/minimum(svdvals(zz*I-A)) for zz in z]
resolvent_output=[opnorm(C*((zz*I-A)\B)) for zz in z]
xs=collect(range(0.88,1.035;length=65));ys=collect(range(-0.25,0.25;length=85))
logres=[-log10(max(minimum(svdvals(complex(x,y)*I-A)),eps())) for x in xs,y in ys]
ev=eigvals(A)
commutator=norm(A'*A-A*A')/norm(A)^2
projection=eigvals(Symmetric(C*B))
@printf("DMD r=%d window=%.2f cycles commutator=%.5f max delay gain=%.3f current-state gain=%.3f\n",
        r,M.τ/n,commutator,maximum(state_gain),maximum(output_gain))
println("Current-block projection eigenvalues: ",projection)
println("True physical peak gains by phase: ",maximum.(truth_gain))
println("Relative response error at one cycle: ",relative_error[n+1])
serialize(joinpath(SMOOTH_DIR,"operators.jls"),
          (;n,K,state_gain,output_gain,truth_gain,relative_error,freq,damping,resolvent_full,resolvent_output,
            xs,ys,logres,ev,commutator,projection,rank=r,delay=M.τ))
