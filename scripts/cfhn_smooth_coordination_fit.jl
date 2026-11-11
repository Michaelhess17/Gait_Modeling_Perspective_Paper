#!/usr/bin/env julia
# Redundant observable ablation. Select using unchanged four-state validation MSE.
include("fig6_relaxation_data.jl")
include("cfhn_smooth_system.jl")
BLAS.set_num_threads(1)
D=deserialize(joinpath(SMOOTH_DIR,"data.jls")); W=170
rawdiff(Y)=(Y[:,2].*D.σ[2].+D.μ[2]).-(Y[:,4].*D.σ[4].+D.μ[4])
δ=reduce(vcat,rawdiff.(D.Ys)); δμ=mean(δ); δσ=std(δ)
augment(Y)=hcat(Y,(rawdiff(Y).-δμ)./δσ)
Ys=augment.(D.Ys)
function score(pred)
    mean(D.V) do Y
        F=pred(Y[1:W,:],size(Y,1)-W)
        F===nothing || any(!isfinite,F) ? Inf : mean(abs2,F.-Y[W+1:end,:])
    end
end
function evaluate(pred)
    map(D.test) do t
        F=pred(t.truth[1:W,:],30D.n); F0=pred(t.oncycle[1:W,:],30D.n)
        ok=F!==nothing && F0!==nothing && all(isfinite,F) && all(isfinite,F0) && max(maximum(abs,F),maximum(abs,F0))<50
        (;ok,F,F0)
    end
end
for kind in ("dmd","esn")
    base=deserialize(joinpath(SMOOTH_DIR,"$kind.jls"))
    function fitpred(g,seed)
        if kind=="dmd"
            m=dmd_model_multi(Ys;g...)
            return (H,n)->dmd_free(m,augment(H),n)[:,1:4]
        end
        m=fit_esn_multi(Ys;g...,seed)
        (H,n)->begin
            S=run_steps(esn_step(m),esn_tf(m,augment(H),size(H,1)),n)
            S===nothing ? nothing : permutedims(S[m.N+1:m.N+4,:])
        end
    end
    fits=map(base.grid) do row
        g=row.g
        val=mean(score(fitpred(g,s)) for s in 1:(kind=="dmd" ? 1 : 2))
        println(kind," augmented ",g," val=",val);flush(stdout)
        (;g,score=val)
    end
    best=fits[argmin(getfield.(fits,:score))]
    R=[(;seed,tests=evaluate(fitpred(best.g,seed))) for seed in 1:(kind=="dmd" ? 1 : 4)]
    serialize(joinpath(SMOOTH_DIR,"$(kind)_coordination.jls"),(;kind,δμ,δσ,best,grid=fits,R))
    println("Saved augmented ",kind);flush(stdout)
end
