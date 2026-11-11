#!/usr/bin/env julia
# Independent validation trajectories select settings; test trajectories are never used for selection.
include("fig6_relaxation_data.jl")
include("cfhn_smooth_system.jl")
BLAS.set_num_threads(1)
D=deserialize(joinpath(SMOOTH_DIR,"data.jls")); W=170; NTEST=30D.n
kind=isempty(ARGS) ? "dmd" : ARGS[1]
function val_score_smooth(pred)
    errors=map(D.V) do Y
        F=pred(Y[1:W,:],size(Y,1)-W)
        F===nothing || any(!isfinite,F) ? Inf : mean(abs2,F.-Y[W+1:end,:])
    end
    mean(errors)
end
function evaluate_smooth(pred)
    map(D.test) do t
        F=pred(t.truth[1:W,:],NTEST)
        F0=pred(t.oncycle[1:W,:],NTEST)
        if F===nothing || F0===nothing || any(!isfinite,F) || maximum(abs,F)>50 || maximum(abs,F0)>50
            return (;ok=false,F,F0)
        end
        E=env(dist_curve(F[1:5D.n,:],D.cycle;closed=true))
        Et=env(dist_curve(t.truth[W+1:W+5D.n,:],D.cycle;closed=true))
        (;ok=true,F,F0,E,Et,offset=sqrt(mean(abs2,dist_curve(F[end-3D.n+1:end,:],D.cycle;closed=true))))
    end
end
esnpred(m)=(H,n)->begin
    S=run_steps(esn_step(m),esn_tf(m,H,size(H,1)),n)
    S===nothing ? nothing : permutedims(S[m.N+1:end,:])
end
if kind=="dmd"
    grid=vec([(;τ,r,variant="ls",ridge) for τ in (27,55,110,165),r in (12,30,60,120),ridge in (1e-6,1e-4)])
    fits=map(grid) do g
        m=dmd_model_multi(D.Ys;g...)
        score=val_score_smooth((H,n)->dmd_free(m,H,n))
        println("DMD ",g," val=",score);flush(stdout)
        (;g,score)
    end
    best=fits[argmin(getfield.(fits,:score))]
    m=dmd_model_multi(D.Ys;best.g...)
    R=[(;seed=1,tests=evaluate_smooth((H,n)->dmd_free(m,H,n)))]
    serialize(joinpath(SMOOTH_DIR,"dmd.jls"),(;kind,grid=fits,best,model=m,R))
elseif kind=="esn"
    grid=vec([(;N=400,ρ,α,λ,σin=0.5) for ρ in (0.6,0.9),α in (0.2,0.5),λ in (1e-6,1e-4)])
    fits=map(grid) do g
        scores=[val_score_smooth(esnpred(fit_esn_multi(D.Ys;g...,seed=s))) for s in 1:2]
        println("ESN ",g," val=",mean(scores));flush(stdout)
        (;g,score=mean(scores))
    end
    best=fits[argmin(getfield.(fits,:score))]
    R=map(1:4) do seed
        m=fit_esn_multi(D.Ys;best.g...,seed)
        (;seed,tests=evaluate_smooth(esnpred(m)))
    end
    serialize(joinpath(SMOOTH_DIR,"esn.jls"),(;kind,grid=fits,best,R))
elseif kind=="node"
    grid=[(;hidden=h,depth=2,epochs=200,lr=2e-3,shoot=25) for h in (32,64)]
    candidates=map(grid) do g
        checkpoint=joinpath(SMOOTH_DIR,"node_candidate_h$(g.hidden).jls")
        if isfile(checkpoint)
            println("Loading completed NODE candidate ",checkpoint);flush(stdout)
            return deserialize(checkpoint)
        end
        vf,loss,weights=fit_node_multi(D.Ys,D.dt;g...,seed=1,return_model=true,report_every=20)
        pred=(H,n)->begin
            S=run_steps(step_of_vf(vf,D.dt),H[end,:],n)
            S===nothing ? nothing : permutedims(S)
        end
        score=val_score_smooth(pred)
        println("NODE ",g," val=",score);flush(stdout)
        candidate=(;g,score,loss,weights)
        serialize(checkpoint,candidate)
        candidate
    end
    best=candidates[argmin(getfield.(candidates,:score))]
    R=map(1:3) do seed
        checkpoint=joinpath(SMOOTH_DIR,"node_seed$(seed).jls")
        if isfile(checkpoint)
            println("Loading completed NODE seed ",seed);flush(stdout)
            return deserialize(checkpoint)
        end
        if seed==1
            weights=best.weights; loss=best.loss
            vf=u->first(weights.model(u,weights.ps,weights.st))
        else
            vf,loss,weights=fit_node_multi(D.Ys,D.dt;best.g...,seed,return_model=true,report_every=20)
        end
        pred=(H,n)->begin
            S=run_steps(step_of_vf(vf,D.dt),H[end,:],n)
            S===nothing ? nothing : permutedims(S)
        end
        row=(;seed,loss,weights,tests=evaluate_smooth(pred))
        serialize(joinpath(SMOOTH_DIR,"node_seed$(seed).jls"),row)
        row
    end
    # Keep serialized weights separately; lightweight summary/trajectory cache loads without Lux.
    plainR=[(;r.seed,r.loss,r.tests) for r in R]
    plainbest=(;best.g,best.score,best.loss)
    serialize(joinpath(SMOOTH_DIR,"node.jls"),(;kind,best=plainbest,R=plainR))
else
    error("unknown model $kind")
end
println("Completed ",kind)
