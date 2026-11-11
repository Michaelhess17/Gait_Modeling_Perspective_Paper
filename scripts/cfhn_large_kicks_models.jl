#!/usr/bin/env julia
include("fig6_relaxation_data.jl")
include("cfhn_smooth_system.jl")
include("lstm_model.jl")
BLAS.set_num_threads(1)
root=joinpath(SMOOTH_DIR,"large_kicks"); D=deserialize(joinpath(root,"data.jls"))
standard(Y)=(Y.-D.μ)./D.σ
δ=reduce(vcat,[t.series[:,2].-t.series[:,4] for t in D.train]);δμ=mean(δ);δσ=std(δ)
# Use the same redundant observable for both models, following the preceding ablation.
observe(Y)=hcat(standard(Y),(Y[:,2].-Y[:,4].-δμ)./δσ)
Ys=[observe(t.series) for t in D.train]
function val_score(pred)
    mean(D.validation) do t
        F=pred(observe(t.history),5D.n)
        F===nothing || any(!isfinite,F) || maximum(abs,F)>50 ? Inf : mean(abs2,F.-standard(t.truth[2:5D.n+1,:]))
    end
end
function evaluate(pred)
    map(D.test) do t
        F=pred(observe(t.history),12D.n)
        control_history=copy(t.history);control_history[end,:]=t.control[1,:]
        F0=pred(observe(control_history),12D.n)
        ok=F!==nothing && all(isfinite,F) && maximum(abs,F)<50
        (;ok,F,F0)
    end
end
if isempty(ARGS) || ARGS[1]=="dmd"
    fits=NamedTuple[];bestscore=Inf;bestmodel=nothing;bestg=nothing
    for τ in (27,55,110)
        Hs=[build_hankel_multi(Y,τ) for Y in Ys]
        # Include histories crossing the kick, but never learn the externally imposed jump as an autonomous transition.
        Xh=reduce(hcat,[h[:,[j for j in 1:size(h,2)-1 if j!=D.H-1-τ]] for h in Hs])
        Zh=reduce(hcat,[h[:,[j+1 for j in 1:size(h,2)-1 if j!=D.H-1-τ]] for h in Hs])
        Hcat=reduce(hcat,Hs);E=eigen(Symmetric(Hcat*Hcat'));Uall=E.vectors[:,end:-1:1]
        for r in (30,60,120),ridge in (1e-6,1e-4)
            U=Uall[:,1:min(r,size(Uall,1))];X=U'*Xh;Z=U'*Zh
            A=Z*X'/(X*X'+ridge*tr(X*X')/size(X,1)*I)
            A=stabilise_unit_clip(A);m=(;A,U,τ,d=5)
            score=val_score((H,n)->dmd_free(m,H,n)[:,1:4]);g=(;τ,r,ridge)
            push!(fits,(;g,score));println("DMD ",g," val=",score);flush(stdout)
            if score<bestscore
                global bestscore=score;global bestmodel=m;global bestg=g
            end
        end
    end
    isfinite(bestscore) || error("No valid DMD candidate")
    R=[(;seed=1,tests=evaluate((H,n)->dmd_free(bestmodel,H,n)[:,1:4]))]
    serialize(joinpath(root,"dmd.jls"),(;best=(g=bestg,score=bestscore),grid=fits,model=bestmodel,R,δμ,δσ))
elseif ARGS[1]=="lstm"
    # Winner et al. (2023) training (lstm_model.jl) on the 24 training sequences together (batch = sequences), same five
    # observables; one-cycle segments (109 samples), the first eight train, the ninth validates. Teacher forcing covers
    # the whole sequence, including the one imposed pre-kick → kicked step. Forecast: driven through the observed
    # history (with the kicked state last), then self-driven. One seed per call: julia ... lstm <seed>
    seed=parse(Int,ARGS[2]); t0=time()
    m=fit_lstm_winner(Ys;H=128,lookback=D.n-1,trainseg=8,valseg=1,seed)
    fit_secs=time()-t0
    println("LSTM seed $seed: best epoch $(m.best_epoch), val $(m.val), $(round(fit_secs/60,digits=1)) min");flush(stdout)
    pred=(H,n)->forecast_lstm(m,H,n)[:,1:4]
    R=[(;seed,tests=evaluate(pred),fit_secs,best_epoch=m.best_epoch,val=m.val)]
    serialize(joinpath(root,"lstm_seed$seed.jls"),(;R,δμ,δσ,settings=(;H=128,lookback=D.n-1,trainseg=8,valseg=1)))
else
    function fit_esn_kicks(;N=400,ρ,α,λ,seed)
        rng=MersenneTwister(seed);d=5
        W=Matrix(sprandn(rng,N,N,10/N));W.*=ρ/maximum(abs,eigvals(W))
        Win=0.5 .* (2rand(rng,N,d).-1);b=0.2 .* (2rand(rng,N).-1)
        Fs=Matrix{Float64}[];Ts=Matrix{Float64}[]
        for Y in Ys
            r=zeros(N);Rs=zeros(N,size(Y,1))
            for t in axes(Y,1);r=(1-α).*r.+α.*tanh.(W*r.+Win*Y[t,:].+b);Rs[:,t]=r;end
            ix=[t for t in 30:size(Y,1)-1 if t!=D.H-1]
            push!(Fs,vcat(Rs[:,ix],Y[ix,:]',ones(1,length(ix))));push!(Ts,Y[ix.+1,:]')
        end
        F=reduce(hcat,Fs);T=reduce(hcat,Ts);O=T*F'/(F*F'+λ*I)
        (;W,Win,b,α,N,d,Wr=O[:,1:N],Wu=O[:,N+1:N+d],c=O[:,end])
    end
    pred(m)=(H,n)->begin
        # esn_step consumes the current observation before applying the readout.
        # Supply the reservoir BEFORE that observation, so the kick is consumed once.
        prior=esn_tf(m,H,size(H,1)-1)
        initial=vcat(prior[1:m.N],H[end,:])
        S=run_steps(esn_step(m),initial,n)
        S===nothing ? nothing : permutedims(S[m.N+1:m.N+4,:])
    end
    grid=[(;N=400,ρ,α,λ) for ρ in (.6,.9),α in (.2,.5),λ in (1e-4,1e-2)]
    fits=map(vec(grid)) do g
        score=mean(val_score(pred(fit_esn_kicks(;g...,seed))) for seed in 1:2)
        println("ESN ",g," val=",score);flush(stdout);(;g,score)
    end
    best=fits[argmin(getfield.(fits,:score))];isfinite(best.score) || error("No valid ESN candidate")
    R=[(;seed,tests=evaluate(pred(fit_esn_kicks(;best.g...,seed)))) for seed in 1:4]
    serialize(joinpath(root,"esn.jls"),(;best,grid=fits,R,δμ,δσ,initialization="consume latest observation once"))
end
println("Completed large-kick fits")
