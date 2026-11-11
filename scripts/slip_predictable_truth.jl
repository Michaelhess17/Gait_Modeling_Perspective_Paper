#!/usr/bin/env julia
# Ceiling for the predictable-response experiment (predictable_response.jl): the noise-free continuation of the
# kicked SLIP walks from each test start, i.e. the best forecast possible without knowing future disturbances.
#
# Runs in SLIP.jl's own environment:
#   julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_predictable_truth.jl
#
# Test starts: walks with seeds 11–20 for κ = 0.0 and 0.1 (cache/truth/slip_walks), 30 starts per walk at steps
# k = 21, 36, …, each at a random phase j ∈ 1:NPHASE within step k. A walk is kicked only at VLO, so between VLO k and
# VLO k + 1 it follows the noise-free flow from the kicked state z_k + kick_k; the continuation from that state with no
# further kicks therefore passes through the observed state at (k, j) and is the noise-free future from there.
# Each continuation step is resampled onto NPHASE points of its own duration, exactly as phase_walk() does.
# → cache/predictable/slip_truth_continuations.jls

include("slip_truth.jl")                 # pattern_params, step_segment, NPHASE, EBAR (main block not run)

const NCONT = 8                          # steps of continuation (covers 3 strides from any phase)
const NSTART = 30

function resample_step(seg)
    tk = seg[:, 1]; τ = range(tk[1], tk[end]; length = NPHASE); out = zeros(NPHASE, 3)
    for d in 1:3, (j, τj) in enumerate(τ)
        m = clamp(searchsortedlast(tk, τj), 1, length(tk) - 1)
        f = (τj - tk[m]) / max(tk[m+1] - tk[m], eps())
        out[j, d] = (1 - f) * seg[m, d+1] + f * seg[m+1, d+1]
    end
    out
end

function continuation(w, k)
    p = pattern_params(; kappa = w.kappa)
    z = w.Z[k, :] .+ w.K[k, :]; stance = isodd(k) ? RIGHT : LEFT          # kicked_walk starts with RIGHT
    out = zeros(NCONT, NPHASE, 3)
    for s in 1:NCONT
        r = step_segment(z[1], z[2], stance, p); r === nothing && return nothing
        y1, θ1, seg, _, stance = r
        out[s, :, :] = resample_step(seg); z = [y1, θ1]
    end
    out
end

res = Dict{String, Any}()
for κ in (0.0, 0.1), seed in 11:20
    w = deserialize(joinpath(ROOT, "cache", "truth", "slip_walks", "walk_k$(κ)_s$(seed).jls"))
    rng = MersenneTwister(1000seed + round(Int, 10κ))
    ks = 21 .+ 15 .* (0:NSTART-1); js = rand(rng, 1:NPHASE, NSTART)
    C = [continuation(w, k) for k in ks]
    # check: the first continuation step must reproduce the walk's own step k
    err = maximum(maximum(abs.(C[i][1, :, :] .- w.S[ks[i], :, :])) for i in eachindex(ks) if C[i] !== nothing)
    @printf("κ %.1f seed %2d: %d/%d continuations, max |step k − walk| = %.2e\n", κ, seed,
            count(!isnothing, C), NSTART, err)
    res["k$(κ)_s$(seed)"] = (; ks, js, C)
end
mkpath(joinpath(ROOT, "cache", "predictable"))
serialize(joinpath(ROOT, "cache", "predictable", "slip_truth_continuations.jls"), res)
println("→ cache/predictable/slip_truth_continuations.jls")
