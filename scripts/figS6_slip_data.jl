#!/usr/bin/env julia
# SLIP inputs for Fig S6 (lattice-constrained models): the lattice fit's components on the 10 three-step decays, the
# Hankel-DMD spectrum in the cycle frame (floquet_lifted.jl part C), and per-start metrics of every model from the
# same 4 off-cycle starts (offcycle_v2.jl caches; floquet_lifted.jl parts B, E):
#   rate4   relaxation rate over the first 4 steps against the model's own on-cycle run (Fig 6B's metric)
#   off25   distance to the TRUE cycle after 25 steps / initial distance (does the free run end on the true cycle?)
#
#   julia --project=. scripts/figS6_slip_data.jl      (writes cache/figS6/slip.jls)

include("floquet_lifted.jl")

rate4(v) = all(isfinite, v) && v[1] > 0 ? (v[5] / v[1])^(1 / 4) : NaN
off25(fc) = fc === nothing ? NaN : (ev = cycenv(nearest(fc, CTRUE)); ev[26] / ev[1])

if abspath(PROGRAM_FILE) == @__FILE__
    B = fit_lattice(YS; Ks = 6); ρ, rot = percycle(B, B.σ, B.ν)
    @printf("lattice fit: %.3f ∠ %.1f° per step, period %.2f samples\n", ρ, rot, B.Pc)
    E(t, ϑ) = [exp((B.σ + im * B.ν) * t + im * k * ϑ) for k in -B.Ks:B.Ks]
    ex = map(eachindex(YS)) do r                                     # each decay: data, cycle, lattice deviation
        t = 0:size(YS[r], 1)-1; ϑ = B.ω .* t .- B.φs[r]
        (; Y = YS[r], cyc = cycmat(B.a, B.K, ϑ), dev = real.(B.ψ[r] .* reduce(hcat, [vec(transpose(E(tt, x)) * B.b) for (tt, x) in zip(t, ϑ)])'),
           amp = abs(B.ψ[r]) .* exp.(B.σ .* t), θ = mod.(ϑ, 2π))
    end
    θs = range(0, 2π; length = 201)
    bθ = reduce(hcat, [vec(transpose([exp(im * k * x) for k in -B.Ks:B.Ks]) * B.b) for x in θs])'
    acyc = cycmat(B.a, B.K, θs)
    dmd = only(c for c in deserialize(cache_path("floquet/lifted_C.jls")) if c.data == "3-step decays").lattice
    truth = (rate4 = [rate4(cycenv(nearest(TL[W_WARM+1:end, :], CTRUE))[1:5]) for TL in TRUTH_L],
             off25 = [(e = cycenv(nearest(TL[W_WARM+1:end, :], CTRUE)); e[26] / e[1]) for TL in TRUTH_L])
    rows = NamedTuple[]
    for m in ("dmd", "esn", "node"), f in sort(filter(f -> startswith(f, "offv2_slip_$(m)_"), readdir(cache_path("floquet"))))
        o = deserialize(cache_path("floquet/" * f))
        push!(rows, (; model = m, seed = o.seed, rate4 = rate4.(o.all_own), off25 = off25.(o.fcs), mult = nothing))
    end
    Bc = only(b for b in deserialize(cache_path("floquet/lifted_B.jls")) if b.data == "3-step decays")
    push!(rows, (; model = "lattice", seed = 0, rate4 = [s.own4 for s in Bc.score], off25 = [s.vs_true25 for s in Bc.score], mult = Bc.mult))
    for e in deserialize(cache_path("floquet/lifted_E.jls"))
        push!(rows, (; model = e.freeze ? "neural_frozen" : "neural_free", seed = e.seed, rate4 = [s.own4 for s in e.score],
                       off25 = [s.vs_true25 for s in e.score], mult = e.mult))
    end
    for r in rows
        @printf("%-14s seed %d  rate4 %s (truth %s) |err| %.3f   off25 %s\n", r.model, r.seed, fmt(r.rate4), fmt(truth.rate4),
                mean(abs.(r.rate4 .- truth.rate4)), fmt(r.off25))
    end
    mkpath(cache_path("figS6"))
    serialize(cache_path("figS6/slip.jls"), (; mult = (ρ, rot), Pc = B.Pc, ex, θs, bθ, acyc, dmd, truth, rows, exact = EXACT, rank1 = B.rank1))
end
