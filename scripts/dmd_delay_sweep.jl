#!/usr/bin/env julia
# Off-cycle relaxation of decay-trained Hankel-DMD vs delay window (offcycle_v2.jl scoring: distance to the
# nearest point of the model's own run from an on-cycle warm-up; RMS per cycle relative to cycle 0).
# Warm-up length W = τ + 10 so long windows can be filled. Same data budget (30 cycles) laid out as
# 10 decays × 3 cycles or 5 decays × 6 cycles; extra args: number of 3-cycle decays (data size).
# Usage: julia --project=. scripts/dmd_delay_sweep.jl <sys> <ntraj>...
include("offcycle_v2.jl")

function decays(sys, ntraj, ncyc; seed = 1)
    σp = sys == "cfhn" ? SIGMA_P : 0.02 * sqrt(SL_SIGMA)
    cs = cycle_states(sys); sdc = vec(std(cs, dims = 1)); rng = MersenneTwister(seed)
    raw = [sim_from(sys, cs[rand(rng, 1:size(cs, 1)), :] .+ PERT .* sdc .* randn(rng, length(sdc)), ncyc * SPC_F;
                    σp, seed = seed * 1000 + k) for k in 1:ntraj]
    allX = reduce(vcat, raw); μ = mean(allX, dims = 1); σ = std(allX, dims = 1)
    [(X .- μ) ./ σ for X in raw], μ, σ
end

function truth_runs(sys, W, off)
    cs = cycle_states(sys); f! = sys == "cfhn" ? cfhn! : sl!; dt = sys_dt(sys); sdc = vec(std(cs, dims = 1))
    run(u0) = permutedims(Array(solve(ODEProblem(f!, u0, (0.0, (W + 6SPC_F) * dt)), Tsit5(); saveat = dt, abstol = 1e-10, reltol = 1e-10)))
    map((1, 28, 55, 83)) do ph
        u = copy(cs[ph, :])
        sys == "sl" ? (u .*= OFFSETS.sl) : (u[3] += off * sdc[3]; u[4] += off * sdc[4])
        (run(u), run(copy(cs[ph, :])))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
sys = ARGS[1]
for (ntraj, ncyc) in [(parse(Int, a), 3) for a in ARGS[2:end]]
    Ys, μ, σ = decays(sys, ntraj, ncyc)
    std_(X) = (X .- μ) ./ σ; Ctrue = std_(cycle_states(sys))
    for τ in round.(Int, SPC_F .* (1.0, 1.5, 2.0, 2.5)), r in (20, 40, 80, 160), variant in ("ls", "unit")
        τ >= ncyc * SPC_F - 20 && continue
        W = τ + 10; runs = truth_runs(sys, W, OFFSETS.cfhn)
        m = try dmd_model_multi(Ys; τ, r, variant, ridge = 1e-6) catch; continue end
        et = Vector{Float64}[]; em = Vector{Float64}[]
        for (T, T0) in runs
            Ts = std_(T); e = env(nearest(Ts[W+1:end, :], Ctrue)); push!(et, e ./ e[1])
            fc = dmd_free(m, Ts[1:W, :], 6SPC_F); f0 = dmd_free(m, std_(T0)[1:W, :], 6SPC_F)
            (all(isfinite, fc) && all(isfinite, f0)) || (push!(em, fill(NaN, 5)); continue)
            a = env(nearest(fc, f0)); push!(em, a ./ a[1])
        end
        med(v) = [median(filter(isfinite, [x[k] for x in v])) for k in 1:5]
        tr, md = med(et), med(em)
        @printf("%s %2d×%d τ=%3d (%.1f cyc) r=%3d %-4s | truth %s | DMD %s | per-cycle rate %.2f (truth %.2f)\n",
                sys, ntraj, ncyc, τ, τ / SPC_F, r, variant, join([@sprintf("%.2f", x) for x in tr], " "),
                join([@sprintf("%.2f", x) for x in md], " "), md[5]^(1/4), tr[5]^(1/4))
    end
end
end
