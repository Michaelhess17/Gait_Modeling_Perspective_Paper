#!/usr/bin/env julia
# R2: how long does each free run stay within ε of the true cycle? (benchmark of Fig2_self_driving.jl)
#
# For every cached free run (8 cycles from the end of the training window; Hankel-DMD as tuned and retuned without a
# spectral constraint, ESN 8 seeds, neural ODE 4 seeds, SSMLearnPy 2 degrees, mean-cycle template), the distance of each
# forecast sample to a reference set, RMS over channels in the cell's z-scored units (cycle-SD units):
#   simulated ODEs   the noise-free limit cycle (densely sampled), normalised exactly as the cell's data
#   SLIP             the settled cycle (last 3 cycles of the noise-free trajectory)
#   gait             the states observed in training (no true cycle is known)
# Exit time = first time the distance exceeds ε (censored at 8 cycles). The held-out data's own distance to the same
# reference gives the scale of real departures (process noise, real stride-to-stride variability).
#
# Usage: julia -t 8 --project=. scripts/r2_tube.jl   → cache/selfdrive/r2_tube.jls

include("dmd_tuning.jl")
using Random

const EPS = (0.25, 0.5, 1.0)
const DIRS = joinpath(dirname(@__DIR__), "cache", "selfdrive")
nearest(x, R) = sqrt(minimum(sum(abs2, R .- x', dims = 2)) / size(R, 2))          # RMS over channels
dist(F, R) = [any(!isfinite, F[i, :]) ? Inf : nearest(F[i, :], R) for i in axes(F, 1)]
exit_time(d, ε, P) = (i = findfirst(>(ε), d); i === nothing ? Inf : (i - 1) / P)

"Reference set and the cell's normalisation, rebuilt exactly as run_cell does."
function reference(system, regime)
    rg = REGIMES[regime]; D = npzread(cache_path("selfdrive/data_$(system)_$(regime).npz"))
    Ytr, Yte, P = D["Ytr"], D["Yte"], D["P"]
    name = startswith(system, "gait") ? "gait:" * system[5:end] : system
    if startswith(system, "gait")
        return (; R = Ytr, Ytr, Yte, P, kind = "training states")
    end
    is_ode = haskey(ODE_SYSTEMS, name)
    X, dt = load_system(name, rg.n_cyc, is_ode ? rg.σp : 0.0)
    Pn = period_samples(X)
    n_tr = min(round(Int, rg.n_cyc * Pn), size(X, 1) - round(Int, N_TEST_CYC * Pn) - 1)
    μ = mean(X[1:n_tr, :], dims = 1); s = std(X[1:n_tr, :], dims = 1)
    Ychk = (X[1:n_tr, :] .- μ) ./ s .+ rg.σm .* randn(MersenneTwister(11), size(X))[1:n_tr, :]   # noise drawn for the whole record, as run_cell
    @assert size(Ychk) == size(Ytr) && maximum(abs.(Ychk .- Ytr)) < 1e-6 "normalisation mismatch for $system $regime"
    R = if is_ode
        sy = ODE_SYSTEMS[name]; ddt = (sy.T / 110) / 10
        (simulate_ode(sy.f, sy.u0, nothing, 3 * 1100, ddt) .- μ) ./ s
    else                                                   # SLIP: settled end of the noise-free trajectory
        (X[end-round(Int, 3Pn)+1:end, :] .- μ) ./ s
    end
    (; R, Ytr, Yte, P, kind = is_ode ? "noise-free cycle" : "settled cycle")
end

function runs(system, regime, P, Ytr)
    out = NamedTuple[]
    f = joinpath(DIRS, "dmdtuned_$(system)_$(regime).jls")
    isfile(f) && push!(out, (; model = "DMD", run = "tuned", Fc = Float64.(deserialize(f).Fc)))
    sb = deserialize(joinpath(DIRS, "stabilization_benchmark.jls"))
    for o in sb
        (o.system == system && o.regime == regime) || continue
        g = o.retuned
        Fc = fit_dmd_variant(Ytr; τ = g.τ, r = g.r, variant = g.variant, P, stabilization = :none)(Ytr, round(Int, 8P))
        push!(out, (; model = "DMD0", run = "retuned, no constraint", Fc))
    end
    f = joinpath(DIRS, "esntuned_$(system)_$(regime).jls")
    isfile(f) && append!(out, [(; model = "ESN", run = "seed $(r.seed)", Fc = Float64.(r.Fc)) for r in deserialize(f).rows])
    f = joinpath(DIRS, "$(system)_$(regime)_node.jls")
    isfile(f) && append!(out, [(; model = "NODE", run = r.variant, Fc = Float64.(r.Fc)) for r in deserialize(f).rows])
    f = joinpath(DIRS, "ssm_$(system)_$(regime).npz")
    if isfile(f)
        S = npzread(f)
        append!(out, [(; model = "SSM", run = "deg $(S["degrees"][k])", Fc = Float64.(S["fc"][k, :, :])) for k in axes(S["fc"], 1)])
    end
    f = joinpath(DIRS, "$(system)_$(regime)_dmd-esn-template.jls")
    isfile(f) && append!(out, [(; model = "template", run = "", Fc = Float64.(r.Fc)) for r in deserialize(f).rows if r.model == "template"])
    out
end

cells = [(m[1], m[2]) for f in readdir(DIRS) for m in (match(r"^dmdtuned_(.+)_(realistic|long)\.jls$", f),) if m !== nothing]
res = Vector{Any}(undef, length(cells))
Threads.@threads for i in eachindex(cells)
    system, regime = cells[i]
    ref = reference(system, regime); P = ref.P; n8 = round(Int, 8P)
    ddata = dist(ref.Yte[1:min(end, n8), :], ref.R)
    recs = map(runs(system, regime, P, ref.Ytr)) do r
        d = dist(r.Fc[1:min(end, n8), :], ref.R)
        (; r.model, r.run, d = Float32.(d), exits = Dict(ε => exit_time(d, ε, P) for ε in EPS), maxd = maximum(d))
    end
    res[i] = (; system, regime, P, kind = ref.kind, data_q95 = quantile(ddata, 0.95), data_max = maximum(ddata),
              data_d = Float32.(ddata), recs)
end
serialize(joinpath(DIRS, "r2_tube.jls"), res)

for o in sort(res; by = o -> (o.system, o.regime))
    @printf("%-9s %-9s %-17s data q95 %.2f max %.2f | ", o.system, o.regime, o.kind, o.data_q95, o.data_max)
    for m in ("DMD", "DMD0", "ESN", "NODE", "SSM", "template")
        rr = [r for r in o.recs if r.model == m]; isempty(rr) && continue
        @printf("%s %d/%d ", m, count(r -> isinf(r.exits[0.5]), rr), length(rr))
    end
    println()
end
println("\nShare of runs within ε of the reference for all 8 cycles, pooled over cells:")
for ε in EPS
    @printf("ε = %.2f: ", ε)
    for m in ("DMD", "DMD0", "ESN", "NODE", "SSM", "template")
        rr = [r for o in res for r in o.recs if r.model == m]
        @printf("%s %3.0f%% (n=%d, median exit %s)  ", m, 100 * count(r -> isinf(r.exits[ε]), rr) / length(rr), length(rr),
                (e = median([r.exits[ε] for r in rr]); isinf(e) ? ">8" : @sprintf("%.1f", e)))
    end
    println()
end
