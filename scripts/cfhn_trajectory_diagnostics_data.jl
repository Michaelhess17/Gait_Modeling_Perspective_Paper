#!/usr/bin/env julia
# Diagnose the exact cached trajectories scored in Fig6_linear_affordances panel A.
# No refitting, new model selection, alignment, or model-specific reference.
include("fig6_relaxation_data.jl")

Ys, μ, σ = raw_training("cfhn", "decays"; seed=1)
cycle = cycle_states("cfhn")
Ct = (cycle .- μ) ./ σ
reference = deserialize(cache_path("fig6/relaxation.jls")).relax["cfhn"]
files = sort(filter(f -> occursin(r"^offv2_cfhn_(dmd|esn|node)_\d+\.jls$", f),
                    readdir(cache_path("floquet"))))
rows = map(files) do f
    o = deserialize(cache_path("floquet/" * f))
    ref = only(filter(r -> r.model == o.model && r.seed == o.seed, reference.rows))
    starts = map(eachindex(o.starts)) do k
        T = o.starts[k]
        F = o.fcs[k]
        @assert F !== nothing && all(isfinite, F)
        F = F[1:5SPC_F, :]
        truth = T[W_WARM+1:W_WARM+5SPC_F, :]
        td = dist_curve((truth .- μ) ./ σ, Ct; closed=true)
        fd = dist_curve(F, Ct; closed=true)
        @assert maximum(abs.(env(fd) .- ref.envs[k])) < 1e-8
        @assert maximum(abs.(env(td) .- reference.truth_envs[k])) < 1e-8
        raw = F .* σ .+ μ
        (; truth, raw, warmup=T[1:W_WARM, :], td, fd,
           truth_env=env(td), model_env=env(fd))
    end
    println(o.model, " seed ", o.seed, ": reproduced Figure 4A envelopes for all four starts")
    (; model=o.model, seed=o.seed, starts)
end
serialize(cache_path("fig6/cfhn_trajectory_diagnostics.jls"),
          (; rows, cycle, μ, σ, spc=SPC_F, warmup=W_WARM, dt=sys_dt("cfhn")))
