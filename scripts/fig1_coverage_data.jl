#!/usr/bin/env julia
# Data for Fig 1A: what the training data cover. Coupled FHN, same budget (30 cycles) as one long on-cycle
# trajectory (process noise only) or as 10 decays × 3 cycles from perturbed starts (offcycle_relaxation.jl,
# raw_training). Saved in raw units with the true cycle, and the slow-variable desynchrony w₁ − w₂ at each
# Poincaré section (upward v₁ crossings of mid-level), whose cycle-to-cycle map has slope |μ| = 0.69.
# Usage: julia --project=. scripts/fig1_coverage_data.jl
include("offcycle_relaxation.jl")

unstd(Y, μ, σ) = Y .* σ .+ μ
Ye, μe, σe = raw_training("cfhn", "excited")
Yd, μd, σd = raw_training("cfhn", "decays")
section_desync(X) = [(r = interp_row(X, c); r[2] - r[4]) for c in crossings(X)]
Xe = unstd(Ye[1], μe, σe); Xd = [unstd(Y, μd, σd) for Y in Yd]
out = (; excited = Xe, decays = Xd, cycle = cycle_states("cfhn"), spc = SPC_F,
         sec_excited = section_desync(Xe), sec_decays = section_desync.(Xd), μ_true = TRUTH.μ_slow)
mkpath(cache_path("fig1")); serialize(cache_path("fig1/coverage_cfhn.jls"), out)
slope(seqs) = (x = reduce(vcat, [q[1:end-1] for q in seqs]); y = reduce(vcat, [q[2:end] for q in seqs]); (x' * y) / (x' * x))
@printf("section w-desync slope: on-cycle %.3f, decays %.3f (truth %.3f)\n",
        slope([out.sec_excited]), slope(out.sec_decays), out.μ_true)
