#!/usr/bin/env julia
# Operator metrics for human walking and fly walking with the rodent decks' own fitting code and design
# (../Rodent_DMD_Analysis: scripts/18, 28, 31, 33; src/RodentDMDAnalysis.jl fit_ridge_dmd), so the speed effects can
# be set beside the lone pups, the Cntnap2 / L7-Tsc1 open field and wheel running (33_speed_consistency.py).
#
# Common to all: 20 delays, full rank (channels × 20), Schur-stabilised ridge DMD (λ = ridge · s₁²), every model fitted
# to exactly N transitions, DRAWS draws; metrics: self-dominance (mean diagonal / mean off-diagonal of the resolvent
# IO gain matrix at f0, scaled by its largest entry), Henrici departure, participation ratio of the Hankel data's
# squared singular values, log transient growth over one period of f0, log resolvent gain at f0, spectral radius.
#
#   human  unit = person, level = trial (speed in cm/s). Real time: 100 Hz, ridge 3e-9, N = 1000 (a 1020-sample
#          window at a random start in the 1500-sample trial), f0 = the trial's stride frequency. Stride time: 25
#          samples per stride, ridge 1e-4, f0 = one cycle per 25 samples, N = 150 (trials of ≥ 7 strides; the rodent
#          stride-time fits use 1000 = 40 strides, which a 15-s trial does not hold).
#   fly    unit = fly, levels = its 2-s walking segments split at their median speed (slow / fast), real time only
#          (no stride rhythm in the tracks), ridge 3e-9, N = 1000 (segments in random order, the last trimmed);
#          f0 = 1 Hz, an arbitrary body-velocity timescale, for the IO and resolvent metrics (no stride frequency).
#          Also one fit per fly over all its walking (N = 1000) for between-strain comparisons.
#
#   OPENBLAS_NUM_THREADS=1 julia -t 16 --project=../Rodent_DMD_Analysis scripts/cross_species_fits.jl
# Writes cache/cross_species/{human_real,human_stride,fly_speed,fly_all}.csv

const RODENT = normpath(joinpath(@__DIR__, "..", "..", "Rodent_DMD_Analysis"))
include(joinpath(RODENT, "src", "RodentDMDAnalysis.jl"))
using .RodentDMDAnalysis
using CSV, DataFrames, DMDAnalysis, LinearAlgebra, Random, Serialization, Statistics, Printf

const OUT = joinpath(dirname(@__DIR__), "cache", "cross_species")
const FS = 100.0
const ND = 20
const DRAWS = 20

function fit_metrics(bouts, f0, ridge)
    p = size(bouts[1], 2)
    model = DMD(rank = RodentDMDAnalysis.rank_rule(p * ND, 1e-6), stabilize = true)
    hr = DMDAnalysis.transform(HankelGenerator(n_delays = ND, delay_interval = 1), bouts)
    res = RodentDMDAnalysis.fit_ridge_dmd(model, hr, ridge)
    G = DMDAnalysis.resolvent_io_matrix(res, f0; fs = FS); M = G ./ maximum(G); n = size(M, 1)
    dd = mean(diag(M)) / mean(M[i, j] for i in 1:n, j in 1:n if i != j)
    hen = analyze(res, MetricSet(Henrici())).henrici
    s2 = Float64.(res.singular_values) .^ 2; s2 ./= sum(s2)
    k = max(1, round(Int, FS / f0))
    tg = try DMDAnalysis.transient_growth(res; kmax = k) catch; NaN end
    rs = DMDAnalysis.resolvent_gain_at(res, [f0]; fs = FS)[1]
    (; diagonal_dominance = dd, henrici = hen, participation_ratio = 1 / sum(s2 .^ 2), log_tg_stride = log10(max(tg, eps())),
       log_res_stride = log10(max(rs, eps())), spectral_radius = maximum(abs.(res.eigenvalues)), n_transitions = res.n_transitions)
end

centre(X) = X .- mean(X, dims = 1)
ntrans(b) = max(size(b, 1) - ND, 0)
"Whole segments in random order until exactly `target` transitions (last one trimmed), as scripts/18 draw();
also the indices of the segments used."
function draw(rng, segs, target)
    out = Matrix{Float64}[]; used = Int[]; total = 0
    for i in randperm(rng, length(segs))
        need = target - total; need <= 0 && break
        b = segs[i]; ntrans(b) == 0 && continue
        if ntrans(b) > need; need < 10 && break; b = b[1:need+ND, :]; end
        push!(out, b); push!(used, i); total += ntrans(b)
    end
    out, total, used
end

tmap(f, xs) = (out = Vector{Any}(undef, length(xs)); Threads.@threads for i in eachindex(xs); out[i] = try f(xs[i]) catch e; e end; end; out)
function collect_rows(rows, name)
    bad = findall(r -> r isa Exception, rows); isempty(bad) || @printf("%s: %d fits failed, first: %s\n", name, length(bad), sprint(showerror, rows[bad[1]]))
    df = DataFrame([r for r in rows if !(r isa Exception)]); CSV.write(joinpath(OUT, name * ".csv"), df)
    @printf("wrote %s.csv (%d rows)\n", name, nrow(df)); df
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    In = deserialize(joinpath(OUT, "inputs.jls"))
    H = In.human
    jobs = [(i, d) for i in eachindex(H) for d in 1:DRAWS]
    rows = tmap(jobs) do (i, d)
        h = H[i]; rng = MersenneTwister(hash((h.subject, h.speed, d))); s = rand(rng, 0:size(h.real, 1)-1020)
        merge((subject = h.subject, group = h.group, speed = h.speed, draw = d, f0 = h.f0),
              fit_metrics([centre(h.real[s+1:s+1020, :])], h.f0, 3e-9))
    end
    collect_rows(rows, "human_real")
    NS = 150
    jobs = [(i, d) for i in eachindex(H) for d in 1:DRAWS if size(H[i].stride, 1) >= NS + ND]
    rows = tmap(jobs) do (i, d)
        h = H[i]; rng = MersenneTwister(hash((h.subject, h.speed, d, :stride))); s = rand(rng, 0:size(h.stride, 1)-(NS+ND))
        merge((subject = h.subject, group = h.group, speed = h.speed, draw = d, f0 = FS / 25),
              fit_metrics([centre(h.stride[s+1:s+NS+ND, :])], FS / 25, 1e-4))
    end
    collect_rows(rows, "human_stride")

    F = In.fly; N = 1000
    jobs = []
    for f in F
        m = median(f.speed)
        for (lvl, sel) in (("slow", f.speed .<= m), ("fast", f.speed .> m))
            segs = f.segs[sel]; sum(ntrans, segs; init = 0) < N && continue
            for d in 1:DRAWS; push!(jobs, (f, lvl, segs, f.speed[sel], d)); end
        end
    end
    rows = tmap(jobs) do (f, lvl, segs, sp, d)
        sel, _, used = draw(MersenneTwister(hash((f.name, lvl, d))), segs, N)
        merge((fly = f.name, strain = f.strain, level = lvl, draw = d, mean_speed = mean(sp[used]), f0 = 1.0),
              fit_metrics(centre.(sel), 1.0, 3e-9))
    end
    collect_rows(rows, "fly_speed")
    jobs = [(f, d) for f in F for d in 1:5 if sum(ntrans, f.segs; init = 0) >= N]
    rows = tmap(jobs) do (f, d)
        sel, _, _ = draw(MersenneTwister(hash((f.name, :all, d))), f.segs, N)
        merge((fly = f.name, strain = f.strain, draw = d, mean_speed = mean(f.speed), walk_s = f.walk_s, trial_s = f.trial_s, f0 = 1.0),
              fit_metrics(centre.(sel), 1.0, 3e-9))
    end
    collect_rows(rows, "fly_all")
end
