#!/usr/bin/env julia
# Which validation criterion picks the models that forecast well and keep the cycle? (OUTLINE.md §3 ¶1 R1, to-do 7)
#
# For every Hankel-DMD setting of dmd_tuning.jl (delay depth × rank × variant, 60) and every ESN setting of
# esn_tuning.jl (ρ × leak × size × ridge, 36, seeds 1–3) in every self-driving benchmark cell (13 system × regime
# cells, the exported data_*.npz), the model is fitted on the training window minus its last two cycles and scored on
# those two held-out cycles by four criteria (errors relative to the RMS of the data they are scored on):
#   train 1-step   in-sample one-step error on the fitting data (the analogue of a training loss)
#   val 1-step     one-step (teacher-forced) error on the held-out cycles: each sample predicted from the true past
#   val rollout    one free run from the end of the fitting data over the two held-out cycles (the tuning criterion
#                  of dmd_tuning.jl / esn_tuning.jl, without DMD's amplitude penalty)
#   val multistart mean error of 1-cycle free runs started at 5 points across the held-out cycles (0, ¼, ½, ¾, 1 cycle
#                  in), each warmed up on the true data before its start
# Each setting is then refitted on the full training window and run freely as in the benchmark (score()): test
# forecast error over the first held-out test cycle (e1, and skill 1 − e1 / e1 of the mean-cycle reference) and the
# 30-cycle fate. Per cell and model class we record which setting each criterion would choose, and the rank
# correlation of each criterion with the test error.
#
# The ESN is refitted with a copy of fit_esn() that also exposes the reservoir for teacher forcing (same random
# draws; checked against fit_esn's free run). DMD uses fit_dmd_variant() from dmd_tuning.jl, whose forecaster runs
# from any supplied history.
#
# Usage: julia --project=. -t 16 scripts/validation_criteria.jl           (writes cache/selfdrive/validation_criteria.jls)
#        julia --project=. scripts/validation_criteria.jl summary           (summary from the cache only)

include("dmd_tuning.jl")

const ESN_GRID = [(; ρ, α, N, λ) for ρ in (0.5, 0.9, 1.2), α in (0.2, 0.5, 1.0), N in (200, 800), λ in (1e-6, 1e-3)]
dmd_grid(P) = [(; τ, r, variant) for τ in unique(max.(4, round.(Int, P .* (0.1, 0.25, 0.5, 1.0)))),
                                     r in (8, 16, 30, 50, 80), variant in ("ls", "fb", "unit")]
const STARTS = (0, 0.25, 0.5, 0.75, 1.0)

relerr(F, Y) = any(!isfinite, F) ? Inf : sqrt(mean(abs2, F .- Y)) / sqrt(mean(abs2, Y))

"fit_esn with the same random draws, returning one-step predictions on any data and free runs from any history."
function fit_esn_tf(Y; N = 400, ρ = 0.9, σin = 0.5, α = 0.5, λ = 1e-6, seed = 1, washout = 100)
    rng = MersenneTwister(seed); d = size(Y, 2)
    W = sprandn(rng, N, N, 10 / N); W .*= ρ / maximum(abs.(eigvals(Matrix(W))))
    Win = σin .* (2 .* rand(rng, N, d) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
    step(r, u) = (1 - α) .* r .+ α .* tanh.(W * r .+ Win * u .+ b)
    n = size(Y, 1); Rs = zeros(N, n); r = zeros(N)
    for t in 1:n; r = step(r, Y[t, :]); Rs[:, t] = r; end
    F = vcat(Rs[:, washout:n-1], Y[washout:n-1, :]', ones(1, n - washout))
    Wout = Y[washout+1:n, :]' * F' / (F * F' + λ * I)
    readout(r, u) = Wout * vcat(r, u, 1.0)
    "States after driving with the true data X (from rest), and the one-step predictions of X[t+1]."
    function drive(X)
        r = zeros(N); R = zeros(N, size(X, 1)); P1 = zeros(size(X, 1), d)
        for t in axes(X, 1); r = step(r, X[t, :]); R[:, t] = r; P1[t, :] = readout(r, X[t, :]); end
        R, P1
    end
    "Free run of nf samples after the true history X (the reservoir is driven through X first)."
    function freerun(X, nf)
        R, _ = drive(X[1:end-1, :]); r = R[:, end]; u = X[end, :]; out = zeros(nf, d)
        for k in 1:nf
            r = step(r, u); u = readout(r, u); out[k, :] = u
            if any(!isfinite, u) || maximum(abs, u) > 1e6; out[k:end, :] .= NaN; break; end
        end
        out
    end
    (; drive, freerun, forecast = (Ytr, nf) -> freerun(Y, nf))
end

"Four validation criteria for a model fitted on Yfit, scored on Yval (the next nv samples)."
function criteria(onestep_fit, onestep_val, rollout, Yfit, Yval, P)
    nv = size(Yval, 1); L = round(Int, P); W = vcat(Yfit, Yval)
    multi = map(STARTS) do s
        k = round(Int, s * P); k + L > nv && return NaN
        relerr(rollout(W[1:size(Yfit, 1)+k, :], L), Yval[k+1:k+L, :])
    end
    (; train1 = onestep_fit, val1 = onestep_val, valroll = relerr(rollout(Yfit, nv), Yval),
       valmulti = mean(filter(!isnan, multi)))
end

function dmd_row(g, Yfit, Yval, Ytr, Yte, P, n_run)
    f = fit_dmd_variant(Yfit; g..., P)
    τ = g.τ; n0 = τ + 1
    one(X, t) = f(X[1:t, :], 1)[1, :]                      # prediction of X[t+1] from X[1:t]
    tr1 = relerr(reduce(hcat, [one(Yfit, t) for t in n0:size(Yfit, 1)-1])', Yfit[n0+1:end, :])
    W = vcat(Yfit, Yval); nf = size(Yfit, 1)
    va1 = relerr(reduce(hcat, [one(W, t) for t in nf:size(W, 1)-1])', Yval)
    c = criteria(tr1, va1, f, Yfit, Yval, P)
    test = score(fit_dmd_variant(Ytr; g..., P)(Ytr, n_run), Yte, Ytr, P)
    (; model = "DMD", setting = g, seed = 0, c..., test...)
end

function esn_row(g, seed, Yfit, Yval, Ytr, Yte, P, n_run)
    m = fit_esn_tf(Yfit; g..., seed)
    _, P1 = m.drive(Yfit); w = 100
    tr1 = relerr(P1[w:end-1, :], Yfit[w+1:end, :])
    W = vcat(Yfit, Yval); nf = size(Yfit, 1); _, Pw = m.drive(W)
    va1 = relerr(Pw[nf:end-1, :], Yval)
    c = criteria(tr1, va1, (X, n) -> m.freerun(X, n), Yfit, Yval, P)
    test = score(fit_esn_tf(Ytr; g..., seed).forecast(Ytr, n_run), Yte, Ytr, P)
    (; model = "ESN", setting = g, seed, c..., test...)
end

function run_cell(file)
    D = npzread(cache_path("selfdrive/" * file))
    Ytr, Yte, P, n_run = D["Ytr"], D["Yte"], D["P"], Int(D["n_run"])
    nv = round(Int, 2P); Yfit, Yval = Ytr[1:end-nv, :], Ytr[end-nv+1:end, :]
    tpl = score(fit_template(Ytr, P)(Ytr, n_run), Yte, Ytr, P)
    jobs = vcat(vec([(:dmd, g, 0) for g in dmd_grid(P)]), [(:esn, g, s) for g in vec(ESN_GRID) for s in 1:3])
    rows = Vector{Any}(undef, length(jobs))
    Threads.@threads for i in eachindex(jobs)
        kind, g, s = jobs[i]
        rows[i] = try
            kind == :dmd ? dmd_row(g, Yfit, Yval, Ytr, Yte, P, n_run) : esn_row(g, s, Yfit, Yval, Ytr, Yte, P, n_run)
        catch err
            nothing
        end
    end
    cell = replace(file, "data_" => "", ".npz" => "")
    (; cell, P, template_e1 = tpl.e1, rows = [r for r in rows if r !== nothing])
end

const CRIT = (:train1 => "train 1-step", :val1 => "val 1-step", :valroll => "val rollout", :valmulti => "val multistart")
"Average ranks (ties share the mean rank, so the many infinite errors of blown-up runs do not get arbitrary order)."
function rk(v)
    o = sortperm(v); r = zeros(length(v)); i = 1
    while i <= length(v)
        j = i; while j < length(v) && v[o[j+1]] == v[o[i]]; j += 1; end
        r[o[i:j]] .= (i + j) / 2; i = j + 1
    end
    r
end
fin(x) = isfinite(x) ? x : 1e9

function summarise(cells)
    for m in ("DMD", "ESN")
        println("\n== $m: the setting each criterion chooses, per cell (test skill of the first cycle against the mean cycle; fate)")
        @printf("%-16s", "cell"); for (_, lab) in CRIT; @printf(" | %-22s", lab); end; @printf(" | %-22s\n", "oracle (test e1)")
        tally = Dict(k => (ok = 0, skill = Float64[]) for k in vcat(first.(collect(CRIT)), :oracle))
        rhos = Dict(k => Float64[] for k in first.(collect(CRIT)))
        for c in cells
            R = [r for r in c.rows if r.model == m]; isempty(R) && continue
            @printf("%-16s", c.cell)
            for k in vcat(first.(collect(CRIT)), :oracle)
                x = k == :oracle ? R[argmin([fin(r.e1) for r in R])] : R[argmin([fin(getfield(r, k)) for r in R])]
                sk = 1 - x.e1 / c.template_e1
                @printf(" | %+7.2f %-14s", sk, x.fate)
                t = tally[k]; tally[k] = (ok = t.ok + (x.fate == "ok"), skill = push!(t.skill, sk))
            end
            println()
            e1 = [fin(r.e1) for r in R]
            for k in first.(collect(CRIT))
                v = [fin(getfield(r, k)) for r in R]; push!(rhos[k], cor(rk(v), rk(e1)))
            end
        end
        n = length(tally[:oracle].skill)
        for k in vcat(first.(collect(CRIT)), :oracle)
            lab = k == :oracle ? "oracle (test e1)" : Dict(CRIT)[k]
            @printf("  %-16s chosen setting ok in %2d of %d cells; median test skill %+.2f", lab, tally[k].ok, n, median(tally[k].skill))
            k == :oracle ? println() : @printf("; rank corr with test e1 across settings, median over cells %+.2f\n", median(rhos[k]))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ARGS, 1, "") == "summary"
        summarise(deserialize(cache_path("selfdrive/validation_criteria.jls")))
    else
        BLAS.set_num_threads(1)
        # fit_esn_tf reproduces fit_esn
        D = npzread(cache_path("selfdrive/data_fhn_realistic.npz")); Y = D["Ytr"]
        @assert maximum(abs, fit_esn(Y; ρ = 0.9, α = 0.5, N = 200, λ = 1e-6, seed = 2)(Y, 50) .-
                             fit_esn_tf(Y; ρ = 0.9, α = 0.5, N = 200, λ = 1e-6, seed = 2).forecast(Y, 50)) < 1e-8
        files = sort(filter(f -> startswith(f, "data_") && endswith(f, ".npz"), readdir(cache_path("selfdrive"))))
        cells = []
        for f in files
            t = @elapsed c = run_cell(f); push!(cells, c)
            @printf("%-16s %4d settings in %5.0f s\n", c.cell, length(c.rows), t); flush(stdout)
        end
        serialize(cache_path("selfdrive/validation_criteria.jls"), cells)
        summarise(cells)
    end
end
