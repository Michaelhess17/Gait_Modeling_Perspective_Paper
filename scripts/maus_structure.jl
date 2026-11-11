#!/usr/bin/env julia
# Structure of the Fourier-in-phase (LTP) operator learned from long running trials (maus_floquet_models.jl), in the
# regime where it is learnable (≥ ~32 strides; maus_learning.jl), and how reproducible it is.
#
# Per trial: deviations of the 6 joint angles from the 61-stride moving-average stride (maus_lattice.jl :ma); one
# operator setting for all fits, SET = (delay 8, rank 64, 12 harmonics, ridge 1e-3), the usual choice at 160 strides.
# Fitted on the whole trial and on each half of its strides. From each fit:
#   multipliers   eigenvalues of the one-stride monodromy Π A(θ_k) (from phase 0), largest moduli
#   modes         the leading Floquet modes carried through the stride: y_j(φ) = C Φ(0→φ) v_j / λ_j^{φ/100}, the
#                 periodic shape each mode has in the joint angles (C: latest delay block of the POD basis)
#   growth(φ)     ‖C Φ(φ→φ+25) C⁺‖: how much a deviation of the joint angles present at phase φ can grow over the next
#                 quarter stride, worst case over deviations lifted into the delay state with C⁺ (an impulse-like state the
#                 data never visit: read with care); persist(φ), route(φ): the same for the trial's own states at φ
#                 (data_struct), the measure to interpret
# Gait events from the treadmill forces (maus_extract.jl): contact = total vertical force > FTH for ≥ 20 frames; the foot
# is the lower one (heel / 5th metatarsal); phases of right and left foot strike and toe-off, circular mean over strides.
# Phase 0 = descending zero crossing of the right thigh angle (strides()).
# Reproducibility: growth profiles and mode shapes (|y| over phase and joint) correlated between the two halves of a
# trial, between whole trials of the same person, and between people. Repetition (6 consecutive 4-min recordings):
# within-person Spearman ρ of the leading multiplier and mean growth with recording number.
#
# Usage: julia --project=. -t 16 scripts/maus_structure.jl      (writes cache/maus/structure.jls)

include("maus_floquet_models.jl")

const SET = (8, 64, 12, 1e-3)
const FTH = 100.0
const QS = 25

"Stride boundaries (fractional frame indices): strides()'s rule."
function stride_bounds(X)
    ϕ = unwrap(angle.(analytic(X[:, 1] .- mean(X[:, 1])))) .- π / 2; b = Float64[]
    for k in ceil(Int, ϕ[1] / 2π)+1:floor(Int, ϕ[end] / 2π)
        i = findfirst(>=(2π * k), ϕ); (i === nothing || i == 1) && continue
        push!(b, i - 1 + (2π * k - ϕ[i-1]) / (ϕ[i] - ϕ[i-1]))
    end
    b
end
cmean(φs) = isempty(φs) ? NaN : mod(angle(mean(cis.(2π .* φs))) / 2π, 1)
"Phases (fraction of the stride) of right and left foot strike and toe-off."
function gait_events(t)
    b = stride_bounds(t.X); c = (t.fz[:, 1] .+ t.fz[:, 2]) .> FTH; ev = Dict(k => Float64[] for k in (:RFS, :RTO, :LFS, :LTO))
    ph(i) = (s = searchsortedlast(b, i); (s < 1 || s >= length(b)) ? nothing : (i - b[s]) / (b[s+1] - b[s]))
    s0 = 0
    for i in 2:length(c)
        if c[i] && !c[i-1]; s0 = i
        elseif !c[i] && c[i-1] && s0 > 0 && i - s0 >= 20
            m = (s0 + i) ÷ 2; right = t.footz[m, 1] < t.footz[m, 2]
            a, e = ph(s0), ph(i - 1)
            a === nothing || push!(ev[right ? :RFS : :LFS], a); e === nothing || push!(ev[right ? :RTO : :LTO], e)
        end
    end
    Dict(k => cmean(v) for (k, v) in ev)
end

"""Data-weighted quarter-stride persistence and routing: the trial's own delay states z at phase φ (POD coordinates),
carried 25 samples by the operator. persist(φ) = ‖C P Z_φ‖ / ‖C Z_φ‖ (typical deviation size a quarter stride later
relative to now, < 1 = decaying); route(φ) = (C P Z_φ)(C Z_φ)⁺, joint j now → joint i a quarter stride later."""
function data_struct(M, D, rows; p = 6)
    C = M.U[end-p+1:end, :]; d = M.d; H = hankel(D[rows, :] .- M.μ, d); Z = M.U' * H
    ph = [mod(rows[1] + c + d - 3, NPH) for c in axes(Z, 2)]; Id = Matrix(1.0I, size(M.Aph[1])...)
    persist = zeros(NPH); route = zeros(p, p, NPH)
    for φ in 0:NPH-1
        Zφ = Z[:, ph .== φ]; P = foldl((X, k) -> M.Aph[mod(k, NPH)+1] * X, φ:φ+QS-1; init = Id)
        Y0 = C * Zφ; Y1 = C * P * Zφ
        persist[φ+1] = norm(Y1) / norm(Y0); route[:, :, φ+1] = Y1 * pinv(Y0)
    end
    (; persist, route)
end

function op_struct(M; p = 6, nmode = 3)
    C = M.U[end-p+1:end, :]; Cp = pinv(C); Id = Matrix(1.0I, size(M.Aph[1])...)
    F = eigen(monodromy(M)); o = sortperm(abs.(F.values), rev = true); λ = F.values[o]; V = F.vectors[:, o]
    modes = map(1:nmode) do j
        x = complex.(V[:, j]); Y = zeros(ComplexF64, NPH, p)
        for φ in 0:NPH-1
            Y[φ+1, :] = C * x ./ λ[j]^(φ / NPH); x = M.Aph[φ+1] * x
        end
        Y ./ maximum(abs, Y)
    end
    Γ = [(P = foldl((X, k) -> M.Aph[mod(k, NPH)+1] * X, φ:φ+QS-1; init = Id); C * P * Cp) for φ in 0:NPH-1]
    (; λ = λ[1:8], modes, growth = [opnorm(G) for G in Γ], routing = cat([abs.(G) for G in Γ]..., dims = 3))
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
    T = A.trials[qc_run.([strides(t.X) for t in A.trials]) .< 5]
    R = tmap(T) do t
        S_ = strides(t.X); n = size(S_, 1); nh = n ÷ 2; _, _, D = split_baseline(S_, :ma)
        fit(rows) = fit_ltp(D, rows, SET[1], SET[2], SET[3], [SET[4]])[1]
        both(rows) = (M = fit(rows); merge(op_struct(M), data_struct(M, D, rows)))
        (; whole = both(1:n*NPH), half1 = both(1:nh*NPH), half2 = both(nh*NPH+1:n*NPH), events = gait_events(t), n)
    end
    bad = count(isnothing, R); subs = [t.subj for t in T]; reps = [t.rep for t in T]
    @printf("%d trials (%d failed)\n", length(R), bad)
    println("\nGait events (phase, % of stride; phase 0 = right thigh vertical, moving back), median over a person's trials:")
    for s in unique(subs)
        I = findall(subs .== s)
        @printf("  subj%d: %s\n", s, join([@sprintf("%s %4.1f", k, 100 * cmean([R[i].events[k] for i in I])) for k in (:RFS, :RTO, :LFS, :LTO)], "  "))
    end
    println("\nMultipliers per stride (whole trial), |λ| of the top 4, and the leading one's rotation:")
    for s in unique(subs)
        I = findall(subs .== s)
        @printf("  subj%d: %s\n", s, join([join([@sprintf("%.2f", abs(x)) for x in R[i].whole.λ[1:4]], "/") * @sprintf("∠%.0f°", abs(rad2deg(angle(R[i].whole.λ[1])))) for i in I], "  "))
    end
    hcor(f) = [cor(f(R[i].half1), f(R[i].half2)) for i in eachindex(R)]
    pcor(f, same) = [cor(f(R[i].whole), f(R[j].whole)) for i in eachindex(R) for j in i+1:length(R) if (subs[i] == subs[j]) == same]
    mi3(v) = @sprintf("%.2f [%.2f, %.2f]", median(v), quantile(v, 0.25), quantile(v, 0.75))
    lgr(o) = log.(o.growth); m1(o) = vec(abs.(o.modes[1])); m2(o) = vec(abs.(o.modes[2]))
    println("\nReproducibility (correlation): halves of a trial | trials of the same person | different people")
    lps(o) = log.(o.persist)
    for (nm, f) in (("log growth profile", lgr), ("persistence profile", lps), ("mode 1 |shape|", m1), ("mode 2 |shape|", m2))
        @printf("  %-20s %s | %s | %s\n", nm, mi3(hcor(f)), mi3(pcor(f, true)), mi3(pcor(f, false)))
    end
    @printf("  leading |λ|: half 1 − half 2 %s; whole %s\n", mi3([abs(R[i].half1.λ[1]) - abs(R[i].half2.λ[1]) for i in eachindex(R)]), mi3([abs(r.whole.λ[1]) for r in R]))
    println("\nLog quarter-stride growth by phase (every 10%), mean over each person's trials:")
    for s in unique(subs)
        g = mean(lgr(R[i].whole) for i in findall(subs .== s)); @printf("  subj%d %s\n", s, join([@sprintf("%+.2f", g[k]) for k in 1:10:100], " "))
    end
    println("\nPersistence of a typical deviation over the next quarter stride, by phase (every 5%), mean over each person's trials:")
    for s in unique(subs)
        g = mean(R[i].whole.persist for i in findall(subs .== s)); @printf("  subj%d %s\n", s, join([@sprintf("%.2f", g[k]) for k in 1:5:100], " "))
    end
    println("\nLeading Floquet mode |shape| by joint (mean over all trials, every 10% of the stride; peak phase):")
    Ms = mean(abs.(r.whole.modes[1]) for r in R)
    for (c, nm) in enumerate(A.channels)
        @printf("  %-8s %s | peak at %d%%\n", nm, join([@sprintf("%.2f", Ms[k, c]) for k in 1:10:100], " "), argmax(Ms[:, c]) - 1)
    end
    println("\nRouting of typical deviations (|route|, mean over trials), stance vs flight (right stance 81→13%, flight 13→31%, left stance 31→64%, flight 64→81%):")
    for (nm, ph) in (("right stance", vcat(81:100, 1:13)), ("flight R→L", 14:31), ("left stance", 32:64), ("flight L→R", 65:81))
        Rm = mean(mean(abs.(r.whole.route[:, :, ph]), dims = 3)[:, :, 1] for r in R)
        @printf("  %-12s diag %s | largest off-diagonal: %s\n", nm, join([@sprintf("%.2f", Rm[i, i]) for i in 1:6], " "),
                join([@sprintf("%s→%s %.2f", A.channels[j], A.channels[i], Rm[i, j]) for (i, j) in Tuple.(partialsortperm(vec(Rm .* (1 .- I(6))), 1:3, rev = true) .|> k -> CartesianIndices(Rm)[k])], ", "))
    end
    println("\nRecording number (6 × 4 min back to back): within-person Spearman ρ")
    rk(v) = invperm(sortperm(v)) .* 1.0
    for (nm, f) in (("leading |λ|", r -> abs(r.whole.λ[1])), ("mean log growth", r -> mean(lgr(r.whole))), ("mean persistence", r -> mean(r.whole.persist)))
        ρ = [cor(rk(Float64.(reps[subs .== s])), rk([f(R[i]) for i in findall(subs .== s)])) for s in unique(subs)]
        @printf("  %-16s %s\n", nm, join([@sprintf("subj%d %+.2f", s, x) for (s, x) in zip(unique(subs), ρ)], "  "))
    end
    serialize(joinpath(ROOT, "cache", "maus", "structure.jls"), (; R, subs, reps, SET, channels = A.channels))
    println("wrote cache/maus/structure.jls")
end
