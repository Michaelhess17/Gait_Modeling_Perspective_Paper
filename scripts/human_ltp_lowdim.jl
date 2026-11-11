#!/usr/bin/env julia
# Can one 15-s walking trial support a Fourier-in-phase operator if the operator is small? (follow-up to human_ltp.jl,
# whose per-trial operators, chosen at rank 12, 4 harmonics, ridge 0.1, had no reliable structure from 8 strides; and
# slip_ltp_recovery.jl, where a ridge of 0.1 shrinks known multipliers by half.)
# Same pipeline (fit_ltp: deviation from the trial's mean stride, channels standardised, Hankel of delay d, POD to rank
# r, z_{t+1} = A(θ_t) z_t with A a Fourier series in phase to Hh harmonics, ridge λ), grid GDL × GRL × GHL × GLL down to
# rank 2 and 1 harmonic, with two phase estimates for cutting strides:
#   hip   the limb-1 hip angle's Hilbert phase (as strides() in fig5_gait_data.jl)
#   pair  the leading oscillatory pair of all six angles: top two left singular vectors of the Hankel matrix of the
#         standardised angles (window half a mean stride), phase = angle of the projection, at the window centre;
#         section shifted to the hip section on average. Uses every channel, so one noisy angle moves it less.
# Between cuts, samples are spaced uniformly in time (as before) in both.
# A. Running (maus_extract.jl; ~320 strides per trial): operators from 8-stride chunks (10 per trial within strides
#    1–240) scored on held-out strides 251–310 of the same trial (R² of the deviation 6, 12, 25 samples ahead, vs the
#    mean stride); against the same setting fitted on strides 1–240: leading multiplier, log quarter-stride growth
#    profile (as human_ltp.jl).
# B. Walking (companion superset): each trial's operator (all its strides, trials with ≥ 5) scored on the same person's
#    other trials within ×/÷ 1.25 of its speed, and on up to 8 trials of other people in its group at such speeds;
#    leading multiplier and growth profile per trial.
# Usage: julia --project=. -t 12 scripts/human_ltp_lowdim.jl      (writes cache/fig5/human_ltp_lowdim.jls)

include("human_ltp_pooled.jl")

const GDL = [4, 8]
const GRL = [2, 3, 4, 6, 8, 12, 16, 24]
const GHL = [0, 1, 2, 3]
const GLL = [1e-4, 1e-3, 1e-2, 1e-1]
const SETL = [(d, r, h, λ) for d in GDL for r in GRL for h in GHL for λ in GLL]
const HZL = [6, 12, 25]
const RATIO_L = 1.25

interp(v, t) = (i = clamp(floor(Int, t), 1, length(v) - 1); a = t - i; (1 - a) * v[i] + a * v[i+1])
hip_phase(X) = (; ϕ = unwrap(angle.(analytic(X[:, 1] .- mean(X[:, 1])))) .- π / 2, tc = collect(1.0:size(X, 1)))
"Phase of the leading oscillatory pair of the angles X (time × channels), at the window centres tc."
function pair_phase(X; frac = 0.5)
    Xs = (X .- mean(X, dims = 1)) ./ std(X, dims = 1); h = hip_phase(X)
    per = 2π * (size(X, 1) - 1) / abs(h.ϕ[end] - h.ϕ[1]); L = max(4, round(Int, frac * per))
    H = hankel(Xs, L); F = eigen(Symmetric(H * H')); V = F.vectors[:, end:-1:end-1]
    sv = sqrt.(max.(F.values[end:-1:end-2], 0)); Z = V' * H
    ψ = unwrap(atan.(Z[2, :], Z[1, :])); ψ[end] < ψ[1] && (ψ = -ψ)
    tc = collect(1:size(H, 2)) .+ (L - 1) / 2
    off = angle(mean(cis.(ψ .- [interp(h.ϕ, t) for t in tc])))
    (; ϕ = ψ .- off, tc, sv, L, back = mean(diff(ψ) .< 0))
end
"Strides cut where ϕ (at times tc) crosses 2πk, samples uniform in time between cuts; also the cut times."
function cut_strides(X, ϕ, tc; nph = NPH)
    b = Float64[]
    for k in ceil(Int, ϕ[1] / 2π)+1:floor(Int, ϕ[end] / 2π)
        i = findfirst(>=(2π * k), ϕ); (i === nothing || i == 1) && continue
        push!(b, tc[i-1] + (tc[i] - tc[i-1]) * (2π * k - ϕ[i-1]) / (ϕ[i] - ϕ[i-1]))
    end
    S = zeros(max(length(b) - 1, 0), nph, size(X, 2))
    for s in 1:length(b)-1, (j, t) in enumerate(range(b[s], b[s+1]; length = nph + 1)[1:nph])
        i = clamp(floor(Int, t), 1, size(X, 1) - 1); a = t - i
        S[s, j, :] = (1 - a) .* X[i, :] .+ a .* X[i+1, :]
    end
    S, b
end
cut(X, how) = (P = how == :hip ? hip_phase(X) : pair_phase(X); cut_strides(X, P.ϕ, P.tc))

"R² of the deviation HZL samples ahead from starts ts."
function skill_l(M, D, ts)
    p = size(D, 2); r = size(M.U, 2); C = M.U[end-p+1:end, :]; Ut = Matrix(M.U'); μ = vec(M.μ); d = M.d
    E = zeros(length(HZL)); R = zeros(length(HZL)); z = zeros(r); w = zeros(r); y = zeros(p); h = zeros(p * d)
    for t in ts
        for l in 0:d-1, j in 1:p; h[l*p+j] = D[t-d+1+l, j] - μ[j]; end
        mul!(z, Ut, h); k = 1
        for s in 1:HZL[end]
            mul!(w, M.Aph[mod(t + s - 2, NPH)+1], z); z, w = w, z
            if s == HZL[k]
                mul!(y, C, z)
                for j in 1:p; E[k] += (y[j] + μ[j] - D[t+s, j])^2; R[k] += D[t+s, j]^2; end
                k += 1
            end
        end
    end
    Float32.(1 .- E ./ R)
end
"Leading monodromy multiplier (modulus) and log quarter-stride growth by phase into the output angles."
function struct_l(M, p)
    m = maximum(abs, eigvals(monodromy(M))); C = M.U[end-p+1:end, :]; Cp = pinv(C); g = zeros(Float32, NPH)
    for φ in 0:NPH-1
        P = Matrix(1.0I, size(M.Aph[1])...); for k in φ:φ+QS-1; P = M.Aph[mod(k, NPH)+1] * P; end
        g[φ+1] = log(opnorm(C * P * Cp))
    end
    Float32(m), g
end

function running_trial(t, how)
    S, _ = cut(t.X, how); size(S, 1) < 310 && return nothing
    Dte = devs(S[251:310, :, :]); ts = 20:3:size(Dte, 1)-HZL[end]
    Df = devs(S[1:240, :, :]); Dc = [devs(S[(k-1)*24+1:(k-1)*24+8, :, :]) for k in 1:10]
    ns = length(SETL); full_sk = zeros(Float32, 3, ns); full_m = zeros(Float32, ns); full_g = zeros(Float32, NPH, ns)
    ch_sk = zeros(Float32, 3, 10, ns); ch_m = zeros(Float32, 10, ns); ch_g = zeros(Float32, NPH, 10, ns); k0 = 0
    for d in GDL, r in GRL, Hh in GHL
        Mf = fit_ltp(Df, 1:240*NPH, d, r, Hh, GLL); Mc = [fit_ltp(D, 1:8*NPH, d, r, Hh, GLL) for D in Dc]
        for li in eachindex(GLL)
            k = k0 + li; full_sk[:, k] = skill_l(Mf[li], Dte, ts); full_m[k], full_g[:, k] = struct_l(Mf[li], 6)
            for c in 1:10; ch_sk[:, c, k] = skill_l(Mc[c][li], Dte, ts); ch_m[c, k], ch_g[:, c, k] = struct_l(Mc[c][li], 6); end
        end
        k0 += length(GLL)
    end
    (; subj = t.subj, rep = t.rep, how, full_sk, full_m, full_g, ch_sk, ch_m, ch_g)
end

function walking_trial(i, Sh, sub, grp, spd, tests)
    near(j) = max(spd[i], spd[j]) / min(spd[i], spd[j]) <= RATIO_L
    own = [j for j in tests if j != i && sub[j] == sub[i] && near(j)]
    oth = [j for j in tests if sub[j] != sub[i] && grp[j] == grp[i] && near(j)]
    length(oth) > 8 && (oth = oth[randperm(MersenneTwister(i), length(oth))[1:8]])
    n = size(Sh[i], 1); D = devs(Sh[i]); Dt = Dict(j => devs(Sh[j]) for j in vcat(own, oth))
    tsj(j) = 20:2:size(Dt[j], 1)-HZL[end]
    ns = length(SETL); own_sk = fill(NaN32, 3, ns); oth_sk = fill(NaN32, 3, ns); m = zeros(Float32, ns); g = zeros(Float32, NPH, ns); k0 = 0
    for d in GDL, r in GRL, Hh in GHL
        Ms = fit_ltp(D, 1:n*NPH, d, r, Hh, GLL)
        for li in eachindex(GLL)
            k = k0 + li; M = Ms[li]
            isempty(own) || (own_sk[:, k] = mean(skill_l(M, Dt[j], tsj(j)) for j in own))
            isempty(oth) || (oth_sk[:, k] = mean(skill_l(M, Dt[j], tsj(j)) for j in oth))
            m[k], g[:, k] = struct_l(M, 6)
        end
        k0 += length(GLL)
    end
    (; i, subject = sub[i], group = grp[i], speed = spd[i], n, nown = length(own), noth = length(oth), own_sk, oth_sk, m, g)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    sub = String.(meta.subject); grp = String.(meta.lf_or_hf); spd = Float64.(meta.speed)
    A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls")); T = A.trials[qc_run.([strides(t.X) for t in A.trials]) .< 5]

    # phase diagnostics
    println("Phase estimates: strides per trial, stride-duration CV, deviation RMS (standardised), leading-pair quality")
    Sh = Dict{Symbol, Any}(); pdiag = Dict{Symbol, Any}()
    for how in (:hip, :pair)
        C = tmap(i -> cut(data[i, :, :], how), collect(axes(data, 1))); Sh[how] = first.(C)
        cv = [std(diff(c[2])) / mean(diff(c[2])) for c in C if length(c[2]) > 3]
        rms = [sqrt(mean(abs2, devs(c[1]))) for c in C if size(c[1], 1) >= 3]
        pdiag[how] = (; cv, rms, nstr = size.(Sh[how], 1))
        @printf("  walking %-4s strides %.0f [%d–%d] | duration CV %.3f | deviation RMS %.3f\n", how, median(size.(Sh[how], 1)),
                extrema(size.(Sh[how], 1))..., median(cv), median(rms))
        for g in ("AB", "HF", "LF")
            I = [k for k in eachindex(C) if grp[k] == g && length(C[k][2]) > 3]
            @printf("      %s: duration CV %.3f, deviation RMS %.3f\n", g, median(std(diff(C[k][2])) / mean(diff(C[k][2])) for k in I),
                    median(sqrt(mean(abs2, devs(C[k][1]))) for k in I if size(C[k][1], 1) >= 3))
        end
    end
    P = tmap(i -> pair_phase(data[i, :, :]), collect(axes(data, 1)))
    @printf("  leading pair: s2/s1 %.2f, s3/s2 %.2f, backward steps %.4f (medians)\n", median(p.sv[2] / p.sv[1] for p in P),
            median(p.sv[3] / p.sv[2] for p in P), median(p.back for p in P))
    for how in (:hip, :pair)
        C = [cut(t.X, how) for t in T[1:4]]
        @printf("  running %-4s duration CV %.3f, deviation RMS %.3f (4 trials)\n", how, median(std(diff(c[2])) / mean(diff(c[2])) for c in C),
                median(sqrt(mean(abs2, devs(c[1]))) for c in C))
    end
    flush(stdout)

    RA = tmap(collect(Iterators.product(eachindex(T), (:hip, :pair)))[:]) do (ti, how)
        running_trial(T[ti], how)
    end
    println("running done"); flush(stdout)
    RB = Dict{Symbol, Any}()
    for how in (:hip, :pair)
        S_ = Sh[how]; tests = [j for j in eachindex(S_) if size(S_[j], 1) >= 3]
        RB[how] = filter(!isnothing, tmap(i -> walking_trial(i, S_, sub, grp, spd, tests), [i for i in eachindex(S_) if size(S_[i], 1) >= 5]))
        println("walking $how done"); flush(stdout)
    end
    serialize(joinpath(ROOT, "cache", "fig5", "human_ltp_lowdim.jls"), (; RA = filter(!isnothing, RA), RB, pdiag, SETL, HZL, RATIO_L))
    println("wrote cache/fig5/human_ltp_lowdim.jls")
end
