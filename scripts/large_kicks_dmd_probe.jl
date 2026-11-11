#!/usr/bin/env julia
# Why does Hankel-DMD forecast the large FHN kicks poorly (Fig_large_kicks) when the same settings forecast small kicks
# almost exactly (cfhn_smooth_models.jl, validation MSE 0.009)? Two candidate causes:
#   capacity  the large-kick grid stopped at τ = 110, r = 120, where validation error was still falling
#   data      every large-kick training sequence holds a 0.4–1.6 SD kick: one linear operator must fit large nonlinear
#             excursions and the cycle at once
# Probes: (1) the large-kick fit over a wider grid; (2) DMD fitted to the near-cycle small-kick data (cfhn_smooth) and
# applied to the large kicks; (3) the large-kick fit with only small training kicks (≤ 0.6 SD), when they exist.
# For each: validation score, the settled distance of the model's own unkicked run to the true cycle (waveform), forecast
# error over cycles 1–3 for each kick kind, and coordination recovery (distance to the true cycle at cycles 1, 6, 12).
# Usage: julia -t 12 --project=. scripts/large_kicks_dmd_probe.jl
include("fig6_relaxation_data.jl")
include("cfhn_smooth_system.jl")
using Printf
BLAS.set_num_threads(1)
root = joinpath(SMOOTH_DIR, "large_kicks"); D = deserialize(joinpath(root, "data.jls")); n = D.n; K = 12n
standard(Y) = (Y .- D.μ) ./ D.σ
δ = reduce(vcat, [t.series[:, 2] .- t.series[:, 4] for t in D.train]); δμ = mean(δ); δσ = std(δ)
observe(Y) = hcat(standard(Y), (Y[:, 2] .- Y[:, 4] .- δμ) ./ δσ)
function clipA(A)
    F = schur(complex(A)); T = copy(F.T)
    for i in axes(T, 1); abs(T[i, i]) > 1 && (T[i, i] /= abs(T[i, i])); end
    real(F.Z * T * F.Z')
end
function dmd_free(m, Yh, nn)
    α = m.U' * vec(permutedims(Yh[end-m.τ:end, :])); out = zeros(nn, m.d)
    for k in 1:nn; α = m.A * α; out[k, :] = (m.U * α)[m.τ*m.d+1:end]; end
    out
end
# multi-trajectory Hankel-DMD; `skip[j]`: column index (per sequence) of the imposed jump to leave out, or 0
function fit(Ys, τ, rs, ridges; skip = zeros(Int, length(Ys)))
    d = size(Ys[1], 2); Hs = [build_hankel_multi(Y, τ) for Y in Ys]
    Xh = reduce(hcat, [h[:, [j for j in 1:size(h, 2)-1 if j != s]] for (h, s) in zip(Hs, skip)])
    Zh = reduce(hcat, [h[:, [j + 1 for j in 1:size(h, 2)-1 if j != s]] for (h, s) in zip(Hs, skip)])
    Hc = reduce(hcat, Hs); E = eigen(Symmetric(Hc * Hc')); Uall = E.vectors[:, end:-1:1]
    [(r = r, ridge = ridge, m = begin
         U = Uall[:, 1:min(r, size(Uall, 1))]; X = U' * Xh; Z = U' * Zh
         A = Z * X' / (X * X' + ridge * tr(X * X') / size(X, 1) * I); (; A = clipA(A), U, τ, d)
     end) for r in rs, ridge in ridges]
end
percycle(e) = [sqrt(mean(abs2, e[(k-1)*n+1:k*n, :])) for k in 1:12]
dcyc(F) = [sqrt(mean(abs2, dist_curve(F[(k-1)*n+1:k*n, :], D.cycle; closed = true))) for k in 1:12]
function assess(pred)              # pred(raw history, nn) → nn × 4 in large-kick standard units
    val = mean(D.validation) do t
        F = pred(t.history, 5n); any(!isfinite, F) || maximum(abs, F) > 50 ? Inf : mean(abs2, F .- standard(t.truth[2:5n+1, :]))
    end
    rows = map(D.test) do t
        F = pred(t.history, K); ch = copy(t.history); ch[end, :] = t.control[1, :]; F0 = pred(ch, K)
        ok = all(isfinite, F) && maximum(abs, F) < 50 && all(isfinite, F0)
        tr_ = standard(t.truth[2:K+1, :])
        (; kind = t.kind, ok, e13 = ok ? sqrt(mean(abs2, F[1:3n, :] .- tr_[1:3n, :])) : Inf,
           wave = ok ? sqrt(mean(abs2, dist_curve(F0[end-n+1:end, :], D.cycle; closed = true))) : Inf,
           rec = ok ? dcyc(F) : fill(Inf, 12), rec_true = dcyc(tr_))
    end
    med(kind, f) = median([f(r) for r in rows if r.kind == kind])
    (; val, wave = median([r.wave for r in rows]), amp13 = med("amplitude", r -> r.e13), coo13 = med("coordination", r -> r.e13),
       rec = [med("coordination", r -> r.rec[k]) for k in (1, 6, 12)], rec_true = [med("coordination", r -> r.rec_true[k]) for k in (1, 6, 12)],
       failed = count(r -> !r.ok, rows))
end
line(lab, a) = @printf("%-46s val %.4f | own cycle off by %.3f | err c1–3 amp %.2f coo %.2f | coord. distance c1/6/12 %.2f %.2f %.2f (truth %.2f %.2f %.2f) | failed %d\n",
                       lab, a.val, a.wave, a.amp13, a.coo13, a.rec..., a.rec_true..., a.failed)

# (1) large-kick data, wider grid
Ys = [observe(t.series) for t in D.train]
taus = (110, 165, 220)               # forecasts start from two cycles (221 samples) of history: τ ≤ 220
rs = (120, 200, 300, 450, 650); ridges = (1e-8, 1e-6, 1e-4)
res = Vector{Any}(undef, length(taus))
Threads.@threads for i in eachindex(taus)
    τ = taus[i]
    res[i] = [(; τ, f.r, f.ridge, a = assess((H, nn) -> dmd_free(f.m, observe(H), nn)[:, 1:4]))
              for f in fit(Ys, τ, rs, ridges; skip = fill(D.H - 1 - τ, length(Ys)))]
end
allr = sort(reduce(vcat, [vec(r) for r in res]), by = x -> x.a.val)
println("(1) large-kick training data, wider grid (best 8 by validation; the original choice was τ 110, r 120, ridge 1e-4: val 0.373)")
for x in allr[1:8]; line("τ $(x.τ), r $(x.r), ridge $(x.ridge)", x.a); end
x = only(filter(x -> x.τ == 110 && x.r == 120 && x.ridge == 1e-4, allr)); line("original: τ 110, r 120, ridge 1e-4", x.a)

# (2) fitted to the near-cycle small-kick data (as cfhn_smooth_models.jl), applied to the large kicks
S = deserialize(joinpath(SMOOTH_DIR, "data.jls"))
toS(H) = (H .- S.μ) ./ S.σ; fromS(F) = standard(F .* S.σ .+ S.μ)
println("\n(2) DMD fitted to near-cycle data (10 runs of 3 cycles, 0.2 SD starts, weak noise), applied to the large kicks")
for τ in (110, 165), f in fit(S.Ys, τ, (120, 200), (1e-4,))
    line("near-cycle data: τ $τ, r $(f.r), ridge 1e-4", assess((H, nn) -> fromS(dmd_free(f.m, toS(H), nn))))
end

# (3) large-kick data restricted to the smaller training kicks
amp = [sqrt(mean(abs2, (t.history[end, :] .- t.series[size(t.history, 1)-1, :]) ./ vec(std(D.c.C[1:end-1, :], dims = 1)))) for t in D.train]
println("\ntraining kick sizes (RMS, cycle SD): ", round.(sort(amp), digits = 2))
small = findall(<(0.9), amp)
if length(small) >= 6
    Ysm = Ys[small]
    for τ in (110, 220), f in fit(Ysm, τ, (120, 300), (1e-6, 1e-4); skip = fill(D.H - 1 - τ, length(Ysm)))
        line("$(length(small)) smaller training kicks: τ $τ, r $(f.r), ridge $(f.ridge)", assess((H, nn) -> dmd_free(f.m, observe(H), nn)[:, 1:4]))
    end
end


# (4) the cycle imposed: at each harmonic k of the cycle, the dominant eigenvalue near k (within 0.05 harmonics) is
# moved to the unit circle at exactly k cycles per period (n samples); all other eigenvalues are left as fitted
function snap_harmonics(A, nper)
    E = eigen(A); λ = copy(E.values); fr = angle.(λ) .* nper ./ (2π)
    for k in 0:floor(Int, nper / 2)
        c = [i for i in eachindex(λ) if abs(fr[i] - k) < 0.05]
        isempty(c) && continue
        i = c[argmax(abs.(λ[c]))]; λ[i] = cis(2π * k / nper)
        j = findfirst(x -> x != i && abs(λ[x] - conj(E.values[i])) < 1e-10, eachindex(λ))   # its conjugate partner
        j === nothing || (λ[j] = conj(λ[i]))
    end
    real(E.vectors * Diagonal(λ) / E.vectors)
end
println("\n(4) large-kick training data, cycle harmonics imposed on the unit circle (best 6 by validation)")
res4 = Vector{Any}(undef, length(taus))
Threads.@threads for i in eachindex(taus)
    τ = taus[i]
    res4[i] = [(; τ, f.r, f.ridge, a = assess((H, nn) -> dmd_free((; f.m..., A = snap_harmonics(f.m.A, n)), observe(H), nn)[:, 1:4]))
               for f in fit(Ys, τ, (120, 200, 300), (1e-6, 1e-4); skip = fill(D.H - 1 - τ, length(Ys)))]
end
all4 = sort(reduce(vcat, [vec(r) for r in res4]), by = x -> x.a.val)
for x in all4[1:6]; line("snapped: τ $(x.τ), r $(x.r), ridge $(x.ridge)", x.a); end
println("near-cycle data, snapped:")
for τ in (110, 165), f in fit(S.Ys, τ, (120, 200), (1e-4,))
    line("near-cycle, snapped: τ $τ, r $(f.r)", assess((H, nn) -> fromS(dmd_free((; f.m..., A = snap_harmonics(f.m.A, n)), toS(H), nn))))
end
serialize(joinpath(root, "dmd_probe", "probe.jls"), (; grid = allr, snapped = all4))
