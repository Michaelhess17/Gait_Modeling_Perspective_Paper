#!/usr/bin/env julia
# Is there a stride rhythm in the fly tracks? (York et al. drosophilid walking data; cache/fly from the R conversion.)
# Flies step at roughly 5–20 Hz, inside the 50-Hz Nyquist band of these 100-Hz tracks. Welch spectra (2-s Hann windows,
# half overlap) of sustained walking only (windows whose smoothed speed stays above 0.3 cm/s), per strain, of
#   forward / lateral body velocity  from the x, y differences rotated into the heading frame
#   heading rate                     from the heading-angle differences
#   vt, vs (as provided)             translational and sideslip velocity
# A stepping rhythm would show as a peak above a smooth background; a lattice (limit-cycle) model needs one.
# Also the running hip angle's spectrum (maus_extract.jl, 250 Hz) for comparison.
#
# Usage: julia --project=. scripts/fly_spectra.jl      (writes cache/fly/spectra.jls)

using Mmap, CSV, DataFrames, Statistics, FFTW, Printf, Serialization

const ROOT = dirname(@__DIR__)
const D = joinpath(ROOT, "cache", "fly")
const L = 200

idx = CSV.read(joinpath(D, "fly_index.csv"), DataFrame); cols = readlines(joinpath(D, "fly_columns.txt")); nc = length(cols)
raw = Mmap.mmap(open(joinpath(D, "fly_tracks.f32")), Vector{Float32}, (sum(idx.n) * nc,))
trial(i) = permutedims(reshape(Float64.(raw[idx.offset[i]*nc+1:(idx.offset[i]+idx.n[i])*nc]), nc, idx.n[i]))
c(n) = findfirst(==(n), cols)
const W = 0.5 .- 0.5 .* cos.(2π .* (0:L-1) ./ L)
const F = (0:L÷2) .* (100 / L)

function signals(X)
    x, y, a = X[:, c("x")], X[:, c("y")], deg2rad.(X[:, c("angle")]); dx, dy = diff(x), diff(y); ah = a[1:end-1]
    (; forward = dx .* cos.(ah) .+ dy .* sin.(ah), lateral = -dx .* sin.(ah) .+ dy .* cos.(ah),
       heading_rate = mod.(diff(rad2deg.(a)) .+ 180, 360) .- 180, vt = X[1:end-1, c("vt")], vs = X[1:end-1, c("vs")],
       speed = X[1:end-1, c("vt_cm_smooth")])
end

const KEYS = (:forward, :lateral, :heading_rate, :vt, :vs)
out = Dict{String, Any}()
for strain in sort(unique(idx.strain))
    acc = Dict(k => zeros(length(F)) for k in KEYS); nw = 0
    for i in findall(==(strain), idx.strain)
        s = signals(trial(i))
        for st in 1:L÷2:length(s.speed)-L+1
            r = st:st+L-1
            (all(isfinite, s.speed[r]) && minimum(s.speed[r]) > 0.3) || continue
            any(k -> any(!isfinite, getfield(s, k)[r]), KEYS) && continue
            for k in KEYS; v = getfield(s, k)[r]; acc[k] .+= abs2.(rfft((v .- mean(v)) .* W)); end
            nw += 1
        end
    end
    nw == 0 && continue
    out[strain] = (; nwin = nw, P = Dict(k => acc[k] ./ sum(acc[k]) for k in KEYS))
    P = out[strain].P[:forward]
    @printf("%-14s %5d windows | forward velocity, dB re total: 2 Hz %5.1f  5 Hz %5.1f  10 Hz %5.1f  15 Hz %5.1f  20 Hz %5.1f\n", strain, nw,
            (10log10(P[findfirst(>=(f), F)]) for f in (2, 5, 10, 15, 20))...)
end
# Peak test: a rhythm needs the spectrum to rise with frequency somewhere above the noise floor. Largest rise
# 10·log10(P(f₂)/P(f₁)), 3 ≤ f₁ < f₂ ≤ 25 Hz, both bins ≥ 6 dB above the floor (median power over 30–50 Hz); and the
# frequency where the spectrum reaches that floor. (A smooth curve fitted through the fall and the floor leaves a
# spurious "excess" at the knee, ~13 Hz here, so a fitted-background test is not used.)
function rise(P)
    fl = median(P[(F .>= 30) .& (F .<= 50)]); ok = findall((F .>= 3) .& (F .<= 25) .& (P .>= 4fl))
    r = 0.0; for (a, i) in enumerate(ok), j in ok[a+1:end]; r = max(r, 10log10(P[j] / P[i])); end
    knee = F[findfirst(i -> F[i] >= 3 && P[i] < 4fl, eachindex(P))]
    r, knee
end
println("largest rise of the spectrum with frequency above the floor, 3–25 Hz (dB) [frequency where the floor is reached]:")
println("                  forward velocity     heading rate")
for (s, v) in sort(collect(out), by = first)
    @printf("  %-14s %5.1f [%4.1f Hz]      %5.1f [%4.1f Hz]\n", s, rise(v.P[:forward])..., rise(v.P[:heading_rate])...)
end
# running hip angle for comparison (250 Hz → same 2-s windows at 250 Hz)
A = deserialize(joinpath(ROOT, "cache", "maus", "angles.jls"))
Lr = 500; Wr = 0.5 .- 0.5 .* cos.(2π .* (0:Lr-1) ./ Lr); Pr = zeros(Lr ÷ 2 + 1)
for t in A.trials, st in 1:Lr÷2:size(t.X, 1)-Lr+1
    v = t.X[st:st+Lr-1, 1]; Pr .+= abs2.(rfft((v .- mean(v)) .* Wr))
end
running = (; f = (0:Lr÷2) .* (250 / Lr), P = Pr ./ sum(Pr))
serialize(joinpath(D, "spectra.jls"), (; f = F, flies = out, running))
println("wrote cache/fly/spectra.jls")
