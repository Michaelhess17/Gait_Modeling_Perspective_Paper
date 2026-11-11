#!/usr/bin/env julia
# Inputs for the cross-species operator metrics (cross_species_fits.jl, run in ../Rodent_DMD_Analysis's environment
# so that the fits and metrics are that project's own code).
#
# Human walking: the companion paper's superset (../Gait_NonNormality_Paper3/data/human_gait_superset.npy, 299
#   trials × 1500 samples at 100 Hz × 6 sagittal angles; 42 people, each at 5–9 treadmill speeds in cm/s).
#   Real time: the trial as recorded; stride frequency from the limb-1 hip Hilbert phase.
#   Stride time: strides cut at that phase (fig5_gait_data.jl strides()), each resampled to 25 samples, concatenated
#   (the rodent decks' CyclePhaseNormalize: one cycle per 25 samples).
# Flies (York et al., "The evolutionary trajectory of drosophilid walking"; data/drosophila_data, converted to
#   cache/fly/ by R): 1 068 trials (one fly each), 29 strains/species, centroid tracks at 100 Hz. State =
#   [vt, vr, vs] smoothed (cm/s, deg/s, cm/s), z-scored per fly over its walking. Walking = smoothed translational
#   speed > 0.3 cm/s for ≥ 1 s; walking bouts cut into 2-s segments (the median fly has only 4 bouts, too few to split
#   by bout as the rodent decks do), each with its mean speed. These tracks carry no stride rhythm (spectra fall
#   monotonically through the 5–20 Hz stepping band), so the fly fits describe body-velocity dynamics only.
#
# Usage: julia --project=. scripts/cross_species_prep.jl      (writes cache/cross_species/inputs.jls)

include("fig5_gait_data.jl")
using Mmap

const FLY = joinpath(ROOT, "cache", "fly")
const SEG = 200            # 2-s fly segments
const VTHR = 0.3           # cm/s
const MINBOUT = 100        # samples

function human_inputs()
    data = npzread(joinpath(P3, "data", "human_gait_superset.npy"))
    meta = CSV.read(joinpath(P3, "data", "human_gait_superset_metadata.csv"), DataFrame)
    map(axes(data, 1)) do i
        X = data[i, :, :]
        ϕ = unwrap(angle.(analytic(X[:, 1] .- mean(X[:, 1])))); f0 = abs(ϕ[end] - ϕ[1]) / 2π / ((size(X, 1) - 1) / 100)
        S = strides(X; nph = 25)
        (; subject = String(meta.subject[i]), group = String(meta.lf_or_hf[i]), speed = Float64(meta.speed[i]),
           real = X, stride = reduce(vcat, [S[k, :, :] for k in axes(S, 1)]), f0, nstrides = size(S, 1))
    end
end

function fly_inputs()
    idx = CSV.read(joinpath(FLY, "fly_index.csv"), DataFrame); cols = readlines(joinpath(FLY, "fly_columns.txt")); nc = length(cols)
    raw = Mmap.mmap(open(joinpath(FLY, "fly_tracks.f32")), Vector{Float32}, (sum(idx.n) * nc,))
    c(n) = findfirst(==(n), cols)
    out = []
    for i in 1:nrow(idx)
        X = permutedims(reshape(Float64.(raw[idx.offset[i]*nc+1:(idx.offset[i]+idx.n[i])*nc]), nc, idx.n[i]))
        V = X[:, [c("vt_cm_smooth"), c("vr_cm_smooth"), c("vs_cm_smooth")]]
        w = vec(all(isfinite, V, dims = 2)) .& (X[:, c("vt_cm_smooth")] .> VTHR)
        bouts = UnitRange{Int}[]; s = 0
        for t in 1:length(w)+1
            on = t <= length(w) && w[t]
            on && s == 0 && (s = t)
            if !on && s > 0; t - s >= MINBOUT && push!(bouts, s:t-1); s = 0; end
        end
        isempty(bouts) && continue
        Wk = reduce(vcat, [V[b, :] for b in bouts]); μ, σ = mean(Wk, dims = 1), std(Wk, dims = 1)
        segs = Matrix{Float64}[]; sp = Float64[]
        for b in bouts, k in 0:length(b)÷SEG-1
            r = b[1]+k*SEG:b[1]+(k+1)*SEG-1; push!(segs, (V[r, :] .- μ) ./ σ); push!(sp, mean(V[r, 1]))
        end
        isempty(segs) && continue
        push!(out, (; name = idx.name[i], strain = idx.strain[i], segs, speed = sp, walk_s = sum(length, bouts) / 100,
                    trial_s = idx.n[i] / 100))
    end
    out
end

if abspath(PROGRAM_FILE) == @__FILE__
    H = human_inputs(); F = fly_inputs()
    @printf("humans: %d trials, %d people; strides per trial %s\n", length(H), length(unique(h.subject for h in H)),
            string(quantile([h.nstrides for h in H], [0, 0.5, 1])))
    @printf("flies: %d with walking, %d with ≥ 12 segments (2 × 1000 transitions possible); segments per fly median %d\n",
            length(F), count(f -> length(f.segs) >= 12, F), median(length.(getfield.(F, :segs))))
    mkpath(joinpath(ROOT, "cache", "cross_species"))
    serialize(joinpath(ROOT, "cache", "cross_species", "inputs.jls"), (; human = H, fly = F))
    println("wrote cache/cross_species/inputs.jls")
end
