#!/usr/bin/env julia
# Human treadmill running (Maus et al. 2015, J R Soc Interface 12:20140899; Dryad doi:10.5061/dryad.r9v30):
# sagittal joint angles from the marker data, for the lattice-constrained model (maus_lattice.jl) and the
# cross-species operator metrics.
#
# Data: data/maus_data/subj<s>_r<r>.zip, unzipped to data/maus_data/extracted/subj<s>_r<r>/ (kin_t1_r<r>.mat,
# 250 Hz, 60 000 frames = 240 s; frc_t1_r<r>.mat, 1000 Hz). 4 subjects (1, 2, 3, 7), 6 recordings each (subject 3
# lacks r5), self-selected speed 2.6-3.0 m/s. Axes: x lateral, y forward (toe ahead of heel), z vertical.
# Marker gaps are stored as reconstructed values (ex_<marker> > 0), not NaN, and are used as they are.
#
# Angles (degrees, sagittal projection onto the y-z plane), right leg then left leg:
#   hip    thigh (trochanter -> lateral knee) from the downward vertical, + = knee ahead of hip. The pelvis markers
#          (ASIS/PSIS) are reconstructed in up to 18% of frames, so the thigh is not referred to the pelvis.
#   knee   thigh angle - shank angle (shank = lateral knee -> ankle), + = flexion
#   ankle  foot angle - shank angle - 90 (foot = heel -> 5th metatarsal head), + = dorsiflexion
# Every segment angle is measured from the downward vertical (+ = distal end forward), where no segment ever points,
# so nothing wraps at ±180° (in early swing the foot points down and back).
# Ankle point = midpoint of the lateral and medial malleoli; subject 7's right lateral malleolus is reconstructed in
# 95-100% of frames, so that side uses the medial malleolus alone (a constant offset; channels are standardised later).
#
# Vertical force under the right and left belt halves (sum of each plate's four sensors, 1000 Hz averaged to the 250-Hz
# frames; assumes the force and kinematic recordings start together, both 240 s): fz = [right left] (N). The plates
# are loaded together 8–15% of the time even above 300 N (feet landing across the belt midline), so contact is taken
# from the total force and the foot from the lower of the two feet (footz: lowest of heel and 5th metatarsal, right, left).
#
# MAT.jl is not in the project environment; this script runs in a temporary environment that adds it.
# Usage: julia scripts/maus_extract.jl      (writes cache/maus/angles.jls)

using Pkg; Pkg.activate(temp = true; io = devnull); Pkg.add(["MAT"]; io = devnull)
using MAT, Statistics, Serialization, Printf

const ROOT = dirname(@__DIR__)
const SRC = joinpath(ROOT, "data", "maus_data", "extracted")

segangle(p, d) = atand.(d[:, 2] .- p[:, 2], -(d[:, 3] .- p[:, 3]))             # from the downward vertical, + forward

function leg(f, s, subj)
    trc, kne, hee, mtv = (read(f, "$(s)_$m") for m in ("trc", "kne", "hee", "mtv"))
    anl, anm = read(f, "$(s)_anl"), read(f, "$(s)_anm")
    ank = (subj == 7 && s == "r") ? anm : (anl .+ anm) ./ 2
    th, sh, ft = segangle(trc, kne), segangle(kne, ank), segangle(hee, mtv)
    hcat(th, th .- sh, ft .- sh .- 90)
end

trials = []
for d in sort(readdir(SRC))
    m = match(r"^subj(\d+)_r(\d+)$", d); m === nothing && continue
    subj, rep = parse.(Int, m.captures)
    f = matopen(joinpath(SRC, d, "kin_t1_r$rep.mat"))
    X = hcat(leg(f, "r", subj), leg(f, "l", subj))
    com = read(f, "com"); phi1, phi2 = vec(read(f, "phi1")), vec(read(f, "phi2"))
    footz = hcat((min.(read(f, "$(s)_hee")[:, 3], read(f, "$(s)_mtv")[:, 3]) for s in ("r", "l"))...)   # lowest foot marker
    recon = Dict(k => mean(read(f, "ex_r_$k") .> 0) for k in ("trc", "kne", "anl", "anm", "hee", "mtv") if haskey(f, "ex_r_$k"))
    close(f)
    g = matopen(joinpath(SRC, d, "frc_t1_r$rep.mat")); speed = read(g, "speed"); vb = vec(read(g, "vb"))
    fl = sum(vec(read(g, "fzl$k")) for k in 1:4); fr = sum(vec(read(g, "fzr$k")) for k in 1:4); close(g)
    ds(x) = vec(mean(reshape(x[1:4*size(X, 1)], 4, :), dims = 1))          # 1000 Hz → 250 Hz (frames aligned)
    fz = hcat(ds(fr), ds(fl))
    @assert !any(isnan, X)
    push!(trials, (; subj, rep, fs = 250.0, X, com, phi1, phi2, speed, belt = mean(vb), recon, fz, footz))
    @printf("subj%d r%d: %d frames, speed %.2f m/s | hip %.0f°, knee %.0f°, ankle %.0f° range (right)\n", subj, rep,
            size(X, 1), speed, (maximum(X[:, k]) - minimum(X[:, k]) for k in 1:3)...)
end
mkpath(joinpath(ROOT, "cache", "maus"))
serialize(joinpath(ROOT, "cache", "maus", "angles.jls"),
          (; trials, channels = ["hip R", "knee R", "ankle R", "hip L", "knee L", "ankle L"]))
println("wrote cache/maus/angles.jls ($(length(trials)) trials)")
