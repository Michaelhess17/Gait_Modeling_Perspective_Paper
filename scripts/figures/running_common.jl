# running_common.jl — data shared by Figure_04.jl (R1, running) and Figure_08.jl (R4): the running-deviation results of
# maus_learning_all.jl (cpu, lstm), maus_node.jl and maus_ssm.py, and the model list (colours as in the other figures).
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, Printf, NPZ

const ROOT = dirname(dirname(@__DIR__))
L = Dict(η => deserialize(joinpath(ROOT, "cache", "maus", "learning_all_cpu_$(η).jls")) for η in (0.0, 0.5, 1.0))
LSTMF = Dict(η => joinpath(ROOT, "cache", "maus", "learning_all_lstm_$(η).jls") for η in (0.0, 0.5, 1.0))
const SZ = L[0.0].SIZES; const HZ = L[0.0].HZ
# model, label, colour, marker, line style. Colours by model as in the other figures (DMD blue, ESN orange, neural ODE
# aqua, SSMLearnPy yellow); the phase-varying and phase-binned linear models in DMD's blue family; nonlinear models dashed
const C_LTP = colorant"#1a4f99"; const C_LPV = colorant"#78b0f5"
MODELS = [(:ltp, "phase-varying linear (Fourier in phase)", C_LTP, :diamond, :solid),
          (:lpv, "phase-binned linear (LPV)", C_LPV, :rect, :solid),
          (:dmd, "time-invariant linear (DMD)", CAT[1], :circle, :solid),
          (:lattice, "lattice (imposed cycle + one mode)", C_REF, :xcross, :solid),
          (:esn, "ESN", CAT[2], :utriangle, :dash)]
haslstm = isfile(LSTMF[0.0])
haslstm && push!(MODELS, (:lstm, "LSTM", CAT[5], :dtriangle, :dash))
NODEF = Dict(η => joinpath(ROOT, "cache", "maus", "node_dev_$(η).jls") for η in (0.0, 0.5, 1.0))
NODE = Dict(η => deserialize(f).out for (η, f) in NODEF if isfile(f))
haskey(NODE, 0.0) && push!(MODELS, (:node, "neural ODE", CAT[3], :hexagon, :dash))
SSMF = joinpath(ROOT, "cache", "maus", "ssm.npz"); SSM = isfile(SSMF) ? npzread(SSMF) : nothing
SSM === nothing || push!(MODELS, (:ssm, "SSM (SSMLearnPy)", CAT[4], :star5, :dash))
const PHASE_INPUT = (:esn, :lstm, :node)          # given the stride phase when modelling the deviation

"R² values over trials for model m, training size n, noise η, horizon index h."
function skillvals(m, n, η, h)
    if m == :lstm
        isfile(LSTMF[η]) || return Float64[]
        O = deserialize(LSTMF[η]).out; haskey(O, n) || return Float64[]
        return [r.r2[h] for r in O[n].rows]
    elseif m == :node
        (haskey(NODE, η) && haskey(NODE[η], n)) || return Float64[]
        return [r.r2[h] for r in NODE[η][n].rows]
    elseif m == :ssm
        SSM === nothing && return Float64[]
        e = findfirst(==(η), SSM["etas"]); k = findfirst(==(n), SSM["sizes"])
        (e === nothing || k === nothing) && return Float64[]
        return filter(isfinite, SSM["dev"][:, e, k, h])
    end
    [r[(m, n)].r2[h] for r in L[η].R if r !== nothing && haskey(r, (m, n))]
end
q(v, p) = isempty(v) ? NaN : quantile(v, p)
"Relative remaining variance, MSE / MSE(average stride) = 1 − R² of the deviation, over trials."
vals(m, n, η, h) = 1 .- skillvals(m, n, η, h)

