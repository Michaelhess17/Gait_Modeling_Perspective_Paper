# r2_common.jl — shared by Figure_05.jl (free-run gallery) and Figure_06.jl (outcomes): model list, outcome classes and
# loading of the benchmark's free runs (cache/selfdrive). Outcomes (self_driving_benchmark.jl): ok (amplitude 0.75–1.33
# of the data's, period within 10 %), decays (right period, 0.25–0.75), wrong rhythm (period off, or amplitude > 1.33),
# collapse (< 0.25), blow-up (non-finite or > 4). Settings are chosen on the training data only (DMD and ESN:
# dmd_tuning.jl, esn_tuning.jl; Fourier in phase: benchmark_more.jl, by the same criterion); the neural ODE uses one
# fixed recipe, SSMLearnPy the configuration tuned on gait, the LSTM the published recipe.
using Pkg; Pkg.activate(dirname(dirname(@__DIR__)); io = devnull)
include(joinpath(@__DIR__, "..", "paper_style.jl"))
using Serialization, Statistics, NPZ, Random, Printf

const ROOTF = dirname(dirname(@__DIR__))
const DIR = joinpath(ROOTF, "cache", "selfdrive")
const C_LTP = colorant"#1a4f99"
const SYSTEMS = [("fhn", "FitzHugh–Nagumo", (1, 2), ("v", "w")),
                 ("cfhn", "coupled FHN", (1, 2), ("v₁", "w₁")),
                 ("vdp2", "coupled Van der Pol", (1, 2), ("x_A", "ẋ_A")),
                 ("slip", "SLIP walking", (1, 3), ("CoM height", "vertical vel.")),
                 ("gait91", "human gait", (1, 2), ("hip", "knee"))]
#            key     label               colour  marker      linestyle
const MODELS = [("FIP", "Fourier in phase", C_LTP, :diamond, :solid), ("DMD", "Hankel-DMD", CAT[1], :circle, :solid),
                ("ESN", "ESN", CAT[2], :utriangle, :dash), ("LSTM", "LSTM", CAT[5], :dtriangle, :dash),
                ("NODE", "neural ODE", CAT[3], :hexagon, :dash), ("SSM", "SSMLearnPy", CAT[4], :star5, :dash)]
const FATES = ["ok", "decays", "wrong", "collapse", "blow-up"]
const FATE_COL = Dict("ok" => colorant"#3d7a5c", "decays" => colorant"#9cc7ae", "wrong" => colorant"#d9b45a",
                      "collapse" => colorant"#a9a8a4", "blow-up" => colorant"#b5523b")

"All free runs of one benchmark cell: template, tuned DMD and ESN, neural ODE, SSMLearnPy, Fourier in phase, LSTM."
function load_cell(sys, regime)
    rows = NamedTuple[]; Yte = nothing; Ytr = nothing; P = NaN
    for tag in ("dmd-esn-template", "node")
        f = joinpath(DIR, "$(sys)_$(regime)_$(tag).jls"); isfile(f) || continue
        o = deserialize(f)
        append!(rows, [(; r.model, r.variant, r.e1, r.fate, Fc = Float64.(r.Fc)) for r in o.rows if !(r.model in ("DMD", "ESN"))])
        Yte = o.Yte; Ytr = o.Ytr; P = o.P
    end
    f = joinpath(DIR, "dmdtuned_$(sys)_$(regime).jls")          # DMD and ESN with settings tuned on the training data
    if isfile(f)
        o = deserialize(f); c = o.chosen
        push!(rows, (; model = "DMD", variant = "τ=$(c.τ) r=$(c.r) $(c.variant)", c.e1, c.fate, Fc = Float64.(o.Fc)))
    end
    f = joinpath(DIR, "esntuned_$(sys)_$(regime).jls")
    isfile(f) && append!(rows, [(; model = "ESN", variant = "seed $(r.seed)", r.e1, r.fate, Fc = Float64.(r.Fc)) for r in deserialize(f).rows])
    f = joinpath(DIR, "ssm_$(sys)_$(regime).npz")
    if isfile(f)
        S = npzread(f)
        append!(rows, [(; model = "SSM", variant = "deg $(S["degrees"][k])", e1 = S["e1"][k], fate = FATES[S["fate_code"][k] + 1],
                          Fc = S["fc"][k, :, :]) for k in eachindex(S["degrees"])])
    end
    for tag in ("fip", "lstm")
        f = joinpath(DIR, "$(tag)_$(sys)_$(regime).jls")
        isfile(f) && append!(rows, [(; r.model, r.variant, r.e1, r.fate, Fc = Float64.(r.Fc)) for r in deserialize(f).rows])
    end
    return rows, Ytr, Yte, P
end

CELLS = [(m[1], m[2]) for f in readdir(DIR) for m in (match(r"^(.+)_(realistic|long)_dmd-esn-template\.jls$", f),) if m !== nothing]
allrows = [(; x.model, x.fate, x.e1, sys, reg) for (sys, reg) in CELLS for x in load_cell(sys, reg)[1]]
fsb = joinpath(DIR, "stabilization_benchmark.jls")              # Hankel-DMD retuned without its spectral constraint
isfile(fsb) && append!(allrows, [(; model = "DMD0", o.retuned.fate, o.retuned.e1, sys = o.system, reg = o.regime) for o in deserialize(fsb)])

