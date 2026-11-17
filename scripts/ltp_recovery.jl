#!/usr/bin/env julia
# R5: can each model class recover known phase-dependent structure, and can its Floquet multipliers be read out?
# Ground truth: a linear time-periodic system z_{t+1} = A(θ_t) z_t + σ ε, 3 states, 100 samples per cycle,
# A(θ) = I + B(θ)/100 with B a Fourier series in θ with harmonics 0–2 only (sparse blocks); true multipliers per cycle
# 0.86 and 0.15 (a complex pair). σ = 0.05; data of 4–128 cycles; 20 simulations per size (10 for the nonlinear models).
#   linear  Fourier in phase (one lifted ridge regression with 4 harmonics), 20-bin phase-binned least squares,
#           time-invariant (H = 0): the error of A(θ) (RMS over phases, relative to A(θ)'s variation) and the multipliers
#           (eigenvalues of the monodromy matrix, the product of the per-sample maps over a cycle)
#   esn     ESN given the phase (cos θ, sin θ), as on the running deviations (300 units; ρ, leak and ridge chosen on an
#           inner split by forecast error 6–50 samples ahead); multipliers: the closed-loop model (reservoir and fed-back
#           output) linearised by finite differences over one cycle along its own periodic orbit; the three largest
#   node    neural ODE given the phase (maus_node.jl, state = the three variables), CPU; multipliers: the Jacobian of its
#           one-cycle flow map along its own periodic orbit
#   lstm    LSTM given the phase (Winner et al. training as on the running deviations, unit-SD channels; GPU, the 10
#           simulations of a size at once); multipliers: the closed loop (cell state and output fed back) linearised by
#           finite differences over one cycle along its own periodic orbit; the three largest
#   export  the simulated data for SSMLearnPy (ltp_recovery_ssm.py: multipliers from the linear part of its reduced map)
# Error of the multipliers: mean distance to the true ones after the best matching (the three largest by modulus).
# Usage: julia -t 8 --project=. scripts/ltp_recovery.jl linear | esn | node | export   → cache/recovery/<mode>.jls
#        NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia -t 8 --project=. scripts/ltp_recovery.jl lstm
const LR_MODE = get(ARGS, 1, "linear")
empty!(ARGS)
if LR_MODE == "lstm"
    using CUDA, cuDNN
end
include("maus_node.jl")
using SparseArrays

const OUTR = joinpath(ROOT, "cache", "recovery"); mkpath(OUTR)
const R3 = 3; const HTRUE = 2; const HFIT = 4; const NB = 20
fb(k, H) = (θ = 2π * k / NPH; vcat(1.0, [f(h * θ) for h in 1:H for f in (cos, sin)]))
Aof(W, k, H, r) = sum(fb(k, H)[i] .* W[:, (i-1)*r+1:i*r] for i in 1:2H+1)
B0 = [-1.0 4.0 0.0; -4.0 -1.0 1.0; 0.0 -1.0 -2.0]
B1c = [0.0 0.0 4.0; 0.0 3.0 0.0; 0.0 0.0 0.0]; B1s = [0.0 -3.0 0.0; 0.0 0.0 0.0; 3.0 0.0 0.0]
B2c = [2.5 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]; B2s = zeros(3, 3); B2s[2, 1] = -2.5
const WTRUE = hcat(I + B0 / NPH, ([B1c, B1s, B2c, B2s] ./ NPH)...)
Atrue(k) = Aof(WTRUE, k, HTRUE, R3)
mono(Af, r) = foldl((P, k) -> Af(k) * P, 0:NPH-1; init = Matrix(1.0I, r, r))
const ρTRUE = eigvals(mono(Atrue, R3))
function simulate(ncyc; σ = 0.05, seed = 1)
    rng = MersenneTwister(seed); n = ncyc * NPH; Z = zeros(R3, n); z = zeros(R3)
    for t in 1:n; Z[:, t] = z; z = Atrue(mod(t - 1, NPH)) * z .+ σ .* randn(rng, R3); end
    Z
end
lifted_fit(Z, H; λ = 1e-6) = (ph = [mod(c - 1, NPH) for c in 1:size(Z, 2)];
    Φ = reduce(hcat, [kron(fb(ph[c], H), Z[:, c]) for c in 1:size(Z, 2)-1]); G = Φ * Φ'; Z[:, 2:end] * Φ' / (G + λ * mean(diag(G)) * I))
binned_fit(Z) = map(1:NB) do b
    cs = [c for c in 1:size(Z, 2)-1 if fld(mod(c - 1, NPH) * NB, NPH) + 1 == b]
    Z[:, cs .+ 1] * Z[:, cs]' / (Z[:, cs] * Z[:, cs]')
end
rate_err(Âf) = sqrt(mean(sum(abs2, NPH .* (Âf(k) .- Atrue(k))) for k in 0:NPH-1)) / sqrt(mean(sum(abs2, NPH .* (Atrue(k) - I)) for k in 0:NPH-1))
function perms(n)
    n == 1 && return [[1]]
    [vcat(p[1:i-1], n, p[i:end]) for p in perms(n - 1) for i in 1:n]
end
"Mean distance between the three largest (by modulus) multipliers e and the true ones, after the best matching."
mult_err(e) = (e3 = sort(e; by = abs, rev = true)[1:3]; minimum(mean(abs.(e3[p] .- ρTRUE)) for p in perms(3)))
const CYC = [4, 8, 16, 32, 64, 128]; const CYC_NL = [16, 64, 128]
seedof(i, rep) = 1000i + rep

# ── ESN given the phase ──
function esn_fit(Z, seed; N = 300)
    D = permutedims(Z); n, p = size(D)
    settings = [(; ρ, α, λ) for ρ in (0.5, 0.9) for α in (0.3, 1.0) for λ in (1e-4, 1e-2)]
    function build(s)
        rng = MersenneTwister(seed)
        W = Matrix(sprandn(rng, N, N, 10 / N)); W .*= s.ρ / maximum(abs, eigvals(W))
        Win = 0.5 .* (2 .* rand(rng, N, p + 2) .- 1); b = 0.2 .* (2 .* rand(rng, N) .- 1)
        R = zeros(N, n); r = zeros(N)
        for t in 1:n; c, si = phase_cs(t); r = (1 - s.α) .* r .+ s.α .* tanh.(W * r .+ Win * vcat(D[t, :], c, si) .+ b); R[:, t] = r; end
        (; W, Win, b, R, α = s.α)
    end
    readout(E, rows, λ) = begin
        ix = collect(first(rows)+50:last(rows)-1); F = vcat(E.R[:, ix], D[ix, :]', ones(1, length(ix)))
        G = F * F'; D[ix .+ 1, :]' * F' / (G + λ * tr(G) / size(G, 1) * I)
    end
    step(E, Wout, r, u, t) = begin
        y = Wout * vcat(r, u, 1.0); c, si = phase_cs(t + 1)
        ((1 - E.α) .* r .+ E.α .* tanh.(E.W * r .+ E.Win * vcat(y, c, si) .+ E.b), y)
    end
    function verr(E, Wout, va)
        e = 0.0
        for t in first(va)+50:5:last(va)-50
            r = E.R[:, t]; u = D[t, :]
            for st in 1:50
                r, u = step(E, Wout, r, u, t + st - 1)
                st in (6, 12, 25, 50) && (e += sum(abs2, u .- D[t+st, :]))
            end
        end
        e
    end
    inner = 1:n÷2; va = n÷2+1:n
    vs = map(settings) do s
        E = build(s); verr(E, readout(E, inner, s.λ), va)
    end
    s = settings[argmin(vs)]; E = build(s); Wout = readout(E, 1:n, s.λ)
    # multipliers: the closed loop (r, u) on its own periodic orbit, linearised by finite differences over one cycle
    cyc(x) = begin                                                             # one cycle, from phase 0
        r = x[1:N]; u = x[N+1:end]
        for k in 0:NPH-1; r, u = step(E, Wout, r, u, k); end
        vcat(r, u)
    end
    x = zeros(N + p); for _ in 1:30; x = cyc(x); end
    ε = 1e-6; f0 = cyc(x)
    J = reduce(hcat, [(cyc(x .+ ε .* (1:N+p .== j)) .- f0) ./ ε for j in 1:N+p])
    (; mult = eigvals(J), setting = s)
end

# ── neural ODE given the phase: multipliers of its one-cycle flow map ──
function node_mult(f)
    p0 = f.p; nrm = f.nrm
    cyc(X) = begin
        τ = zeros(Float32, size(X, 2))
        for _ in 1:NPH; X = rk4(p0, X, τ, nrm, true); τ = τ .+ DT; end
        X
    end
    x = zeros(Float32, R3, 1); for _ in 1:30; x = cyc(x); end
    ε = 1f-3; X = hcat(x, x .+ ε .* Matrix{Float32}(I, R3, R3)); Y = cyc(X)
    J = Float64.((Y[:, 2:end] .- Y[:, 1]) ./ ε)
    eigvals(J)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    BLAS.set_num_threads(1)
    @printf("true multipliers %s\n", round.(ρTRUE, digits = 3))
    if LR_MODE == "linear"
        NREP = 20; curves = Dict(m => zeros(length(CYC), NREP) for m in (:lifted, :binned, :dmd))
        EIG = Dict(m => Matrix{Vector{ComplexF64}}(undef, length(CYC), NREP) for m in (:lifted, :binned, :dmd))
        ERR = Dict(m => zeros(length(CYC), NREP) for m in (:lifted, :binned, :dmd)); fits64 = Any[]
        for (i, nc) in enumerate(CYC), rep in 1:NREP
            Z = simulate(nc; seed = seedof(i, rep)); Wl = lifted_fit(Z, HFIT); Ab = binned_fit(Z); W0 = lifted_fit(Z, 0)
            for (m, Af) in ((:lifted, k -> Aof(Wl, k, HFIT, R3)), (:binned, k -> Ab[fld(k * NB, NPH) + 1]), (:dmd, k -> W0))
                curves[m][i, rep] = rate_err(Af); e = eigvals(mono(Af, R3)); EIG[m][i, rep] = e; ERR[m][i, rep] = mult_err(e)
            end
            nc == 64 && push!(fits64, (; Wl, Ab, W0))
        end
        for m in (:lifted, :binned, :dmd)
            @printf("%-7s A(θ) error %s | multiplier error %s\n", m, join([@sprintf("%.2f", median(curves[m][i, :])) for i in eachindex(CYC)], " "),
                    join([@sprintf("%.3f", median(ERR[m][i, :])) for i in eachindex(CYC)], " "))
        end
        serialize(joinpath(OUTR, "linear.jls"), (; CYC, curves, EIG, ERR, fits64, WTRUE, HTRUE, HFIT, NB, ρTRUE,
                                                 Atrue = [Atrue(k) for k in 0:NPH-1]))
    elseif LR_MODE == "esn"
        NREP = 10; jobs = [(i, nc, rep) for (i, nc) in enumerate(CYC) if nc in CYC_NL for rep in 1:NREP]
        res = tmap(j -> (Z = simulate(j[2]; seed = seedof(j[1], j[3])); r = esn_fit(Z, j[3]); (; nc = j[2], rep = j[3], r.mult, err = mult_err(r.mult), r.setting)), jobs)
        for nc in CYC_NL; @printf("esn %3d cycles: multiplier error %.3f\n", nc, median(x.err for x in res if x !== nothing && x.nc == nc)); end
        serialize(joinpath(OUTR, "esn.jls"), (; res, CYC_NL))
    elseif LR_MODE == "node"
        NREP = 10; out = Any[]
        for (i, nc) in enumerate(CYC)
            nc in CYC_NL || continue
            Ys = [permutedims(simulate(nc; seed = seedof(i, rep))) for rep in 1:NREP]
            bidx = [(s - 1) * NPH + 1 for s in 1:nc+1]; S = [node_split(Y, bidx, nc) for Y in Ys]
            t0 = time(); fits = fit_node_many(Ys, Ys, S, 1, R3, true; maxup = 6000, patience = 1500, dev = identity)
            for (rep, f) in enumerate(fits)
                e = node_mult(f); push!(out, (; nc, rep, mult = e, err = mult_err(e)))
            end
            @printf("node %3d cycles: multiplier error %.3f | %.1f min\n", nc, median(x.err for x in out if x.nc == nc), (time() - t0) / 60); flush(stdout)
            serialize(joinpath(OUTR, "node.jls"), (; res = out, CYC_NL))
        end
    elseif LR_MODE == "lstm"
        NREP = 10; out = Any[]; dev = gpu_device()
        for (i, nc) in enumerate(CYC)
            nc in CYC_NL || continue
            Zs = [simulate(nc; seed = seedof(i, rep)) for rep in 1:NREP]; n = nc * NPH; L = 99
            sds = [vec(std(Z, dims = 2)) for Z in Zs]
            Xs = [Float32.(hcat(permutedims(Z ./ sd), reduce(vcat, [collect(phase_cs(t))' for t in 1:n]))) for (Z, sd) in zip(Zs, sds)]
            trainseg = (n - 1 - L) ÷ (L + 1); t0 = time()
            ms = fit_lstm_winner_many(Xs; H = 128, lookback = L, trainseg, valseg = 1, seed = 1, device = dev, verbose = false,
                                      patience = cld(2000, trainseg), max_epochs = cld(40000, trainseg))
            for (rep, m) in enumerate(ms)
                # closed loop on the cell state: feed back the predicted variables with the exact phase of the predicted sample
                cyc(hc) = begin
                    h = hc[1:m.H, :]; c = hc[m.H+1:end, :]
                    for k in 0:NPH-1
                        u, _ = m.head(h, m.ps.h, m.sh); cs, sn = phase_cs(k + 1)
                        x = vcat(u[1:R3, :], fill(Float32(cs), 1, size(h, 2)), fill(Float32(sn), 1, size(h, 2)))
                        (o, carry), _ = m.cell((x, (h, c)), m.ps.c, m.sc); h, c = carry
                    end
                    vcat(h, c)
                end
                x = zeros(Float32, 2m.H, 1); for _ in 1:30; x = cyc(x); end
                ε = 1f-3; X = hcat(x, x .+ ε .* Matrix{Float32}(I, 2m.H, 2m.H)); Y = cyc(X)
                e = eigvals(Float64.((Y[:, 2:end] .- Y[:, 1]) ./ ε))
                push!(out, (; nc, rep, mult = e, err = mult_err(e)))
            end
            @printf("lstm %3d cycles: multiplier error %.3f | %.1f min\n", nc, median(x.err for x in out if x.nc == nc), (time() - t0) / 60); flush(stdout)
            serialize(joinpath(OUTR, "lstm.jls"), (; res = out, CYC_NL))
        end
    elseif LR_MODE == "export"
        dir = joinpath(OUTR, "ssm_io"); mkpath(dir)
        for (i, nc) in enumerate(CYC), rep in 1:10
            nc in CYC_NL || continue
            npzwrite(joinpath(dir, "sim_$(nc)_$(rep).npz"), Dict("Z" => permutedims(simulate(nc; seed = seedof(i, rep))), "rho" => ComplexF64.(ρTRUE)))
        end
        println("exported")
    end
end
