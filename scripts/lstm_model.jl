# lstm_model.jl — recurrent-network (LSTM) baseline, as used for gait signatures (Winner et al. 2023).
#
# One LSTM layer (H units) with a linear readout predicting the next observation. Trained sequence-to-sequence for
# autonomous use: on each window the network is teacher-forced over a warm-up of W samples, then fed its own
# predictions for K samples; the loss is the mean squared error of that free rollout (K grows from K/4 to K over the
# first half of training). Adam with gradient clipping, learning rate decayed tenfold. Forecasting warms up on the
# supplied history in the same way and then runs freely.
#
#   m = fit_lstm(Ys; H, seed, W, K)      Ys: vector of training series (T × d), z-scored
#   F = forecast_lstm(m, Yhist, n)       n × d free rollout after warming up on Yhist (≥ 1 sample)

using Lux, Zygote, Random, Statistics, Optimisers, LinearAlgebra

# Initialisation as Keras' LSTM / CuDNNLSTM and Dense (what Winner et al. used): input weights Glorot-uniform, recurrent
# weights orthogonal, biases zero with the forget-gate bias 1, readout Glorot-uniform with zero bias. Lux's own default
# for recurrent cells (Lux 1.31, init_rnn_weight) draws randn where rand is meant, so every weight and bias is Gaussian
# with mean −1/√H; for H = 128–512 this saturates the cell and leaves training on a plateau far above persistence.
# Lux's gate order is (input, forget, cell, output). Lux's cell also carries two bias vectors (bias_ih, bias_hh, as
# PyTorch); Keras has one, so bias_hh is kept at zero (frozen in training).
function keras_init(rng, d, H)
    glorot(m, n) = (2 .* rand(rng, Float32, m, n) .- 1) .* sqrt(6f0 / (m + n))
    A = randn(rng, Float32, 4H, H); F = qr(A); Q = Matrix(F.Q) .* sign.(diag(F.R))'
    bih = zeros(Float32, 4H); bih[H+1:2H] .= 1
    (; c = (; weight_ih = glorot(4H, d), weight_hh = Matrix{Float32}(Q), bias_ih = bih, bias_hh = zeros(Float32, 4H)),
       h = (; weight = glorot(d, H), bias = zeros(Float32, d)))
end

function fit_lstm(Ys::Vector{<:AbstractMatrix}; H = 64, seed = 1, W = 100, K = 100, iters = 1500, batch = 32,
                  lr = 3e-3, clip = 1.0)
    rng = Xoshiro(seed); d = size(Ys[1], 2)
    cell = LSTMCell(d => H); head = Dense(H => d)
    _, sc = Lux.setup(rng, cell); _, sh = Lux.setup(rng, head)
    ps = keras_init(rng, d, H)
    data = [Float32.(permutedims(Y)) for Y in Ys]                        # d × T each
    valid = [(i, s) for (i, X) in enumerate(data) for s in 1:size(X, 2)-W-K+1]
    isempty(valid) && error("training series shorter than W + K")
    function rollout(ps, X, Kk)
        (y, carry), _ = cell(X[:, 1, :], ps.c, sc)
        for t in 2:W
            (y, carry), _ = cell((X[:, t, :], carry), ps.c, sc)
        end
        u, _ = head(y, ps.h, sh)
        out = Zygote.Buffer(zeros(Float32, d, Kk, size(X, 3))); out[:, 1, :] = u
        for k in 2:Kk
            (y, carry), _ = cell((u, carry), ps.c, sc)
            u, _ = head(y, ps.h, sh); out[:, k, :] = u
        end
        copy(out)
    end
    opt = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.ClipNorm(clip), Optimisers.Adam(lr)), ps)
    Optimisers.freeze!(opt.c.bias_hh)                     # one bias per gate, as Keras (Lux carries two)
    for it in 1:iters
        Kk = it < iters ÷ 2 ? max(2, round(Int, K * (0.25 + 1.5 * it / iters))) : K
        idx = [valid[rand(rng, 1:length(valid))] for _ in 1:batch]
        X = cat([data[i][:, s:s+W+Kk-1] for (i, s) in idx]...; dims = 3)
        l, back = Zygote.pullback(p -> mean(abs2, rollout(p, X, Kk) .- X[:, W+1:W+Kk, :]), ps)
        isfinite(l) || continue
        Optimisers.adjust!(opt, lr * 0.1^(it / iters))
        opt, ps = Optimisers.update(opt, ps, back(1f0)[1])
    end
    (; cell, head, ps, sc, sh, d, H, W)
end
fit_lstm(Y::AbstractMatrix; kw...) = fit_lstm([Y]; kw...)

function forecast_lstm(m, Yhist::AbstractMatrix, n)
    X = Float32.(permutedims(Yhist)); out = zeros(n, m.d)
    (y, carry), _ = m.cell(X[:, 1:1], m.ps.c, m.sc)
    for t in 2:size(X, 2)
        (y, carry), _ = m.cell((X[:, t:t], carry), m.ps.c, m.sc)
    end
    u, _ = m.head(y, m.ps.h, m.sh)
    for k in 1:n
        out[k, :] = u
        if any(!isfinite, u) || maximum(abs, u) > 1e3
            out[k:end, :] .= NaN; break
        end
        k == n && break
        (y, carry), _ = m.cell((u, carry), m.ps.c, m.sc)
        u, _ = m.head(y, m.ps.h, m.sh)
    end
    out
end

# ── Winner et al. (2023) training, as in bermanlabemory/gait_signatures (Script 1) ───────────────────────────────────
# One LSTM layer + linear readout, trained teacher-forced to predict the next sample (MSE over every sample of a
# segment). Trials are batched together (batch = trials) and each trial is cut into consecutive segments of `lookback`
# samples; segments are processed in order with the hidden and cell states carried from one segment to the next
# (Keras `stateful=True`), gradients truncated at segment boundaries. The first `trainseg` segments train, the next
# `valseg` validate (one-step error, states carried through the training segments); Adam, early stopping on the
# validation loss (10000 epochs, patience 500, Adam 1e-3, Keras initialisation: keras_init). States start from zero at every epoch (Keras leaves this to the user; Winner et al. reset to zero
# when driving the trained model). At use, the state is reset to zero, the model is driven by the observed history
# (externally driven), and then fed its own predictions (self-driven): forecast_lstm.
# `device = gpu_device()` trains on the GPU (needs `using CUDA, cuDNN`); the returned parameters are on the CPU.
function fit_lstm_winner(trials::Vector{<:AbstractMatrix}; H = 512, lookback = 124, trainseg = 3, valseg = 1,
                         seed = 1, max_epochs = 10000, patience = 500, lr = 1e-3, device = cpu_device(), verbose = false)
    rng = Xoshiro(seed); d = size(trials[1], 2)
    cell = LSTMCell(d => H); head = Dense(H => d)
    _, sc = Lux.setup(rng, cell); _, sh = Lux.setup(rng, head)
    ps = keras_init(rng, d, H) |> device; sc = sc |> device; sh = sh |> device
    X = cat([Float32.(permutedims(Y)) for Y in trials]...; dims = 3) |> device    # d × T × B
    B = size(X, 3); L = lookback; need = (L + 1) * trainseg + valseg * L + 1
    size(X, 2) >= need || error("trials of $(size(X, 2)) samples are shorter than the $(need) needed")
    vstart = (L + 1) * trainseg                                                   # as in Script 1
    segs = [(X[:, i*L+1:(i+1)*L, :], X[:, i*L+2:(i+1)*L+1, :]) for i in 0:trainseg-1]
    vsegs = [(X[:, vstart+i*L+1:vstart+(i+1)*L, :], X[:, vstart+i*L+2:vstart+(i+1)*L+1, :]) for i in 0:valseg-1]
    zero_carry() = (fill!(similar(X, H, B), 0f0), fill!(similar(X, H, B), 0f0))
    function run(ps, xin, y, carry)                                               # teacher-forced segment: Σ squared error
        tot = 0f0
        for t in axes(xin, 2)
            (o, carry), _ = cell((xin[:, t, :], carry), ps.c, sc)
            u, _ = head(o, ps.h, sh); tot += sum(abs2, u .- y[:, t, :])
        end
        tot / length(y), carry
    end
    function val_loss(ps)
        carry = zero_carry()
        for (xin, y) in segs; _, carry = run(ps, xin, y, carry); end
        tot = 0.0
        for (xin, y) in vsegs; l, carry = run(ps, xin, y, carry); tot += l; end
        tot / valseg
    end
    opt = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)
    Optimisers.freeze!(opt.c.bias_hh)                     # one bias per gate, as Keras (Lux carries two)
    best = (Inf, ps, 0); history = Float64[]
    for ep in 1:max_epochs
        carry = zero_carry()
        for (xin, y) in segs
            c0 = carry
            (_, carry), back = Zygote.pullback(p -> run(p, xin, y, c0), ps)
            carry = (copy(carry[1]), copy(carry[2]))                              # truncate: no gradient across segments
            opt, ps = Optimisers.update(opt, ps, back((1f0, nothing))[1])
        end
        v = val_loss(ps); push!(history, v)
        v < best[1] && (best = (v, deepcopy(ps), ep))
        verbose && ep % 100 == 0 && println("epoch $ep val $(round(v, sigdigits = 4)) best $(round(best[1], sigdigits = 4)) @ $(best[3])")
        ep - best[3] > patience && break
    end
    cdev = cpu_device()
    (; cell, head, ps = best[2] |> cdev, sc = sc |> cdev, sh = sh |> cdev, d, H, best_epoch = best[3], val = best[1],
       history, kind = :winner)
end
fit_lstm_winner(Y::AbstractMatrix; kw...) = fit_lstm_winner([Y]; kw...)

"""
    fit_lstm_winner_many(trials; H = 128, kw...)

One independent `fit_lstm_winner` per trial (each model sees only its own trial; same segments, validation, early
stopping and initialisation as a standalone fit with this seed), trained simultaneously as one batched computation with
the models along the third dimension, so that a GPU is used efficiently. The loss is the sum of the models' own losses
and Adam acts elementwise, so every model follows its standalone trajectory; a model's best parameters stop being
updated once it has gone `patience` epochs without improving. Returns a vector of models for forecast_lstm.
"""
function fit_lstm_winner_many(trials::Vector{<:AbstractMatrix}; H = 128, lookback = 124, trainseg = 3, valseg = 1,
                              seed = 1, max_epochs = 10000, patience = 500, lr = 1e-3, device = cpu_device(),
                              verbose = false)
    bmul = Lux.NNlib.batched_mul
    d = size(trials[1], 2); M = length(trials)
    cell = LSTMCell(d => H); head = Dense(H => d)
    inits = map(1:M) do _
        rng = Xoshiro(seed); Lux.setup(rng, cell); Lux.setup(rng, head); keras_init(rng, d, H)   # as fit_lstm_winner
    end
    st3(f) = cat([f(p) for p in inits]...; dims = 3)
    ps = (; Wih = st3(p -> p.c.weight_ih), Whh = st3(p -> p.c.weight_hh),
            b = st3(p -> reshape(p.c.bias_ih .+ p.c.bias_hh, :, 1)),
            Wo = st3(p -> p.h.weight), bo = st3(p -> reshape(p.h.bias, :, 1))) |> device
    X = cat([Float32.(permutedims(Y)) for Y in trials]...; dims = 3) |> device    # d × T × M
    L = lookback; need = (L + 1) * trainseg + valseg * L + 1
    size(X, 2) >= need || error("trials of $(size(X, 2)) samples are shorter than the $(need) needed")
    vstart = (L + 1) * trainseg
    segs = [(X[:, i*L+1:(i+1)*L, :], X[:, i*L+2:(i+1)*L+1, :]) for i in 0:trainseg-1]
    vsegs = [(X[:, vstart+i*L+1:vstart+(i+1)*L, :], X[:, vstart+i*L+2:vstart+(i+1)*L+1, :]) for i in 0:valseg-1]
    zero_carry() = (fill!(similar(X, H, 1, M), 0f0), fill!(similar(X, H, 1, M), 0f0))
    sig(x) = inv(1 + exp(-x))
    function run(ps, xin, y, carry)                                   # per-model mean squared error (1 × 1 × M)
        h, c = carry; tot = 0f0
        for t in axes(xin, 2)
            g = bmul(ps.Wih, xin[:, t:t, :]) .+ bmul(ps.Whh, h) .+ ps.b
            c = sig.(g[H+1:2H, :, :]) .* c .+ sig.(g[1:H, :, :]) .* tanh.(g[2H+1:3H, :, :])
            h = sig.(g[3H+1:4H, :, :]) .* tanh.(c)
            u = bmul(ps.Wo, h) .+ ps.bo
            tot = tot .+ sum(abs2, u .- y[:, t:t, :]; dims = (1, 2))
        end
        tot ./ (d * size(y, 2)), (h, c)
    end
    function val_loss(ps)
        carry = zero_carry()
        for (xin, y) in segs; _, carry = run(ps, xin, y, carry); end
        tot = 0f0
        for (xin, y) in vsegs; l, carry = run(ps, xin, y, carry); tot = tot .+ l; end
        vec(Array(tot)) ./ valseg
    end
    opt = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)
    best = map(copy, ps); best_val = fill(Inf, M); best_ep = zeros(Int, M); stopped = falses(M)
    for ep in 1:max_epochs
        carry = zero_carry()
        for (xin, y) in segs
            c0 = carry
            (_, carry), back = Zygote.pullback(p -> (r = run(p, xin, y, c0); (sum(r[1]), r[2])), ps)
            carry = (copy(carry[1]), copy(carry[2]))
            opt, ps = Optimisers.update(opt, ps, back((1f0, nothing))[1])
        end
        v = val_loss(ps)
        imp = (v .< best_val) .& .!stopped
        if any(imp)
            mk = reshape(Float32.(imp), 1, 1, M) |> device
            best = map((b, p) -> b .* (1 .- mk) .+ p .* mk, best, ps)
            best_val[imp] .= v[imp]; best_ep[imp] .= ep
        end
        stopped .|= (ep .- best_ep) .> patience
        verbose && ep % 100 == 0 && println("epoch $ep: $(count(.!stopped)) of $M still training, median best val ",
                                            round(median(best_val), sigdigits = 4)); flush(stdout)
        all(stopped) && break
    end
    best = map(Array, best)
    _, sc = Lux.setup(Xoshiro(seed), cell); _, sh = Lux.setup(Xoshiro(seed), head)
    [(; cell, head, ps = (; c = (; weight_ih = best.Wih[:, :, m], weight_hh = best.Whh[:, :, m], bias_ih = best.b[:, 1, m],
                                 bias_hh = zeros(Float32, 4H)),
                            h = (; weight = best.Wo[:, :, m], bias = best.bo[:, 1, m])),
       sc, sh, d, H, best_epoch = best_ep[m], val = best_val[m], kind = :winner) for m in 1:M]
end
