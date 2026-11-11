# Heterogeneity of a pool of recordings, model-free: how different are the units a group model must fit at once,
# measured against how different two stretches of one unit's own data are?
#
# Every unit (a trial, or an individual's training trials) contributes two disjoint stretches A and B of about five
# cycles, in the pool's own units (pooled z-scoring, as the group fits see them). Each stretch is delay-embedded with
# q delays spanning one (median) cycle, and the pool's delay vectors are projected onto their leading principal
# components (95% of variance): the "Hankel PCA" space of the group fit. The distance between two stretches is the
# symmetric mean distance from each state of one to the nearest state of the other (the ε-tube distance of r2_tube.jl).
#
#   H = median over pairs u ≠ v of  D(A_u, B_v) / sqrt(D(A_u, B_u) · D(A_v, B_v))
#
# H ≈ 1: two units are as alike as two stretches of the same recording, so pooling only adds data. H ≫ 1: the group
# model has to represent distinct cycles (period, shape, amplitude) at once.

using LinearAlgebra, Statistics, Random

"Delay vectors (rows) of Y: q delays spanning about one cycle of P samples."
function delay_cloud(Y, P; q = 8)
    ℓ = max(1, round(Int, P / q)); n, d = size(Y); span = (q - 1) * ℓ
    X = zeros(n - span, q * d)
    for t in 1:n-span, j in 0:q-1
        X[t, j*d+1:(j+1)*d] = Y[t+span-j*ℓ, :]
    end
    X
end

"Symmetric mean nearest-neighbour distance between the rows of A and of B."
function nn_dist(A, B)
    D = sum(abs2, A, dims = 2) .+ sum(abs2, B, dims = 2)' .- 2 .* (A * B')
    D .= sqrt.(max.(D, 0))
    0.5 * (mean(minimum(D, dims = 2)) + mean(minimum(D, dims = 1)))
end

"""
    heterogeneity(halves; P, q = 8, var = 0.95, maxpairs = 4000)

`halves[u] = (A, B)`: two disjoint stretches of unit u's data (samples × channels, pooled units), each about five
cycles plus one cycle for the embedding. Returns `H` (median ratio over unit pairs), all ratios, the within-unit
distances `w`, the number of principal components `k`.
"""
function heterogeneity(halves; P, q = 8, var = 0.95, maxpairs = 4000, rng = MersenneTwister(1))
    clouds = [(delay_cloud(A, P; q), delay_cloud(B, P; q)) for (A, B) in halves]
    allX = reduce(vcat, [vcat(A, B) for (A, B) in clouds])
    μ = mean(allX, dims = 1); F = svd(allX .- μ)
    k = findfirst(>=(var), cumsum(F.S .^ 2) ./ sum(F.S .^ 2)); V = F.V[:, 1:k]
    proj = [((A .- μ) * V, (B .- μ) * V) for (A, B) in clouds]
    N = length(proj)
    w = [nn_dist(A, B) for (A, B) in proj]
    N == 1 && return (; H = 1.0, ratios = Float64[], w, k)
    pairs = [(u, v) for u in 1:N for v in 1:N if u != v]
    length(pairs) > maxpairs && (pairs = pairs[randperm(rng, length(pairs))[1:maxpairs]])
    r = zeros(length(pairs))
    Threads.@threads for i in eachindex(pairs)
        u, v = pairs[i]; r[i] = nn_dist(proj[u][1], proj[v][2]) / sqrt(w[u] * w[v])
    end
    (; H = median(r), ratios = r, w, k)
end
