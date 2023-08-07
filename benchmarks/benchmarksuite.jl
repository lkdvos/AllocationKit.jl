using BenchmarkTools
using TensorOperations
using AllocationKit

suite = BenchmarkGroup()

## parameters
d = 2 # physical dimension
V = 3 # mpo virtual dimension
Ds = [8, 16, 64, 128] # bond dimension
T = ComplexF64
repeats = 30


suite["default"] = BenchmarkGroup()
function contract_default(A, GL, O, GR)
    @tensor A′[-1, -2, -3] := A[1, 3, 4] * GL[-1, 2, 1] * O[2, -2, 3, 5] * GR[4, 5, -3]
    return A′
end

suite["malloc"] = BenchmarkGroup()
function contract_malloc(A, GL, O, GR)
    @tensor allocator=malloc A′[-1, -2, -3] := A[1, 3, 4] * GL[-1, 2, 1] * O[2, -2, 3, 5] * GR[4, 5, -3]
    return A′
end

suite["safemalloc"] = BenchmarkGroup()
function contract_safemalloc(A, GL, O, GR)
    @tensor allocator = safemalloc A′[-1, -2, -3] := A[1, 3, 4] * GL[-1, 2, 1] * O[2, -2, 3, 5] * GR[4, 5, -3]
    return A′
end

O = randn(T, V, d, d, V)
for D in Ds
    GL = randn(T, D, V, D)
    GR = randn(T, D, V, D)
    A = randn(T, D, d, D)
    suite["default"][D] = @benchmarkable contract_default($A, $GL, $O, $GR) evals = 30 samples = 500 seconds = Inf
    suite["malloc"][D] = @benchmarkable contract_malloc($A, $GL, $O, $GR) evals = 30 samples = 500 seconds = Inf
    suite["safemalloc"][D] = @benchmarkable contract_safemalloc($A, $GL, $O, $GR) evals = 30 samples = 500 seconds = Inf
end

result = run(suite; verbose=true)
