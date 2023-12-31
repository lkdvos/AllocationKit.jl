# AllocationKit

[![Build Status](https://github.com/lkdvos/AllocationKit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lkdvos/AllocationKit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

This is an in-progress package to provide a set of tools for the allocation of intermediate
tensors within [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl).

## Rationale

The problem of allocating intermediate tensors is a difficult one. The optimal allocation
strategy seems to be dependent on the size of the tensors, the number of operations, the
number of threads, the memory bandwidth, and the cache size. This package aims to provide a
set of tools to experiment with, in order to hopefully provide an efficient allocation
strategy for a given problem.

The expressions that are generated by TensorOperations allocate new tensors in a variety of
ways. The first and foremost is the allocation of output tensors, i.e. objects that are used
further down the line in the program, and should thus in almost all cases be just allocated
like any julia variable, and handled by the garbage collector. If a user desires to change
this behaviour, this is probably something that should be handled externally, and not by
TensorOperations, thus this type of allocation is not focussed on by this package. The code
pattern that achieves this would just use in-place operations for output tensors, and would
look something like this:

```julia
C = my_allocator()
@tensor C[a,b,c] = A[a,b,d] * B[d,c]
```

The second type of allocation is the intermediate tensors that arise from the pairwise
handling of larger tensor networks. I will refer to these as
**external intermediate tensors**. In practise, these are very hard to remove, and in
general settings, it seems that these should and could also be handled by the native garbage
collection mechanism. However, often the same network is evaluated many times with the same
size of input tensors, which arises for example when building up a Krylov subspace for a
linear solver or an eigenvalue solver. In these cases, it is possible to reuse the same
intermediate tensors, and thus hoist the allocation out of the loop. For this case, we aim
to provide some solutions that differ in the amount of additional work/code a user is
willing to supply, as well as the performance gains that can be achieved.

Finally, there is the case of the intermediate tensors that arise from the pairwise
contraction of tensors when using the common pattern of TTGT
(transpose-transpose-gemm-transpose), where the input tensors are first permuted to a form
where the contraction can be dispatched to a BLAS routine. I will refer to these as
**internal intermediate tensors**. Note that this is implementation-dependent, and various
alternatives exist which already aim to handle this problem. For example, [tblis]() does not
require these additional intermediate tensors, and other implementations exist
(https://arxiv.org/abs/1607.00145), ... Additionally, these intermediate tensors are
shorter-lived, and thus may benefit from a different strategy than the intermediate tensors
that arise from the pairwise contraction of tensors. We will also provide some examples of
these strategies, but this should be considered separately.

## Benchmarking

As the problem is inherently dependent on the network, the size of the tensors, the
hardware, and the surrounding code (e.g. the number of threads, the available memory, ...),
I currently only provide a limited set of benchmarks, but feel free to open an issue and
I'll gladly consider adding it to the benchmark suite.

Because of the aforementioned reasons, the benchmarks will be run without envoking the
garbage collector inbetween different samples, and we will focus on the repeated evaluation
of the same tensor network. This should hopefully replicate a more realistic setting than
simply isolating a single contraction.

Please also note that my experience with benchmarking allocations is limited, so any
suggestions or notes are more than welcome.

## Solutions

### Manual allocation and deallocation

One of the simplest solutions to try is to manually allocate and deallocate the intermediate
tensors. While this will not reduce the amount of allocations, it will bypass the garbage
collector, and at the very least alleviate these costs. Additionally, it might reduce the
total memory usage at a given time, as memory will be freed as early as possible.

Nevertheless, this process might be error-prone, as interruption of the program might lead
to memory-leaks, while explicit finalizers encur some performance cost.