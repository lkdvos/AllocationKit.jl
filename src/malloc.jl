const MallocBackend = TensorOperations.Backend{:malloc}

function TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, istemp, ::MallocBackend) where {T,N}
    if istemp
        @assert isbitstype(T)
        ptr = Base.Libc.malloc(prod(structure) * sizeof(T))
        return unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
    else
        return tensoralloc(Array{T,N}, structure, istemp)
    end
end

function TensorOperations.tensorfree!(t::Array, ::MallocBackend)
    Base.Libc.free(pointer(t))
    return nothing
end

const SafeMallocBackend = TensorOperations.Backend{:safemalloc}

function TensorOperations.tensoralloc(::Type{Array{T,N}}, structure, istemp, ::SafeMallocBackend) where {T,N}
    if istemp
        @assert isbitstype(T)
        ptr = Base.Libc.malloc(prod(structure) * sizeof(T))
        A = unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
        finalizer(Base.Fix2(TensorOperations.tensorfree!, MallocBackend()), A)
    else
        return tensoralloc(Array{T,N}, structure, istemp)
    end
end

function TensorOperations.tensorfree!(t::Array, ::SafeMallocBackend)
    finalize(t)
    return nothing
end
