const MallocBackend = TensorOperations.Backend{:malloc}

function TensorOperations.tensoralloc(ttype::Type{<:AbstractArray{T}}, structure, istemp, ::MallocBackend) where {T}
    if istemp
        ptr = Base.Libc.malloc(prod(structure) * sizeof(T))
        return unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
    else
        return tensoralloc(ttype, structure, istemp)
    end
end

function TensorOperations.tensorfree!(t::AbstractArray, ::MallocBackend)
    Base.Libc.free(pointer(t))
    return nothing
end

const SafeMallocBackend = TensorOperations.Backend{:safemalloc}

function TensorOperations.tensoralloc(ttype::Type{<:AbstractArray{T}}, structure, istemp, ::SafeMallocBackend) where {T}
    if istemp
        ptr = Base.Libc.malloc(prod(structure) * sizeof(T))
        A = unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
        finalizer(A) do x
            Base.Libc.free(pointer(x))
        end
        return A
    else
        return tensoralloc(ttype, structure, istemp)
    end
end

function TensorOperations.tensorfree!(t::AbstractArray, ::SafeMallocBackend)
    finalize(t)
    return nothing
end