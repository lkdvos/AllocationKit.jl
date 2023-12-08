using Bumper

const BumperBackend = TensorOperations.Backend{:bumper}

function TensorOperations.tensoralloc(ttype::Type{Array{T,N}}, structure, istemp, ::BumperBackend) where {T,N}
    if istemp
        buffer = Bumper.default_buffer()
        nbytes = prod(structure) * sizeof(T)
        ptr = Bumper.alloc_ptr!(buffer, nbytes)
        return unsafe_wrap(Array, convert(Ptr{T}, ptr), structure)
    else
        return tensoralloc(ttype, structure, istemp)
    end
end

TensorOperations.tensorfree!(::Array{T,N}, ::BumperBackend) where {T,N} = nothing

# The approach below does not work, because freeing is not done in reverse order of allocation! For now, you have to manually insert @no_escape in front of the @tensor macro.

# function TensorOperations.tensorfree!(t::AbstractArray, ::BumperBackend)
#     buffer = Bumper.default_buffer(AllocBuffer)
#     @info "Starting deallocation ($(size(t))): $(buffer.offset)"
#     buffer.offset = pointer(t) - pointer(buffer.buf)
#     @info "Done deallocation: $(buffer.offset)"
#     return nothing
# end

export @no_escape
