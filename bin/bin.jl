
########################################### LAYER ABSTRACT TYPE ##############################

abstract type Layer{T} end

########################################### LINEAR LAYER #####################################

struct Linear{T<:Real} <: Layer{T} 
    weight::Matrix{T}
    bias::Vector{T}
end

function Linear{T}(in_to_out::Pair{Int,Int}; std=1/sqrt(in_to_out[1])) where {T<:Real}
    Linear{T}(std*randn(T,in_to_out[2],in_to_out[1]),zeros(T,in_to_out[2]))
end

function Linear(in_to_out::Pair{Int,Int})
    Linear{Float32}(in_to_out)
end

function (l::Linear{T})(v::Union{Vector{T},BitVector}) where {T<:Real}
    l.weight*v+l.bias
end

function (l::Linear{T})(v::Union{Matrix{T},BitMatrix}) where {T<:Real}
    l.weight*v.+l.bias
end

function (l::Linear{T})(v::Union{Array{T},BitArray}) where {T<:Real}
    sz=size(v)
    v=reshape(v,sz[1],:)
    reshape(l.weight*v.+l.bias,sz)
end

########################################### COMPOSITION #####################################

struct Composition{T<:Real} <: Layer{T}
    layers::Vector{Layer{T}} 

    Composition{T}(layers::Vector{Layer{T}}) where {T<:Real} =new{T}(layers)
    Composition{T}(layers::Layer{T}...) where {T<:Real}=new{T}(collect(layers)) 
end

Composition(layers::Vector{Layer{T}}) where {T<:Real} =Composition{T}(layers)
Composition(layers::Layer{T}...) where {T<:Real}=Composition{T}(layers...) 

function (cmp::Composition{T})(v::Array{T}) where {T<: Real}
    for l in cmp.layers
        v=l(v)
    end
    return v
end

####################################### OneHotEncoder ################################

struct OneHotEncoder{T}
    encoder::Dict{T,BitVector}

    function OneHotEncoder(vec::Vector{T}) where {T}
        encoder=Dict{T,BitVector}()
        dim=length(vec)
        for (index,element) in enumerate(vec)
            v=zeros(Bool,dim)
            v[index]=true
            encoder[element]=v
        end
        new{T}(encoder)
    end

end

function OneHotEncoder(range::AbstractRange)
    OneHotEncoder(collect(range))
end

function (enc::OneHotEncoder{T})(obj::T) where {T}
    enc.encoder[obj]
end

##########################
# section: MISCELLANEOUS #
##########################

struct i32 end
import Base:*
(*)(n, ::Type{i32}) = Int32(n)
