using Random, Statistics, CairoMakie, MakieShorthands, Flux


function vec_of_vec_to_matrix(v::Vector{Vector{T}}) where {T}
    width=length(v)
    height=length(v[1])
    m=zeros(T,height, width)
    for i in eachindex(v)
        m[:,i].=v[i]
    end
    return m
end

function variation(vec::Vector)
    findmax(abs.(vec[2:end]-vec[1:end-1]))[1]
end

function histogram_forward(model::Chain, layers, X; bins=100)
    fig=Figure()
    ax=Axis(fig[1,1])
    for layer in layers
        hist!(ax,(model[1:layer](X))[:], label="after layer: $layer",bins=bins, normalization=:probability)
    end
    Legend(fig[1,2],ax)
    return fig
end

function histogram_forward(model::Chain, layers, X, neurons; bins=100)
    fig=Figure()
    ax=Axis(fig[1,1])
    for layer in layers
        for nr in neurons
            hist!(ax,(model[1:layer](X))[nr,:], label="after layer: $layer, at neuron: $nr",bins=bins, normalization=:probability)
        end
    end
    Legend(fig[1,2],ax)
    return fig
end

