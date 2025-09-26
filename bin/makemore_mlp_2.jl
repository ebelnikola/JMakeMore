#######################
# section 0: PACKAGES #
#######################

using Flux, OneHotArrays, CUDA, Statistics, LinearAlgebra, CairoMakie, Random, Distributions, Serialization

include("Tools.jl")


#####################################
# section 1: LOADING OF THE DATASET #
#####################################

words=readlines(open("names.txt","r"));

############################################################
# section 2: LOOK UP TABLES BETWEEN CHARACTERS AND INDICES #
############################################################

const global chars=append!(['.'],Set(join(words)) |> collect |> sort);
const global char_to_index=Dict(enumerate(chars) .|> x->(x[2],x[1]));
const global index_to_char=Dict(collect(char_to_index) .|> x-> (x[2],x[1])) 

######################################
# section 3: BUILDING UP THE DATASET #
######################################

#####################################
# subsection 3.1: BUILDER FUNCTION  #
#####################################

function build_dataset(words; context_window_size=3)
    cyclic_perm=push!(collect(2:context_window_size),1)
    X=Vector{Int32}[]
    Y=Int32[]
    for w in words
        context=ones(Int32,context_window_size)
        for ch in w*"."
            index=char_to_index[ch]
            push!(X,context)
            push!(Y,index)
            context=context[cyclic_perm]
            context[end]=index
        end
    end
    X=vec_of_vec_to_matrix(X); 
    Y=onehotbatch(Y,1:27);
    return X,Y
end


###############################
# subsection 3.2: PLAYGROUND  #
###############################

#PARS##PARS##PARS##PARS#
 context_window_size=3 #
#PARS##PARS##PARS##PARS#

shuffle!(words);
n1=floor(0.8*length(words)) |> Int32;
n2=floor(0.9*length(words)) |> Int32;

Xtr,Ytr=build_dataset(words[1:n1]);
Xtst,Ytst=build_dataset(words[n1+1:n2]);
Xdev,Ydev=build_dataset(words[n2+1:end]);


#########################
# section 4: THE MODEL  #
#########################

######################################
# subsection 4.1: TRAINING FUNCTION  #
######################################

function train_model!(X,Y,model,optim)
    loss,grads=Flux.withgradient(model) do m
        probs=m(X)
        Flux.crossentropy(probs,Y)
    end
    Flux.update!(optim,model,grads...)
    return loss
end


########################################
# subsection 4.2: MODEL INITIALISATION #
########################################


#PASR##PARS##PARS##PARS#
embedding_dimension=10 #
middle_layer_size=200  #
#PARS##PARS##PARS##PARS#


emb=Flux.Embedding(27=>embedding_dimension; init=randn32)
layer1=Dense(context_window_size*embedding_dimension=>middle_layer_size)
layer2=BatchNorm(middle_layer_size; momentum=0.0001f0);
layer3=Dense(middle_layer_size=>27)


model=Flux.Chain(emb,
                 x->(reshape(x,context_window_size*embedding_dimension,:)),
                 layer1,
                 layer2,
                 tanh_fast,
                 layer3,
                 softmax
                )



#######################################
# subsection 4.3: TRAINING PLAYGROUND #
#######################################

batchsize=30
η=0.3
η_min=5e-5
loss_min_variation=5e-5

optim=Flux.setup(Descent(η),model)

loader=Flux.DataLoader((Xtr,Ytr);batchsize=batchsize);

loss_history_in_batches=Float32[]
loss_history=Float32[]
layer2_μ_1_history=Float32[]
layer2_σ²_1_history=Float32[]

f=1.0

for epoch=1:100
    for _=1:5
        for (x,y) in loader
            push!(loss_history_in_batches,train_model!(x,y,model,optim))
        end
    end
    push!(loss_history,Flux.crossentropy(model(Xtr),Ytr))
    print("loss: ", round(loss_history[end]; digits=5))
    if length(loss_history)>=2
        println( " loss chage:  ", abs(loss_history[end]-loss_history[end-1]))
    else
        print("\n")
    end
    
    if length(loss_history)>3
        if variation(loss_history[end-3:end])<loss_min_variation 
            if η/(f*sqrt(2))>η_min
                f*=sqrt(2);
                println("  ")
                println("f update, new f: $f")
                println("  ")
            end
            Flux.adjust!(optim,η/f)
        end
    end 
end

lines(loss_history)

Flux.crossentropy(model(Xdev),Ydev)

#####################################
# section 5: NAME SAMPLING FUNCTION #
#####################################

############################
# subsection 5.1: FUNCTION #
############################

Flux.testmode!(model)

function mpl_name_sample()
    out=Char[]
    slide=push!(collect(2:context_window_size),1)
    context=ones(Int32,context_window_size)
    while true
        p=reshape(model(context),:)
        distr=Distributions.Categorical(p);
        index=rand(distr)
        push!(out,index_to_char[index])
        if index==1
            break
        end
        context=context[slide]
        context[end]=index
    end
    return join(out)
end

##############################
# subsection 5.2: PLAYGROUND #
##############################

for _=1:20
    println(mpl_name_sample())
end



#=

mutable struct BatchNormLayer
    γ::Vector{Float32}
    β::Vector{Float32}
    train_mode::Bool
    p::Float32
    μ::Vector{Float32}
    σ²::Vector{Float32} 
    ϵ::Float32 
end

BatchNormLayer(size::Int64; p=0.1f0, ϵ=0.0001f0) = BatchNormLayer(ones32(size),zeros32(size),true,p,zeros32(size),zeros32(size),ϵ)

Flux.@layer BatchNormLayer trainable=(β,γ)

function (l::BatchNormLayer)(X)
    if l.train_mode
        X̂=mean(X;dims=2)
        σX²=var(X;dims=2)
        l.μ=l.p*l.μ+(1-l.p)*(X̂[:])
        l.σ²=l.p*l.σ²+(1-l.p)*(σX²[:])
    else
        X̂=l.μ
        σX²=l.σ²
    end
    X_normal = @. (X-X̂)/sqrt(σX²+l.ϵ)
    return @. l.γ*X_normal+l.β
end

function Base.show(io::IO, l::BatchNormLayer)
    print(io, "BatchNormLayer($(length(l.γ)))")
end
=#