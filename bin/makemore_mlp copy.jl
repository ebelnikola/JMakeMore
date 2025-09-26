#######################
# section 0: PACKAGES #
#######################
include("Tools.jl")


#####################################
# section 1: LOADING THE DATASET #
#####################################

file=open("names.txt")
const words=readlines(file)
close(file)
file=nothing

const chars=text_to_unique_chars(words;additional_chars=".")
const char_to_index=Dict(enumerate(chars) .|> x->(x[2],x[1]));
const index_to_char=Dict(enumerate(chars) .|> x-> (x[1],x[2])); 



const context_window_size=3 

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
middle_layer_size=150  #
#PARS##PARS##PARS##PARS#


emb=Flux.Embedding(27=>embedding_dimension; init=randn32)
layer1=Dense(context_window_size*embedding_dimension=>middle_layer_size,tanh)
layer2=Dense(middle_layer_size=>27)

model=Flux.Chain(emb,
                 x->(reshape(x,context_window_size*embedding_dimension,:)),
                 layer1,
                 layer2,
                 softmax
                )

#######################################
# subsection 4.3: TRAINING PLAYGROUND #
#######################################

batchsize=80
η=0.3

optim=Flux.setup(Descent(η),model)

loader=Flux.DataLoader((Xtr,Ytr);batchsize=batchsize);

loss_history_in_batches=Float32[]
loss_history=Float32[]
f=1
for epoch=1:100
    Flux.adjust!(optim,η/f)
    for _=1:5
        for (x,y) in loader
            push!(loss_history_in_batches,train_model!(x,y,model,optim))
        end
    end
    push!(loss_history,Flux.crossentropy(model(Xtr),Ytr))
    if length(loss_history)>=2
        println(abs(loss_history[end]-loss_history[end-1]))
    end
    if variation(loss_history[end-3:end]) <5e-5
        if η/(f*1.5) > 1e-6
            f=f*1.5
            println("new factor: ", f)
        end
    end
end

lines(loss_history_in_batches)
lines(loss_history)


#####################################
# section 5: NAME SAMPLING FUNCTION #
#####################################

############################
# subsection 5.1: FUNCTION #
############################

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