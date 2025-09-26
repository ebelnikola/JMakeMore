# sections:                                                     important:
# section 0: PACKAGES
# section 1: LOADING OF THE DATASET                             words::Vector{String} - the dataset
# section 2: LOOK UP TABLES BETWEEN CHARACTERS AND INDICES      chars::Vector{Char}, char_to_index::Dict(Char => Int64), index_to_char::Dict(Int64 => Char)
# section 3: BIGRAM LANGUAGE MODEL                               
#   subsection 3.1: MODEL AND SAMPLING FUNCTIONS                the_bigram_counts_matrix_plot::Function, bigram_name_sample::Function, uniform_name_sample::Function
#   subsection 3.2: PLAYGROUND                                  some name examples are printed here  
# section 4: NEGATIVE LOG LIKELYHOOD                                
#   subsection 4.1: METHODS                                     neg_log_likelyhood::Function - two methods
#   subsection 4.2: PLAYGROUND                                  negative logarithmic likelyhood for the bigram model is printed here
# section 5: NETWORK APPROACH              
#   subsection 5.1: TRAINING SET                                X,Y - the dataset 
#   subsection 5.2: MODEL                                       model - the model with one dense layer 27 => 27 and a softmax layer
#   subsection 5.3: TRAINING                                    
#      subsection 5.3.1: TRAINING FUNCTION                      train!::Function
#      subsection 5.3.2: TRAINING PLAYGROUND                    training happends here
#      subsection 5.3.3: TRAINING RESULT                        model_bigram_probabilities::Matrix
#   subsection 5.4: PLAYGROUND                                  some name examples are printed here


#######################
# section 0: PACKAGES #
#######################

using Distributions, CairoMakie
using Flux, OneHotArrays, CUDA, Statistics, LinearAlgebra

#####################################
# section 1: LOADING OF THE DATASET #
#####################################

const global words=readlines(open("names.txt","r"));


############################################################
# section 2: LOOK UP TABLES BETWEEN CHARACTERS AND INDICES #
############################################################

const global chars=Set(join(words)) |> collect |> sort;
const global char_to_index=Dict(enumerate(chars) .|> x->(x[2],x[1]+1));
char_to_index['.']=1; 
const global index_to_char=Dict(collect(char_to_index) .|> x-> (x[2],x[1])) 


####################################
# section 3: BIGRAM LANGUAGE MODEL #
####################################

################################################
# subsection 3.1: MODEL AND SAMPLING FUNCTIONS #
################################################



const global bigram_counts=zeros(Int32,27,27);
for w in words
    characters=vcat(['.'], collect(w), ['.'])
    for (ch1,ch2) in zip(characters,characters[2:end])
        index=(char_to_index[ch1],char_to_index[ch2])
        bigram_counts[index...]+=1
    end
end

function the_bigram_counts_matrix_plot()
    fig=Figure(;size=(1200,1200));
    ax=Axis(fig[1,1], 
            yreversed=true,
            xticksvisible=false,
            xticklabelsvisible=false,
            yticksvisible=false,
            yticklabelsvisible=false);

    heatmap!(ax,bigram_counts',colormap=:blues);

    for i=1:27
        for j=1:27
            label=join([index_to_char[i],index_to_char[j]])
            text!((j,i); text=label, align=(:center,:bottom),color=:gray)
            text!((j,i); text=string(bigram_counts[i,j]), align=(:center,:top),color=:gray)
        end
    end
    fig
end


const global bigram_probabilities = mapslices(x->x/sum(x), Float32.(bigram_counts.+1);dims=2);


function bigram_name_sample(bigram_probabilities=bigram_probabilities)
    out=Char[]
    index=1
    while true
        p=bigram_probabilities[index,:]
        distr=Distributions.Categorical(p);
        index=rand(distr)
        push!(out,index_to_char[index])
        if index==1
            break
        end
    end
    return join(out)
end

function uniform_name_sample()
    out=Char[]
    index=1
    while true
        p=ones(Float32,27)
        p/=sum(p);
        distr=Distributions.Categorical(p);
        index=rand(distr)
        push!(out,index_to_char[index])
        if index==1
            break
        end
    end
    return join(out)
end

##############################
# subsection 3.2: PLAYGROUND #        
##############################

for _=1:10
    println(bigram_name_sample())
end

for _=1:10
    println(uniform_name_sample())
end



######################################
# section 4: NEGATIVE LOG LIKELYHOOD #                           
######################################

###########################
# subsection 4.1: METHODS #
###########################

function neg_log_likelyhood(probs::Matrix, words::Vector{String})
    log_likelyhood=0.0f0
    n=0
    for w in words
        characters=vcat(['.'], collect(w), ['.'])
        for bigram in zip(characters,characters[2:end])
            index1,index2=char_to_index[bigram[1]],char_to_index[bigram[2]]
            prob=probs[index1,index2]
            logprob=log(prob)
            log_likelyhood+=logprob;
            n+=1
        end
    end
    return -log_likelyhood/n    
end

function neg_log_likelyhood(probs::Matrix, Y::OneHotMatrix)
    -sum(log.(probs) .* Y)/size(probs,2)
end

##############################
# subsection 4.2: PLAYGROUND #
##############################

println("negative log likelyhood for the bigram model: ", neg_log_likelyhood(bigram_probabilities,words))

###############################
# section 5: NETWORK APPROACH #                           
###############################

################################
# subsection 5.1: TRAINING SET #                           
################################
X_tmp,Y_tmp=UInt32[],UInt32[]

for w in words
    characters=vcat(['.'], collect(w), ['.'])
    for bigram in zip(characters,characters[2:end])
        index1,index2=char_to_index[bigram[1]],char_to_index[bigram[2]]
        push!(X_tmp,index1)
        push!(Y_tmp,index2)
    end
end

X=onehotbatch(X_tmp,1:27);
Y=onehotbatch(Y_tmp,1:27);


#########################
# subsection 5.2: MODEL #                           
#########################

model=Chain(Dense(27 => 27), softmax);

############################
# subsection 5.3: TRAINING #                           
############################

#######################################
# subsection 5.3.1: TRAINING FUNCTION #                           
#######################################


function train!(X,Y,model,optim)
    loss, grads = Flux.withgradient(model) do m
        probs=m(X)  
        loss=neg_log_likelyhood(probs,Y)
        w=Flux.params(model)[1]
        loss+0.01*mean(w.^2)
    end
    Flux.update!(optim,model,grads...)
    return loss
end


#########################################
# subsection 5.3.2: TRAINING PLAYGROUND #                           
#########################################

optim=Flux.setup(Flux.Descent(2),model)
loss_history=Float32[]
for epoch in 1:1000
    loss=train!(X,Y,model,optim)
    println(loss)
    push!(loss_history,loss)
end

lines(loss_history)

#####################################
# subsection 5.3.3: TRAINING RESULT #                           
#####################################

model_bigram_probabilities=(onehotbatch(collect(1:27),1:27) |> model)';

using Serialization

serialize("bigrams.model", model_bigram_probabilities)

##############################
# subsection 5.4: PLAYGROUND #                           
##############################

for _=1:10
    println(bigram_name_sample(model_bigram_probabilities))
end
