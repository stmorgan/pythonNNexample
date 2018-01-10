# ===============================================================================================
# python(with numpy)                    13.909520864486694
# julia                               ÷ 3.449227
#                                    ______________________
#                        julia runs:    4.032648725 times faster
# ===============================================================================================

function nonlin(x, deriv=false)
    deriv==true && return (x*(1-x))
    return 1/(1+exp(-x))
end

# Wraped in a function so it runs faster... keep code out of global scope.
function train()
    
    x = [ 0 0 1 ; 0 1 1 ; 1 0 1 ; 1 1 1  ]  # input
    y = [0; 1; 1; 0]                        # output

    rng = MersenneTwister(1234);
    syn0 = randn(rng, (3,4)) # 3x4 matrix of normalised weights.
    syn1 = randn(rng, (4,1)) # 4x1 matrix of normalised weights.

    for j in 1:60000

        # Calculate forward through the network.
            l0 = x
            l1 = nonlin.(l0 * syn0)
            l2 = nonlin.(l1 * syn1) 

        # Back propagation of errors using the chain rule.
            l2_error = y - l2            
            (j % 10000) == 0  && println("Error: ", mean(abs.(l2_error)) ) # print error every 10000 steps.

            l2_delta = l2_error .* nonlin.(l2, true) 
            l1_error = l2_delta .* syn1' 
            l1_delta = l1_error .* nonlin.(l1,true) 

        # Update weights (no learning rate term)
            syn1 += l1' * l2_delta 
            syn0 += l0' * l1_delta

            println("Output after training")
            println(l2)
    end
end
