# This runs twice as fast on my machine. Of course the objective was not really to compare speed.
# Hope this is usefull to someone ;)
# ===============================================================================================
# julia>                               finished in 10.75 seconds
#
# python >>> (with numpy)              finished in 20.24 seconds
# ===============================================================================================

function nonlin(x, deriv=false)
    if deriv==true
        return (x*(1-x))
    end
    return 1/(1+exp(-x))
end


# input
x = [ 0 0 1 ; 0 1 1 ; 1 0 1 ; 1 1 1  ]
# output
y = [0; 1; 1; 0]

syn0 = 2*rand(3,4)-1 # 3x4 matrix of weights. ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*rand(4,1)-1 # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.


for j in 1:60000

    # Calculate forward through the network.
        l0 = x
        l1 = nonlin.(l0 * syn0)
        l2 = nonlin.(l1 * syn1) 

    # Back propagation of errors using the chain rule.
        l2_error = y - l2 
        if(j % 10000) == 0   # Only print the error every 10000 steps.
            println("Error: ", mean(abs.(l2_error)) ) 
        end

        l2_delta = l2_error .* nonlin.(l2, true) 
        l1_error = l2_delta .* transpose(syn1) 
        l1_delta = l1_error .* nonlin.(l1,true) 

    # Update weights (no learning rate term)
        syn1 += transpose(l1) * l2_delta 
        syn0 += transpose(l0) * l1_delta

        println("Output after training")
        println(l2)
end
