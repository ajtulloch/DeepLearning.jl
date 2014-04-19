immutable Net
    layers::Vector{Layer}

    function Net(layers::Vector{Layer})
        for i in 1:length(layers) - 1
            @assert inputSize(layers[i+1]) == outputSize(layers[i])
        end
        new(layers)
    end
end

gradientStates(net::Net) = map(gradientStates, net.layers)

function forward(net::Net, inputActivation::Vol)
    act = forward(net.layers[1], inputActivation)
    for i in 2:length(net.layers)
        act = forward(net.layers[i], act)
    end
    invariants(net)
    return act
end

function backward(net::Net, class::Int)
    invariants(net)
    numLayers = length(net.layers)
    loss = backward(net.layers[numLayers], class)
    for i = numLayers-1:-1:1
        backward(net.layers[i])
    end
    invariants(net)
    return loss
end

function invariants(net::Net)
    for i in 1:length(net.layers) - 1
        @assert is(net.layers[i].outputActivation, net.layers[i+1].inputActivation)
        @assert is(net.layers[i].outputActivation.w, net.layers[i+1].inputActivation.w)
        @assert is(net.layers[i].outputActivation.dw, net.layers[i+1].inputActivation.dw)
        if i + 2 <= length(net.layers)
            @assert !is(net.layers[i].outputActivation, net.layers[i+2].inputActivation)
            @assert !is(net.layers[i].outputActivation.w, net.layers[i+2].inputActivation.w)
            @assert !is(net.layers[i].outputActivation.dw, net.layers[i+2].inputActivation.dw)
        end
    end
end

function train(net::Net, inputActivation::Vol, class::Int)
    forward(net, inputActivation)
    result = backward(net, class)
    invariants(net)
    return result
end

function predict(net::Net, inputActivation::Vol)
    forward(net, inputActivation)
    _, index = findmax(last(net.layers).outputActivation.w)
    invariants(net)
    return index
end
