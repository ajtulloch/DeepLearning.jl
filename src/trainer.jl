immutable Trainer
    net::Net
    sgdUpdaters::Vector{Vector{SGDUpdater}}

    function Trainer(net::Net, learningRate::Float64)
        buildUpdater(state) = MomentumUpdater(learningRate, 0.0, 0.0, 0.0, zeros(length(state.vol)))
        new(net, map(layer -> [buildUpdater(state) for state in gradientStates(layer)],  net.layers))
    end
end

train(t::Trainer, inputActivation::Vol, class::Int) = train(t.net, inputActivation, class)
predict(t::Trainer, inputActivation::Vol) = predict(t.net, inputActivation)

function updateGradients(t::Trainer, batchSize::Int)
    for (updaters, layer) in zip(t.sgdUpdaters, t.net.layers)
        priorW = copy(gradientStates(layer))
        println("Prior state: ", priorW)
        for (updater, state) in zip(updaters, gradientStates(layer))
            updateGradients(updater, state, batchSize)
        end
        postW = copy(gradientStates(layer))
        println("Post state: ", postW)
        @assert length(priorW) == 0 || priorW != postW
    end
end
