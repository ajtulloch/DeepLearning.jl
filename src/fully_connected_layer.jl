type FullyConnectedLayer <: Layer
    filters::Vector{Vol}
    biases::Vol
    inputActivation::Vol
    outputActivation::Vol

    function FullyConnectedLayer(numOutputs::Int, bias::Float64, dimensions)
        filters = [Vol(dimensions) for _ in 1:numOutputs]
        biases = Vol((1, 1, numOutputs), bias)
        new(filters, biases, Vol(dimensions, 0.0), Vol((1, 1, numOutputs), 0.0))
    end
end

function forward(self::FullyConnectedLayer, inputActivation::Vol)
    @assert size(self.inputActivation) == size(inputActivation)
    self.inputActivation = inputActivation
    for i in 1:length(self.filters)
        filter = self.filters[i]
        @assert size(self.inputActivation) == size(filter)
        self.outputActivation.w[1, 1, i] = sum(filter.w .* inputActivation.w) + self.biases.w[1, 1, i]
    end
    return self.outputActivation
end

function backward(self::FullyConnectedLayer)
    fill!(self.inputActivation.dw, 0.0)
    for i in 1:length(self.filters)
        filter = self.filters[i]
        chainGrad = self.outputActivation.dw[1, 1, i]

        self.inputActivation.dw += filter.w * chainGrad
        filter.dw += self.inputActivation.w * chainGrad
        self.biases.dw[1, 1, i] += chainGrad
        @printf("Filter %s, chainGrad: %s, inputActivation.w: %s\n", i, chainGrad, flatten(self.inputActivation.w))
    end
end

function gradientStates(self::FullyConnectedLayer)
    grads = [GradientState(filter, 1.0, 1.0) for filter in self.filters]
    append!(grads, [GradientState(self.biases, 0.0, 0.0)])
    return grads
end
