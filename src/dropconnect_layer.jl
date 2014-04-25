type DropConnectLayer <: Layer
    filters::Vector{Vol}
    biases::Vol
    inputActivation::Vol
    outputActivation::Vol

    keepProbability::Float64
    filterMasks::Vector{BitArray{3}}
    biasMask::BitArray{3}

    function DropConnectLayer(numOutputs::Int, bias::Float64, keepProbability::Float64, dimensions)
        filters = [Vol(dimensions) for _ in 1:numOutputs]
        masks = [BitArray(dimensions) for _ in 1:numOutputs]
        biases = Vol((1, 1, numOutputs), bias)
        new(filters, biases, Vol(dimensions, 0.0), Vol((1, 1, numOutputs), 0.0), keepProbability, masks)
    end
end

function forward(self::DropConnectLayer, inputActivation::Vol)
    @assert size(self.inputActivation) == size(inputActivation)
    self.inputActivation = inputActivation
    self.biasMask = rand(size(biases)) .< self.keepProbability

    for i in 1:length(self.filters)
        filter = self.filters[i]
        self.filterMasks[i] = rand(size(filter)) .< self.keepProbability
        @assert size(self.inputActivation) == size(filter)

        self.outputActivation.w[1, 1, i] =
            sum(filter.w .* inputActivation.w .* self.filterMasks[i]) +
            self.biases.w[1, 1, i] * self.biasMask[1, 1, i]
    end
    return self.outputActivation
end

function backward(self::DropConnectLayer)
    fill!(self.inputActivation.dw, 0.0)
    for i in 1:length(self.filters)
        filter = self.filters[i]
        chainGrad = self.outputActivation.dw[1, 1, i]

        self.inputActivation.dw += filter.w .* self.filterMasks[i] * chainGrad
        filter.dw += self.inputActivation.w .* self.filterMasks[i] * chainGrad
        self.biases.dw[1, 1, i] += chainGrad * self.biasMask[1, 1, i]
    end
end

function gradientStates(self::DropConnectLayer)
    grads = [GradientState(filter, 1.0, 1.0) for filter in self.filters]
    append!(grads, [GradientState(self.biases, 0.0, 0.0)])
    return grads
end
