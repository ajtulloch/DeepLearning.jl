type DropoutLayer <: Layer
    inputActivation::Vol
    outputActivation::Vol
    keepProbability::Float64
    keptWeights::BitArray{3}

    function DropoutLayer(keepProbability::Float64, dimensions)
        new(Vol(dimensions), Vol(dimensions), keepProbability, BitArray(dimensions))
    end
end

function forward(self::DropoutLayer, inputActivation::Vol)
    self.inputActivation = inputActivation
    self.keptWeights = rand(size(self.inputActivation)) <. self.keepProbability
    self.outputActivation = self.inputActivation .* self.keptWeights
    return self.outputActivation
end

function backward(self::DropoutLayer)
    fill!(self.inputActivation.dw, 0.0)
    self.inputActivation.dw = self.outputActivation.dw .* self.keptWeights
end

gradientStates(self::DropoutLayer) = []
