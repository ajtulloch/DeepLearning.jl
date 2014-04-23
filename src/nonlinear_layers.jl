type SigmoidLayer <: Layer
    inputActivation::Vol
    outputActivation::Vol

    function SigmoidLayer(dimensions)
        new(Vol(dimensions, 0.0), Vol(dimensions, 0.0))
    end
end

function forward(self::SigmoidLayer, inputActivation::Vol)
    @assert inputSize(self) == outputSize(self)
    self.inputActivation = inputActivation
    self.outputActivation.w = logistic(inputActivation.w)
    return self.outputActivation
end

function backward(self::SigmoidLayer)
    @assert inputSize(self) == outputSize(self)
    self.inputActivation.dw = self.outputActivation.w .* (1.0 .- self.outputActivation.w) .* self.outputActivation.dw
end

gradientStates(self::SigmoidLayer) = []
weights(self::SigmoidLayer) = []

type RectifiedLinearLayer
    inputActivation::Vol
    outputActivation::Vol

    function RectifiedLinearLayer(dimensions)
        new(Vol(dimensions, 0.0), Vol(dimensions, 0.0))
    end
end

function forward(self::RectifiedLinearLayer, inputActivation::Vol)
    @assert size(inputActivation) == size(self.inputActivation)
    self.inputActivation = inputActivation
    self.outputActivation = max(self.inputActivation, 0)
    return self.outputActivation
end

function backward(self::RectifiedLinearLayer)
    @assert inputSize(self) == outputSize(self)
    fill!(self.inputActivation.dw, 0.0)
    self.inputActivation.dw = (0 .< self.outputActivation.w) .* self.outputActivation.dw
end

gradientStates(self::RectifiedLinearLayer) = []
