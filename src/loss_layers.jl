type SoftMaxLayer <: Layer
    inputActivation::Vol
    outputActivation::Vol

    function SoftMaxLayer(numOutputs::Int)
        dimensions = (1, 1, numOutputs)
        new(Vol(dimensions, 0.0), Vol(dimensions, 0.0))
    end
end

function forward(self::SoftMaxLayer, inputActivation::Vol)
    @assert inputSize(self) == outputSize(self)
    self.inputActivation = inputActivation
    exponentials = exp(self.inputActivation.w .- maximum(self.inputActivation.w))
    exponentials /= sum(exponentials)

    self.outputActivation.w = exponentials
    @assert abs(sum(self.outputActivation.w) - 1.0) < 1E-5
    return self.outputActivation
end

function backward(self::SoftMaxLayer, class::Int)
    @assert inputSize(self) == outputSize(self)
    fill!(self.inputActivation.dw, 0.0)
    for i in 1:length(self.outputActivation)
        indicator = ((i == class) ? 1.0 : 0.0)
        mul = -(indicator - self.outputActivation.w[1, 1, i])
        self.inputActivation.dw[1, 1, i] = mul
    end
    @assert self.inputActivation.dw |> sum |> abs < 1E-5
    return -log(self.outputActivation.w[1, 1, class])
end

gradientStates(self::SoftMaxLayer) = []
