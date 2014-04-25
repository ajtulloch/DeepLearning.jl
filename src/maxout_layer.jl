type MaxoutLayer <: Layer
    inputActivation::Vol
    outputActivation::Vol
    groupSize::Int
    argmax::Array{Int64, 3}

    function MaxoutLayer(groupSize::Int, dimensions)
        (inputX, inputY, inputDepth) = dimensions
        outputDepth = floor(inputDepth / groupSize)
        outputDimensions = (inputX, inputY, outputDepth)
        new(zeros(dimensions), zeros(outputDimensions), groupSize, zeros(Int64, outputDimensions))
    end
end

function forward(self::MaxoutLayer, inputActivation::Vol)
    self.inputActivation = inputActivation
    (outputX, outputY, outputDepth) = size(self.outputActivation)
    for d in 1:outputDepth, y in 1:outputY, x in 1:outputX
        maxValue, maxIndex = findmax([V[x, y, ds] for ds in (d - 1) * groupSize + 1:d * groupSize])
        self.outputActivation.dw[x, y, d] = maxValue
        self.argmax[x, y, d] = maxIndex
    end
    return self.outputActivation
end

function backward(self::MaxoutLayer)
    fill!(self.inputActivation.dw, 0.0)
    (outputX, outputY, outputDepth) = size(self.outputActivation)
    for d in 1:outputDepth, y in 1:outputY, x in 1:outputX
        chainGrad = self.outputActivation.dw[x, y, d]
        maxIndex = self.argmax[x, y, d]
        maxInputIndex = (d-1) * groupSize + maxIndex
        self.inputActivation.dw[x, y, maxinputIndex]  = chainGrad
    end
end
