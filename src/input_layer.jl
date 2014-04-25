type InputLayer
    inputActivation::Vol
    outputActivation::Vol

    function InputLayer(dimensions)
        new(zeros(dimensions), zeros(dimensions))
    end
end

function foward(self::InputLayer, inputActivation::Vol)
    self.outputActivation = copy(self.inputActivation)
    self.outputActivation.w = copy(self.inputActivation.w)
end

function backward(self::InputLayer)
    self.inputActivation.dw = copy(self.inputActivation.dw)
end

gradientStates(self::InputLayer) = []
