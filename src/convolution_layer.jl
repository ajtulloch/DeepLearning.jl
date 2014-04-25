type ConvolutionLayer
    filters::Vector{Vol}
    biases::Vol
    stride::Int
    padding::Int
    inputActivation::Vol
    outputActivation::Vol

    function ConvolutionLayer(numOutputs::Int, stride::Int, padding::Int, filterDimensions, inputDimensions)
        (filterX, filterY, filterDepth) = filterDimensions
        (inputX, inputY, inputDepth) = inputDimensions
        outputX = floor((inputX + 2 * padding - filterX) / stride + 1)
        outputY = floor((inputY + 2 * padding - filterY) / stride + 1)
        filters = [Vol(dimensions) for _ in 1:numOutputs]
        biases = Vol((1, 1, numOutputs))
        inputActivation = Vol(inputDimensions)
        outputActivation = Vol((outputX, outputY, numOutputs))
        new(filters, biases, stride, padding, inputActivation, outputActivation)
    end
end

function foward(self::ConvolutionLayer, inputActivation::Vol)
    (inputX, inputY, inputDepth) = size(l.inputActivation)
    (outputX, outputY, numOutputs) = size(l.outputActivation)
    self.inputActivation = inputActivation
    for d in 1:length(self.filters)
        filter = self.filters[d]
        filterX, filterY, filterDepth = size(filters)
        for ay in 1:outputY, ax in 1:outputX
            x, y = (stride - 1) * ax - padding, (stride - 1) * ay - padding
            a = 0.0
            for fd in 1:filterDepth, fy in 1:filterY, fx in 1:filterX
                ox = x + fx
                oy = y + fy
                if 1 <= ox <= inputX && 1 <= oy <= inputY && 1 <= fd <= inputDepth
                    a += filter.w[fx, fy, fd] * inputActivation.w[ox, oy, fd]
                end
            end
            a += self.biases.w[1, 1, d]
            self.outputActivation.w[ax, ay, d] = a
        end
    end
    return self.outputActivation
end


function backward(self::ConvolutionLayer)
    (inputX, inputY, inputDepth) = size(self.inputActivation)
    (outputX, outputY, numOutputs) = size(self.outputActivation)
    fill!(self.inputActivation.dw, 0.0)
    for d in 1:length(self.filters)
        filter = self.filters[d]
        filterX, filterY, filterDepth = size(filters)
        for ay in 1:outputY, ax in 1:outputX
            x, y = (stride - 1) * ax - padding, (stride - 1) * ay - padding
            chainGrad = this.outputActivation.dw[ax, ay, d]
            for fd in 1:filterDepth, fy in 1:filterY, fx in 1:filterX
                ox = x + fx
                oy = y + fy
                if 1 <= ox <= inputX && 1 <= oy <= inputY && 1 <= fd <= inputDepth
                    filter.dw[fx, fy, d] += self.inputActivation[ox, oy, d] * chainGrad
                    self.inputActivation.dw[ox, oy, d] += filter.w[fx, fy, d] * chainGrad
                end
            end
            self.biases.dw[1, 1, d] += chain_grad
        end
    end
end

function gradientState(self::ConvolutionLayer)
    grads = [GradientState(filter.w, filter.dw,  1.0, 1.0)
             for filter in self.filters]
    append!(grads, [GradientState(self.biases.w, self.biases.dw, 0.0, 0.0)])
    return grads
end
