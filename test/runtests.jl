using ConvNet

CN = ConvNet
using Base.Test

function test_fc_forward()
    dimensions = (1, 1, 3)
    input = CN.Vol(reshape(float64([1.0, 2.0, 0.0]), dimensions), zeros(dimensions))
    layer = CN.FullyConnectedLayer(2, 1.0, dimensions)
    CN.flatten(layer.filters[1].w)[:] = [-5.0, 2.0, 3.0]
    CN.flatten(layer.filters[2].w)[:] = [3.0, 2.0, 5.0]
    CN.forward(layer, input)
    @test layer.inputActivation == input
    @test size(layer.outputActivation.w) == (1, 1, 2)
    @test CN.flatten(layer.outputActivation.w) == [0.0, 8.0]
end

function test_sig_forward()
    dimensions = (1, 1, 2)
    input = CN.Vol(reshape(float64([1.0, 5.0]), dimensions), zeros(dimensions))
    layer = CN.SigmoidLayer(dimensions)
    CN.forward(layer, input)
    @test layer.inputActivation == input
    @test size(layer.outputActivation.w) == dimensions
    @test CN.flatten(layer.outputActivation.w) == [0.7310585786300049, 0.9933071490757153]
end

function test_softmax_forward()
    dimensions = (1, 1, 5)
    input = CN.Vol(reshape(float64([1.0, 2.0, 3.0, 4.0, 5.0]), dimensions), zeros(dimensions))
    layer = CN.SoftMaxLayer(5)
    output = CN.forward(layer, input)
    @test output == layer.outputActivation
    @test layer.inputActivation == input
    @test size(layer.outputActivation.w) == dimensions
    @test layer.outputActivation.w |> CN.flatten |> x -> round(x, 2) == [0.01, 0.03, 0.09, 0.23, 0.64]
end

function test_softmax_backward()
    dimensions = (1, 1, 2)
    input = CN.Vol(reshape(float64([1.0, 5.0]), dimensions), zeros(dimensions))
    layer = CN.SoftMaxLayer(2)
    output = CN.forward(layer, input)

    @test layer.inputActivation == input
    @test CN.backward(layer, 2) < CN.backward(layer, 1)

    CN.backward(layer, 2)
    oldDw = copy(layer.inputActivation.dw)
    CN.backward(layer, 2)
    @test oldDw == layer.inputActivation.dw
    @test layer.inputActivation == input
    println(layer.inputActivation)
    @test CN.flatten(layer.dw) == [0.017986209962091555, -0.01798620996209155]
end

function test_softmax_backward()
    numOutputs = 2
    dimensions = (1, 1, numOutputs)

    input = CN.Vol(reshape(float64([1.0, 5.0]), dimensions), zeros(dimensions))
    layer = CN.SoftMaxLayer(numOutputs)
    output = CN.forward(layer, input)

    @test layer.inputActivation == input
    @test CN.backward(layer, 2) < CN.backward(layer, 1)

    CN.backward(layer, 2)
    oldDw = copy(layer.inputActivation.dw)
    CN.backward(layer, 2)
    @test oldDw == layer.inputActivation.dw
    @test layer.inputActivation == input
    @test CN.flatten(layer.inputActivation.dw) == [0.017986209962091555, -0.01798620996209155]
end

function test_xor()
    numInputs = 2
    numHidden = 2
    numClasses = 2
    net = CN.Net([CN.FullyConnectedLayer(numHidden, 1.0, (1, 1, numInputs)),
                  CN.FullyConnectedLayer(numClasses, 1.0, (1, 1, numHidden)),
                  CN.SoftMaxLayer(numClasses)])
    function instance()
        samples = rand(0:1, (1, 1, 2)) |> float64
        return CN.Vol(samples, zeros(size(samples))), (sum(samples) == 1.0 ? 1 : 2)
    end
    trainer = CN.Trainer(net, 0.01)
    iterations = 5
    for i in 1:iterations
        inputActivation, class = instance()
        @printf("Iteration %s\n\n", i)
        @printf("Instance: %s\nclass: %s\nprediction: %s\n\n",
                inputActivation, class, CN.predict(trainer.net, inputActivation))
        println("Before training")
        for (i, s) in enumerate(CN.gradientStates(trainer.net))
            @printf("Layer: %s, Gradients: %s\n", i, s)
        end

        CN.train(trainer, inputActivation, class)
        println("After training")
        for (i, s) in enumerate(CN.gradientStates(trainer.net))
            @printf("Layer: %s, Gradients: %s\n", i, s)
        end
        CN.updateGradients(trainer, 1)

        println("After BackPropagation")
        for (i, s) in enumerate(CN.gradientStates(trainer.net))
            @printf("Layer: %s, Gradients: %s\n", i, s)
        end
    end

    evaluations = 10
    evaluationSamples = [instance() for _ in 1:evaluations]
    errors = 0
    for _ in 1:evaluations
        inputActivation, class = instance()
        prediction = CN.predict(trainer, inputActivation)
        errors += int(prediction != class)
        println("Prediction: ", prediction)
        println("Class: ", class)
    end
    @printf("Accuracy: %s", errors / evaluations)
end


test_fc_forward()
test_sig_forward()
test_softmax_forward()
test_softmax_backward()
test_xor()
