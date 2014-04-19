abstract SGDUpdater

immutable MomentumUpdater <: SGDUpdater
    learningRate::Float64
    decayL1::Float64
    decayL2::Float64
    momentum::Float64
    lastUpdate::Vector{Float64}
end


function updateGradients(m::MomentumUpdater, state::GradientState, batchSize::Int)
    rateL1 = m.decayL1 * state.multiplierL1
    rateL2 = m.decayL2 * state.multiplierL2
    for i in 1:length(state.vol.w)
        gradL1 = rateL1 * (state.vol.w[i] > 0 ? 1 : -1)
        gradL2 = rateL2 * (state.vol.w[i] * state.vol.w[i] / 2.0)

        rawGradient = (gradL1 + gradL2 + state.vol.dw[i]) / batchSize
        momentumGradient = m.momentum * m.lastUpdate[i] - m.learningRate * rawGradient

        state.vol.w[i] += momentumGradient
        m.lastUpdate[i] = momentumGradient
    end
    fill!(state.vol.dw, 0.0)
end
