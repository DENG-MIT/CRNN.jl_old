using Flux: Params, gradient, mse, glorot_normal, throttle
using Flux.Optimise: update!
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2
    dydt[2] = k[1] * y[1]^2
end

u0 = Float64[1.0;0.0]
datasize = 100
tspan = Float64[0.0, 20.0]
tsteps = range(tspan[1], tspan[2], length = datasize)

k = Float64[0.1]
alg = Rosenbrock23(autodiff = false)

prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
ode_data = Array(solve(prob_trueode, alg, saveat = tsteps))


# Struct to define model
mutable struct model
    W1::AbstractArray
    b::AbstractArray
    W2::AbstractArray
end

n_in = 2
n_out = 2
nr = 1

mW1 = convert(Array{Float64},glorot_normal(n_in, nr))
mb = convert(Array{Float64},glorot_normal(1, nr))
mW2 = convert(Array{Float64},glorot_normal(nr, n_out))

m = model((mW1), (mb), (mW2))
theta = params([m.W1, m.b, m.W2])

function neuralode_func(dydt, y, m, t)
    dydt = exp.(y' * m.W1 .+ m.b) * m.W2
end

prob_neuralode = ODEProblem(neuralode_func, u0, tspan, m)

function loss_neuralode()
    pred = Array(solve(prob_neuralode, alg, saveat = tsteps))
    loss = mse(pred, ode_data)
    return loss
end

m.W1 = clamp.(m.W1, 0, 2.1)

loss_neuralode()

grad = gradient(() -> loss_neuralode(), theta)


opt = ADAM(0.01)
for epoch in 1:10
    m.W1 = clamp.(m.W1, 0, 2.1)
    grad = gradient(() -> loss_neuralode(), theta)
    update!(opt, theta, grad)
    m.W1 = clamp.(m.W1, 0, 2.1)
end
