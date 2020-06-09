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

dudt2 = FastChain((x, p)->log.(clamp.(x, 1e-30, 10)),
                  FastDense(2, 1, exp),
                  FastDense(1, 2))

prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps)
prob_neuralode.p[1:2] .= clamp.(prob_neuralode.p[1:2], 0, 2)
p = prob_neuralode.p

function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Callback function to observe training
list_plots = []
iter = 0
cb = function (p, l, pred; doplot = true)
    global list_plots, iter

    p[1:2] .= clamp.(p[1:2], 0, 2)

    if iter == 0
        list_plots = []
    end
    iter += 1

    display(l)
    display(iter)

    # plot current prediction against data
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    push!(list_plots, plt)
    if doplot
        display(plot(plt))
    end

    return false
end

pstart = DiffEqFlux.sciml_train(loss_neuralode, p, ADAM(0.001), cb = cb, maxiters = 10000).minimizer

pstart = DiffEqFlux.sciml_train(loss_neuralode, pstart, ADAM(0.01), cb = cb, maxiters = 10000).minimizer

#pmin = DiffEqFlux.sciml_train(loss_neuralode, pstart, cb = cb, Optim.KrylovTrustRegion(), maxiters = 100)
