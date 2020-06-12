using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2
    dydt[2] = k[1] * y[1]^2
end

u0 = Float64[1.0;0.0]
datasize = 40
tspan = Float64[0.0, 20.0]
tsteps = range(tspan[1], tspan[2], length = datasize)

k = Float64[0.1]
alg = Rosenbrock23(autodiff = false)

prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
ode_data = Array(solve(prob_trueode, alg, saveat = tsteps))

lb = 1e-5
ub = 10.0

dudt2 = FastChain((x, p)->log.(clamp.(x, lb, ub)),
                  FastDense(2, 1, exp),
                  FastDense(1, 2))

prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps)
p = prob_neuralode.p
p[1:2] .= clamp.(p[1:2], 0, 2.5)
p[6] = 0
p[7] = 0

function predict_neuralode(p)
    return clamp.(Array(prob_neuralode(u0, p)), -ub, ub)
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Callback function to observe training
list_plots = []
list_loss = []
iter = 0
cb = function (p, l, pred; doplot = true)
    global list_plots, iter

    p[1:2] .= clamp.(p[1:2], 0, 2.5)
    p[6] = 0
    p[7] = 0

    if iter == 0
        list_plots = []
    end
    iter += 1

    push!(list_loss, l)

    # plot current prediction against data
    plt = scatter(tsteps, ode_data[1,:], label = "data1")
    plot!(plt, tsteps, pred[1,:], label = "prediction1")
    scatter!(plt, tsteps, ode_data[2,:], label = "data2")
    plot!(plt, tsteps, pred[2,:], label = "prediction2")

    if iter < 2000
        plt_loss = plot(list_loss, yscale = :log10, label="loss")
    else
        plt_loss = plot(list_loss, xscale = :log10, yscale = :log10, label="loss")
    end

    plt_all = plot(plt, plt_loss, layout = 2, legend = true)

    #push!(list_plots, plt)
    if doplot & (iter % 1000 == 0)
        display(plt_all)
        display(iter)
        display(l)
        display(p)
    end

    return false
end

pstart = DiffEqFlux.sciml_train(loss_neuralode, p, ADAM(0.01), cb = cb, maxiters = 100000).minimizer

#pstart = DiffEqFlux.sciml_train(loss_neuralode, pstart, ADAM(0.001), cb = cb, maxiters = 100000).minimizer

#pmin = DiffEqFlux.sciml_train(loss_neuralode, pstart, cb = cb, Optim.KrylovTrustRegion(), maxiters = 10000)
