using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1]
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4]
    dydt[3] = k[2] * y[1] - k[3] * y[3]
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4]
    dydt[5] = k[4] * y[2] * y[4]
end

u0 = Float64[1.0;1.0;0.0;0.0;0.0]
datasize = 20
tspan = Float64[0.0, 20.0]
tsteps = range(tspan[1], tspan[2], length = datasize)

k = Float64[0.1, 0.2, 0.13, 0.3]
alg = Rosenbrock23(autodiff = false)

prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
ode_data = Array(solve(prob_trueode, alg, saveat = tsteps))

lb = 1e-5
ub = 10.0

dudt2 = FastChain((x, p)->log.(clamp.(x, lb, ub)),
                  FastDense(5, 4, exp),
                  FastDense(4, 5))

prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps)
p = prob_neuralode.p
# layer1: 5x4+4 = 24
# layer2: 4x5+5 = 25
p[1:20] .= clamp.(p[1:20], 0, 2.5)
p[45:49] .= 0.0 * p[45:49]

function predict_neuralode(p)
    return clamp.(Array(prob_neuralode(u0, p)), -ub, ub)
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

function display_p(p)
    println("r1")
    println(p[1:5])
    println("r2")
    println(p[6:10])
    println("r3")
    println(p[11:15])
    println("r4")
    println(p[16:20])
end

# Callback function to observe training
list_plots = []
list_loss = []
iter = 0
p_now = p
cb = function (p, l, pred; doplot = true)
    global list_plots, iter, p_now

    p[1:20] .= clamp.(p[1:20], 0, 2.5)
    p[45:49] .= 0.0 * p[45:49]

    p_now = p

    if iter == 0
        list_plots = []
    end
    iter += 1

    push!(list_loss, l)

    # plot current prediction against data

    list_plt = []

    for i in 1:5
        plt = scatter(tsteps, ode_data[i,:], label = string("data",i))
        plot!(plt, tsteps, pred[i,:], label = string("pred",i))
        push!(list_plt, plt)
    end

    if iter < 2000
        plt_loss = plot(list_loss, yscale = :log10, label="loss")
    else
        plt_loss = plot(list_loss, xscale = :log10, yscale = :log10, label="loss")
    end

    push!(list_plt, plt_loss)

    plt_all = plot(list_plt..., layout = (2,3), legend = true)

    #push!(list_plots, plt)
    if doplot & (iter % 100 == 0)
        display(plt_all)
        display(iter)
        display(l)
        display_p(p)
    end

    return false
end

pstart = DiffEqFlux.sciml_train(loss_neuralode, p, ADAM(0.001), cb = cb, maxiters = 10000).minimizer

pstart = DiffEqFlux.sciml_train(loss_neuralode, pstart, ADAM(0.001), cb = cb, maxiters = 200000).minimizer

pmin = DiffEqFlux.sciml_train(loss_neuralode, p, cb = cb, Optim.KrylovTrustRegion(), maxiters = 1000)
