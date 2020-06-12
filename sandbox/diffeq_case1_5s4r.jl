using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using Flux: gradient
using Flux.Optimise: update!

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1]
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4]
    dydt[3] = k[2] * y[1] - k[3] * y[3]
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4]
    dydt[5] = k[4] * y[2] * y[4]
end

n_exp = 10
ns = 5
nr = 4
u0_list = rand(Float64, (n_exp, ns))
u0_list[:, 3:ns] .= 0

datasize = 20
tspan = Float64[0.0, 20.0]
tsteps = range(tspan[1], tspan[2], length = datasize)
k = Float64[0.1, 0.2, 0.13, 0.3]
alg = Rosenbrock23(autodiff = false)

ode_data_list = []

for i in 1:n_exp
    u0 = u0_list[i, :]
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat = tsteps))
    push!(ode_data_list, ode_data)
end

lb = 1e-5
ub = 10.0

dudt2 = FastChain((x, p)->log.(clamp.(x, lb, ub)),
                  FastDense(5, 4, exp),
                  FastDense(4, 5))

prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps)
p = prob_neuralode.p

function clip_p(p)
    # layer1: 5x4+4 = 24
    # layer2: 4x5+5 = 25
    p[1:20] .= clamp.(p[1:20], 0, 2.5)
    p[45:49] .= 0

    return p
end

function predict_neuralode(u0, p)
    pred = clamp.(Array(prob_neuralode(u0, p)), -ub, ub)
    return pred
end

function loss_neuralode(p, i_exp)
    u0 = u0_list[i_exp, :]
    ode_data = ode_data_list[i_exp]
    pred = predict_neuralode(u0, p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

function loss_pred_neuralode(p, i_exp)
    u0 = u0_list[i_exp, :]
    ode_data = ode_data_list[i_exp]
    pred = predict_neuralode(u0, p)
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
cb = function (p, i_exp; doplot = true)
    global list_plots, iter

    p = clip_p(p)

    loss, pred = loss_pred_neuralode(p, i_exp)

    if iter == 0
        list_plots = []
    end
    iter += 1

    push!(list_loss, loss)

    list_plt = []

    ode_data = ode_data_list[i_exp]

    for i in 1:5
        plt = scatter(tsteps, ode_data[i,:], title = string(i), label = string("data",i))
        plot!(plt, tsteps, pred[i,:], label = string("pred",i))
        push!(list_plt, plt)
    end

    if iter < 2000
        plt_loss = plot(list_loss, yscale = :log10, label="loss", title = string("exp", i_exp))
    else
        plt_loss = plot(list_loss, xscale = :log10, yscale = :log10, label="loss", title = string("exp", i_exp))
    end

    push!(list_plt, plt_loss)

    plt_all = plot(list_plt..., layout = (2,3), legend = false)

    #push!(list_plots, plt_all)
    if iter % 100 == 0
        println(string("\n", "iter = ", iter, " i_exp = ", i_exp, " loss = ", loss, "\n"))
        display_p(p)
        display(plt_all)
    end

    return false
end


opt = ADAM(0.001)

p = clip_p(p)

for epoch in 1:100000
    global p, iter
    i_exp = rand(1:n_exp)
    grad = gradient(loss_neuralode, p, i_exp)[1]
    update!(opt, p, grad)
    p = clip_p(p)
    #println(string("\n", "iter ", epoch, " i_exp ", i_exp))
    #display_p(p)
    cb(p, i_exp)
end
