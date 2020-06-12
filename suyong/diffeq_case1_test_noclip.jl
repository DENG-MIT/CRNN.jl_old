using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, JLD

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1]
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4]
    dydt[3] = k[2] * y[1] - k[3] * y[3]
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4]
    dydt[5] = k[4] * y[2] * y[4]
end

u0 = Float64[1.0;1.0;0.0;0.0;0.0]
datasize = 30
tspan = Float64[0.0, 20.0]
tsteps = range(tspan[1], tspan[2], length = datasize)

k = Float64[0.1, 0.2, 0.13, 0.3]
alg = Rosenbrock23(autodiff = false)

prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
ode_data = Array(solve(prob_trueode, alg, saveat = tsteps))

dudt2 = FastChain((x, p)->log.(clamp.(x, 1e-5, 10)),
                  FastDense(5, 3, exp),
                  FastDense(3, 5))

prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps)
p0 = prob_neuralode.p

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
iter = 1
track_loss = []
cb = function (p, l, pred; doplot = false, save_file = true)
    global list_plots, iter, track_loss

    append!(track_loss, l)

    # plot current prediction against data
    plt0 = plot(track_loss, yscale = :log10)
    plt1 = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt1, tsteps, pred[1,:], label = "prediction")
    plt2 = scatter(tsteps, ode_data[2,:])
    scatter!(plt2, tsteps, pred[2,:])
    plt3 = scatter(tsteps, ode_data[3,:])
    scatter!(plt3, tsteps, pred[3,:])
    plt4 = scatter(tsteps, ode_data[4,:])
    scatter!(plt4, tsteps, pred[4,:])
    plt5 = scatter(tsteps, ode_data[5,:])
    scatter!(plt5, tsteps, pred[5,:])
    plt = plot(plt0, plt1, plt2, plt3, plt4, plt5, layout=(2,3))
    push!(list_plots, plt)

    if doplot
        display(plot(plt))
    end
    
    if save_file
        savefig(plt,"Results_loss_y_t.png")
    end

    iter += 1
    return false
end

p_iter = []
p_temp = DiffEqFlux.sciml_train(loss_neuralode, p0, ADAM(0.01), cb = cb, maxiters = 2000).minimizer
push!(p_iter, p_temp)

for i in 2:10
    p_temp = DiffEqFlux.sciml_train(loss_neuralode, p_iter[i-1], ADAM(0.01), cb = cb, maxiters = 1000).minimizer
    push!(p_iter, p_temp)
end

for i in 11:15
    p_temp = DiffEqFlux.sciml_train(loss_neuralode, p_iter[i-1], ADAM(0.01), cb = cb, maxiters = 1000).minimizer
    push!(p_iter, p_temp)
end

save("CRNN.jld", "p_iter", p_iter)
# CRNN = load("CRNN.jld")
