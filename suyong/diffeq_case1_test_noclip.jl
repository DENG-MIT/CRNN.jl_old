using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

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

    push!(track_loss, l)
    p0 = plot(iter, track_loss)

    # plot current prediction against data
    p1 = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(p1, tsteps, pred[1,:], label = "prediction")
    p2 = scatter(tsteps, ode_data[2,:], label = "data")
    scatter!(p2, tsteps, pred[2,:], label = "prediction")
    p3 = scatter(tsteps, ode_data[3,:], label = "data")
    scatter!(p3, tsteps, pred[3,:], label = "prediction")
    p4 = scatter(tsteps, ode_data[4,:], label = "data")
    scatter!(p4, tsteps, pred[4,:], label = "prediction")
    plt = plot(p1,p2,p3,p4,layout=(2,2))
    push!(list_plots, plt)
    if doplot
        display(plot(plt))
    end
    if save_file
        savefig(plt,"y_t.png")
        savefig(p0, "loss.png")
    end

    iter += 1
    return false
end

"
pstart = DiffEqFlux.sciml_train(loss_neuralode, p, ADAM(1), cb = cb, maxiters = 100).minimizer

p1 = DiffEqFlux.sciml_train(loss_neuralode, p, ADAM(0.1), cb = cb, maxiters = 100).minimizer
p2 = DiffEqFlux.sciml_train(loss_neuralode, p1, ADAM(0.1), cb = cb, maxiters = 100).minimizer
p2_2 = DiffEqFlux.sciml_train(loss_neuralode, p1, ADAM(0.01), cb = cb, maxiters = 100).minimizer
p2_3 = DiffEqFlux.sciml_train(loss_neuralode, p1, ADAM(0.01), cb = cb, maxiters = 1000).minimizer
p3 = DiffEqFlux.sciml_train(loss_neuralode, p2_3, ADAM(0.01), cb = cb, maxiters = 1000).minimizer

p_second_1 = DiffEqFlux.sciml_train(loss_neuralode, p3, cb = cb, Optim.KrylovTrustRegion(), maxiters = 100)

p_another = DiffEqFlux.sciml_train(loss_neuralode, p, ADAM(0.01), cb = cb, maxiters = 2000).minimizer
"

p1 = DiffEqFlux.sciml_train(loss_neuralode, p0, ADAM(0.01), cb = cb, maxiters = 2000).minimizer
"
p2 = DiffEqFlux.sciml_train(loss_neuralode, p1, ADAM(0.1), cb = cb, maxiters = 1000).minimizer
p3 = DiffEqFlux.sciml_train(loss_neuralode, p2, ADAM(0.01), cb = cb, maxiters = 1000).minimizer
""

