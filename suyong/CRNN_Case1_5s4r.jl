using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, JLD, Random, Distributions

# Target ODE
function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1]
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4]
    dydt[3] = k[2] * y[1] - k[3] * y[3]
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4]
    dydt[5] = k[4] * y[2] * y[4]
end
k = Float32[0.1, 0.2, 0.13, 0.3]

# User Settings
n_exp = 50
n_iter = 2000
n_train = n_exp*n_iter
ns = 5
nr = 4
# initial_species = [1,2]
file_name = "Alr001_Alr0001_n50_i2000"
is_restart = false
# opt = [GradientDescent(), ADAM(0.01), Optim.KrylovTrustRegion()]
# maxiters_list = [100, n_iter, n_iter]
opt = [ADAM(0.01), ADAM(0.001)]
maxiters_list = [n_train, n_train]
description = string(opt, "Iterations: ", maxiters_list)

# ODE Grid
datasize = 50
tspan = Float32[0.0, 30.0]
tsteps = range(tspan[1], tspan[2], length = datasize)

# ODE Solver Algorithm
alg = Rosenbrock23(autodiff = false)

# Initial Conditions
if is_restart == true
    CRNN = load(string("CRNN_", file_name, ".jld"))
    u0_list = CRNN["u0_list"]
    u0_list_train = CRNN["u0_list_train"]
    iter = CRNN["iter"]
    p_iter = CRNN["p_iter"]
    p = p_iter[end]
    track_loss = CRNN["track_loss"]
else
    u0_list = []
    u0_list_train = []
    for i in 1:n_exp
       u0_temp = rand(Float32, (1,ns)) #rand(Uniform(), (1,ns))
       u0_temp[1,3:ns] .= 0
       push!(u0_list, u0_temp)
    end
    for i in 1:n_iter
        global u0_list_train = vcat(u0_list_train,shuffle(u0_list))
    end
    iter = 1
    p_iter = []
end

# True Solutions
ode_data = Any[]
for (i,u) in enumerate(u0_list)
    prob_trueode = ODEProblem(trueODEfunc, u[:], tspan, k)
    push!(ode_data, Array(solve(prob_trueode, alg, saveat = tsteps)))
end

# Chemical Reaction Neural Network (CRNN) - Fast Chain with DiffEqFlux
dudt = FastChain((x, p)->log.(clamp.(x, 1e-5, 10)),
                  FastDense(5, 4, exp),
                  FastDense(4, 5))
prob_neuralode = NeuralODE(dudt, tspan, alg, saveat = tsteps) #, atol=1e-5, rtol=1e-5)
p = prob_neuralode.p

# Chemical Reaction Neural Network (CRNN) - Normal Chain with Flux
# dudt = Chain(x->log.(clamp.(x, 1e-5, 10)),
#                   Dense(5, 4, exp),
#                   Dense(4, 5))
# prob_neuralode = NeuralODE(dudt, tspan, alg, saveat = tsteps, atol=1e-5, rtol=1e-5)
# p = prob_neuralode.p

function predict_neuralode(u, p)
    Array(prob_neuralode(u, p))
end

function loss_neuralode(p, u...)
    i = findall(x->x==hcat(collect(u))', u0_list)[1]
    pred = predict_neuralode(u0_list[i][:], p)
    loss = sum(abs2, ode_data[i] .- pred)
    return loss, pred
end

# Weight and Bias Constraints
function clip_p(p)
    # layer1: 5x4+4 = 24
    # layer2: 4x5+5 = 25
    p[1:20] .= clamp.(p[1:20], 0, 2.5)
    p[45:49] .= 0
    
    return p
end
p = clip_p(p)

# Callback function to observe training
list_plots = []
track_loss = []
cb = function (p, l, pred; doplot = false, save_file = true)
    global list_plots, iter, track_loss, u0_list, u0_list_train, ode_data

    append!(track_loss, l)

    # if doplot
    #     display(plot(plt))
    # end

    if save_file && rem(iter,100) == 0
        global iter_loop = iter
        for i in maxiters_list
            if iter_loop > i
                iter_loop -= i
            end
        end
        i = findall(x->x==u0_list_train[iter_loop],u0_list)[1]
        ode_data_temp = ode_data[i]

        # plot current prediction against data
        plt0 = plot(track_loss, xscale = :log10, yscale = :log10)
        plt1 = plot(tsteps, ode_data_temp[1,:], label = "data")
        scatter!(plt1, tsteps, pred[1,:], label = "prediction")
        plt2 = plot(tsteps, ode_data_temp[2,:], label = "data")
        scatter!(plt2, tsteps, pred[2,:], label = "prediction")
        plt3 = plot(tsteps, ode_data_temp[3,:], label = "data")
        scatter!(plt3, tsteps, pred[3,:], label = "prediction")
        plt4 = plot(tsteps, ode_data_temp[4,:], label = "data")
        scatter!(plt4, tsteps, pred[4,:], label = "prediction")
        plt5 = plot(tsteps, ode_data_temp[5,:], label = "data")
        scatter!(plt5, tsteps, pred[5,:], label = "prediction")
        plt = plot(plt0, plt1, plt2, plt3, plt4, plt5, layout=(2,3))
        push!(list_plots, plt)
        savefig(plt,string("CRNN_", file_name, ".png"))
    end

    if save_file && rem(iter,500) == 0
        push!(p_iter, p)
        save(string("CRNN_", file_name, ".jld"), "p_iter", p_iter, "p_current", p, "n_exp", n_exp, "n_iter", n_iter, "iter", iter, "u0_list", u0_list, "u0_list_train", u0_list_train, "track_loss", track_loss, "discription", description)
    end

    iter += 1

    p = clip_p(p)

    return false
end

Network = []
for (i, opt_current) in enumerate(opt)
    global p
    if i > 1
        p = Network[i-1].minimizer
    end
    Net_temp = DiffEqFlux.sciml_train(loss_neuralode, p, opt_current, u0_list_train, cb = cb, maxiters = maxiters_list[i])
    push!(Network, Net_temp)
end

save(string("CRNN_", file_name, "_finished.jld"), "Network", Network, "p_iter", p_iter, "p_current", p, "n_exp", n_exp, "n_iter", n_iter, "iter", iter, "u0_list", u0_list, "u0_list_train", u0_list_train, "discription", description)
