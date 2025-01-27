using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, JLD, Random

# Target ODE
function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1]
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4]
    dydt[3] = k[2] * y[1] - k[3] * y[3]
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4]
    dydt[5] = k[4] * y[2] * y[4]
end
k = Float64[0.1, 0.2, 0.13, 0.3]

# User Settings
n_exp = 10
n_iter = 5000
n_train = n_exp*n_iter
ns = 5
nr = 4
file_name = "MI_Alr0001_K_n10_i5000_clip"
is_restart = false
# opt = [GradientDescent(), ADAM(0.01), Optim.KrylovTrustRegion()]
# maxiters_list = [100, n_iter, n_iter]
opt = [ADAM(0.001), Optim.KrylovTrustRegion()]
maxiters_list = [n_train, n_train]
description = string(opt, "Iterations: ", maxiters_list)

# ODE Grid
datasize = 20
tspan = Float64[0.0, 20.0]
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
    p0 = p_iter[end]
else
    u0_list = []
    u0_list_train = []
    for i in 1:n_exp
        u0_temp = rand(Float64, (1,ns))
        u0_temp[1,3:ns] .= 0
        push!(u0_list, u0_temp)
        for j in 1:n_iter
            push!(u0_list_train, u0_temp)
        end
    end
    u0_list_train = shuffle(u0_list_train)
    iter = 1
    p_iter = []
end

# True Solutions
ode_data = Any[]
for (i,u) in enumerate(u0_list)
    prob_trueode = ODEProblem(trueODEfunc, u[:], tspan, k)
    push!(ode_data, Array(solve(prob_trueode, alg, saveat = tsteps)))
end

# Chemical Reaction Neural Network (CRNN)
dudt2 = FastChain((x, p)->log.(clamp.(x, 1e-5, 10)),
                  FastDense(5, 4, exp),
                  FastDense(4, 5))
prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps)
p0 = prob_neuralode.p

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
p0 = clip_p(p0)

# Callback function to observe training
list_plots = []
track_loss = []
cb = function (p, l, pred; doplot = false, save_file = true)
    global list_plots, iter, track_loss, u0_list, u0_list_train, ode_data

    i = findall(x->x==u0_list_train[iter],u0_list)[1]
    ode_data_temp = ode_data[i]
    append!(track_loss, l)

    # plot current prediction against data
    plt0 = plot(track_loss, yscale = :log10)
    plt1 = scatter(tsteps, ode_data_temp[1,:], label = "data")
    scatter!(plt1, tsteps, pred[1,:], label = "prediction")
    plt2 = scatter(tsteps, ode_data_temp[2,:], label = "data")
    scatter!(plt2, tsteps, pred[2,:], label = "prediction")
    plt3 = scatter(tsteps, ode_data_temp[3,:], label = "data")
    scatter!(plt3, tsteps, pred[3,:], label = "prediction")
    plt4 = scatter(tsteps, ode_data_temp[4,:], label = "data")
    scatter!(plt4, tsteps, pred[4,:], label = "prediction")
    plt5 = scatter(tsteps, ode_data_temp[5,:], label = "data")
    scatter!(plt5, tsteps, pred[5,:], label = "prediction")
    plt = plot(plt0, plt1, plt2, plt3, plt4, plt5, layout=(2,3))
    push!(list_plots, plt)

    if doplot
        display(plot(plt))
    end

    if save_file && rem(iter,50)==0
        savefig(plt,string("CRNN_", file_name, ".png"))
        save(string("CRNN_", file_name, ".jld"), "p_iter", p_iter, "n_exp", n_exp, "n_iter", n_iter, "iter", iter, "u0_list", u0_list, "u0_list_train", u0_list_train, "discription", description)
    end

    iter += 1
    
    p = clip_p(p)

    return false
end

for (i, opt_current) in enumerate(opt)
    p_temp = DiffEqFlux.sciml_train(loss_neuralode, p0, opt_current, u0_list_train, cb = cb, maxiters = maxiters_list[i]).minimizer
    push!(p_iter, p_temp)
end
