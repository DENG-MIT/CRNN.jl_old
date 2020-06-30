using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using Flux: gradient
using Flux.Optimise: update!
using Random
using BSON: @save, @load
using Statistics
using RecursiveArrayTools
using LinearAlgebra:norm

n_plot = 10
n_epoch = 1000000
n_exp = 1

model_dir = "robertson/"

function trueODEfunc(du, u, k, t)
    R1 = k[1] * u[1]
    R2 = k[2] * u[2]^2
    R3 = k[3] * u[2] * u[3]
    du[1] = -R1 + R3
    du[2] = R1 - 2R2 - R3
    du[3] = R2
end

ns = 3
nr = 3
u0_list = rand(Float64, (n_exp, ns))
u0_list[:, 1] .= u0_list[:, 1] .* 1.0 .+ 1.0
u0_list[:, 2:3] .= 0

datasize = 40
tspan = (0.0, 1e5)
tsteps = 10 .^ (range(-1, stop = log10(tspan[2]), length = datasize))
k = Float64[4e-2, 3e7, 1e4]
alg = Rosenbrock23(autodiff = false)
#alg = Tsit5()
#alg = AutoTsit5(Rosenbrock23(autodiff = false))

ode_data_list = []

for i = 1:n_exp
    u0 = u0_list[i, :]
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(
        prob_trueode,
        alg,
        saveat = tsteps,
    # reltol = 1e-3,
    # abstol = 1e-8,
    ))
    display(size(ode_data))
    push!(ode_data_list, ode_data)
end

VA = VectorOfArray(ode_data_list)
arrVA = reshape(VA, (ns, datasize * n_exp))

arrVA_max = maximum(arrVA, dims=2)[:,1]

println("Entering neural ode")

lb = 1e-7
ub = 2.5e6

arrlogVA = log.(clamp.(arrVA, lb, ub))

log_x_mean = mean(arrlogVA, dims=2)
log_x_std = std(arrlogVA, dims=2)

dudt2 = FastChain(
    (x, p) -> x,
    FastDense(ns, 10, tanh),
    FastDense(10, 10, tanh),
    FastDense(10, ns),
)

condition(u,t,integrator) = ( (t > tsteps[3]) & ((maximum(u)>ub) || (minimum(u) < -ub)) )
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition,affect!)

prob_neuralode = NeuralODE(
    dudt2,
    tspan,
    alg,
    saveat = tsteps,
    reltol = 1e-3,
    abstol = 1e-6,
    maxiters = 10000,
    # callback = cb,
)
p = prob_neuralode.p

# prob_neuralode(u0_list[1, :], p)

function clip_p(p)
    # layer1: 5x8+8 = 48
    # layer2: 8x5+5 = 45
    np = ns * nr * 2 + ns + nr
    p[1:ns*nr] .= clamp.(p[1:ns*nr], 0, 2.5)
    p[np-ns+1:np] .= 0
    return p
end

function predict_neuralode(u0, p)
    pred = clamp.(Array(prob_neuralode(u0, p)), -ub, ub)
    return pred
end

function loss_pred_neuralode(p, i_exp)
    u0 = u0_list[i_exp, :]
    ode_data = ode_data_list[i_exp]
    pred = predict_neuralode(u0, p)
    last_index = size(pred)[2]
    loss = sum(abs2, (ode_data[:, 1:last_index] .- pred))/ns/last_index
    loss = loss + sum(abs2, pred .* (pred .< 0.0))/ns/last_index
    return loss, pred
end

function loss_neuralode(p, i_exp)
    return loss_pred_neuralode(p, i_exp)[1]
end

function display_p(p)
    println("R1 | R2 | R3")
    for i = 1:ns
        println(p[nr*i-nr+1:nr*i])
    end
end


function set_p(p)
    np = ns * nr * 2 + ns + nr
    p = zeros(np)
    p[1:3] = [1, 0, 0]
    p[4:6] = [0, 2, 1]
    p[7:9] = [0, 0, 1]
    p[10:12] .= log.(k)
    p[13:15] .= [-1, 1, 0]
    p[16:18] .= [0, -2, 1]
    p[19:21] .= [1, -1, 0]
    return p
end

# Callback function to observe training
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    list_plt = []
    ENV["GKSwstype"]="100"
    for i = 1:ns
        plt = scatter(
            tsteps,
            ode_data[i, :],
            title = string(i),
            label = string("data_", i),
        )
        plot!(
            plt,
            tsteps[1:size(pred)[2]],
            pred[i, :],
            label = string("pred_", i),
            xscale = :log10,
            tspan = (1e-6, tspan[2]),
        )
        push!(list_plt, plt)
    end
    plt_all = plot(list_plt..., legend = false)
    png(plt_all, string(model_dir, "i_exp_", i_exp))
    return false
end

list_loss = []
iter = 0
cb = function (p, loss_mean)
    global list_loss, iter

    ENV["GKSwstype"]="100"

    push!(list_loss, loss_mean)
    if iter % n_plot == 0
        println(string("\n", "iter = ", iter, " loss = ", loss_mean, "\n"))
        display_p(p)
        for i_exp = 1:10:n_exp
            cbi(p, i_exp)
        end
        if iter < 2000
            plt_loss = plot(list_loss, yscale = :log10, label = "loss")
        else
            plt_loss = plot(
                list_loss,
                xscale = :log10,
                yscale = :log10,
                label = "loss",
            )
        end
        png(plt_loss, string(model_dir, "loss"))
        @save string(model_dir, "dudt2.bson") dudt2
    end
    iter += 1
end

opt = ADAM(0.001)

# p = clip_p(p)

loss_all = zeros(Float64, n_exp)

i_exp = 1
# p = set_p(p)
# iter = 0

for epoch = 1:n_epoch
    global p, iter
    for i_exp in randperm(n_exp)
        loss, back = Flux.pullback(loss_neuralode, p, i_exp)

        if isnan(loss)
            continue
        end
        grad = back(one(loss))[1]
        update!(opt, p, grad)
        # p = clip_p(p)
        loss_all[i_exp] = loss
        println(string("epoch=", iter, " i_exp=", i_exp, " loss=", loss, " norm=", norm(grad, 2)))
    end
    loss_mean = sum(loss_all) / n_exp
    cb(p, loss_mean)
end
