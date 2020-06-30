using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using Flux: gradient
using Flux.Optimise: update!
using Random
using BSON: @save, @load
using Distributed: pmap

n_plot = 10
n_epoch = 100000
n_exp = 100

model_dir = "fig_missing_s3_case11/"

# SC = zeros(Float32, (4, 5))
# R = zeros(Float32, 4)

function trueODEfunc(dydt, y, k, t)

    # SC[1, :] .= [-2,  1,  0,  0,  0]
    # SC[2, :] .= [-1, -1,  1,  0,  0]
    # SC[3, :] .= [ 1, -1, -1,  1,  0]
    # SC[4, :] .= [ 0,  1, -1, -1,  1]
    # SC[5, :] .= [ 1,  1, -1,  0,  0]
    # SC[6, :] .= [-1,  1,  1, -1,  0]
    # SC[7, :] .= [ 0, -1,  1,  1, -1]

    R1 = k[1] * y[1]^2
    R2 = k[2] * y[1] * y[2]
    R3 = k[3] * y[2] * y[3]
    R4 = k[4] * y[3] * y[4]
    R5 = k[5] * y[3]
    R6 = k[6] * y[1] * y[4]
    R7 = k[7] * y[2] * y[5]
    R8 = k[8] * y[2] * y[4]

    dydt[1] = -2R1 - R2 + R3 + R5 - R6
    dydt[2] = R1 - R2 - R3 + R4 + R5 + R6 - R7 - R8
    dydt[3] = R2 - R3 - R4 - R5 + R6 + R7
    dydt[4] = R3 - R4 - R6 + R7 - R8
    dydt[5] = R4 - R7 + R8

end

ns = 5
nr = 8
u0_list = rand(Float32, (n_exp, ns))
u0_list[:, 1] .= u0_list[:, 1] .* 1.5 .+ 0.5
u0_list[:, 3:5] .= 0

datasize = 100
tspan = Float32[0.0, 100.0]
tsteps = range(tspan[1], tspan[2], length = datasize)
k = Float32[0.1, 0.2, 0.13, 0.3, 0.2, 0.13, 0.3, 0.3]
# alg = Rosenbrock23(autodiff = false)
alg = Tsit5()
#alg = AutoTsit5(Rosenbrock23(autodiff = false))

ode_data_list = []

for i in 1:n_exp
    u0 = u0_list[i, :]
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat = tsteps))
    push!(ode_data_list, ode_data)
end

lb = 1e-3
ub = 5.0

dudt2 = FastChain((x, p)->log.(clamp.(x, lb, ub)),
                  FastDense(ns, nr, exp),
                  FastDense(nr, ns))

prob_neuralode = NeuralODE(dudt2, tspan, alg, saveat = tsteps, abstol = 1e-6, reltol = 1e-3)
p = prob_neuralode.p

function clip_p(p)
    # layer1: 5x8+8 = 48
    # layer2: 8x5+5 = 45
    np = ns*nr*2+ns+nr
    p[1:ns*nr] .= clamp.(p[1:ns*nr], 0, 2.5)
    p[np - ns + 1:np] .= 0
    return p
end

function set_p(p)
    p = zeros(49)
    p[1:4] = [2,1,0,0]
    p[5:8] = [0,0,0,1]
    p[9:12] = [0,0,1,0]
    p[13:16] = [0,0,0,1]
    p[17:20] = [0,0,0,0]

    p[21:24] .= log.([0.1, 0.2, 0.13, 0.3])

    p[25:29] .= [-2,  1,  0,  0, 0]
    p[30:34] .= [-1,  0,  1,  0, 0]
    p[35:39] .= [ 0,  0, -1,  1, 0]
    p[40:44] .= [ 0, -1,  0, -1, 1]
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
    loss = sum(abs2, (ode_data .- pred)[[1,2,4,5], :])
    loss = loss + sum(abs2, pred .* (pred .< 0.0))
    return loss, pred
end

function loss_neuralode(p, i_exp)
    return loss_pred_neuralode(p, i_exp)[1]
end

function display_p(p)
    println("S1 | S2 | S3 | S4 | S5")
    for i in 1:ns
        println(p[nr*i-nr + 1:nr*i])
    end
end

# Callback function to observe training
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    list_plt = []
    for i in 1:ns
        plt = scatter(tsteps, ode_data[i,:], title = string(i), label = string("data_", i))
        plot!(plt, tsteps, pred[i,:], label = string("pred_", i))
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
    push!(list_loss, loss_mean)
    if iter % n_plot == 0
        println(string("\n", "iter = ", iter, " loss = ", loss_mean, "\n"))
        display_p(p)
        pmap(1:10:n_exp) do i_exp
            cbi(p, i_exp)
        end
        if iter < 2000
            plt_loss = plot(list_loss, yscale = :log10, label = "loss")
        else
            plt_loss = plot(list_loss, xscale = :log10, yscale = :log10, label = "loss")
        end
        png(plt_loss, string(model_dir, "loss"))
        @save string(model_dir, "dudt2.bson") dudt2
    end
    iter += 1
end

opt = ADAM(0.001)

p = clip_p(p)

loss_all = zeros(Float32, n_exp)

for epoch in 1:n_epoch
    global p, iter
    for i_exp in randperm(n_exp)
        loss, back = Flux.pullback(loss_neuralode, p, i_exp)
        update!(opt, p, back(one(loss))[1])
        p = clip_p(p)
        loss_all[i_exp] = loss
        # println(string("epoch=", iter, " i_exp=", i_exp, " loss=", loss))
    end
    loss_mean = sum(loss_all) / n_exp
    cb(p, loss_mean)
end
