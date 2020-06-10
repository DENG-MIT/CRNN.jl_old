using Flux, Statistics, DelimitedFiles
using Flux: Params, gradient, mse, glorot_normal, throttle
using Flux.Optimise: update!
using Flux.Data: DataLoader
using DelimitedFiles, Statistics
using Parameters: @with_kw
using CUDAapi

using NPZ
using BSON: @save, @load
using ProgressBars

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    import CuArrays		# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end

# Struct to define hyperparameters
@with_kw mutable struct Hyperparams
    restart::Bool = false
    batchsize::Int = 256
    throttle::Int = 1000
    epochs::Int = 1000000
    nr::Int = 3
    lr::Float64 = 0.0005		# learning rate
    eps1::Float64 = 1e-30
    device::Function = cpu  # set as gpu, if gpu available
end

function get_processed_data(args)

    X_train_np = npzread("data/stack/X_train.npy")
    Y_train_np = npzread("data/stack/Y_train.npy")

    X_test_np = npzread("data/stack/X_test.npy")
    Y_test_np = npzread("data/stack/Y_test.npy")

    eps1 = args.eps1

    X_train_log = log.(X_train_np .+ eps1)
    X_test_log = log.(X_test_np .+ eps1)

    X_mean = mean(X_train_log, dims = 1)
    X_std = std(X_train_log, dims = 1) .+ eps1
    Y_mean = mean(Y_train_np, dims = 1) .* 0
    # Y_std = maximum(abs.(Y_train_np), dims = 1) .+ eps1
    Y_std = std(Y_train_np, dims = 1) .+ eps1

    x_train = (X_train_log .- X_mean) ./ X_std
    x_test = (X_test_log .- X_mean) ./ X_std

    y_train = (Y_train_np .- Y_mean) ./ Y_std
    y_test = (Y_test_np .- Y_mean) ./ Y_std

    train_data = (x_train, y_train)
    test_data = (args.device(x_test), args.device(y_test))

    scale = (args.device(X_mean), args.device(X_std), args.device(Y_mean), args.device(Y_std))

    return train_data, test_data, scale
end

# Struct to define model
mutable struct model
    W1::AbstractArray
    b::AbstractArray
    W2::AbstractArray
    alpha::AbstractArray
end

# Function to predict output from given parameters
predict(x, m) = (exp.((x * m.W1 .+ m.b) .* (m.alpha) .* 10)) * m.W2
    
function train(; kws...)
    # Initialize the Hyperparamters
    args = Hyperparams(; kws...)
    
    # Load the data
    train, test, scale = get_processed_data(args)

    train_loader = DataLoader(train[1]', train[2]', batchsize = args.batchsize, shuffle = true) 

    train_loader = args.device(train_loader)

    println("size of training data: ", size(train[1]), " test data: ", size(test[1]))

    n_in = size(train[1], 2)
    n_out = size(train[2], 2)
    nr = args.nr

    mW1 = args.device(glorot_normal(n_in, nr))
    mb = args.device(glorot_normal(1, nr))
    mW2 = args.device(glorot_normal(nr, n_out))
    malpha = args.device([0.1])
    
    # The model
    m = model((mW1), (mb), (mW2), (malpha))

    # function loss(x, y)
    #     pred = predict(x, m)
    #     pred_log = log.(abs.(pred) .+ args.eps1)
    #     y_log = log.(abs.(y) .+ args.eps1)
    #     return mse(pred .+ pred_log ./ 3e5, y .+ y_log ./ 3e5)
    # end

    loss(x, y) = mse(predict(x, m), y)

    ## Training
    theta = params([m.W1, m.b, m.W2, m.alpha])

    opt = ADAM(args.lr)

    train_log = Dict()
    for key in ["epoch", "loss_train", "loss_test", "alpha"]
        train_log[key] = []
    end

    if args.restart == true
        @load "model-checkpoint.bson" m opt train_log
        epoch_old = train_log["epoch"][end]
    end

    @info("Training....")

    m.W1[1:end - 2, :] = clamp.(m.W1[1:end - 2, :], 0, 100)
    for epoch in ProgressBar(1:args.epochs)
        if args.restart == true
            if epoch <= epoch_old
                continue
            end
        end

        for (x, y) in train_loader
            grad = gradient(()->loss(x', y'), theta)
            update!(opt, theta, grad)
            m.W1[1:end - 2, :] = clamp.(m.W1[1:end - 2, :], 0, 100)
        end

        if epoch % args.throttle == 0

            push!(train_log["epoch"], epoch)
            push!(train_log["alpha"], m.alpha[1] * 10)
            push!(train_log["loss_train"], loss(train[1], train[2]))
            push!(train_log["loss_test"], loss(test[1], test[2]))

            @show epoch round.(m.alpha; digits = 3) train_log["loss_train"][end] train_log["loss_test"][end]

            @show "input layer" round.(m.W1 .* (m.alpha .* 10) ./ scale[2]'; digits = 3)

            V_output = m.W2 .* scale[4]

            @show "output layer" round.((V_output ./ V_output[:, 1])'; digits = 3)

            @save "model-checkpoint.bson" m opt train_log

            npzwrite("log/model/W1.npy", m.W1)
            npzwrite("log/model/b.npy", m.b)
            npzwrite("log/model/W2.npy", m.W2)
            npzwrite("log/model/alpha.npy", m.alpha)
            
        end
    end
end

cd(@__DIR__)
train() 