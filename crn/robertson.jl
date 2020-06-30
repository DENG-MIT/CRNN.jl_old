using DifferentialEquations, Plots

function trueODEfunc(du, u, k, t)
    R1 = k[1] * u[1]
    R2 = k[2] * u[2]^2
    R3 = k[3] * u[2] * u[3]
    du[1] = -R1 + R3
    du[2] = R1 - 2R2 - R3
    du[3] = R2
end

datasize = 100
alg = Rosenbrock23(autodiff = false)
# alg = Tsit5()
tspan = (0.0, 1.0e5)
tsteps = 10 .^ (range(-6, stop=log10(tspan[2]), length = datasize))

prob = ODEProblem(rober,[1.0,0.0,0.0],tspan,(4e-2,3e7,1e4))
sol = solve(prob, alg, saveat=tsteps)
plot(sol, xscale=:log10, tspan=(1e-6, tspan[2]), layout=(3,1))
