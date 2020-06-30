using DiffEqBiological, DifferentialEquations, Plots

rs = @reaction_network begin
  c1, S + E --> SE
  c2, SE --> S + E
  c3, SE --> P + E
end c1 c2 c3
# p = (0.00166,0.0001,0.1)
p = (1.0, 0.5, 1.0)
tspan = (0., 10.)
u0 = [1., 1., 0., 0.]  # S = 301, E = 100, SE = 0, P = 0

# solve ODEs
prob = ODEProblem(rs, u0, tspan, p)
sol  = solve(prob, Tsit5())
