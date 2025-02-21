# perform cd to PEPSDI directory
cd("/home/henhagg/Documents/PEPSDI")

# Required packages for formulating a model and do inference 
using Distributions # For placing priors 
using Random # For setting seed 
using LinearAlgebra # For matrix operations 
using Plots
using Revise
tmp = push!(LOAD_PATH, pwd() * "/src") # Push PEPSDI into load-path 
using PEPSDI # Load PEPSDI 

################################# SDE MOD #################################
# The drift vector need to have arguments du, u, p, t (simular to julia differential equations)
# du : alpha-vector (not allocating new du is efficient)
# u : current state-values 
# p : unknown model-quantites. p.c gives acces to individual parameters c, 
#     while p.kappa gives acces to cell-constants for multi-individual inference.  
# t : current time-value 
function gfp_alpha(du, u, p, t)
    c = p.c

    @views du[1] = -c[1] * u[1]
    @views du[2] = c[3] * u[1] - c[2] * u[2]
end

# The diffusion matrix has the same arguments as the drift-vector. However, here du = beta (diffusion matrix)
function gfp_beta(du, u, p, t)
    c = p.c
    
    @views du[1, 1] = c[1] * u[1]
    @views du[1, 2] = 0
    @views du[2, 1] = 0
    @views du[2, 2] = c[3] * u[1] + c[2] * u[2]
end


# The initial-value function needs to have the arguments u0, p
# u0 : vector with initial values (not allocating new du is efficient)
# p : as above (this allows initial values to be inferred)
function gfp_u0!(u0, p) 
    kappa = p.kappa
    println("u0")
    
    u0[1] = kappa[1]
    u0[2] = 0
end


# The observation function y = g(X, p) must always have the arguments 
# y_mod, u, p, t
# y_mod : vector with model values at time t 
# u : state-values at time t
# p : as above (this allows y to depend on the parameters)
# t : current time-value 
function gfp_h(y_mod, u, p, t)
    println("h")
    kappa = p.kappa
    y_mod[1] = log(kappa[2] * u[2] + kappa[3])
end


# The function for the probability to observe y_mod must always have the arguments 
# y_obs, y_mod, error_param, t, dim_obs. 
# y_obs : vector with observed values at time t (dimension = dim_obs)
# y_obs : vector with model values at time t (dimension = dim_obs)
# error_param : vector with error-parameters xi 
# t : current t-value 
# dim_obs, dimension of the observation vector y. 
function gfp_g(y_obs, y_mod, error_param, t, dim_obs)
    println("g")
    
    # Since y_obs ~ N(y_mod, xi^2) the likelihood can be calculated 
    # via the normal distribution. Perform calculations on log-scale 
    # for stabillity. 
    prob::FLOAT = 0.0
    noise = error_param[1]
    error_dist = Normal(0.0, noise)
    diff = y_obs[1] - y_mod[1]
    # println("diff = ", diff, "noise = ", noise)
    prob = logpdf(error_dist, diff)

    return exp(prob)
end


# P-matrix is the identity matrix here 
# P_mat = [1]
sde_mod = init_sde_model(gfp_alpha, 
                         gfp_beta, 
                         gfp_u0!, 
                         gfp_h, 
                         gfp_g, 
                         2,         # Model dimension dim(X)  
                         1)         # Dimension of observation model dim(Y)
                        #  P_mat)

# time_span = [0, 30]
# u0 = [exp(5.704), 0]
# p = (c = exp.([-0.694, -3, 0.027]), kappa = (exp(5.704), 0))
# dt = 0.01
# tvec, sde_sol = solve_sde_em(sde_mod, time_span, u0, p, dt)
# observed_sol = log.(exp(0.751) .* sde_sol[2,:] .+ exp(2.079))
# observed_sol_with_noise = observed_sol + rand(Normal(0.0, exp(-1.6)), 3001)
# plot(observed_sol)
# println(observed_sol[1:3])

error_dist = Normal(0.0, exp(-1.6))
x0 = [exp(5.704), 0]
tvec = range(0, 30, length = 61)
# p = (c = exp.([-0.694, -3, 0.027]), kappa = exp.([5.704, 0.751, 2.079]))
p = (c = exp.([-0.9421551678066991,-2.467263109757441,2.787283665693973]), kappa = exp.([4.594671068439563,1.7624472563165896,2.1224905960971356]))
tvec, yvec = simulate_data_sde(sde_mod, error_dist, tvec, p, x0, dt = 0.01)
plot(yvec[1,:])