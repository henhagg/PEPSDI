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

# tvec, sde_sol = solve_sde_model_n_times(sde_mod, [0,30+0.01], [exp(5.704), 0], exp.([-0.694, -7.014, 0.027]), 0.01)
# observed_sol = log.(exp(0.751) .* sde_sol[2,:] .+ exp(2.079))
# plot(observed_sol)
# println(observed_sol[1:3])

#######################################################################################################
# Prior for population parameters η = (μ, τ). Note, the priors can be almost any univariate distribution, 
# but they most be provided as arrays. 
prior_mean = [Normal(-0.694, 1.0), Normal(-3, 1.0), Normal(0.027, 1.0)]
prior_scale = [Gamma(5.0, 0.5), Gamma(5.0, 0.5), Gamma(5.0, 0.5)]

# Prior for strength of measurement error ξ
prior_sigma = [Normal(-1.5, 1.0)]

# Prior for cell-constant parameters ĸ. Priors on log-scale 
# since we infer ĸ on the log-scale 
prior_kappa = [Normal(5, 1.0), Normal(1, 1.0), Normal(3, 1.0)]
    
# Inference options for η, ĸ and ξ
pop_param_info = init_pop_param_info(prior_mean, 
                                     prior_scale, 
                                     prior_sigma, 
                                     prior_pop_kappa = prior_kappa, # ĸ not always used hence priors must be made explicit  
                                     pos_pop_kappa = false, # ĸ not constrained to positive  
                                     log_pop_kappa = true, # ĸ inferred on log-scale 
                                     pos_pop_sigma = false,
                                     log_pop_sigma = true) # ξ inferred to be positive (and default not on log-scale)

# Set up opitons for individual parameters c_i
ind_val = [-0.694, -3, 0.027] # Starting value for each individual 
ind_param_info = init_ind_param_info(ind_val,         # Starting value (can also be mean, median, random to sample prior)
                                     3,               # Number of individual parameters  
                                     log_scale=true,  # Individual parameters inferred on log-scale 
                                     pos_param=false) # Individual parameters not constrained to be positive 

# Choosing a particle filter 
dt = 0.01 # Step-length when simulating the model 
rho = 0.99 # Correlation level between particles 
# Use the modified diffusion bridge filter 
# filter_opt = init_filter(ModDiffusion(), dt, rho=rho)
filter_opt = init_filter(BootstrapEm(), dt, rho=rho)

# Choose adaptive mcmc-scheme when proposing parameters (ĸ_i, ξ_i) and c_i
cov_mat_ci = diagm([0.16, 0.16, 0.16])
cov_mat_kappa_sigma = diagm([0.25, 0.25, 0.25, 0.25]) ./ 10
# As seen in the code multiple options can be provided for the RAM-sampler 
mcmc_sampler_ci = init_mcmc(RamSampler(), ind_param_info, cov_mat=cov_mat_ci, step_before_update=500)
mcmc_sampler_kappa_sigma = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_kappa_sigma, step_before_update=500)

# Define the distributions for (ĸ_pop, ξ_pop) and η
pop_sampler_opt = init_pop_sampler_opt(PopOrnstein(), n_warm_up=50) # η
# (ĸ_pop, ξ_pop) with ε = 0.01
# kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances = [0.01, 0.01 ,0.01]) # 

# Set up struct that stores all file-locations 
path_data = pwd() * "/Intermediate/Simulated_data/SSA/Multiple_ind/gfp/observations_pepsdi.csv"
# Multiple_ind = true -> stored in intermediate under Multiple_individual folder
file_loc = init_file_loc(path_data, "Example/Gfp_model", multiple_ind=true) 

import Random
Random.seed!(123)
tune_part_data = init_pilot_run_info(pop_param_info,
                                        n_particles_pilot=300,
                                        n_samples_pilot=500, 
                                        rho_list=[0.99],
                                        n_times_run_filter=50,
                                        init_kappa=[5.704, 0.751, 2.079],
                                        init_sigma = [-1.6])

# Function for tuning the number of particles for a mixed-effects model. 
tune_particles_opt1(tune_part_data, pop_param_info, ind_param_info,
        file_loc, sde_mod, filter_opt, mcmc_sampler_ci, mcmc_sampler_kappa_sigma, pop_sampler_opt)

n_samples = 20000
stuff = run_PEPSDI_opt1(n_samples, pop_param_info, ind_param_info, file_loc, sde_mod, 
        filter_opt, mcmc_sampler_ci, mcmc_sampler_kappa_sigma, pop_sampler_opt, pilot_id=1)
status = "done"
print("Done")


