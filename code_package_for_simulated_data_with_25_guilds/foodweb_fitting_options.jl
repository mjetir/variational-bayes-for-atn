
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


module FittingOptions


using LinearAlgebra
using Random, Distributions
using Statistics

include("foodweb_utils.jl")
include("foodweb_model_ODE.jl")


# OPTIONS FOR OLS
# nPar : the number of parameters to be fitted
# p_init: initial values for the parameters (untransformed, in R^n)
# param_min, param_max: parameter bounds (for transformed parameters)
# bc_min, bc_max: bounds for the boxconstraint in optimization
# max_oscillation: max CV for abundances
# model : the model to be fitted, exploitation or not; this is a function
struct OLSOptions
    nPar::Int64 
    p_init::Vector{Float64} 
    param_min::Vector{Float64}
    param_max::Vector{Float64}
    bc_min::Vector{Float64}
    bc_max::Vector{Float64}
    max_oscillation::Float64
    training_data
    model 
end

# OPTIONS FOR BAYES ESTIMATION
# nPar : the number of parameters to be fitted
# p_init: initial values for the parameters (untransformed, in R^n)
# prior_dist: prior distributions for the parameters, in R^n
# epsilon_dist: the distribution to be used in the variational Bayes algorithm
# param_min, param_max: parameter bounds (for transformed parameters)
# bc_min, bc_max: bounds for the boxconstraint in optimization
# max_oscillation: max CV for abundances
# model : the model to be fitted, exploitation or not; this is a function
struct BayesOptions
    nPar::Int64
    p_init::Vector{Float64} # transformed (inverse)
    prior_dist#::IsoNormal
    epsilon_dist#::IsoNormal
    param_min::Vector{Float64}
    param_max::Vector{Float64}
    bc_min::Vector{Float64}
    bc_max::Vector{Float64}
    max_oscillation::Float64 
    training_data
    model 
end

# INITIALIZE AN OLSOPTIONS STRUCT
# fw_init: initial food web structure and parameters
function initialize_OLS_options(fw_init,training_data;kwargs...)

    nPar = fw_init.nLinks

    var_ini = fw_init.fr_half_sat[fw_init.I_ind]

    # half-saturation constants
    param_min::Vector{Float64} = 0.05.*ones(nPar)

    param_max::Vector{Float64} = ones(nPar)
    
    model = FoodWebModel.foodweb_model

    bc_min = copy(param_min)
    bc_max = copy(param_max)

    return OLSOptions(nPar,var_ini,param_min,param_max,bc_min,bc_max,0.001,training_data,model)
end



# INITIALIZE A BAYESOPTIONS STRUCT
# fw_init: initial food web structure and parameters
# rho: std of variational distribution 
function initialize_bayes_options(fw_init,rho,training_data;kwargs...)

    # the number of model parameters to be estimated
    nPar = fw_init.nLinks

    par_means_ini = fw_init.fr_half_sat[fw_init.I_ind]

    par_stds_ini = rho.*ones(nPar)

    param_min::Vector{Float64} = zeros(2*nPar)
        
    # half-saturation constants
    param_min[1:fw_init.nLinks] .= 0.05
      
    param_max::Vector{Float64} = ones(2*nPar)     # includes upper bounds for rhos, too
    # rhos not restricted from above:
    param_max[nPar+1:end] .= Inf
    
    # initialize
    p_init::Vector{Float64} = zeros(2*nPar)

    # transform to R^n
    p_init[1:nPar] = 
        FoodWebUtils.inverse_activation1(par_means_ini,
                                   param_min[1:nPar],
                                   param_max[1:nPar])

    p_init[nPar+1:end] = FoodWebUtils.inverse_rho_transform(par_stds_ini)

    prior_dist = [Normal(p_init[i],par_stds_ini[i]) for i=1:nPar]
    
    epsilon_dist = MvNormal(zeros(nPar),1.0)

    model = FoodWebModel.foodweb_model

    println("Parameter lower bounds:")
    println(param_min)
    println("Parameter upper bounds:")
    println(param_max)

    bc_min = copy(param_min)
    bc_max=copy(param_max)
    bc_min .= -Inf
    bc_max .= Inf

    return BayesOptions(nPar,p_init,prior_dist,
    epsilon_dist, param_min, param_max, bc_min, bc_max, 0.001, training_data, model)
end


# variables for avoiding temporary allocations during estimation
mutable struct OLSContainers
    prediction::Matrix{Float64}
end

# variables for avoiding temporary allocations during estimation
mutable struct BayesContainers
    epsilons::Vector{Float64}
    sampled_parameters::Vector{Float64}
    log_likelihood::Float64
    log_prior::Float64
    log_variational::Float64
    prediction::Matrix{Float64}
end


# INITIALIZE A CONTAINER FOR OLS
# fw_param: a foodweb struct containing the nnumber of feeding links, the number of species
function initialize_OLS_container(fw_param)
    prediction = zeros(fw_param.nSpec,length(fw_param.time_grid))

    return OLSContainers(prediction)
end


# INITIALIZE A CONTAINER FOR BAYESIAN FITTING
# fw_param: a foodweb struct containing the nnumber of feeding links, the number of species
function initialize_bayes_container(fw_param,f_opt)
    epsilons = zeros(f_opt.nPar)
    sampled_parameters = zeros(f_opt.nPar)

    log_likelihood = 0.0
    log_prior = 0.0
    log_variational = 0.0

    prediction = zeros(fw_param.nSpec,length(fw_param.time_grid))

    return BayesContainers(epsilons,
        sampled_parameters,
        log_likelihood, log_prior, log_variational, prediction)
end


end # module
