
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


# Minimize the loss function w.r.t. the functional response
# parameters using evolutionary optimization
module FitFoodWebModel

using DifferentialEquations
using StatsPlots
using LinearAlgebra
using Random, Distributions, Statistics
using CSV, DataFrames
using BioEnergeticFoodWebs
using Evolutionary

include("set_foodweb_parameters.jl")
include("foodweb_model_ODE.jl")
include("foodweb_utils.jl")
include("foodweb_fitting_options.jl")
include("foodweb_loss_functions.jl")
include("plot_results_for_foodwebs_public.jl")


# STRUCT FOR SAVING RESULTS
# loss: initial loss and losses after running EO
# param_final: parameter values corresponding to the minimum of loss, untransformed values
# param_init: intial values of parameters, untransformed
# method: Bayes or OLS
# iterations_restart: the numebr of times EO was restarted
# f_options: fitting options struct
struct FittingResults
    loss#::Vector{Float64}
    param#::Matrix{Float64} #the best candidate in each iteration step
    param_final#::Vector{Float64} #best candidate overall
    pop_final#::Vector{Vector{Float64}} #the population where the best candidate belongs to
    param_init#::Vector{Float64}
    method::String
    iterations_restart::Int64
    offspring::Int64
    f_options # OLSOptions or BayesOptions struct
end

function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::CMAES, options)
    record["population"] = population
end    

# MINIMIZE THE LOSS FUNCTION USING CMAES, STARTING FROM A GIVEN VALUE OF PARAMETERS
function EO_minimize(loss_function,fw_tmp,f_opt,f_cont,param_init,n_offspring)

    bc = BoxConstraints(f_opt.bc_min, f_opt.bc_max)

    res = Evolutionary.optimize(x->loss_function(x,fw_tmp,f_opt,f_cont),bc,param_init,CMAES(lambda=n_offspring),
        Evolutionary.Options(iterations = 3000, store_trace = true))

    println("EO result:")
    println(res)
            
    p_tmp = Evolutionary.minimizer(res);

    pop_tmp = get(res.trace[end].metadata,"population",:)

    if in(p_tmp,pop_tmp)
        0
    else 
        pop_tmp = 0
        for k = 1:res.iterations
            pop_tmp_ = get(res.trace[k].metadata,"population",:)
            if in(p_tmp,pop_tmp_)
                pop_tmp = pop_tmp_
            end
        end
    end

    loss = loss_function(p_tmp,fw_tmp,f_opt,f_cont)

    return p_tmp, pop_tmp, loss
end

# MINIMIZE THE LOSS FUNCTION USING CMAES, STARTING FROM A GIVEN VALUE OF PARAMETERS
# restart using the result
function EO_minimize_with_restart(loss_function,fw_tmp,f_opt,f_cont,param_init,n_iter_restart,saving_interval,n_offspring,sub_results_to)
            
    Random.seed!(1526436)

    func_minimize = EO_minimize

    # initialize, includes initial loss
    loss_in_iteration::Vector{Float64}=zeros(n_iter_restart+2)

    # initialize, includes initial guess
    parameters_in_iteration::Matrix{Float64}=zeros(n_iter_restart+2,length(param_init))

    # initialize
    min_loss = loss_function(param_init,fw_tmp,f_opt,f_cont)

    # intialize
    best_solution=copy(param_init)

    # initialize
    population_of_best_solution = 0

    loss_in_iteration[1] = copy(min_loss) # initial loss
    parameters_in_iteration[1,:] = copy(param_init)

    print("Initial loss: ")
    println(min_loss)    

    res = 0
    p_tmp= 0
    pop_tmp = 0

    sub_execution_time = @elapsed begin
        p_tmp, pop_tmp, loss_in_iteration[2] = func_minimize(loss_function,fw_tmp,f_opt,f_cont,copy(param_init),n_offspring)
    end

    parameters_in_iteration[2,:] = copy(p_tmp)
    if(loss_in_iteration[2]<min_loss)
        best_solution = copy(p_tmp)
        population_of_best_solution = copy(pop_tmp)
        min_loss = copy(loss_in_iteration[2])
    end

    ### Construct a FittingResults struct
    sub_results = FittingResults(loss_in_iteration[1:2],
            parameters_in_iteration[1:2,:],
            best_solution, 
            population_of_best_solution,
            [0],
            "_",
            n_iter_restart,
            n_offspring,
            f_opt)

    # write the execution time to file
    FoodWebUtils.write_ex_time_to_file(sub_execution_time,            
        sub_results_to*"first_call/execution_time")

    # write the untransformed (in R^n) parameters to file that minimize the loss function
    FoodWebUtils.write_parameters_to_file(sub_results,
        sub_results_to*"first_call/EO_parameter_estimates")
                        
    FoodWebUtils.write_losses_to_file(sub_results,
        sub_results_to*"first_call/EO_loss")

    FoodWebPlots.plot_losses(sub_results,            
        sub_results_to*"first_call/EO_loss")


    print("=== First round of optimization executed. ")
    print("Loss: ")
    println(loss_in_iteration[2])


    for k=0:Int(n_iter_restart/saving_interval)-1
        sub_execution_time = @elapsed begin
            for i = k*saving_interval + 1 : k*saving_interval + saving_interval
                p_tmp, pop_tmp, loss_in_iteration[i+2] = func_minimize(loss_function,fw_tmp,f_opt,f_cont,p_tmp,n_offspring)
                parameters_in_iteration[i+2,:] = copy(p_tmp)

                print("=== Restarted, the proportion of iterations executed: ")
                print(i/n_iter_restart)
                print(". Loss: ")
                println(loss_in_iteration[i+2])
                
                if(loss_in_iteration[i+2]<min_loss)
                    best_solution = copy(p_tmp)
                    population_of_best_solution = copy(pop_tmp)
                    min_loss = copy(loss_in_iteration[i+2])
                    println("    ###### Restart provided better fit.")
                end     
            
            end
        end
        ### Construct a FittingResults struct
        sub_results = FittingResults(loss_in_iteration[1:k*saving_interval + saving_interval+2],
                parameters_in_iteration[1:k*saving_interval + saving_interval+2,:],
                best_solution, 
                population_of_best_solution,
                [0],
                "_",
                n_iter_restart,
                n_offspring,
                f_opt)

        # write the execution time to file
        FoodWebUtils.write_ex_time_to_file(sub_execution_time,            
            sub_results_to*string(k*saving_interval + saving_interval)*"_restarts/execution_time")
    
        # write the untransformed (in R^n) parameters to file that minimize the loss function
        FoodWebUtils.write_parameters_to_file(sub_results,
            sub_results_to*string(k*saving_interval + saving_interval)*"_restarts/EO_parameter_estimates")
                            
        FoodWebUtils.write_losses_to_file(sub_results,
            sub_results_to*string(k*saving_interval + saving_interval)*"_restarts/EO_loss")
    
        FoodWebPlots.plot_losses(sub_results,            
            sub_results_to*string(k*saving_interval + saving_interval)*"_restarts/EO_loss")

    end
    return best_solution, parameters_in_iteration, population_of_best_solution, min_loss, loss_in_iteration
end

# FIT A FOODWEB MODEL 
# fw_init: initial FoodWeb struct, obtain initial parameter values from this
# n_iter_restart: number of iterations in fitting
# saving_interval: for saving subresults during iteration
# rho: for variational Bayesian estimation, std of the variational priors, same for all parameters 
# kwargs: lower and upper bounds for the parameters
function fit_foodweb(method::String,training_data,fw_init,
    n_iter_restart,saving_interval,n_offspring,rho,sub_results_to;kwargs...)

    #### Set fitting options, give initial values for the parameters (untransformed)
    if(cmp(method,"bayes")==0)
        f_opt = FittingOptions.initialize_bayes_options(fw_init,rho,training_data)
        
        f_cont = FittingOptions.initialize_bayes_container(fw_init,f_opt)
        
        loss_function = LossFunctions.loss_BBP
    elseif(cmp(method,"MLE")==0)
        f_opt = FittingOptions.initialize_MLE_options(fw_init,training_data)
        
        f_cont = FittingOptions.initialize_OLS_container(fw_init)
        
        loss_function = LossFunctions.loss_MLE

    else        
        f_opt = FittingOptions.initialize_OLS_options(fw_init,training_data)
        
        f_cont = FittingOptions.initialize_OLS_container(fw_init)
        
        loss_function = LossFunctions.loss_OLS
    end

    fw_tmp = SetFoodWebParameters.copy_foodweb(fw_init)

    param_init = copy(f_opt.p_init)

    best_solution, solutions_in_iteration, population_of_best_solution, min_loss, losses = EO_minimize_with_restart(loss_function,fw_tmp,f_opt,f_cont,
                            param_init, n_iter_restart, saving_interval, n_offspring,sub_results_to)

    # for returning the results:
    param_final = best_solution 
    
    pop_final = population_of_best_solution

    param = solutions_in_iteration
    
    iterations_restart = n_iter_restart
    
    offspring = n_offspring

    ### Construct a FittingResults struct
    return FittingResults(losses,
    param,
    param_final, 
    pop_final,
    param_init,
    method,
    iterations_restart,
    offspring,
    f_opt)
end


end #module


