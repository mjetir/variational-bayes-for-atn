
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


module MethodValidation

using DifferentialEquations
using StatsPlots
using LinearAlgebra
using Random, Distributions, Statistics
using CSV, DataFrames
using DelimitedFiles
using BioEnergeticFoodWebs
using Evolutionary


include("fit_foodweb_parameters_EO.jl")
include("set_foodweb_parameters.jl")
include("foodweb_model_ODE.jl")
include("foodweb_utils.jl")
include("plot_results_for_foodwebs_public.jl")


# FIT SIMULATED FOODWEBS
# for method validation
function fit_noisy_data(data_folder_name,CV_suffix,
        n_iter_restart, saving_interval, n_offspring, 
        half_sat_ini, method, rho, sub_results_to)

    I = 0
    q = 0
    K = 0

    try
        I = readdlm(data_folder_name*"/I.txt",Int64)
        q = readdlm(data_folder_name*"/q.txt",Float64)
        K = readdlm(data_folder_name*"/K.txt",Float64)[1]
    catch
        println("No appropriate data files in the folder "*data_folder_name*"!")
        return 0
    end

    n_spec = size(I,1)

    ### Read training data
    training_data_with_noise_df = CSV.read(data_folder_name*"/training_data_"*CV_suffix*".txt",DataFrame)
    training_data_with_noise = transpose(Array(training_data_with_noise_df[:,2:end]))

    # set std to the sample standard deviation of the data
    std_set=zeros(n_spec)
    for i=1:n_spec
        std_set[i] = Statistics.std(training_data_with_noise[i,:])
    end

    #### Initialize a FoodWeb struct    
    fw_init = SetFoodWebParameters.initialize_generic_foodweb("data",
            I, ones(n_spec), copy(half_sat_ini), q, K, std_set)
    fw_init.time_grid = training_data_with_noise_df.Year

    results = FitFoodWebModel.fit_foodweb(method,
            training_data_with_noise,fw_init,
            n_iter_restart, saving_interval, n_offspring,
            rho, sub_results_to)

    return results
end


function fit_simulated_data_sets(data_ind,parent_folder_name,cv_rand, 
        n_iter_restart, saving_interval, n_offspring, half_sat_ini, method, rho)

    cd(parent_folder_name)

    data_folder_names = ["dataset"*string(i) for i=1:3]
    
    if(cv_rand==0.3)
        suffix="CV03"
    elseif(cv_rand==0.7)
        suffix="CV07"
    else 
        suffix=""
    end

    for i=data_ind
        
    
        if(cmp(method,"bayes")==0)
            
            results_to = method*"_fit_only_B0s_"*data_folder_names[i]*"_results_with_noise_"*suffix*"_"*string(n_offspring)*"offspring_"*string(n_iter_restart+1)*"_iterations"
            sub_results_to = method*"_fit_only_B0s_"*data_folder_names[i]*"_results_with_noise_"*suffix*"_"*string(n_offspring)*"offspring_after_"

            execution_time = @elapsed results = fit_noisy_data(
                data_folder_names[i], suffix, n_iter_restart, saving_interval, n_offspring, copy(half_sat_ini), method, copy(rho), sub_results_to)#;kwargs...)
    
            # write the execution time to file
            FoodWebUtils.write_ex_time_to_file(execution_time,            
                results_to*"/execution_time")

            # write the untransformed (in R^n) parameters to file that minimize the loss function
            FoodWebUtils.write_parameters_to_file(results,
                results_to*"/EO_parameter_estimates")
                        
            FoodWebUtils.write_losses_to_file(results,
                results_to*"/EO_loss")

            FoodWebPlots.plot_losses(results,            
                results_to*"/EO_loss")

        else

            results_to = method*"_fit_only_B0s_"*data_folder_names[i]*"_results_with_noise_"*suffix*"_"*string(n_offspring)*"offspring_"*string(n_iter_restart+1)*"_iterations"
            sub_results_to = method*"_fit_only_B0s_"*data_folder_names[i]*"_results_with_noise_"*suffix*"_"*string(n_offspring)*"offspring_after_"

            execution_time = @elapsed results = fit_noisy_data(
                data_folder_names[i], suffix, n_iter_restart, saving_interval, n_offspring, copy(half_sat_ini), method, copy(rho), sub_results_to)#;kwargs...)
    

            # write the execution time to file
            FoodWebUtils.write_ex_time_to_file(execution_time,            
                results_to*"/execution_time")

            FoodWebUtils.write_parameters_to_file(results,
                results_to*"/EO_parameter_estimates")
            
            FoodWebUtils.write_losses_to_file(results,
                results_to*"/EO_loss")

            FoodWebPlots.plot_losses(results,            
                results_to*"/EO_loss")

        end
    end

    cd("../")
    cd("../")
end

end # module



#MethodValidation.fit_simulated_data_sets([1],"ATN_simulated_webs/25_guilds_data",0.3,25,5,100,
#            0.5.*ones(25,25),"bayes",1.0)

#MethodValidation.fit_simulated_data_sets([1],"ATN_simulated_webs/25_guilds_data",0.3,50,5,100,
#            0.5.*ones(25,25),"OLS",1.0)
