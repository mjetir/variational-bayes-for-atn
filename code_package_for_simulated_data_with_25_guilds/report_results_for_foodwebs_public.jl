
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


# Functions for plotting
module FoodWebPlots

using StatsPlots
using Colors
using Random, Distributions
using Statistics
using CSV
using DataFrames
using DelimitedFiles

gr(dpi=600)

include("foodweb_utils.jl")
include("foodweb_model_ODE.jl")
include("set_foodweb_parameters.jl")
include("foodweb_fitting_options.jl")

my_green = RGB(0.6549, 0.77255, 0.09412)
my_pink = RGB(0.996,0.008,0.482)
my_gray = RGB(0.9216,0.9255,0.9412)

my_gray_1 = RGB(0.749,0.749,0.749)
my_gray_2 = RGB(0.498,0.498,0.498)
my_gray_3 = RGB(0.251,0.251,0.251)

my_turq = RGB(0,198/255,207/255)
my_purple = RGB(155/255,5/255,76/255)
my_red = RGB(242/255,9/255,4/255)
my_orange = RGB(254/255,105/255,2/255)
my_blue = RGB(41/255, 99/255, 162/255)

half_sat_ini = 0.5


# Plot a numerical solution of the foodweb model 
# sol: simulated foodweb
# fw: a foodweb struct
# used in simulation of synthetic foodwebs
function plot_model(sol,sol_ini2,sol_ini3,sol_ini4,sol_ini5,fw,folder_name)
    println("Plotting to "*folder_name)
    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    l = @layout [a ; b c]

    sol_arr = Array(sol)
    sol_arr_min=0
    sol_arr_max=0
    for i in 1:fw.nSpec
        sol_arr_max_ = maximum(sol_arr[i,1950:2000])
        sol_arr_min_ = minimum(sol_arr[i,1950:2000])

        p1=StatsPlots.plot(sol, lw=1,vars=(0,i), 
        linecolor = my_green,label="Dynamics to be fitted",title = "Overall dynamics")
        annotate!(p1,1000, sol_arr_min, "CV at the end: "*string(Statistics.std(sol_arr[i,1950:2000])/Statistics.mean(sol_arr[i,1950:2000])),8,
        fontfamily=:"Computer Modern")
        
        p2 = StatsPlots.plot(sol,lw=1,label="Dynamics to be fitted",vars=(0,i), linecolor = my_green,
        xlim=(0,50.0))

        StatsPlots.plot!(p2,sol_ini2,lw=2,title = "Monostability", label=:none,vars=(0,i), linecolor = :black,
        xlim=(0,50.0), fontfamily=:"Computer Modern")
        try
            StatsPlots.plot!(p2,sol_ini3,lw=1, legend=:topright,label="Different initial values",vars=(0,i), linecolor = :black,
            xlim=(0,50.0))
        catch
        end
        try
            StatsPlots.plot!(p2,sol_ini4,lw=1,label=:none,vars=(0,i), linecolor = :black,
            xlim=(0,50.0))
        catch
        end
        try
            StatsPlots.plot!(p2,sol_ini5,lw=1,label=:none,vars=(0,i), linecolor = :black,
            xlim=(0,50.0))
        catch
        end

        p4 = StatsPlots.plot(sol,lw=2,legend=:bottomright,label="Dynamics to be fitted",vars=(0,i),title="Stabilized dynamics", linecolor = my_green,
        xlim=(fw.tspan[2]-50.0,fw.tspan[2]),ylim=(sol_arr_min_,sol_arr_max_), fontfamily=:"Computer Modern")
        StatsPlots.plot!(p4,sol_simple,lw=2,legend=:bottomright,label="Simpler solver",vars=(0,i), linecolor =:red,
        xlim=(fw.tspan[2]-50.0,fw.tspan[2]),ylim=(sol_arr_min_,sol_arr_max_), fontfamily=:"Computer Modern")
    
        plot(p1, p2, p4, layout = l)
        savefig(folder_name*"/guild"*string(i)*".png")
    end
end



function compute_average_rho_for_parameters(parent_folder)

    CVs = ["CV03"]
    sets = ["1"]
    colors = [my_gray]

    data_names = ["TN4"]
    ndatas = length(data_names)
    conf_names = ["0.3"]
    nconf = length(conf_names)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    average_rhos_25 = zeros(nconf, ndatas)

    for i=1
        CV = CVs[i]
            for j=1
                set = sets[j]

                guild = "25"

                I = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/I.txt",Int64)
                fr_half_sat_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/B0.txt",Float64)
                fr_shape_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/q.txt",Float64)
                K_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/K.txt",Float64)

                I_ind = findall(x->(x .> 0),I)
                n_spec = size(I,1)
                nLinks = length(I_ind)
                
                param_bayes = readdlm(parent_folder*"/"*guild*"_guilds_data/bayes_fit_only_B0s_dataset1_results_with_noise_CV03_100offspring_after_25_restarts/EO_parameter_estimates/parameter_estimates.txt",Float64)
                
                rhos = [FoodWebUtils.rho_transform(param_bayes[nLinks+i])[1] 
                       for i=1:nLinks]

                average_rhos_25[i,j] = Statistics.mean(rhos)


            end
    end

    println(average_rhos_25)

end



function compute_parameter_estimate_errors(parent_folder,half_sat_ini)

    CVs = ["CV03"]
    CVs_num = [0.3]

    sets = ["1"]
    colors = [my_gray]

    data_names = ["TN4"]
    ndatas = length(data_names)
    conf_names = ["0.3"]
    nconf = length(conf_names)

    guilds = ["25"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    errors_ini_half_sat = zeros(nconf,ndatas,nguilds)
    errors_ini_half_sat_rel = zeros(nconf,ndatas,nguilds)
    
    errors_bayes_half_sat = zeros(nconf,ndatas,nguilds)
    errors_bayes_half_sat_rel = zeros(nconf,ndatas,nguilds)

    errors_OLS_half_sat = zeros(nconf,ndatas,nguilds)
    errors_OLS_half_sat_rel = zeros(nconf,ndatas,nguilds)


    for i=1
        CV = CVs[i]
            for j=1
                set = sets[j]

                for k in 1:1
                    
                    guild = guilds[k]

                    I = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/I.txt",Int64)
                    fr_half_sat_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/B0.txt",Float64)
                    fr_shape_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/q.txt",Float64)
                    K_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/K.txt",Float64)

                    I_ind = findall(x->(x .> 0),I)
                    n_spec = size(I,1)
                    nLinks = length(I_ind)

                    ### Initial guess

                    fr_half_sat_ini = zeros(n_spec,n_spec)
                    fr_half_sat_ini[I_ind] .= half_sat_ini

                    errors_ini_half_sat[i,j,k] = sum(abs,fr_half_sat_true[I_ind].-fr_half_sat_ini[I_ind])/nLinks
                    errors_ini_half_sat_rel[i,j,k] = sum(abs.(fr_half_sat_true[I_ind].-fr_half_sat_ini[I_ind])./fr_half_sat_true[I_ind])/nLinks

                    training_data_with_noise_df = CSV.read(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                    training_data = transpose(Array(training_data_with_noise_df[:,2:end])) 
    
                    fw_init = SetFoodWebParameters.initialize_generic_foodweb("init",I,ones(n_spec),
                        fr_half_sat_ini,fr_shape_true,K_true[1],
                        zeros(n_spec))

                    b_options = FittingOptions.initialize_bayes_options(fw_init,1.0,training_data)

                    ### Bayes

                    param_bayes = readdlm(parent_folder*"/"*guild*"_guilds_data/bayes_fit_only_B0s_dataset1_results_with_noise_CV03_100offspring_after_25_restarts/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    fr_half_sat_pred = zeros(n_spec,n_spec)

                    fr_half_sat_pred[I_ind] = FoodWebUtils.activation1(param_bayes[1:nLinks],b_options.param_min[1:nLinks],b_options.param_max[1:nLinks])
                    

                    errors_bayes_half_sat[i,j,k] = sum(abs,fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])/nLinks
                    errors_bayes_half_sat_rel[i,j,k] = sum(abs.(fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])./fr_half_sat_true[I_ind])/nLinks
 
                    ### OLS

                    param_OLS = readdlm(parent_folder*"/"*guild*"_guilds_data/OLS_fit_only_B0s_dataset1_results_with_noise_CV03_100offspring_51_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    fr_half_sat_pred[I_ind] = param_OLS[1:nLinks]

                    errors_OLS_half_sat[i,j,k] = sum(abs,fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])/nLinks
                    errors_OLS_half_sat_rel[i,j,k] = sum(abs.(fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])./fr_half_sat_true[I_ind])/nLinks

                end

            end
    end

    println("Half-sat. absolute errors:")
    println("Initial:")
    println(errors_ini_half_sat[:,:,1])
    println("Bayes:")
    println(errors_bayes_half_sat[:,:,1])
    println("OLS:")
    println(errors_OLS_half_sat[:,:,1])

    println()

    println("Half-sat. squared errors:")
    println("Initial:")
    println(errors_ini_half_sat_squared[:,:,1])
    println("Bayes:")
    println(errors_bayes_half_sat_squared[:,:,1])
    println("OLS:")
    println(errors_OLS_half_sat_squared[:,:,1])

    println()

    println("Half-sat. relative errors:")
    println("Initial:")
    println(errors_ini_half_sat_rel[:,:,1])
    println("Bayes:")
    println(errors_bayes_half_sat_rel[:,:,1])
    println("OLS:")
    println(errors_OLS_half_sat_rel[:,:,1])

end


function compute_posterior_and_prior_predictive_errors(parent_folder,half_sat_ini)

    CVs = ["CV03"]
    sets = ["1"]
    colors = [my_gray]

    data_names = ["TN4"]
    ndatas = length(data_names)
    conf_names = ["0.3"]
    nconf = length(conf_names)

    guilds = ["25"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    errors_ini = zeros(nconf,ndatas,nguilds)
    errors_expl_ini = zeros(nconf,ndatas,nguilds)

    errors_bayes = zeros(nconf,ndatas,nguilds)
    errors_bayes_expl = zeros(nconf,ndatas,nguilds)

    errors_OLS = zeros(nconf,ndatas,nguilds)
    errors_OLS_expl = zeros(nconf,ndatas,nguilds)

    prop_errors_ini = zeros(nconf,ndatas,nguilds)
    prop_errors_ini_expl = zeros(nconf,ndatas,nguilds)

    prop_errors_bayes = zeros(nconf,ndatas,nguilds)
    prop_errors_bayes_expl = zeros(nconf,ndatas,nguilds)

    prop_errors_OLS = zeros(nconf,ndatas,nguilds)
    prop_errors_OLS_expl = zeros(nconf,ndatas,nguilds)

    for i=1
        CV = CVs[i]
            for j=1
                set = sets[j]

                for k in 1:nguilds
                    
                    guild = guilds[k]
    
                    ### the true foodweb and dynamics

                    I = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/I.txt",Int64)
                    fr_half_sat_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/B0.txt",Float64)
                    fr_shape_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/q.txt",Float64)
                    K_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/K.txt",Float64)

                    I_ind = findall(x->(x .> 0),I)
                    n_spec = size(I,1)
                    nLinks = length(I_ind)

                    fw_true = SetFoodWebParameters.initialize_generic_foodweb("true",I,ones(n_spec),fr_half_sat_true,fr_shape_true,
                        K_true[1],zeros(n_spec))
                    true_abundance = FoodWebModel.foodweb_model(fw_true)
                    truth = zeros(n_spec,length(fw_true.time_grid))
                    for i in 1:length(fw_true.time_grid)
                        truth[:,i]=true_abundance[FoodWebUtils.closest_index(
                                true_abundance.t,fw_true.time_grid[i])] 
                    end

                    test_data_with_noise_df = CSV.read(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/test_data_"*CV*".txt",DataFrame)

                    fw_true_expl = SetFoodWebParameters.copy_foodweb(fw_true)
                    fw_true_expl.tspan = (0.0,2030.0)
                    fw_true_expl.time_grid = test_data_with_noise_df.Year
                
                    true_abundance_expl = FoodWebModel.foodweb_model_exploited(fw_true_expl)
                    truth_expl = zeros(n_spec,length(fw_true_expl.time_grid))
                    for i in 1:length(fw_true_expl.time_grid)
                        truth_expl[:,i]=true_abundance_expl[FoodWebUtils.closest_index(
                                true_abundance_expl.t,fw_true_expl.time_grid[i])] 
                    end

                    ### initial guess about the dynamics

                    fw_ini = SetFoodWebParameters.copy_foodweb(fw_true)
                    fw_ini.fr_half_sat[fw_true.I_ind] .= half_sat_ini

                    training_data_with_noise_df = CSV.read(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/training_data_"*CVs[i]*".txt",DataFrame)
                    training_data_with_noise = transpose(Array(training_data_with_noise_df[:,2:end])) 

                    fw_ini.std = [Statistics.std(training_data_with_noise[i,:]) for i=1:n_spec]

                    ini_abundance = FoodWebModel.foodweb_model_extinct(fw_ini)
                    ini_ab = zeros(n_spec,length(fw_true.time_grid))
                    for i in 1:length(fw_true.time_grid)
                        ini_ab[:,i]= ini_abundance[FoodWebUtils.closest_index(
                                ini_abundance.t,fw_true.time_grid[i])] 
                    end

                    fw_ini_expl = SetFoodWebParameters.copy_foodweb(fw_true_expl)
                    fw_ini_expl.fr_half_sat[fw_true.I_ind] .= half_sat_ini
                
                    ini_abundance_expl = FoodWebModel.foodweb_model_exploited(fw_ini_expl)
                    ini_ab_expl = zeros(n_spec,length(fw_true_expl.time_grid))
                    for i in 1:length(fw_true_expl.time_grid)
                        ini_ab_expl[:,i]=ini_abundance_expl[FoodWebUtils.closest_index(
                                ini_abundance_expl.t,fw_true_expl.time_grid[i])] 
                    end

                    errors_ini[i,j,k] = sum(abs,truth-ini_ab)/length(truth)
                    errors_expl_ini[i,j,k] = sum(abs,truth_expl-ini_ab_expl)/length(truth_expl)

                    prop_errors_ini[i,j,k] = sum(abs.(truth .- ini_ab)./truth)/length(truth)
                    prop_errors_ini_expl[i,j,k] = sum(abs.(truth_expl .- ini_ab_expl)./truth_expl)/length(truth_expl)

                    ### predicted abundances after fitting

                    # Bayes
                    param_bayes = readdlm(parent_folder*"/"*guild*"_guilds_data/bayes_fit_only_B0s_dataset1_results_with_noise_CV03_100offspring_after_25_restarts/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    b_opt = FittingOptions.initialize_bayes_options(fw_ini,1.0,training_data_with_noise)
        
                    fr_half_sat_pred = zeros(n_spec,n_spec)
                    fr_half_sat_pred[I_ind] = FoodWebUtils.activation1(param_bayes[1:nLinks],b_opt.param_min[1:nLinks],b_opt.param_max[1:nLinks])
                    
                    fw_pred = SetFoodWebParameters.initialize_generic_foodweb("prediction",I,ones(n_spec),
                        fr_half_sat_pred,fr_shape_true,K_true[1],zeros(n_spec))
                    predicted_abundance = FoodWebModel.foodweb_model_extinct(fw_pred)
                    prediction = zeros(n_spec,length(fw_true.time_grid))
                    for i in 1:length(fw_true.time_grid)
                        prediction[:,i]=predicted_abundance[FoodWebUtils.closest_index(
                                predicted_abundance.t,fw_true.time_grid[i])] 
                    end

                    errors_bayes[i,j,k] = sum(abs,truth-prediction)/length(truth)
                    prop_errors_bayes[i,j,k] = sum(abs.(truth .- prediction)./truth)/length(truth)

                    fw_pred_expl =SetFoodWebParameters.copy_foodweb(fw_pred)
                    fw_pred_expl.tspan = fw_true_expl.tspan
                    fw_pred_expl.time_grid = fw_true_expl.time_grid

                    predicted_abundance_expl = FoodWebModel.foodweb_model_exploited(fw_pred_expl)
                    prediction_expl = zeros(n_spec,length(fw_true_expl.time_grid))
                    for i in 1:length(fw_true_expl.time_grid)
                        prediction_expl[:,i]=predicted_abundance_expl[FoodWebUtils.closest_index(
                                predicted_abundance_expl.t,fw_true_expl.time_grid[i])] 
                    end
            
                    errors_bayes_expl[i,j,k] = sum(abs,truth_expl-prediction_expl)/length(truth_expl)
                    prop_errors_bayes_expl[i,j,k] = sum(abs.(truth_expl .- prediction_expl)./truth_expl)/length(truth_expl)

                    # OLS
                    param_OLS = readdlm(parent_folder*"/"*guild*"_guilds_data/OLS_fit_only_B0s_dataset1_results_with_noise_CV03_100offspring_51_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    fr_half_sat_pred[I_ind] = param_OLS[1:nLinks]
                    
                    fw_pred = SetFoodWebParameters.initialize_generic_foodweb("OLS prediction",I,ones(n_spec),
                        fr_half_sat_pred,fr_shape_true,K_true[1],zeros(n_spec))
                    predicted_abundance = FoodWebModel.foodweb_model_extinct(fw_pred)
                    prediction = zeros(n_spec,length(fw_true.time_grid))
                    for i in 1:length(fw_true.time_grid)
                        prediction[:,i]=predicted_abundance[FoodWebUtils.closest_index(
                                predicted_abundance.t,fw_true.time_grid[i])] 
                    end
            
                    errors_OLS[i,j,k] = sum(abs,truth-prediction)/length(truth)
                    prop_errors_OLS[i,j,k] = sum(abs.(truth .- prediction)./truth)/length(truth)

                    fw_pred_expl = SetFoodWebParameters.copy_foodweb(fw_pred)
                    fw_pred_expl.tspan = fw_true_expl.tspan
                    fw_pred_expl.time_grid = fw_true_expl.time_grid

                    predicted_abundance_expl = FoodWebModel.foodweb_model_exploited(fw_pred_expl)
                    prediction_expl = zeros(n_spec,length(fw_true_expl.time_grid))
                    for i in 1:length(fw_true_expl.time_grid)
                        prediction_expl[:,i]=predicted_abundance_expl[FoodWebUtils.closest_index(
                                predicted_abundance_expl.t,fw_true_expl.time_grid[i])] 
                    end
            
                    errors_OLS_expl[i,j,k] = sum(abs,truth_expl-prediction_expl)/length(truth_expl)
                    prop_errors_OLS_expl[i,j,k] = sum(abs.(truth_expl .- prediction_expl)./truth_expl)/length(truth_expl)

                end

            end
    end
    
    println("Initial errors, training set:")
    println(errors_ini)

    println("Initial errors, test set:")
    println(errors_expl_ini)

    println("Bayes errors, training set:")
    println(errors_bayes)
    
    println("Bayes errors, test set:")
    println(errors_bayes_expl)
    
    println("OLS errors, training set:")
    println(errors_OLS)
    
    println("OLS errors, test set:")
    println(errors_OLS_expl)

    println()


    println("Relative initial errors, training set:")
    println(prop_errors_ini)
        
    println("Relative initial errors, test set:")
    println(prop_errors_ini_expl)
    
    println("Relative Bayes errors, training set:")
    println(prop_errors_bayes)
    
    println("Relative Bayes errors, test set:")
    println(prop_errors_bayes_expl)

    println("Relative OLS errors, training set:")
    println(prop_errors_OLS)
    
    println("Relative OLS errors, test set:")
    println(prop_errors_OLS_expl)


end


function compute_90CPI_coverage_for_parameters(parent_folder)

    CVs_num = [0.3]
    CVs = ["CV03"]
    sets = ["1"]
    colors = [my_gray]

    data_names = ["TN4"]
    ndatas = length(data_names)
    conf_names = ["0.3"]
    nconf = length(conf_names)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    coverage_bayes_25_ = zeros(nconf, ndatas)

    for i in 1
        CV = CVs[i]
            for j=1
                set = sets[j]

                guild = "25"

                I = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/I.txt",Int64)
                fr_half_sat_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/B0.txt",Float64)
                fr_shape_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/q.txt",Float64)
                K_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/K.txt",Float64)

                I_ind = findall(x->(x .> 0),I)
                n_spec = size(I,1)
                nLinks = length(I_ind)
                
                param_bayes = readdlm(parent_folder*"/"*guild*"_guilds_data/bayes_fit_only_B0s_dataset1_results_with_noise_CV03_100offspring_after_25_restarts/EO_parameter_estimates/parameter_estimates.txt",Float64)
                
                I_ind = findall(x->(x .> 0),I)
                n_spec = size(I,1)
                nLinks = length(I_ind)

                ######################
                fw_true = SetFoodWebParameters.initialize_generic_foodweb("true",I,ones(n_spec),
                fr_half_sat_true,fr_shape_true,K_true[1],
                zeros(n_spec))
        
                ground_truth =  FoodWebModel.foodweb_model(fw_true)
                ground_truth_array = Array(ground_truth)

                ### Read training data 

                training_data_with_noise_df = CSV.read(parent_folder*"/10_guilds_data/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                training_data_with_noise = transpose(Array(training_data_with_noise_df[:,2:end])) 

                b_opt = FittingOptions.initialize_bayes_options(fw_true,1.0,training_data_with_noise)
                
                
                post_dist = [Normal(param_bayes[i], 
                    FoodWebUtils.rho_transform(param_bayes[nLinks+i])[1]) 
                       for i=1:nLinks]
                lower_q = [quantile.(post_dist[i],0.05) for i in 1:nLinks]
                upper_q = [quantile.(post_dist[i],0.95) for i in 1:nLinks]

                fr_half_sat_true_inv = FoodWebUtils.inverse_activation1(fr_half_sat_true[I_ind], b_opt.param_min[1:nLinks], b_opt.param_max[1:nLinks])
                coverage_bayes_25_[i,j] = sum((fr_half_sat_true_inv.<upper_q[1:nLinks]) .& (fr_half_sat_true_inv.>lower_q[1:nLinks]))/nLinks

            end
    end
    println(coverage_bayes_25_)

end


function plot_90_CPI_coverage_for_posterior_predictives(parent_folder,n_samples)

    CVs = ["CV03"]
    sets = ["1"]
    colors = [my_gray]

    data_names = ["TN4"]
    ndatas = length(data_names)
    conf_names = ["0.3"]
    nconf = length(conf_names)
    guilds = ["25"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    coverage = zeros(nconf,ndatas,nguilds)
    coverage_expl = zeros(nconf,ndatas,nguilds)

    coverage_total = zeros(nconf,ndatas,nguilds)
    coverage_expl_total = zeros(nconf,ndatas,nguilds)

    Random.seed!(3758743) #seed 1: Random.seed!(58743)

    for i=1
        CV = CVs[i]
            for j=1
                set = sets[j]
                for k in 1:nguilds
                    
                guild = guilds[k]

                I = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/I.txt",Int64)
                fr_half_sat_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/B0.txt",Float64)
                fr_shape_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/q.txt",Float64)
                K_true = readdlm(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/K.txt",Float64)

                I_ind = findall(x->(x .> 0),I)
                n_spec = size(I,1)
                nLinks = length(I_ind)
                
                fw_true = SetFoodWebParameters.initialize_generic_foodweb("true",I,ones(n_spec),fr_half_sat_true,fr_shape_true,K_true[1],zeros(n_spec))
                true_abundance = FoodWebModel.foodweb_model_extinct(fw_true)
                truth = zeros(n_spec,length(fw_true.time_grid))
                for i in 1:length(fw_true.time_grid)
                    truth[:,i]=true_abundance[FoodWebUtils.closest_index(
                            true_abundance.t,fw_true.time_grid[i])] 
                end

                training_data_with_noise_df = CSV.read(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                training_data = transpose(Array(training_data_with_noise_df[:,2:end])) 

                test_data_with_noise_df = CSV.read(parent_folder*"/"*guild*"_guilds_data/dataset"*set*"/test_data_"*CV*".txt",DataFrame)
                test_data = transpose(Array(test_data_with_noise_df[:,2:end])) 

                fw_true_expl = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_true_expl.tspan = (0.0,2030.0)
                fw_true_expl.time_grid = test_data_with_noise_df.Year
                
                true_abundance_expl = FoodWebModel.foodweb_model_exploited(fw_true_expl)
                truth_expl = zeros(n_spec,length(fw_true_expl.time_grid))
                for i in 1:length(fw_true_expl.time_grid)
                    truth_expl[:,i]=true_abundance_expl[FoodWebUtils.closest_index(
                            true_abundance_expl.t,fw_true_expl.time_grid[i])] 
                end

                param_bayes = readdlm(parent_folder*"/"*guild*"_guilds_data/bayes_fit_only_B0s_dataset1_results_with_noise_CV03_100offspring_after_25_restarts/EO_parameter_estimates/parameter_estimates.txt",Float64)

                post_dist = [Normal(param_bayes[i], 
                    FoodWebUtils.rho_transform(param_bayes[nLinks+i])[1]) 
                    for i=1:nLinks]

                sampled_param = zeros(nLinks,n_samples)
                for i=1:nLinks
                    sampled_param[i,:]=rand(post_dist[i],n_samples)
                end

                # Bayes options ---> bounds for parameters
                fw_initial = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_initial.fr_half_sat[fw_true.I_ind].=half_sat_ini
                fw_initial.std = [Statistics.std(training_data[i,:]) for i=1:n_spec]

                b_opt = FittingOptions.initialize_bayes_options(fw_initial,1.0,training_data)
                
                transformed_param = FoodWebUtils.activation1(sampled_param,repeat(b_opt.param_min[1:nLinks],1,n_samples),
                    repeat(b_opt.param_max[1:nLinks],1,n_samples))  
                
                post_preds=zeros(n_spec,length(fw_true.time_grid),n_samples)
                post_preds_expl=zeros(n_spec,length(fw_true_expl.time_grid),n_samples)

                post_preds_total=zeros(n_spec,length(fw_true.time_grid),n_samples)
                post_preds_expl_total=zeros(n_spec,length(fw_true_expl.time_grid),n_samples)

                fw_tmp = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_tmp.std = copy(fw_initial.std)

                fw_tmp_expl = SetFoodWebParameters.copy_foodweb(fw_true_expl)
                
                prob_post = 0
                prediction = zeros(n_spec,length(fw_true.time_grid))
                prob_post_expl = 0
                prediction_expl = zeros(n_spec,length(fw_true_expl.time_grid))

                cv_tmp = zeros(n_spec)

                for i=1:n_samples
                        fw_tmp.fr_half_sat[fw_tmp.I_ind] = transformed_param[1:nLinks,i]

                        fw_tmp_expl.fr_half_sat[fw_tmp.I_ind] = transformed_param[1:nLinks,i]
                                    
                        prob_post = FoodWebModel.foodweb_model_extinct(fw_tmp)
                        prob_post_expl = FoodWebModel.foodweb_model_exploited(fw_tmp_expl)

                        # can be approximated in different ways
                        for i in 1:n_spec
                            greater_than_zero = prob_post[i,:].> 0.0 # ignore those years when biomass is zero
                            cv_tmp[i] = Statistics.mean(fw_tmp.std[i]./prob_post[i,greater_than_zero])
                        end

                        if maximum(prob_post.t) >= fw_true.tspan[end]
                            for i in 1:length(fw_true.time_grid)
                                prediction[:,i]=prob_post[FoodWebUtils.closest_index(prob_post.t,fw_true.time_grid[i])] 
                            end
                        end
                        if(size(prediction)==size(post_preds[:,:,1]))                
                                post_preds[:,:,i] = prediction

                                ### Normal noise:
                                std_dist = MvNormal(zeros(fw_true.nSpec),fw_tmp.std)
                                post_preds_total[:,:,i] = prediction .+ rand(std_dist,length(fw_true.time_grid))
                        end    

                        if maximum(prob_post_expl.t) >= fw_true_expl.tspan[end]
                        for i in 1:length(fw_true_expl.time_grid)
                            prediction_expl[:,i]=prob_post_expl[FoodWebUtils.closest_index(prob_post_expl.t,
                            fw_true_expl.time_grid[i])] 
                        end
                        end
                        if(size(prediction_expl)==size(post_preds_expl[:,:,1]))                
                                post_preds_expl[:,:,i] = prediction_expl
                                                                
                                ### Normal noise:
                                cv_dist = MvNormal(zeros(fw_true.nSpec),cv_tmp)
                                noise = rand(cv_dist,length(fw_true_expl.time_grid))
                                post_preds_expl_total[:,:,i] = (1.0 .+ noise) .* prediction_expl                                
                        end

                end
                lower_quantiles = copy(prediction)
                upper_quantiles = copy(prediction)
                lower_quantiles_expl = copy(prediction_expl)
                upper_quantiles_expl = copy(prediction_expl)

                lower_quantiles_total = copy(prediction)
                upper_quantiles_total = copy(prediction)
                lower_quantiles_expl_total = copy(prediction_expl)
                upper_quantiles_expl_total = copy(prediction_expl)

                for i in 1:n_spec
                        for j in 1:length(fw_true.time_grid)
                            lower_quantiles[i,j] = quantile(post_preds[i,j,:],0.05)
                            upper_quantiles[i,j] = quantile(post_preds[i,j,:],0.95)

                            lower_quantiles_total[i,j] = quantile(post_preds_total[i,j,:],0.05)
                            upper_quantiles_total[i,j] = quantile(post_preds_total[i,j,:],0.95)
                        end
                        for j in 1:length(fw_true_expl.time_grid)
                            lower_quantiles_expl[i,j] = quantile(post_preds_expl[i,j,:],0.05)
                            upper_quantiles_expl[i,j] = quantile(post_preds_expl[i,j,:],0.95)

                            lower_quantiles_expl_total[i,j] = quantile(post_preds_expl_total[i,j,:],0.05)
                            upper_quantiles_expl_total[i,j] = quantile(post_preds_expl_total[i,j,:],0.95)

                        end
                end   
                
                coverage[i,j,k] = sum((truth.<upper_quantiles) .& (truth.>lower_quantiles))/length(truth)
                coverage_expl[i,j,k] = sum((truth_expl.<upper_quantiles_expl) .& 
                        (truth_expl.>lower_quantiles_expl))/length(truth_expl)

                coverage_total[i,j,k] = sum((training_data.<upper_quantiles_total) .& (training_data.>lower_quantiles_total))/length(training_data)
                coverage_expl_total[i,j,k] = sum((test_data.<upper_quantiles_expl_total) .& 
                                (test_data.>lower_quantiles_expl_total))/length(test_data)
                end                 
            end 
    end     

    println("90 % CPI coverage, ATN dynamics, training:")
    println(coverage[:,:,1])

    println("90 % CPI coverage, ATN dynamics, test:")
    println(coverage_expl[:,:,1])

    println("90 % CPI coverage, total abundance, training:")
    println(coverage_total[:,:,1])

    println("90 % CPI coverage, total abundance, test:")
    println(coverage_expl_total[:,:,1])

end




end # module


