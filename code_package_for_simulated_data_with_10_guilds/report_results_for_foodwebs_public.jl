
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

high_ab_threshold = 0.01
high_ab_threshold_2 = 0.05

half_sat_ini = 0.5
shape_ini = 0.65


# Plot a numerical solution of the foodweb model 
# sol*: simulated foodweb
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
        linecolor = my_green,label="Dynamics to be fitted")
        annotate!(p1,1500, sol_arr_min, "CV at the end: "*string(Statistics.std(sol_arr[i,1950:2000])/Statistics.mean(sol_arr[i,1950:2000])),8,
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
    
        plot(p1, p2, p4, layout = l)
        savefig(folder_name*"/guild"*string(i)*".png")
    end
end

### FOR PLOTTING RESULTS OF MODEL FITTING ###
#############################################

# Plot the value of the loss function in iteration
function plot_losses(f_res,folder_name)
    println("Plotting to "*folder_name)
    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    l = @layout [a ; b]
    p1 = StatsPlots.plot(1:length(f_res.loss),f_res.loss,lw=2,color=:black,xlabel="Iteration",
     ylabel="Loss", label = "", xticks = 0:5:length(f_res.loss)-1)
    p2 = StatsPlots.plot(30:length(f_res.loss),f_res.loss[30:end],lw=2,color=:black,xlabel="Iteration",
    ylabel="Loss", label = "", xticks = 30:5:length(f_res.loss))
    plot(p1, p2, layout = l)
    savefig(folder_name*"/loss_in_iteration.png")
end

# Plot the number of the iteration round at which the minimum of the loss function is obtained
function plot_iterations_for_minimum_loss(parent_folder)

    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]

    data_names = ["TN1", "TN2", "TN3"]
    ndatas = length(data_names)
    conf_names = ["0.3", "0.7"]
    nconf = length(conf_names)

    guilds = ["10"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    iterations_bayes = zeros(nconf,ndatas,nguilds)
    iterations_OLS = zeros(nconf,ndatas,nguilds)

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]

                for k in 1:nguilds
                    guild = guilds[k]
                    
                    losses_ = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_loss/losses.txt",Float64)

                    println(findmin(losses_)[2][1])
                    iterations_bayes[i,j,k] = findmin(losses_)[2][1]-1
    
                    losses_ = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations_OLS/EO_loss/losses.txt",Float64)
                    
                    iterations_OLS[i,j,k] = findmin(losses_)[2][1]-1

                end

            end
    end
    
    println(iterations_bayes[:,:,1])
    println(iterations_OLS[:,:,1])

    p1 = groupedbar(nam, iterations_bayes[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Iterations for minimum loss",
        title = "VI", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,50), color = [my_gray_1 my_gray_2 my_gray_3],fontfamily=:"Modern Computer")
    p2 = groupedbar(nam, iterations_OLS[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Iterations for minimum loss",
        title = "OLS", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:bottomleft, ylim=(0,50), color = [my_gray_1 my_gray_2 my_gray_3])

    l = @layout [a ; b]    
    plot(p1, p2, layout = l, size=(500,500))

    savefig(parent_folder*"/iterations_for_min_loss.png")
    savefig(parent_folder*"/iterations_for_min_loss.svg")

end

# Plot the time the execution of the training code took
function plot_execution_times(parent_folder)

    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]

    data_names = ["TN1", "TN2", "TN3"]
    ndatas = length(data_names)
    conf_names = ["0.3", "0.7"]
    nconf = length(conf_names)

    guilds = ["10"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    ex_time_bayes = zeros(nconf,ndatas,nguilds)
    ex_time_OLS = zeros(nconf,ndatas,nguilds)

    ex_time_bayes_50_offspring = zeros(nconf,ndatas,nguilds)
    ex_time_OLS_50_offspring = zeros(nconf,ndatas,nguilds)

    ex_time_bayes_200_offspring = zeros(nconf,ndatas,nguilds)
    ex_time_OLS_200_offspring = zeros(nconf,ndatas,nguilds)

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]

                for k in 1:nguilds
                    guild = guilds[k]
                    
                    ex_time_bayes[i,j,k] = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/execution_time/execution_time.txt",Float64)[1]/3600.0
    
                    ex_time_OLS[i,j,k] = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations_OLS/execution_time/execution_time.txt",Float64)[1]/3600.0

                    ex_time_bayes_50_offspring[i,j,k] = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_50offspring_50_iterations/execution_time/execution_time.txt",Float64)[1]/3600.0
    
                    ex_time_OLS_50_offspring[i,j,k] = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_50offspring_50_iterations_OLS/execution_time/execution_time.txt",Float64)[1]/3600.0

                    ex_time_bayes_200_offspring[i,j,k] = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_200offspring_100_iterations/execution_time/execution_time.txt",Float64)[1]/3600.0
    
                    ex_time_OLS_200_offspring[i,j,k] = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_200offspring_100_iterations_OLS/execution_time/execution_time.txt",Float64)[1]/3600.0

                end

            end
    end
    
    println("Execution times:")
    println(ex_time_bayes[:,:,1])
    println(ex_time_OLS[:,:,1])

    p1 = groupedbar(nam, ex_time_bayes[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Execution time [h]",
        title = "VI", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, color = [my_gray_1 my_gray_2 my_gray_3],fontfamily=:"Modern Computer")
    p2 = groupedbar(nam, ex_time_OLS[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Execution time [h]",
        title = "OLS", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:bottomleft, color = [my_gray_1 my_gray_2 my_gray_3])

    l = @layout [a ; b]    
    plot(p1, p2, layout = l, size=(500,500))

    savefig(parent_folder*"/execution_times.png")
    savefig(parent_folder*"/execution_times.svg")

    p1 = groupedbar(nam, ex_time_bayes_50_offspring[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Execution time [h]",
        title = "VI", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, color = [my_gray_1 my_gray_2 my_gray_3],fontfamily=:"Modern Computer")
    p2 = groupedbar(nam, ex_time_OLS_50_offspring[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Execution time [h]",
        title = "OLS", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:bottomleft, color = [my_gray_1 my_gray_2 my_gray_3])

    l = @layout [a ; b]    
    plot(p1, p2, layout = l, size=(500,500))

    savefig(parent_folder*"/execution_times_50_offspring.png")
    savefig(parent_folder*"/execution_times_50_offspring.svg")

    p1 = groupedbar(nam, ex_time_bayes_200_offspring[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Execution time [h]",
        title = "VI", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, color = [my_gray_1 my_gray_2 my_gray_3],fontfamily=:"Modern Computer")
    p2 = groupedbar(nam, ex_time_OLS_200_offspring[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Execution time [h]",
        title = "OLS", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:bottomleft, color = [my_gray_1 my_gray_2 my_gray_3])

    l = @layout [a ; b]    
    plot(p1, p2, layout = l, size=(500,500))

    savefig(parent_folder*"/execution_times_200_offspring.png")
    savefig(parent_folder*"/execution_times_200_offspring.svg")

end

# Plot predictions by VI and OLS
# n_samples: the number of samples from the variational posteriors to be used 
function plot_abundance_estimates(parent_folder,n_samples)
            
    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]                    
                
                guild = "10"

                Random.seed!(8758743)

                I = readdlm(parent_folder*"/dataset"*set*"/I.txt",Int64)
                fr_half_sat = readdlm(parent_folder*"/dataset"*set*"/B0.txt",Float64)
                fr_shape = readdlm(parent_folder*"/dataset"*set*"/q.txt",Float64)
            
                n_spec = length(I[1,:])

                ### Read training and test data 

                training_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                training_data_with_noise = transpose(Array(training_data_with_noise_df[:,2:end])) 

                test_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/test_data_"*CV*".txt",DataFrame)
                test_data_with_noise = transpose(Array(test_data_with_noise_df[:,2:end])) 
            
                # Estimated std

                std_in_data = zeros(n_spec)
                for i=1:n_spec
                    std_in_data[i] = Statistics.std(training_data_with_noise[i,:])
                end

                ### Generate the true foodweb and biomasses
                            
                fw_true = SetFoodWebParameters.initialize_generic_foodweb("true_foodweb",I,ones(n_spec),
                                    fr_half_sat,fr_shape,
                                    std_in_data)
                            
                ground_truth =  FoodWebModel.foodweb_model(fw_true)
                ground_truth_array = Array(ground_truth)
                    
                fw_true_exploited = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_true_exploited.tspan = (0.0,2030.0)
                fw_true_exploited.time_grid = test_data_with_noise_df.Year
                        
                ground_truth_exploited =  FoodWebModel.foodweb_model_exploited(fw_true_exploited)
                ground_truth_exploited_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                for i in 1:length(fw_true_exploited.time_grid)
                        ground_truth_exploited_array[:,i]=ground_truth_exploited[
                            FoodWebUtils.closest_index(ground_truth_exploited.t,fw_true_exploited.time_grid[i])] 
                end

                # Bayes options ---> bounds for parameters
                
                b_opt = FittingOptions.initialize_bayes_options(fw_true,1.0,training_data_with_noise)

                ### Read parameter estimates
                param_est = 0
                try
                    # try to read the untransformed (in R^n) parameter estimates
                    param_est = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)
                catch
                    println("No appropriate files in the given folder.")
                    return 0
                end

                ### Read OLS parameter estimates
                param_est_OLS = 0
                try
                    # try to read the untransformed (in R^n) parameter estimates
                    param_est_OLS = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations_OLS/EO_parameter_estimates/parameter_estimates.txt",Float64)
                catch
                    println("No appropriate files in the given folder.")
                    return 0
                end

                ### Predict abundances
                # Final guess, OLS

                fw_final_OLS = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_final_OLS.fr_half_sat[fw_final_OLS.I_ind] = param_est_OLS[1:fw_true.nLinks]
                fw_final_OLS.fr_shape[fw_final_OLS.I_ind] = param_est_OLS[fw_true.nLinks+1:2*fw_true.nLinks]
                
                model_predicted_OLS = FoodWebModel.foodweb_model(fw_final_OLS)
                model_predicted_OLS_array = Array(model_predicted_OLS)

                fw_expl_OLS = SetFoodWebParameters.copy_foodweb(fw_true_exploited)
                fw_expl_OLS.fr_half_sat[fw_expl_OLS.I_ind] = param_est_OLS[1:fw_true.nLinks]
                fw_expl_OLS.fr_shape[fw_final_OLS.I_ind] = param_est_OLS[fw_true.nLinks+1:2*fw_true.nLinks]
                
                model_expl_OLS = FoodWebModel.foodweb_model_exploited(fw_expl_OLS)
                model_expl_OLS_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                for i in 1:length(fw_true_exploited.time_grid)
                        model_expl_OLS_array[:,i]=model_expl_OLS[
                            FoodWebUtils.closest_index(model_expl_OLS.t,fw_true_exploited.time_grid[i])] 
                end

                # Final guess, Bayes
                param_final_transformed = FoodWebUtils.activation1(param_est[1:2*fw_true.nLinks+fw_true.nSpec],b_opt.param_min[1:2*fw_true.nLinks+fw_true.nSpec],
                    b_opt.param_max[1:2*fw_true.nLinks+fw_true.nSpec]) 

                fw_final = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_final.fr_half_sat[fw_final.I_ind] = param_final_transformed[1:fw_true.nLinks]
                fw_final.fr_shape[fw_final.I_ind] = param_final_transformed[fw_true.nLinks+1:2*fw_true.nLinks]

                model_predicted = FoodWebModel.foodweb_model(fw_final)
                model_predicted_array = Array(model_predicted)

                fw_expl = SetFoodWebParameters.copy_foodweb(fw_true_exploited)
                fw_expl.fr_half_sat[fw_expl.I_ind] = param_final_transformed[1:fw_true.nLinks]
                fw_expl.fr_shape[fw_final.I_ind] = param_final_transformed[fw_true.nLinks+1:2*fw_true.nLinks]
                
                model_expl = FoodWebModel.foodweb_model_exploited(fw_expl)
                model_expl_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                for i in 1:length(fw_true_exploited.time_grid)
                        model_expl_array[:,i]=model_expl[
                            FoodWebUtils.closest_index(model_expl.t,fw_true_exploited.time_grid[i])] 
                end

                # Posterior
                post_dist = [Normal(param_est[i], 
                    FoodWebUtils.rho_transform(param_est[2*fw_true.nLinks+fw_true.nSpec+i])[1]) 
                    for i=1:2*fw_true.nLinks+fw_true.nSpec]

                sampled_param = zeros(2*fw_true.nLinks+fw_true.nSpec,n_samples)
                for i=1:2*fw_true.nLinks+fw_true.nSpec
                    sampled_param[i,:]=rand(post_dist[i],n_samples)
                end

                transformed_param = FoodWebUtils.activation1(sampled_param,repeat(b_opt.param_min[1:2*fw_true.nLinks+fw_true.nSpec],1,n_samples),
                                repeat(b_opt.param_max[1:2*fw_true.nLinks+fw_true.nSpec],1,n_samples))  
    
                post_preds_mean=zeros(fw_true.nSpec,length(fw_true.time_grid),n_samples)
                post_preds=zeros(fw_true.nSpec,length(fw_true.time_grid),n_samples)

                post_preds_expl_mean=zeros(fw_true.nSpec,length(fw_true_exploited.time_grid),n_samples)
                post_preds_expl_total=zeros(fw_true.nSpec,length(fw_true_exploited.time_grid),n_samples)

                fw_tmp = SetFoodWebParameters.copy_foodweb(fw_true)
                
                fw_tmp_exploited = SetFoodWebParameters.copy_foodweb(fw_tmp)
                fw_tmp_exploited.tspan = fw_true_exploited.tspan
                fw_tmp_exploited.time_grid = fw_true_exploited.time_grid

                prob_post = 0
                prob_post_expl = 0
                prediction = zeros(fw_true.nSpec,length(fw_true.time_grid))
                prediction_expl = zeros(fw_true.nSpec,length(fw_true_exploited.time_grid))
                for i=1:n_samples
                    fw_tmp.fr_half_sat[fw_tmp.I_ind] = transformed_param[1:fw_true.nLinks,i]
                    fw_tmp.fr_shape[fw_tmp.I_ind] = transformed_param[fw_true.nLinks+1:2*fw_true.nLinks,i]
                    fw_tmp.std = transformed_param[2*fw_true.nLinks+1:2*fw_true.nLinks+fw_true.nSpec,i]

                    fw_tmp_exploited.fr_half_sat = fw_tmp.fr_half_sat
                    fw_tmp_exploited.fr_shape = fw_tmp.fr_shape
                
                    prob_post = FoodWebModel.foodweb_model_extinct(fw_tmp)
                    prob_post_expl = FoodWebModel.foodweb_model_exploited(fw_tmp_exploited)
                    if maximum(prob_post.t) >= fw_true.tspan[end]
                        for i in 1:length(fw_true.time_grid)
                            prediction[:,i]=prob_post[FoodWebUtils.closest_index(prob_post.t,fw_true.time_grid[i])] 
                        end
                    end
                    if(size(prediction)==size(post_preds[:,:,1]))
                            ### Normal noise:
                            std_dist = MvNormal(zeros(fw_true.nSpec),
                                    transformed_param[2*fw_true.nLinks+1:end,i])
                            post_preds[:,:,i] = prediction .+ rand(std_dist,length(fw_true.time_grid))

                            post_preds_mean[:,:,i] = prediction
                    end
                    
                    cv_tmp = zeros(fw_true.nSpec)

                    # can be approximated in different ways
                    for i in 1:fw_true.nSpec
                                greater_than_zero = prob_post[i,:].> 0.0 # ignore those years when biomass is zero
                                cv_tmp[i] = Statistics.mean(fw_tmp.std[i]./prob_post[i,greater_than_zero])
                    end

                    if maximum(prob_post_expl.t) >= fw_true_exploited.tspan[end]
                        for i in 1:length(fw_true_exploited.time_grid)
                            prediction_expl[:,i]=prob_post_expl[FoodWebUtils.closest_index(prob_post_expl.t,
                                fw_true_exploited.time_grid[i])] 
                        end
                    end
                    if(size(prediction_expl)==size(post_preds_expl_mean[:,:,1]))        
                        post_preds_expl_mean[:,:,i] = prediction_expl  

                        ### Normal noise:
                        cv_dist = MvNormal(zeros(fw_true.nSpec),cv_tmp)
                        noise = rand(cv_dist,length(fw_true_exploited.time_grid))
                        post_preds_expl_total[:,:,i] = (1.0 .+ noise) .* prediction_expl                                
                    end

                end

                medians_ = copy(prediction)
                lower_quantiles = copy(prediction)
                upper_quantiles = copy(prediction)

                medians_mean = copy(prediction)
                lower_quantiles_mean = copy(prediction)
                upper_quantiles_mean = copy(prediction)

                medians_expl = copy(prediction_expl)
                lower_quantiles_expl = copy(prediction_expl)
                upper_quantiles_expl = copy(prediction_expl)

                medians_mean_expl = copy(prediction_expl)
                lower_quantiles_mean_expl = copy(prediction_expl)
                upper_quantiles_mean_expl = copy(prediction_expl)

                for i in 1:fw_true.nSpec
                    for j in 1:length(fw_true.time_grid)
                        medians_[i,j] = quantile(post_preds[i,j,:],0.5)
                        lower_quantiles[i,j] = quantile(post_preds[i,j,:],0.05)
                        upper_quantiles[i,j] = quantile(post_preds[i,j,:],0.95)
                        
                        medians_mean[i,j] = quantile(post_preds_mean[i,j,:],0.5)
                        lower_quantiles_mean[i,j] = quantile(post_preds_mean[i,j,:],0.05)
                        upper_quantiles_mean[i,j] = quantile(post_preds_mean[i,j,:],0.95)
                    end
                    for j in 1:length(fw_true_exploited.time_grid)
                        medians_mean_expl[i,j] = quantile(post_preds_expl_mean[i,j,:],0.5)
                        lower_quantiles_mean_expl[i,j] = quantile(post_preds_expl_mean[i,j,:],0.05)
                        upper_quantiles_mean_expl[i,j] = quantile(post_preds_expl_mean[i,j,:],0.95)
                        
                        medians_expl[i,j] = quantile(post_preds_expl_total[i,j,:],0.5)
                        lower_quantiles_expl[i,j] = quantile(post_preds_expl_total[i,j,:],0.05)
                        upper_quantiles_expl[i,j] = quantile(post_preds_expl_total[i,j,:],0.95)
                        
                    end
                end

                plot_ = StatsPlots.plot(zeros(3),ones(3))
                plots_training = [plot_ for i=1:fw_true.nSpec]
                plots_test = [plot_ for i=1:fw_true.nSpec]
                for i in 1:fw_true.nSpec
                    if i==1 || i==6
                        title1 = "Training"
                        title2 = "Test"
                    else 
                        title1 = ""
                        title2 = ""
                    end   
                    legend1 = :none
                    legend2 = :none 
                    show_ = true

                    y_min = 0.8*minimum([model_predicted_array[i,:] 
                    training_data_with_noise[i,:] 
                    lower_quantiles[i,:] 
                    model_expl_array[i,:]
                    test_data_with_noise[i,:] 
                    lower_quantiles_mean_expl[i,:]
                    lower_quantiles_expl[i,:]])
                    y_max = 1.15*maximum([model_predicted_array[i,:] 
                    training_data_with_noise[i,:]
                    model_expl_array[i,:] 
                    test_data_with_noise[i,:]
                    upper_quantiles[i,:]
                    upper_quantiles_mean_expl[i,:]                
                    upper_quantiles_expl[i,:]])

                    p1=StatsPlots.plot(fw_true.time_grid, lower_quantiles[i,:], fillrange = upper_quantiles[i,:], 
                            fillalpha = 0.2, linealpha=0,
                            ylim=(y_min,y_max), 
                            xlim=(fw_true.time_grid[1],fw_true_exploited.time_grid[end]), color = my_turq, 
                            label = "Abundance: 90 % CPI of posterior", legend = legend1, title=title1, titleloc=:left, 
                            ylabel="G"*string(i)*" BM",xlabel="",
                            fontfamily="Computer Modern",
                            left_margin = 5Plots.mm, right_margin = 1Plots.mm, bottom_margin = 0.5Plots.mm,
                            grid = show_, showaxis = show_)
                    StatsPlots.plot!(p1,fw_true.time_grid, lower_quantiles_mean[i,:], fillrange = upper_quantiles_mean[i,:], 
                            fillalpha = 0.4, linealpha=0, color = my_turq, label = "Internal dyn.: 90 % CPI of posterior", legend = :none)
                    StatsPlots.plot!(p1,fw_true.time_grid,medians_mean[i,:], lw=3, label="Internal dyn.: posterior median",
                            color=my_blue)

                    StatsPlots.plot!(p1,model_predicted_OLS, lw=4, vars=(0,i),
                            color=my_green,label="Internal dyn.: final guess (OLS)",linestyle = :dot)
                    StatsPlots.plot!(p1,ground_truth, lw=2,vars=(0,i),
                            color=:black,label="Internal dyn.: true",linestyle = :solid)

                    StatsPlots.scatter!(p1,fw_true.time_grid,training_data_with_noise[i,:],legend=:none,label="Training data",color=my_gray_1,
                            xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),xlabel="")
                            
                    p2 = StatsPlots.plot(fw_true_exploited.time_grid, lower_quantiles_expl[i,:], 
                            fillrange = upper_quantiles_expl[i,:], 
                            fillalpha = 0.2, linealpha=0, color = my_purple, label = "Abundance with disruption: 90 % CPI of posterior", 
                            legend = legend2,
                            ylim=(y_min,y_max),
                            title=title2,titleloc=:left, 
                            ylabel="",xlabel="Year",
                            fontfamily=:"Computer Modern", 
                            grid = show_, showaxis = show_,
                            left_margin =3Plots.mm, right_margin = 5Plots.mm,bottom_margin = 0.5Plots.mm)
                    StatsPlots.plot!(p2,fw_true_exploited.time_grid, lower_quantiles_mean_expl[i,:],
                            fillrange = upper_quantiles_mean_expl[i,:], fillalpha =0.4, linealpha=0,  color = my_purple, label = "Disturbed internal dyn.: 90 % CPI of posterior",
                            legend = :none)
                    StatsPlots.plot!(p2,fw_true_exploited.time_grid,medians_mean_expl[i,:], lw=3, 
                            label="Disturbed internal dyn.: posterior median",
                            color=my_red)
                    
                    StatsPlots.plot!(p2,fw_true_exploited.time_grid,model_expl_OLS_array[i,:], lw=4,
                            color=my_orange,label="Disturbed internal dyn.: prediction (OLS)",linestyle = :dot)

                    StatsPlots.plot!(p2,fw_true_exploited.time_grid,ground_truth_exploited_array[i,:], lw=2,
                            color=:black,label="Disturbed internal dyn.: true",linestyle=:solid)

                    StatsPlots.scatter!(p2,fw_true_exploited.time_grid,test_data_with_noise[i,:],markershape=:diamond,markersize=3,legend=:none,label="Test data",color=my_gray_1,xlabel="")

                    plots_training[i] = p1 
                    plots_test[i] = p2
                    
                end    

                plot(plots_training[1],plots_test[1],plots_training[2],plots_test[2],plots_training[3],plots_test[3],
                    plots_training[4],plots_test[4], plots_training[5],plots_test[5],
                    layout=grid(5, 2,heights=0.2.*ones(10),widths=0.5.*ones(10)),
                    size=(500,700))
    
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_all_guilds_a_"*string(n_samples)*"_samples.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_all_guilds_a_"*string(n_samples)*"_samples.svg")

                plot(plots_training[6],plots_test[6],
                    plots_training[7],plots_test[7], plots_training[8],plots_test[8], plots_training[9],plots_test[9],
                    plots_training[10],plots_test[10],
                    layout=grid(5, 2,heights=0.2.*ones(10),widths=0.5.*ones(10)),
                    size=(500,700))
    
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_all_guilds_b_"*string(n_samples)*"_samples.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_all_guilds_b_"*string(n_samples)*"_samples.svg")


                i=1   
                title1 = "Training"
                title2 = "Test"
                legend1 = :topleft
                legend2 = :topleft 
                show_ = false

                p1=StatsPlots.plot([fw_true.time_grid, lower_quantiles[i,:]], fillrange = upper_quantiles[i,:], 
                            fillalpha = 0.2, linealpha=0,
                            xlim=(fw_true.time_grid[1],fw_true_exploited.time_grid[end]), color = my_turq, 
                            label = "Abundance: 90 % CPI of posterior", legend = legend1, title=title1, titleloc=:left, 
                            ylabel="",xlabel="",
                            fontfamily=:"Computer Modern",
                            left_margin = 0Plots.mm, right_margin = 1Plots.mm, bottom_margin = 0.5Plots.mm,
                            grid = show_, showaxis = show_)
                StatsPlots.plot!(p1,fw_true.time_grid, lower_quantiles_mean[i,:], fillrange = upper_quantiles_mean[i,:], 
                            fillalpha = 0.4,  linealpha=0, color = my_turq, label = "Internal dyn.: 90 % CPI of posterior")
                StatsPlots.plot!(p1,fw_true.time_grid,medians_mean[i,:], lw=3, label="Internal dyn.: posterior median",
                            color=my_blue,xticks=:none,yticks=:none)
                StatsPlots.plot!(p1,model_predicted_OLS, lw=4, vars=(0,i),
                            color=my_green,label="Internal dyn.: OLS fit",linestyle = :dot,xticks=:none,yticks=:none)
                StatsPlots.plot!(p1,ground_truth, lw=2,vars=(0,i),
                            color=:black,label="Internal dyn.: true",linestyle = :solid,
                            grid = show_, showaxis = show_)
                StatsPlots.scatter!(p1,fw_true.time_grid,training_data_with_noise[i,:],label="Training data",color=my_gray_1,
                            xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),xlabel="",
                            grid = show_, showaxis = show_,xticks=:none,yticks=:none)

                p2 = StatsPlots.plot(fw_true_exploited.time_grid, lower_quantiles_expl[i,:], 
                            fillrange = upper_quantiles_expl[i,:], 
                            fillalpha = 0.2,linealpha=0, color = my_purple, label = "Abundance: 90 % CPI of posterior", 
                            legend = legend2,
                            title=title2,titleloc=:left, 
                            ylabel="",xlabel="Year",
                            fontfamily=:"Computer Modern", yticks=:none,
                            grid = show_, showaxis = show_,
                            left_margin =0Plots.mm, right_margin = 1Plots.mm,bottom_margin = 0.5Plots.mm)                
                StatsPlots.plot!(p2,fw_true_exploited.time_grid, lower_quantiles_mean_expl[i,:],
                            fillrange = upper_quantiles_mean_expl[i,:], fillalpha =0.4, linealpha=0, yticks=:none, 
                            color = my_purple, label = "Internal dyn.: 90 % CPI of posterior")
                StatsPlots.plot!(p2,fw_true_exploited.time_grid,medians_mean_expl[i,:], lw=3, 
                            label="Internal dyn.: posterior median",
                            color=my_red, yticks=:none)
                StatsPlots.plot!(p2,fw_true_exploited.time_grid,model_expl_OLS_array[i,:], lw=4, 
                            color=my_orange,label="Internal dyn.: OLS fit",linestyle = :dot, xticks=:none, yticks=:none)
                StatsPlots.plot!(p2,fw_true_exploited.time_grid,ground_truth_exploited_array[i,: ], lw=2,
                            color=:black,label="Internal dyn.: true",linestyle=:solid, xticks=:none, yticks=:none,)
                StatsPlots.scatter!(p2,fw_true_exploited.time_grid,test_data_with_noise[i,:],markershape=:diamond,markersize=3,label="Test data",color=my_gray_1,xlabel="", xticks=:none, yticks=:none)

                plot(p1,p2,
                    layout=grid(1, 2,heights=0.16.*ones(2),widths=0.5.*ones(2)),
                    size=(650,150))    

                # these need tidying up
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictive_distr_legends.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_distr_legends.svg")

            end
    end        
end

# Plot errors when using 1) the initial guess of parameters and 2) the means of the variational posteriors (in VI) or the 
# OLS estimates to predict the internal dynamics 
function plot_posterior_and_prior_predictive_errors(parent_folder)

    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]
    colors = [my_gray, my_green, my_pink]

    data_names = ["TN1", "TN2", "TN3"]
    ndatas = length(data_names)
    conf_names = ["0.3", "0.7"]
    nconf = length(conf_names)

    guilds = ["10"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    errors_ini = zeros(nconf,ndatas,nguilds)
    errors_expl_ini = zeros(nconf,ndatas,nguilds)

    errors_bayes = zeros(nconf,ndatas,nguilds)
    errors_bayes_expl = zeros(nconf,ndatas,nguilds)

    errors_OLS = zeros(nconf,ndatas,nguilds)
    errors_OLS_expl = zeros(nconf,ndatas,nguilds)

    prop_errors_bayes = zeros(nconf,ndatas,nguilds)
    prop_errors_bayes_expl = zeros(nconf,ndatas,nguilds)

    prop_errors_OLS = zeros(nconf,ndatas,nguilds)
    prop_errors_OLS_expl = zeros(nconf,ndatas,nguilds)

    prop_errors_bayes_2 = zeros(nconf,ndatas,nguilds)
    prop_errors_bayes_expl_2 = zeros(nconf,ndatas,nguilds)

    prop_errors_OLS_2 = zeros(nconf,ndatas,nguilds)
    prop_errors_OLS_expl_2 = zeros(nconf,ndatas,nguilds)

    prop_errors_bayes_3 = zeros(nconf,ndatas,nguilds)
    prop_errors_bayes_expl_3 = zeros(nconf,ndatas,nguilds)

    prop_errors_OLS_3 = zeros(nconf,ndatas,nguilds)
    prop_errors_OLS_expl_3 = zeros(nconf,ndatas,nguilds)

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]

                for k in 1:nguilds
                    
                    guild = guilds[k]
    
                    ### the true foodweb and dynamics

                    I = readdlm(parent_folder*"/dataset"*set*"/I.txt",Int64)
                    fr_half_sat = readdlm(parent_folder*"/dataset"*set*"/B0.txt",Float64)
                    fr_shape = readdlm(parent_folder*"/dataset"*set*"/q.txt",Float64)

                    I_ind = findall(x->(x .> 0),I)
                    n_spec = size(I,1)
                    nLinks = length(I_ind)

                    fw_true = SetFoodWebParameters.initialize_generic_foodweb("true",I,ones(n_spec),fr_half_sat,fr_shape,zeros(n_spec))
                    true_abundance = FoodWebModel.foodweb_model(fw_true)
                    truth = zeros(n_spec,length(fw_true.time_grid))
                    for i in 1:length(fw_true.time_grid)
                        truth[:,i]=true_abundance[FoodWebUtils.closest_index(
                                true_abundance.t,fw_true.time_grid[i])] 
                    end

                    high_ab_guilds = findall(x->(x .> high_ab_threshold),repeat(minimum(truth,dims=2),1,length(fw_true.time_grid)))
                    high_ab_guilds_2 = findall(x->(x .> high_ab_threshold_2),repeat(minimum(truth,dims=2),1,length(fw_true.time_grid)))

                    test_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/test_data_"*CV*".txt",DataFrame)

                    fw_true_expl = SetFoodWebParameters.copy_foodweb(fw_true)
                    fw_true_expl.tspan = (0.0,2030.0)
                    fw_true_expl.time_grid = test_data_with_noise_df.Year
                
                    true_abundance_expl = FoodWebModel.foodweb_model_exploited(fw_true_expl)
                    truth_expl = zeros(n_spec,length(fw_true_expl.time_grid))
                    for i in 1:length(fw_true_expl.time_grid)
                        truth_expl[:,i]=true_abundance_expl[FoodWebUtils.closest_index(
                                true_abundance_expl.t,fw_true_expl.time_grid[i])] 
                    end

                    high_ab_guilds_expl = findall(x->(x .> high_ab_threshold),repeat(minimum(truth,dims=2),1,length(fw_true_expl.time_grid)))
                    high_ab_guilds_expl_2 = findall(x->(x .> high_ab_threshold_2),repeat(minimum(truth,dims=2),1,length(fw_true_expl.time_grid)))
                    # note that we use here unexploited years

                    ### initial guess about the dynamics

                    fw_ini = SetFoodWebParameters.copy_foodweb(fw_true)
                    fw_ini.fr_half_sat[fw_true.I_ind] .= half_sat_ini
                    fw_ini.fr_shape[fw_true.I_ind] .= shape_ini

                    training_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/training_data_"*CVs[i]*".txt",DataFrame)
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
                    fw_ini_expl.fr_shape[fw_true.I_ind] .= shape_ini
                
                    ini_abundance_expl = FoodWebModel.foodweb_model_exploited(fw_ini_expl)
                    ini_ab_expl = zeros(n_spec,length(fw_true_expl.time_grid))
                    for i in 1:length(fw_true_expl.time_grid)
                        ini_ab_expl[:,i]=ini_abundance_expl[FoodWebUtils.closest_index(
                                ini_abundance_expl.t,fw_true_expl.time_grid[i])] 
                    end

                    errors_ini[i,j,k] = sum(abs,truth-ini_ab)/length(truth)
                    errors_expl_ini[i,j,k] = sum(abs,truth_expl-ini_ab_expl)/length(truth_expl)

                    ### predicted abundances after fitting

                    # Bayes
                    param_bayes = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    b_opt = FittingOptions.initialize_bayes_options(fw_ini,1.0,training_data_with_noise)
        
                    fr_half_sat_pred = zeros(n_spec,n_spec)
                    fr_half_sat_pred[I_ind] = FoodWebUtils.activation1(param_bayes[1:nLinks],b_opt.param_min[1:nLinks],b_opt.param_max[1:nLinks])
                    fr_shape_pred = zeros(n_spec,n_spec)
                    fr_shape_pred[I_ind] = FoodWebUtils.activation1(param_bayes[nLinks+1:2*nLinks],b_opt.param_min[nLinks+1:2*nLinks],b_opt.param_max[nLinks+1:2*nLinks])
                    
                    fw_pred = SetFoodWebParameters.initialize_generic_foodweb("prediction",I,ones(n_spec),
                        fr_half_sat_pred,fr_shape_pred,zeros(n_spec))
                    predicted_abundance = FoodWebModel.foodweb_model_extinct(fw_pred)
                    prediction = zeros(n_spec,length(fw_true.time_grid))
                    for i in 1:length(fw_true.time_grid)
                        prediction[:,i]=predicted_abundance[FoodWebUtils.closest_index(
                                predicted_abundance.t,fw_true.time_grid[i])] 
                    end

                    errors_bayes[i,j,k] = sum(abs,truth-prediction)/length(truth)
                    prop_errors_bayes[i,j,k] = sum(abs,(truth .- prediction)./truth)/length(truth)
                    prop_errors_bayes_2[i,j,k] = sum(abs,(truth[high_ab_guilds] .- prediction[high_ab_guilds])./truth[high_ab_guilds])/length(high_ab_guilds)
                    prop_errors_bayes_3[i,j,k] = sum(abs,(truth[high_ab_guilds_2] .- prediction[high_ab_guilds_2])./truth[high_ab_guilds_2])/length(high_ab_guilds_2)

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
                    prop_errors_bayes_expl[i,j,k] = sum(abs,(truth_expl .- prediction_expl)./truth_expl)/length(truth_expl)
                    prop_errors_bayes_expl_2[i,j,k] = sum(abs,(truth_expl[high_ab_guilds_expl] .- prediction_expl[high_ab_guilds_expl])./truth_expl[high_ab_guilds_expl])/length(high_ab_guilds_expl)
                    prop_errors_bayes_expl_3[i,j,k] = sum(abs,(truth_expl[high_ab_guilds_expl_2] .- prediction_expl[high_ab_guilds_expl_2])./truth_expl[high_ab_guilds_expl_2])/length(high_ab_guilds_expl_2)

                    # OLS
                    param_OLS = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations_OLS/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    fr_half_sat_pred[I_ind] = param_OLS[1:nLinks]
                    fr_shape_pred[I_ind] = param_OLS[nLinks+1:2*nLinks]
                    
                    fw_pred = SetFoodWebParameters.initialize_generic_foodweb("OLS prediction",I,ones(n_spec),
                        fr_half_sat_pred,fr_shape_pred,zeros(n_spec))
                    predicted_abundance = FoodWebModel.foodweb_model_extinct(fw_pred)
                    prediction = zeros(n_spec,length(fw_true.time_grid))
                    for i in 1:length(fw_true.time_grid)
                        prediction[:,i]=predicted_abundance[FoodWebUtils.closest_index(
                                predicted_abundance.t,fw_true.time_grid[i])] 
                    end
            
                    errors_OLS[i,j,k] = sum(abs,truth-prediction)/length(truth)
                    prop_errors_OLS[i,j,k] = sum(abs,(truth .- prediction)./truth)/length(truth)
                    prop_errors_OLS_2[i,j,k] = sum(abs,(truth[high_ab_guilds] .- prediction[high_ab_guilds])./truth[high_ab_guilds])/length(high_ab_guilds)
                    prop_errors_OLS_3[i,j,k] = sum(abs,(truth[high_ab_guilds_2] .- prediction[high_ab_guilds_2])./truth[high_ab_guilds_2])/length(high_ab_guilds_2)

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
                    prop_errors_OLS_expl[i,j,k] = sum(abs,(truth_expl .- prediction_expl)./truth_expl)/length(truth_expl)
                    prop_errors_OLS_expl_2[i,j,k] = sum(abs,(truth_expl[high_ab_guilds_expl] .- prediction_expl[high_ab_guilds_expl])./truth_expl[high_ab_guilds_expl])/length(high_ab_guilds_expl)
                    prop_errors_OLS_expl_3[i,j,k] = sum(abs,(truth_expl[high_ab_guilds_expl_2] .- prediction_expl[high_ab_guilds_expl_2])./truth_expl[high_ab_guilds_expl_2])/length(high_ab_guilds_expl_2)

                end

            end
    end
    
    p1 = groupedbar(nam, errors_ini[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "A. Initial guess, training", fontfamily=:"Computer Modern",
        left_margin = 3Plots.mm, right_margin = 1Plots.mm, 
        bar_width = 0.67, 
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.1), color = [my_gray_1 my_gray_2 my_gray_3])
    p1b = groupedbar(nam, errors_expl_ini[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "B. Initial guess, test",fontfamily=:"Computer Modern",
        left_margin = 3Plots.mm, right_margin = 1Plots.mm,bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.1), color = [my_gray_1 my_gray_2 my_gray_3])

    println("Initial errors, training set:")
    println(errors_ini)

    println("Initial errors, test set:")
    println(errors_expl_ini)

    p2 = groupedbar(nam, errors_bayes[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "C. VI, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.1), color = [my_gray_1 my_gray_2 my_gray_3])
    p2b = groupedbar(nam, errors_bayes_expl[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "D. VI, test", fontfamily=:"Computer Modern",
        left_margin = 3Plots.mm, right_margin = 1Plots.mm,bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.1), color = [my_gray_1 my_gray_2 my_gray_3])

    println("Bayes errors, training set:")
    println(errors_bayes)
    
    println("Bayes errors, test set:")
    println(errors_bayes_expl)
    
    p3 = groupedbar(nam, errors_OLS[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "E. OLS, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:topleft, ylim=(0,0.1), color = [my_gray_1 my_gray_2 my_gray_3])
    p3b = groupedbar(nam, errors_OLS_expl[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "F. OLS, test", fontfamily=:"Computer Modern",
        left_margin = 3Plots.mm, right_margin = 1Plots.mm,bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.1), color = [my_gray_1 my_gray_2 my_gray_3])

    println("OLS errors, training set:")
    println(errors_OLS)
    
    println("OLS errors, test set:")
    println(errors_OLS_expl)

    plot(p1, p1b, p2, p2b,p3,p3b, layout=grid(3, 2,heights=0.33.*ones(6),widths=0.5.*ones(6)),
        size=(650,440))

    savefig(parent_folder*"/predictive_errors_100offspring.png")
    savefig(parent_folder*"/predictive_errors_100offspring.svg")
    
    p3 = groupedbar(nam, prop_errors_bayes[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "A. VI, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:topleft, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p3b = groupedbar(nam, prop_errors_bayes_expl[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "B. VI, test", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p4 = groupedbar(nam, prop_errors_OLS[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "C. OLS, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,5), color = [my_gray_1 my_gray_2 my_gray_3])
    p4b = groupedbar(nam, prop_errors_OLS_expl[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "D. OLS, test", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,5), color = [my_gray_1 my_gray_2 my_gray_3])

    plot(p3,p3b,p4,p4b, layout=grid(2, 2,heights=0.4.*ones(4),widths=0.5.*ones(4)),
        size=(650,400))

    println("Relative Bayes errors, training set:")
    println(prop_errors_bayes)
    
    println("Relative Bayes errors, test set:")
    println(prop_errors_bayes_expl)

    println("Relative OLS errors, training set:")
    println(prop_errors_OLS)
    
    println("Relative OLS errors, test set:")
    println(prop_errors_OLS_expl)


    savefig(parent_folder*"/relative_predictive_errors_100offspring.png")
    savefig(parent_folder*"/relative_predictive_errors_100offspring.svg")

    p3 = groupedbar(nam, prop_errors_bayes_2[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "A. VI, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p3b = groupedbar(nam, prop_errors_bayes_expl_2[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "B. VI, test", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p4 = groupedbar(nam, prop_errors_OLS_2[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "C. OLS, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p4b = groupedbar(nam, prop_errors_OLS_expl_2[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "D. OLS, test", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])

    plot(p3,p3b,p4,p4b, layout=grid(2, 2,heights=0.4.*ones(4),widths=0.5.*ones(4)),
        size=(650,400))

    savefig(parent_folder*"/relative_predictive_errors_100offspring_2.png")
    savefig(parent_folder*"/relative_predictive_errors_100offspring_2.svg")

    p3 = groupedbar(nam, prop_errors_bayes_3[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "A. VI, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p3b = groupedbar(nam, prop_errors_bayes_expl_3[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "B. VI, test", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p4 = groupedbar(nam, prop_errors_OLS_3[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "C. OLS, training", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p4b = groupedbar(nam, prop_errors_OLS_expl_3[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. abs. error",
        title = "D. OLS, test", bar_width = 0.67,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])

    plot(p3,p3b,p4,p4b, layout=grid(2, 2,heights=0.4.*ones(4),widths=0.5.*ones(4)),
        size=(650,400))

    savefig(parent_folder*"/relative_predictive_errors_100offspring_3.png")
    savefig(parent_folder*"/relative_predictive_errors_100offspring_3.svg")

end

# Plot the share of 90 % CPIs that cover the data when predicting 
# the internal dynamics and the total abundances using VI estimates
function plot_90_CPI_coverage_for_posterior_predictives(parent_folder,n_samples)

    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]
    colors = [my_gray, my_green, my_pink]

    data_names = ["TN1", "TN2", "TN3"]
    ndatas = length(data_names)
    conf_names = ["0.3", "0.7"]
    nconf = length(conf_names)
    guilds = ["10"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    coverage = zeros(nconf,ndatas,nguilds)
    coverage_expl = zeros(nconf,ndatas,nguilds)

    coverage_total = zeros(nconf,ndatas,nguilds)
    coverage_expl_total = zeros(nconf,ndatas,nguilds)

    Random.seed!(3758743) #seed 1: Random.seed!(58743)

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]
                for k in 1:nguilds
                    
                guild = guilds[k]

                I = readdlm(parent_folder*"/dataset"*set*"/I.txt",Int64)
                fr_half_sat = readdlm(parent_folder*"/dataset"*set*"/B0.txt",Float64)
                fr_shape = readdlm(parent_folder*"/dataset"*set*"/q.txt",Float64)

                I_ind = findall(x->(x .> 0),I)
                n_spec = size(I,1)
                nLinks = length(I_ind)

                fw_true = SetFoodWebParameters.initialize_generic_foodweb("true",I,ones(n_spec),fr_half_sat,fr_shape,zeros(n_spec))
                true_abundance = FoodWebModel.foodweb_model_extinct(fw_true)
                truth = zeros(n_spec,length(fw_true.time_grid))
                for i in 1:length(fw_true.time_grid)
                    truth[:,i]=true_abundance[FoodWebUtils.closest_index(
                            true_abundance.t,fw_true.time_grid[i])] 
                end

                training_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                training_data = transpose(Array(training_data_with_noise_df[:,2:end])) 

                test_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/test_data_"*CV*".txt",DataFrame)
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

                param_bayes = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)

                post_dist = [Normal(param_bayes[i], 
                    FoodWebUtils.rho_transform(param_bayes[2*nLinks+n_spec+i])[1]) 
                    for i=1:2*nLinks+n_spec]

                sampled_param = zeros(2*nLinks+n_spec,n_samples)
                for i=1:2*nLinks+n_spec
                    sampled_param[i,:]=rand(post_dist[i],n_samples)
                end

                # Bayes options ---> bounds for parameters
                fw_initial = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_initial.fr_half_sat[fw_true.I_ind].=half_sat_ini
                fw_initial.fr_shape[fw_true.I_ind].=shape_ini
                std_ini=zeros(n_spec)
                for i=1:n_spec
                    std_ini[i] = Statistics.std(training_data[i,:])
                end
                fw_initial.std=std_ini

                b_opt = FittingOptions.initialize_bayes_options(fw_initial,1.0,training_data)
                
                transformed_param = FoodWebUtils.activation1(sampled_param,repeat(b_opt.param_min[1:2*nLinks+n_spec],1,n_samples),
                    repeat(b_opt.param_max[1:2*nLinks+n_spec],1,n_samples))  
                
                post_preds=zeros(n_spec,length(fw_true.time_grid),n_samples)
                post_preds_expl=zeros(n_spec,length(fw_true_expl.time_grid),n_samples)

                post_preds_total=zeros(n_spec,length(fw_true.time_grid),n_samples)
                post_preds_expl_total=zeros(n_spec,length(fw_true_expl.time_grid),n_samples)

                fw_tmp = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_tmp_expl = SetFoodWebParameters.copy_foodweb(fw_true_expl)
                
                prob_post = 0
                prediction = zeros(n_spec,length(fw_true.time_grid))
                prob_post_expl = 0
                prediction_expl = zeros(n_spec,length(fw_true_expl.time_grid))

                cv_tmp = zeros(n_spec)

                for i=1:n_samples
                        fw_tmp.fr_half_sat[fw_tmp.I_ind] = transformed_param[1:nLinks,i]
                        fw_tmp.fr_shape[fw_tmp.I_ind] = transformed_param[nLinks+1:2*nLinks,i]

                        fw_tmp_expl.fr_half_sat[fw_tmp.I_ind] = transformed_param[1:nLinks,i]
                        fw_tmp_expl.fr_shape[fw_tmp.I_ind] = transformed_param[nLinks+1:2*nLinks,i]

                        fw_tmp.std = transformed_param[2*nLinks+1:2*nLinks+n_spec,i]
                                    
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
                                std_dist = MvNormal(zeros(fw_true.nSpec),transformed_param[2*fw_true.nLinks+1:2*fw_true.nLinks+fw_true.nSpec,i])
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

        #######
 
    p1 = groupedbar(nam, coverage[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Coverage",
         title = "C. Internal dynamics, training",fontfamily=:"Computer Modern",
         left_margin = 5Plots.mm, right_margin = 1Plots.mm,bar_width = 0.67,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:bottomleft, ylim=(0,1.0), color = [my_gray_1 my_gray_2 my_gray_3])
    plot!(p1,[0.0, 2.0],0.9.*ones(2), color=:black, linestyle=:dash, label=:none)
    
    println("C. Internal dynamics, training")
    println(coverage[:,:,1])

    p2 = groupedbar(nam, coverage_expl[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Coverage",
         title = "D. Internal dynamics, test", fontfamily=:"Computer Modern",
         left_margin = 5Plots.mm, right_margin = 1Plots.mm,bar_width = 0.67,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,1.0), color = [my_gray_1 my_gray_2 my_gray_3])
    plot!(p2,[0.0, 2.0],0.9.*ones(2), color=:black, linestyle=:dash, label=:none)

    println("D. Internal dynamics, test")
    println(coverage_expl[:,:,1])

    p3 = groupedbar(nam, coverage_total[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Coverage",
        title = "A. Total abundance, training", fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bar_width = 0.67,bottom_margin = 5Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,1.0), color = [my_gray_1 my_gray_2 my_gray_3])
    plot!(p3,[0.0, 2.0],0.9.*ones(2), color=:black, linestyle=:dash, label=:none)
    
    println("A. Total abundance, training")
    println(coverage_total[:,:,1])

    p4 = groupedbar(nam, coverage_expl_total[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Coverage",
        title = "B. Total abundance, test", fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bar_width = 0.67,bottom_margin = 5Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,1.0), color = [my_gray_1 my_gray_2 my_gray_3])
    plot!(p4,[0.0, 2.0],0.9.*ones(2), color=:black, linestyle=:dash, label=:none)

    println("B. Total abundance, test")
    println(coverage_expl_total[:,:,1])

    plot(p3, p4, p1, p2, layout=grid(2, 2,heights=0.5.*ones(4),widths=0.5.*ones(4)),
        size=(700,300))

    savefig(parent_folder*"/posterior_predictive_90_CPI_coverage_"*string(n_samples)*"_samples.png")
    savefig(parent_folder*"/posterior_predictive_90_CPI_coverage_"*string(n_samples)*"_samples.svg")

end


# Plot predictions by VI and OLS using the whole population of solutions provided by the evolutionary optimization method
# Plots only mean predictions
function plot_abundance_estimates_by_population(parent_folder)
            
    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]                    
                
                guild = "10"

                Random.seed!(8758743)

                I = readdlm(parent_folder*"/dataset"*set*"/I.txt",Int64)
                fr_half_sat = readdlm(parent_folder*"/dataset"*set*"/B0.txt",Float64)
                fr_shape = readdlm(parent_folder*"/dataset"*set*"/q.txt",Float64)
            
                n_spec = length(I[1,:])

                ### Read training and test data 

                training_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                training_data_with_noise = transpose(Array(training_data_with_noise_df[:,2:end])) 

                test_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/test_data_"*CV*".txt",DataFrame)
                test_data_with_noise = transpose(Array(test_data_with_noise_df[:,2:end])) 

                # Estimated std

                std_in_data = zeros(n_spec)
                for i=1:n_spec
                    std_in_data[i] = Statistics.std(training_data_with_noise[i,:])
                end

                ### Generate the true foodweb and biomasses
                            
                fw_true = SetFoodWebParameters.initialize_generic_foodweb("true_foodweb",I,ones(n_spec),
                                    fr_half_sat,fr_shape,
                                    std_in_data)
                            
                ground_truth =  FoodWebModel.foodweb_model(fw_true)
                ground_truth_array = Array(ground_truth)
                    
                fw_true_exploited = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_true_exploited.tspan = (0.0,2030.0)
                fw_true_exploited.time_grid = test_data_with_noise_df.Year
                        
                ground_truth_exploited =  FoodWebModel.foodweb_model_exploited(fw_true_exploited)
                ground_truth_exploited_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                for i in 1:length(fw_true_exploited.time_grid)
                        ground_truth_exploited_array[:,i]=ground_truth_exploited[
                            FoodWebUtils.closest_index(ground_truth_exploited.t,fw_true_exploited.time_grid[i])] 
                end

                # Bayes options ---> bounds for parameters
                
                b_opt = FittingOptions.initialize_bayes_options(fw_true,1.0,training_data_with_noise)

                ### Read parameter estimates
                param_est_bayes_pop = 0
                param_est_bayes_best = 0
                try
                    # try to read the untransformed (in R^n) parameter estimates                    
                    param_est_bayes_best = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)
                    param_est_bayes_pop = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameters_in_iteration.txt",Float64)
                catch
                    println("No appropriate files in the given folder.")
                    return 0
                end

                ### Read OLS parameter estimates
                param_est_OLS_pop = 0
                param_est_OLS_best = 0
                try
                    # try to read the untransformed (in R^n) parameter estimates
                    param_est_OLS_best = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations_OLS/EO_parameter_estimates/parameter_estimates.txt",Float64)
                    param_est_OLS_pop = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations_OLS/EO_parameter_estimates/parameters_in_iteration.txt",Float64)
                catch
                    println("No appropriate files in the given folder.")
                    return 0
                end

                pop_size = size(param_est_bayes_pop,1) #but contains initial values

                ### Predict abundances
                # Final guess, OLS

                fw_final_OLS = SetFoodWebParameters.copy_foodweb(fw_true)

                prediction_OLS = zeros(n_spec,length(fw_true.time_grid),pop_size)
                prediction_expl_OLS = zeros(n_spec,length(fw_true_exploited.time_grid),pop_size)

                prediction_OLS_best = zeros(n_spec,length(fw_true.time_grid))
                prediction_expl_OLS_best = zeros(n_spec,length(fw_true_exploited.time_grid))

                prediction_bayes = zeros(fw_true.nSpec,length(fw_true.time_grid),pop_size)
                prediction_expl_bayes = zeros(fw_true.nSpec,length(fw_true_exploited.time_grid),pop_size)

                prediction_bayes_best = zeros(fw_true.nSpec,length(fw_true.time_grid))
                prediction_expl_bayes_best = zeros(fw_true.nSpec,length(fw_true_exploited.time_grid))

                ############## best predictions #######################

                fw_final_OLS.fr_half_sat[fw_final_OLS.I_ind] = param_est_OLS_best[1:fw_true.nLinks]
                fw_final_OLS.fr_shape[fw_final_OLS.I_ind] = param_est_OLS_best[fw_true.nLinks+1:2*fw_true.nLinks]
                
                model_predicted_OLS = FoodWebModel.foodweb_model(fw_final_OLS)
                try
                    prediction_OLS_best[:,:] = Array(model_predicted_OLS)
                catch 
                    0
                end

                fw_expl_OLS = SetFoodWebParameters.copy_foodweb(fw_true_exploited)
                fw_expl_OLS.fr_half_sat[fw_expl_OLS.I_ind] = param_est_OLS_best[1:fw_true.nLinks]
                fw_expl_OLS.fr_shape[fw_final_OLS.I_ind] = param_est_OLS_best[fw_true.nLinks+1:2*fw_true.nLinks]
                
                model_expl_OLS = FoodWebModel.foodweb_model_exploited(fw_expl_OLS)
                model_expl_OLS_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                for i in 1:length(fw_true_exploited.time_grid)
                        model_expl_OLS_array[:,i]=model_expl_OLS[
                            FoodWebUtils.closest_index(model_expl_OLS.t,fw_true_exploited.time_grid[i])] 
                end
                prediction_expl_OLS_best[:,:] = model_expl_OLS_array

                param_final_transformed = FoodWebUtils.activation1(param_est_bayes_best[1:2*fw_true.nLinks+fw_true.nSpec],b_opt.param_min[1:2*fw_true.nLinks+fw_true.nSpec],
                b_opt.param_max[1:2*fw_true.nLinks+fw_true.nSpec]) 
           
                fw_final = SetFoodWebParameters.copy_foodweb(fw_true)
                fw_final.fr_half_sat[fw_final.I_ind] = param_final_transformed[1:fw_true.nLinks]
                fw_final.fr_shape[fw_final.I_ind] = param_final_transformed[fw_true.nLinks+1:2*fw_true.nLinks]

                model_predicted = FoodWebModel.foodweb_model(fw_final)
                try
                    prediction_bayes_best[:,:] = Array(model_predicted)
                catch 
                    0
                end

                fw_expl = SetFoodWebParameters.copy_foodweb(fw_true_exploited)
                fw_expl.fr_half_sat[fw_expl.I_ind] = param_final_transformed[1:fw_true.nLinks]
                fw_expl.fr_shape[fw_final.I_ind] = param_final_transformed[fw_true.nLinks+1:2*fw_true.nLinks]
                
                model_expl = FoodWebModel.foodweb_model_exploited(fw_expl)
                model_expl_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                for i in 1:length(fw_true_exploited.time_grid)
                        model_expl_array[:,i]=model_expl[
                            FoodWebUtils.closest_index(model_expl.t,fw_true_exploited.time_grid[i])] 
                end
                prediction_expl_bayes_best[:,:] = model_expl_array

                for m=2:pop_size

                    fw_final_OLS.fr_half_sat[fw_final_OLS.I_ind] = param_est_OLS_pop[m,1:fw_true.nLinks]
                    fw_final_OLS.fr_shape[fw_final_OLS.I_ind] = param_est_OLS_pop[m,fw_true.nLinks+1:2*fw_true.nLinks]
                    
                    model_predicted_OLS = FoodWebModel.foodweb_model(fw_final_OLS)
                    try
                        prediction_OLS[:,:,m] = Array(model_predicted_OLS)
                    catch 
                        0
                    end

                    fw_expl_OLS = SetFoodWebParameters.copy_foodweb(fw_true_exploited)
                    fw_expl_OLS.fr_half_sat[fw_expl_OLS.I_ind] = param_est_OLS_pop[m,1:fw_true.nLinks]
                    fw_expl_OLS.fr_shape[fw_final_OLS.I_ind] = param_est_OLS_pop[m,fw_true.nLinks+1:2*fw_true.nLinks]
                    
                    model_expl_OLS = FoodWebModel.foodweb_model_exploited(fw_expl_OLS)
                    model_expl_OLS_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                    for i in 1:length(fw_true_exploited.time_grid)
                            model_expl_OLS_array[:,i]=model_expl_OLS[
                                FoodWebUtils.closest_index(model_expl_OLS.t,fw_true_exploited.time_grid[i])] 
                    end
                    prediction_expl_OLS[:,:,m] = model_expl_OLS_array

                    # Final guess, Bayes
                    param_final_transformed = FoodWebUtils.activation1(param_est_bayes_pop[m,1:2*fw_true.nLinks+fw_true.nSpec],b_opt.param_min[1:2*fw_true.nLinks+fw_true.nSpec],
                         b_opt.param_max[1:2*fw_true.nLinks+fw_true.nSpec]) 
                    
                    fw_final = SetFoodWebParameters.copy_foodweb(fw_true)
                    fw_final.fr_half_sat[fw_final.I_ind] = param_final_transformed[1:fw_true.nLinks]
                    fw_final.fr_shape[fw_final.I_ind] = param_final_transformed[fw_true.nLinks+1:2*fw_true.nLinks]

                    model_predicted = FoodWebModel.foodweb_model(fw_final)
                    try
                        prediction_bayes[:,:,m] = Array(model_predicted)
                    catch 
                        0
                    end

                    fw_expl = SetFoodWebParameters.copy_foodweb(fw_true_exploited)
                    fw_expl.fr_half_sat[fw_expl.I_ind] = param_final_transformed[1:fw_true.nLinks]
                    fw_expl.fr_shape[fw_final.I_ind] = param_final_transformed[fw_true.nLinks+1:2*fw_true.nLinks]
                    
                    model_expl = FoodWebModel.foodweb_model_exploited(fw_expl)
                    model_expl_array = zeros(fw_true_exploited.nSpec,length(fw_true_exploited.time_grid))
                    for i in 1:length(fw_true_exploited.time_grid)
                            model_expl_array[:,i]=model_expl[
                                FoodWebUtils.closest_index(model_expl.t,fw_true_exploited.time_grid[i])] 
                    end
                    prediction_expl_bayes[:,:,m] = model_expl_array

                end

                plot_ = StatsPlots.plot(zeros(3),ones(3))
                plots_training_OLS = [plot_ for i=1:fw_true.nSpec]
                plots_test_OLS = [plot_ for i=1:fw_true.nSpec]

                plots_training_VI = [plot_ for i=1:fw_true.nSpec]
                plots_test_VI = [plot_ for i=1:fw_true.nSpec]

                for i in 1:fw_true.nSpec
                    if i==1 || i==6
                        title1 = "Training (OLS)"
                        title2 = "Test (OLS)"
                        title3 = "Training (VI)"
                        title4 = "Test (VI)"
                    else 
                        title1 = ""
                        title2 = ""
                        title3 = ""
                        title4 = ""
                    end   
                    legend1 = :none
                    legend2 = :none 
                    show_ = true

                    y_min = 0.8*minimum([minimum([training_data_with_noise[i,:] 
                            test_data_with_noise[i,:]]) minimum(prediction_OLS[i,:,:]) minimum(prediction_expl_OLS[i,:,:]) minimum(prediction_bayes[i,:,:]) minimum(prediction_expl_bayes[i,:,:])])                
                    
                    y_max = 1.5*maximum([maximum([training_data_with_noise[i,:] 
                            test_data_with_noise[i,:]]) maximum(prediction_OLS[i,:,:]) maximum(prediction_expl_OLS[i,:,:]) maximum(prediction_bayes[i,:,:]) maximum(prediction_expl_bayes[i,:,:])])                
                            

                    p1 = StatsPlots.plot(fw_true.time_grid, prediction_OLS[i,:,2], lw=2, vars=(0,i),
                             color=my_gray_2,label="Population of predictions (OLS)")

                    for k=3:pop_size
                         StatsPlots.plot!(p1,fw_true.time_grid, prediction_OLS[i,:,k], lw=2, vars=(0,i),
                                 color=my_gray_2,label="")
                    end 

                    StatsPlots.plot!(p1,fw_true.time_grid, prediction_OLS_best[i,:], lw=2, vars=(0,i),
                             color=my_green,label="Internal dyn.: best prediction (OLS)",
#                             ylim=(y_min,y_max), 
                             xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),
                             legend = legend1, title=title1, titleloc=:left, 
                             ylabel="G"*string(i)*" BM",xlabel="",
                             fontfamily="Computer Modern",
                             left_margin = 5Plots.mm, right_margin = 1Plots.mm, bottom_margin = 0.5Plots.mm,
                             grid = show_, showaxis = show_)


                    p1b = StatsPlots.plot(fw_true.time_grid, prediction_bayes[i,:,2], lw=1, vars=(0,i),
                             color=my_gray_2,label="Population of predictions (VI)")

                    for k=3:pop_size
                         StatsPlots.plot!(p1b,fw_true.time_grid, prediction_bayes[i,:,k], lw=1, vars=(0,i),
                                 color=my_gray_2,label="")
                    end 

                    StatsPlots.plot!(p1b,fw_true.time_grid, prediction_bayes_best[i,:], lw=2, vars=(0,i),
                             color=my_turq,label="Internal dyn.: best prediction (VI)",
 #                            ylim=(y_min,y_max), 
                             xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),
                             legend = legend1, title=title3, titleloc=:left, 
                             ylabel="G"*string(i)*" BM",xlabel="",
                             fontfamily="Computer Modern",
                             left_margin = 5Plots.mm, right_margin = 1Plots.mm, bottom_margin = 0.5Plots.mm,
                             grid = show_, showaxis = show_)


                     StatsPlots.scatter!(p1,fw_true.time_grid,training_data_with_noise[i,:],legend=:none,label="Training data",color=my_gray_1,
                             xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),xlabel="")

                     StatsPlots.scatter!(p1b,fw_true.time_grid,training_data_with_noise[i,:],legend=:none,label="Training data",color=my_gray_1,
                             xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),xlabel="")

    #                 ### results on test data ############


                    p2 = StatsPlots.plot(fw_true_exploited.time_grid, prediction_expl_OLS[i,:,2], lw=1, 
                         color=my_gray_2,label="Population of predictions (OLS)")

                    for k=3:pop_size
                        StatsPlots.plot!(p2,fw_true_exploited.time_grid, prediction_expl_OLS[i,:,k], lw=1, 
                            color=my_gray_2,label="")
                    end 

                    StatsPlots.plot!(p2,fw_true_exploited.time_grid, prediction_expl_OLS_best[i,:], lw=2, 
                        color=my_orange,label="Disturbed internal dyn.: best prediction (OLS)",                             
#                        ylim=(y_min,y_max), 
                        xlim=(fw_true_exploited.time_grid[1],fw_true_exploited.time_grid[end]), 
                        legend = legend1, title=title2, titleloc=:left, 
                        ylabel="G"*string(i)*" BM",xlabel="",
                        fontfamily="Computer Modern",
                        left_margin = 5Plots.mm, right_margin = 3Plots.mm, bottom_margin = 0.5Plots.mm,
                        grid = show_, showaxis = show_)


                    p2b = StatsPlots.plot(fw_true_exploited.time_grid, prediction_expl_bayes[i,:,2], lw=1, 
                        color=my_gray_2,label="Population of predictions (VI)")

                   for k=3:pop_size
                       StatsPlots.plot!(p2b,fw_true_exploited.time_grid, prediction_expl_bayes[i,:,k], lw=1, 
                           color=my_gray_2,label="")
                   end 

                   StatsPlots.plot!(p2b,fw_true_exploited.time_grid, prediction_expl_bayes_best[i,:], lw=2, 
                       color=my_purple,label="Disturbed internal dyn.: best prediction (VI)",                             
#                       ylim=(y_min,y_max), 
                       xlim=(fw_true_exploited.time_grid[1],fw_true_exploited.time_grid[end]), 
                       legend = legend1, title=title4, titleloc=:left, 
                       ylabel="G"*string(i)*" BM",xlabel="",
                       fontfamily="Computer Modern",
                       left_margin = 5Plots.mm, right_margin = 3Plots.mm, bottom_margin = 0.5Plots.mm,
                       grid = show_, showaxis = show_)

                    StatsPlots.scatter!(p2,fw_true_exploited.time_grid,test_data_with_noise[i,:],markershape=:diamond,markersize=3,legend=:none,label="Test data",color=my_gray_1,xlabel="")
                    StatsPlots.scatter!(p2b,fw_true_exploited.time_grid,test_data_with_noise[i,:],markershape=:diamond,markersize=3,legend=:none,label="Test data",color=my_gray_1,xlabel="")

                    plots_training_OLS[i] = p1 
                    plots_test_OLS[i] = p2

                    plots_training_VI[i] = p1b 
                    plots_test_VI[i] = p2b
                    
                end  
                
                l_all = @layout [a b ; c d ; e f ; g h ; i j ; k l ; m n ; o p ; q r ; s t]

                plot(plots_training_OLS[1],plots_test_OLS[1],plots_training_OLS[2],plots_test_OLS[2],plots_training_OLS[3],plots_test_OLS[3],
                    plots_training_OLS[4],plots_test_OLS[4], plots_training_OLS[5],plots_test_OLS[5],
                    layout=grid(5, 2,heights=0.2.*ones(10),widths=0.5.*ones(10)),
                    size=(500,700))    

                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_a_OLS.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_a_OLS.svg")

                plot(plots_training_OLS[6],plots_test_OLS[6],
                    plots_training_OLS[7],plots_test_OLS[7], plots_training_OLS[8],plots_test_OLS[8], plots_training_OLS[9],plots_test_OLS[9],
                    plots_training_OLS[10],plots_test_OLS[10],
                layout=grid(5, 2,heights=0.2.*ones(10),widths=0.5.*ones(10)),
                size=(500,700))

                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_b_OLS.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_b_OLS.svg")

                l_all = @layout [a b ; c d ; e f ; g h ; i j ; k l ; m n ; o p ; q r ; s t]

                plot(plots_training_VI[1],plots_test_VI[1],plots_training_VI[2],plots_test_VI[2],plots_training_VI[3],plots_test_VI[3],
                    plots_training_VI[4],plots_test_VI[4], plots_training_VI[5],plots_test_VI[5],
                    layout=grid(5, 2,heights=0.2.*ones(10),widths=0.5.*ones(10)),
                    size=(500,700))    

                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_a.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_a.svg")

                plot(plots_training_VI[6],plots_test_VI[6],
                    plots_training_VI[7],plots_test_VI[7], plots_training_VI[8],plots_test_VI[8], plots_training_VI[9],plots_test_VI[9],
                    plots_training_VI[10],plots_test_VI[10],
                layout=grid(5, 2,heights=0.2.*ones(10),widths=0.5.*ones(10)),
                size=(500,700))

                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_b.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_by_population_all_guilds_b.svg")

                # figure legends separately

                i=1   
                title1 = "Training"
                title2 = "Test"
                legend1 = :topleft
                legend2 = :topleft 
                show_ = false

                p1 = StatsPlots.plot(fw_true.time_grid, prediction_OLS[i,:,2], lw=1, vars=(0,i),
                             color=my_gray_2,label="Population of predictions (OLS)")

                StatsPlots.plot!(p1,fw_true.time_grid, prediction_OLS_best[i,:], lw=2, vars=(0,i),
                color=my_green,label="Internal dyn.: best prediction (OLS)",
                xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),
                legend = legend1, title=title1, titleloc=:left, 
                ylabel="",xlabel="",
                fontfamily="Computer Modern",
                left_margin = 5Plots.mm, right_margin = 1Plots.mm, bottom_margin = 0.5Plots.mm,
                grid = show_, showaxis = show_)

                StatsPlots.plot!(p1,fw_true.time_grid, prediction_bayes[i,:,2], lw=1, vars=(0,i),
                color=my_gray_2,label="Population of predictions (VI)")

                StatsPlots.plot!(p1,fw_true.time_grid, prediction_bayes_best[i,:], lw=2, vars=(0,i),
                color=my_turq,label="Internal dyn.: best prediction (VI)",
                xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),
                legend = legend1, title=title2, titleloc=:left, 
                ylabel="",xlabel="",
                fontfamily="Computer Modern",
                left_margin = 5Plots.mm, right_margin = 1Plots.mm, bottom_margin = 0.5Plots.mm,
                grid = show_, showaxis = show_)

                StatsPlots.scatter!(p1,fw_true.time_grid,training_data_with_noise[i,:],label="Training data",color=my_gray_1,
                        xlim=(fw_true.time_grid[1],fw_true.time_grid[end]),xlabel="")
                
                p2 = StatsPlots.plot(fw_true_exploited.time_grid, prediction_expl_OLS[i,:,2], lw=1, 
                    color=my_gray_2,label="Population of predictions (OLS)")

                StatsPlots.plot!(p2,fw_true_exploited.time_grid, prediction_expl_OLS_best[i,:], lw=2, 
                color=my_orange,label="Disturbed internal dyn.: best prediction (OLS)",
                xlim=(fw_true_exploited.time_grid[1],fw_true_exploited.time_grid[end]), 
                legend = legend1, title=title1, titleloc=:left, 
                ylabel="",xlabel="",
                fontfamily="Computer Modern",
                left_margin = 5Plots.mm, right_margin = 50Plots.mm, bottom_margin = 0.5Plots.mm,
                grid = show_, showaxis = show_)

                StatsPlots.plot!(p2,fw_true_exploited.time_grid, prediction_expl_bayes[i,:,2], lw=1, 
                color=my_gray_2,label="Population of predictions (VI)")

                StatsPlots.plot!(p2,fw_true_exploited.time_grid, prediction_expl_bayes_best[i,:], lw=2, 
                color=my_purple,label="Disturbed internal dyn.: best prediction (VI)",
                xlim=(fw_true_exploited.time_grid[1],fw_true_exploited.time_grid[end]), 
                legend = legend1, title=title1, titleloc=:left, 
                ylabel="",xlabel="",
                fontfamily="Computer Modern",
                left_margin = 5Plots.mm, right_margin = 50Plots.mm, bottom_margin = 0.5Plots.mm,
                grid = show_, showaxis = show_)

                StatsPlots.scatter!(p2,fw_true_exploited.time_grid,test_data_with_noise[i,:],markershape=:diamond,markersize=3,label="Test data",color=my_gray_1,xlabel="")

                plot(p1,p2,
                    layout=grid(1, 2,heights=0.16.*ones(2),widths=0.5.*ones(2)),
                    size=(650,100))    

                # these need tidying up
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_legends.png")
                savefig(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/posterior_predictives_legends.svg")

            end
    end        
end


############ EXPLORE PARAMETER ESTIMATES #########################

# Plot posterior distibutions and OLS estimates of the parameters
# transformed
function plot_parameter_estimates(parent_folder,n_samples)

    CVs = ["CV03", "CV07"]
    CVs_num = [0.3, 0.7]
    sets = ["1", "2", "3"]
    colors = [my_gray, my_green, my_pink]

    guild = "10"
    for k=3 #datasets
        for j=1 #CVs

            Random.seed!(8758743)

            # Read foodweb configuration and true parameter values

            I = 0
            fr_half_sat = 0
            fr_shape = 0
            try
                I = readdlm(parent_folder*"/dataset"*sets[k]*"/I.txt",Int64)
                fr_half_sat = readdlm(parent_folder*"/dataset"*sets[k]*"/B0.txt",Float64)
                fr_shape = readdlm(parent_folder*"/dataset"*sets[k]*"/q.txt",Float64)
            catch
                println("No appropriate data files in the folder "*parent_folder*"/dataset"*sets[k]*"!")
                return 0
            end
            
            n_spec = length(I[1,:])

            ### Generate the true foodweb and biomasses
                        
            fw_true = SetFoodWebParameters.initialize_generic_foodweb("true",I,ones(n_spec),
                                fr_half_sat,fr_shape,
                                zeros(n_spec))
                        
            ground_truth =  FoodWebModel.foodweb_model(fw_true)
            ground_truth_array = Array(ground_truth)

            std_true = zeros(n_spec)
            for i=1:n_spec
                std_true[i] = CVs_num[j]*Statistics.mean(ground_truth_array[i,:])
            end
            fw_true.std = std_true

            true_param=vcat(fw_true.fr_half_sat[fw_true.I_ind],fw_true.fr_shape[fw_true.I_ind],fw_true.std)

            ### Read training data 

            training_data_with_noise_df = CSV.read(parent_folder*"/dataset"*string(k)*"/training_data_"*CVs[j]*".txt",DataFrame)
            training_data_with_noise = transpose(Array(training_data_with_noise_df[:,2:end])) 

            # initialize std based on the scatter in the data
            std_ini=zeros(n_spec)
            for i=1:n_spec
                std_ini[i] = Statistics.std(training_data_with_noise[i,:])
            end
                
            # Bayes options ---> bounds for parameters
            fw_initial = SetFoodWebParameters.copy_foodweb(fw_true)
            fw_initial.fr_half_sat[fw_true.I_ind].=half_sat_ini
            fw_initial.fr_shape[fw_true.I_ind].=shape_ini
            fw_initial.std=std_ini

            b_opt = FittingOptions.initialize_bayes_options(fw_initial,1.0,training_data_with_noise)

            param_init=FoodWebUtils.activation1(b_opt.p_init[1:2*fw_true.nLinks+fw_true.nSpec],b_opt.param_min[1:2*fw_true.nLinks+fw_true.nSpec],b_opt.param_max[1:2*fw_true.nLinks+fw_true.nSpec])

            ### Read parameter estimates
            param_est = 0
            try
                # try to read the untransformed (in R^n) parameter estimates
                param_est = readdlm(parent_folder*"/dataset"*string(k)*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)
            catch
                println("No appropriate files in the folder EO_parameter_estimates.")
                return 0
            end

            ### Read OLS parameter estimates
            param_est_OLS = 0
            try
                # try to read the untransformed (in R^n) parameter estimates
                param_est_OLS = readdlm(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations_OLS/EO_parameter_estimates/parameter_estimates.txt",Float64)
            catch
                println("No appropriate files in the folder EO_parameter_estimates.")
                return 0
            end

            # Final guess, OLS
            param_final_transformed_OLS = FoodWebUtils.activation1(param_est_OLS[1:2*fw_true.nLinks],b_opt.param_min[1:2*fw_true.nLinks],b_opt.param_max[1:2*fw_true.nLinks]) 

            # Final guess, Bayes
            param_final_transformed = FoodWebUtils.activation1(param_est[1:2*fw_true.nLinks+fw_true.nSpec],b_opt.param_min[1:2*fw_true.nLinks+fw_true.nSpec],b_opt.param_max[1:2*fw_true.nLinks+fw_true.nSpec]) 

            # Posterior
            post_dist = [Normal(param_est[i], 
                FoodWebUtils.rho_transform(param_est[2*fw_true.nLinks+fw_true.nSpec+i])[1]) 
                for i=1:2*fw_true.nLinks+fw_true.nSpec]
            
            println("Plotting")
            if ~ispath(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates")
                    mkpath(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates")
            end

            for i in 1:b_opt.nPar
                    p_min = b_opt.param_min[i]
                    p_max = b_opt.param_max[i]

                    println(p_min)
            
                    prior_sample = rand(b_opt.prior_dist[i],n_samples)
                    posterior_sample=rand(post_dist[i],n_samples)
            
                    prior_sample_transformed = [FoodWebUtils.activation1(prior_sample[j],p_min,p_max) for j=1:n_samples]
                    posterior_sample_transformed = [FoodWebUtils.activation1(posterior_sample[j],p_min,p_max) for j=1:n_samples]
            
                    StatsPlots.histogram(prior_sample_transformed, label="Prior",legend=:bottomright, color=my_gray_1, xlim=(p_min,1.8*p_max),nbins=20,size=(400,300))
                    StatsPlots.histogram!(posterior_sample_transformed, label="Posterior", color=my_purple, fillalpha = 0.6, xlim=(p_min,1.8*p_max), 
                    fontfamily=:"Computer Modern")
            
                    StatsPlots.scatter!([param_final_transformed[i]],[0.0],label="Final guess (VI)", color=my_purple, markersize=7)
                    if(i<=2*fw_true.nLinks)
                        StatsPlots.scatter!([param_final_transformed_OLS[i]],[0.0],label="Final guess (OLS)", markershape=:square, markersize=4,color=my_orange)
                    end
                    StatsPlots.scatter!([param_init[i]],[0.0],label="Initial value", color=my_gray_2, markershape=:diamond,markersize=4)
                    StatsPlots.scatter!([true_param[i]],[0.0],label="True value", color=my_green, markersize = 5)
            
                    if in(i,1:fw_true.nLinks)
                        savefig(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates/parameter_estimate_BO"*string(fw_true.I_ind[i][1])*"_"*string(fw_true.I_ind[i][2])*"_"*string(n_samples)*"_samples.png")
                        savefig(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates/parameter_estimate_BO"*string(fw_true.I_ind[i][1])*"_"*string(fw_true.I_ind[i][2])*"_"*string(n_samples)*"_samples.svg")
                    elseif in(i,fw_true.nLinks+1:2*fw_true.nLinks)
                        savefig(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates/parameter_estimate_q"*string(fw_true.I_ind[i-fw_true.nLinks][1])*"_"
                            *string(fw_true.I_ind[i-fw_true.nLinks][2])*"_"*string(n_samples)*"_samples.png")
                        savefig(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates/parameter_estimate_q"*string(fw_true.I_ind[i-fw_true.nLinks][1])*"_"
                            *string(fw_true.I_ind[i-fw_true.nLinks][2])*"_"*string(n_samples)*"_samples.svg")
                    elseif in(i, 2*fw_true.nLinks+1:2*fw_true.nLinks+fw_true.nSpec)
                        savefig(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates/parameter_estimate_std"*string(i-2*fw_true.nLinks)*"_"*string(n_samples)*"_samples.png")
                        savefig(parent_folder*"/dataset"*sets[k]*"_results_with_noise_"*CVs[j]*"_100offspring_50_iterations/plot_parameter_estimates/parameter_estimate_std"*string(i-2*fw_true.nLinks)*"_"*string(n_samples)*"_samples.svg")
                    else
                        0
                    end     
            end
        end                    
    end
end


function plot_parameter_estimate_errors(parent_folder)

    CVs = ["CV03", "CV07"]
    CVs_num = [0.3, 0.7]

    sets = ["1", "2", "3"]
    colors = [my_gray, my_green, my_pink]

    data_names = ["TN1", "TN2", "TN3"]
    ndatas = length(data_names)
    conf_names = ["0.3", "0.7"]
    nconf = length(conf_names)

    guilds = ["10"]
    nguilds = length(guilds)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    errors_ini_half_sat = zeros(nconf,ndatas,nguilds)
    errors_ini_shape = zeros(nconf,ndatas,nguilds)
    errors_ini_cv = zeros(nconf,ndatas,nguilds)
    
    errors_bayes_half_sat = zeros(nconf,ndatas,nguilds)
    errors_bayes_shape = zeros(nconf,ndatas,nguilds)
    errors_bayes_cv = zeros(nconf,ndatas,nguilds)

    errors_OLS_half_sat = zeros(nconf,ndatas,nguilds)
    errors_OLS_shape = zeros(nconf,ndatas,nguilds)

    errors_ini_half_sat_rel = zeros(nconf,ndatas,nguilds)
    errors_ini_shape_rel = zeros(nconf,ndatas,nguilds)
    
    errors_bayes_half_sat_rel = zeros(nconf,ndatas,nguilds)
    errors_bayes_shape_rel = zeros(nconf,ndatas,nguilds)

    errors_OLS_half_sat_rel = zeros(nconf,ndatas,nguilds)
    errors_OLS_shape_rel = zeros(nconf,ndatas,nguilds)

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]

                for k in 1:1
                    
                    guild = guilds[k]

                    I = readdlm(parent_folder*"/dataset"*set*"/I.txt",Int64)
                    fr_half_sat_true = readdlm(parent_folder*"/dataset"*set*"/B0.txt",Float64)
                    fr_shape_true = readdlm(parent_folder*"/dataset"*set*"/q.txt",Float64)

                    I_ind = findall(x->(x .> 0),I)
                    n_spec = size(I,1)
                    nLinks = length(I_ind)

                    ### Initial guess

                    fr_half_sat_ini = zeros(n_spec,n_spec)
                    fr_half_sat_ini[I_ind] .= half_sat_ini

                    fr_shape_ini = zeros(n_spec,n_spec)
                    fr_shape_ini[I_ind] .= shape_ini

                    errors_ini_half_sat[i,j,k] = sum(abs,fr_half_sat_true[I_ind].-fr_half_sat_ini[I_ind])/nLinks
                    errors_ini_half_sat_rel[i,j,k] = sum(abs.(fr_half_sat_true[I_ind].-fr_half_sat_ini[I_ind])./fr_half_sat_true[I_ind])/nLinks

                    errors_ini_shape[i,j,k] = sum(abs,fr_shape_true[I_ind].-fr_shape_ini[I_ind])/nLinks
                    errors_ini_shape_rel[i,j,k] = sum(abs.(fr_shape_true[I_ind].-fr_shape_ini[I_ind])./fr_shape_true[I_ind])/nLinks

                    training_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                    training_data = transpose(Array(training_data_with_noise_df[:,2:end])) 
    
                    cv_ini=zeros(n_spec)
                    std_ini = zeros(n_spec)
                    for i=1:n_spec
                        std_ini[i] = Statistics.std(training_data[i,:])
                        cv_ini[i] = std_ini[i]/Statistics.mean(training_data[i,:])
                    end    

                    errors_ini_cv[i,j,k] = sum(abs, CVs_num[i] .- cv_ini)/n_spec

                    fw_init = SetFoodWebParameters.initialize_generic_foodweb("init",I,ones(n_spec),
                        fr_half_sat_ini,fr_shape_ini,
                        std_ini)

                    b_options = FittingOptions.initialize_bayes_options(fw_init,1.0,training_data)

                    ### Bayes

                    param_bayes = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    fr_half_sat_pred = zeros(n_spec,n_spec)
                    std_pred = zeros(n_spec)
                    fr_shape_pred = zeros(n_spec,n_spec)

                    fr_half_sat_pred[I_ind] = FoodWebUtils.activation1(param_bayes[1:nLinks],b_options.param_min[1:nLinks],b_options.param_max[1:nLinks])
                    
                    fr_shape_pred[I_ind] = FoodWebUtils.activation1(param_bayes[nLinks+1:2*nLinks],b_options.param_min[nLinks+1:2*nLinks],b_options.param_max[nLinks+1:2*nLinks])

                    std_pred = FoodWebUtils.activation1(param_bayes[2*nLinks+1:2*nLinks+n_spec],b_options.param_min[2*nLinks+1:2*nLinks+n_spec],
                        b_options.param_max[2*nLinks+1:2*nLinks+n_spec])

                    fw_pred = SetFoodWebParameters.initialize_generic_foodweb("predicted",I,ones(n_spec),
                        fr_half_sat_pred,fr_shape_pred,zeros(n_spec))
        
                    model_pred =  FoodWebModel.foodweb_model(fw_pred)
                    model_pred_array = Array(model_pred)

                    cv_pred = zeros(n_spec)
                    for i=1:n_spec
                        cv_pred[i] = std_pred[i]/Statistics.mean(model_pred_array[i,:])
                    end
                                
                    errors_bayes_half_sat[i,j,k] = sum(abs,fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])/nLinks
                    errors_bayes_half_sat_rel[i,j,k] = sum(abs.(fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])./fr_half_sat_true[I_ind])/nLinks

                    errors_bayes_shape[i,j,k] = sum(abs,fr_shape_true[I_ind].-fr_shape_pred[I_ind])/nLinks
                    errors_bayes_shape_rel[i,j,k] = sum(abs.(fr_shape_true[I_ind].-fr_shape_pred[I_ind])./fr_shape_true[I_ind])/nLinks

                    errors_bayes_cv[i,j,k] = sum(abs,CVs_num[i] .- cv_pred)/n_spec
    
                    ### OLS

                    param_OLS = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations_OLS/EO_parameter_estimates/parameter_estimates.txt",Float64)

                    fr_half_sat_pred[I_ind] = param_OLS[1:nLinks]
                    fr_shape_pred[I_ind] = param_OLS[nLinks+1:2*nLinks]

                    errors_OLS_half_sat[i,j,k] = sum(abs,fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])/nLinks
                    errors_OLS_half_sat_rel[i,j,k] = sum(abs.(fr_half_sat_true[I_ind].-fr_half_sat_pred[I_ind])./fr_half_sat_true[I_ind])/nLinks

                    errors_OLS_shape[i,j,k] = sum(abs,fr_shape_true[I_ind].-fr_shape_pred[I_ind])/nLinks
                    errors_OLS_shape_rel[i,j,k] = sum(abs.(fr_shape_true[I_ind].-fr_shape_pred[I_ind])./fr_shape_true[I_ind])/nLinks

                end

            end
    end

    p1 = groupedbar(nam, errors_ini_half_sat[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "A. Initial value", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
 
    p3 = groupedbar(nam, errors_bayes_half_sat[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "C. VI", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:topright, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p5 = groupedbar(nam, errors_OLS_half_sat[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "E. OLS", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])

    p2 = groupedbar(nam, errors_ini_half_sat_rel[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. error",
        title = "B. Initial value", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none,  color = [my_gray_1 my_gray_2 my_gray_3], ylim=(0,1.25))
 
    p4 = groupedbar(nam, errors_bayes_half_sat_rel[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. error",
        title = "D. VI", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none,color = [my_gray_1 my_gray_2 my_gray_3], ylim=(0,1.25))
    p6 = groupedbar(nam, errors_OLS_half_sat_rel[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. error",
        title = "F. OLS", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none,  color = [my_gray_1 my_gray_2 my_gray_3], ylim=(0,1.25) )


    plot(p1, p2, p3, p4, p5, p6, layout=grid(3, 2,heights=0.3.*ones(6),widths=0.48.*ones(6)),
        size=(600,500))

    savefig(parent_folder*"/error_half_sat_100offspring.png")
    savefig(parent_folder*"/error_half_sat_100offspring.svg")


    p1 = groupedbar(nam, errors_ini_shape[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "A. Initial value", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])

    p3 = groupedbar(nam, errors_bayes_shape[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "C. VI", bar_width = 0.67,
        fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])
    p5 = groupedbar(nam, errors_OLS_shape[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
        title = "E. OLS", bar_width = 0.67,
        fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:topleft, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])

    p2 = groupedbar(nam, errors_ini_shape_rel[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. error",
        title = "B. Initial value", bar_width = 0.67,fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, color = [my_gray_1 my_gray_2 my_gray_3], ylim=(0,0.5))

    p4 = groupedbar(nam, errors_bayes_shape_rel[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. error",
        title = "D. VI", bar_width = 0.67,
        fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, color = [my_gray_1 my_gray_2 my_gray_3], ylim=(0,0.5))
    
    p6 = groupedbar(nam, errors_OLS_shape_rel[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean rel. error",
        title = "F. OLS", bar_width = 0.67,
        fontfamily=:"Computer Modern",
        left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
        lw = 0, framestyle = :box, legend=:none, color = [my_gray_1 my_gray_2 my_gray_3], ylim=(0,0.5))

    plot(p1, p2, p3, p4, p5, p6, layout=grid(3, 2,heights=0.3.*ones(6),widths=0.48.*ones(6)),
        size=(600,500))
    

    savefig(parent_folder*"/error_shape_100offspring.png")
    savefig(parent_folder*"/error_shape_100offspring.svg")

    p1 = groupedbar(nam, errors_ini_cv[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
    title = "A. Initial value", bar_width = 0.67,fontfamily=:"Computer Modern",
    left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
    lw = 0, framestyle = :box, legend=:none, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])

    p2 = groupedbar(nam, errors_bayes_cv[:,:,1], group = ctg, xlabel = "CV of data", ylabel = "Mean abs. error",
    title = "B. VI", bar_width = 0.67,
    fontfamily=:"Computer Modern",
    left_margin = 5Plots.mm, right_margin = 1Plots.mm,bottom_margin = 3Plots.mm,
    lw = 0, framestyle = :box, legend=:topleft, ylim=(0,0.5), color = [my_gray_1 my_gray_2 my_gray_3])

    plot(p1, p2, layout=grid(2, 1,heights=0.48.*ones(2),widths=0.98.*ones(2)),
    size=(300,330))


    savefig(parent_folder*"/error_cv_100offspring.png") 
    savefig(parent_folder*"/error_cv_100offspring.svg")

end


# Plot the share of 90 % CPIs that covered the true parameter value
function plot_90CPI_coverage_for_parameters(parent_folder)

    CVs_num = [0.3, 0.7]
    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]
    colors = [my_gray, my_green, my_pink]

    data_names = ["TN1", "TN2", "TN3"]
    ndatas = length(data_names)
    conf_names = ["0.3", "0.7"]
    nconf = length(conf_names)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    coverage_bayes_10_ = zeros(nconf, ndatas)
    coverage_bayes_10_2 = zeros(nconf, ndatas)
    coverage_bayes_10_3 = zeros(nconf, ndatas)

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]
                # ten guilds:

                I = readdlm(parent_folder*"/dataset"*set*"/I.txt",Int64)
                fr_half_sat = readdlm(parent_folder*"/dataset"*set*"/B0.txt",Float64)
                fr_shape = readdlm(parent_folder*"/dataset"*set*"/q.txt",Float64)

                I_ind = findall(x->(x .> 0),I)
                n_spec = size(I,1)
                nLinks = length(I_ind)

                ######################
                fw_true = SetFoodWebParameters.initialize_generic_foodweb("true",I,ones(n_spec),
                fr_half_sat,fr_shape,
                zeros(n_spec))
        
                ground_truth =  FoodWebModel.foodweb_model(fw_true)
                ground_truth_array = Array(ground_truth)

                std_true = zeros(n_spec)
                for k=1:n_spec
                    std_true[k] = CVs_num[i]*Statistics.mean(ground_truth_array[k,:])
                end
                fw_true.std = std_true

                ### Read training data 

                training_data_with_noise_df = CSV.read(parent_folder*"/dataset"*set*"/training_data_"*CV*".txt",DataFrame)
                training_data_with_noise = transpose(Array(training_data_with_noise_df[:,2:end])) 

                b_opt = FittingOptions.initialize_bayes_options(fw_true,1.0,training_data_with_noise)
                
                param_bayes_10 = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt")
                
                post_dist = [Normal(param_bayes_10[i], 
                    FoodWebUtils.rho_transform(param_bayes_10[2*nLinks+n_spec+i])[1]) 
                       for i=1:2*nLinks+n_spec]
                lower_q = [quantile.(post_dist[i],0.05) for i in 1:2*nLinks+n_spec]
                upper_q = [quantile.(post_dist[i],0.95) for i in 1:2*nLinks+n_spec]

                fr_half_sat_true_inv = FoodWebUtils.inverse_activation1(fr_half_sat[I_ind], b_opt.param_min[1:nLinks], b_opt.param_max[1:nLinks])
                coverage_bayes_10_[i,j] = sum((fr_half_sat_true_inv.<upper_q[1:nLinks]) .& (fr_half_sat_true_inv.>lower_q[1:nLinks]))/nLinks

                fr_shape_true_inv = FoodWebUtils.inverse_activation1(fr_shape[I_ind], b_opt.param_min[nLinks+1:2*nLinks], b_opt.param_max[nLinks+1:2*nLinks])
                coverage_bayes_10_2[i,j] = sum((fr_shape_true_inv.<upper_q[nLinks+1:2*nLinks]) .& (fr_shape_true_inv.>lower_q[nLinks+1:2*nLinks]))/nLinks

                std_true_inv = FoodWebUtils.inverse_activation1(std_true, b_opt.param_min[2*nLinks+1:2*nLinks+n_spec], b_opt.param_max[2*nLinks+1:2*nLinks+n_spec])
                coverage_bayes_10_3[i,j] = sum((std_true_inv.<upper_q[2*nLinks+1:2*nLinks+n_spec]) .& (std_true_inv.>lower_q[2*nLinks+1:2*nLinks+n_spec]))/n_spec

            end
    end
    p1 = groupedbar(nam, coverage_bayes_10_, group = ctg, title = "Half-saturation", xlabel = "CV of data", ylabel = "Coverage by VP", 
        bar_width = 0.67,        
        left_margin = 7Plots.mm, right_margin = 1Plots.mm,bottom_margin = 1Plots.mm,
        lw = 0, framestyle = :box, legend=:topright, ylim=(0,0.75), color = [my_gray_1 my_gray_2 my_gray_3])

    println("Coverage by VPs:")
    println("Half-saturation:")
    println(coverage_bayes_10_)

    p2 = groupedbar(nam, coverage_bayes_10_2, group = ctg, title = "Exponent", xlabel = "CV of data", ylabel = "Coverage by VP",
        bar_width = 0.67,
        left_margin = 7Plots.mm, right_margin = 1Plots.mm,bottom_margin = 1Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.75), color = [my_gray_1 my_gray_2 my_gray_3])
    println("Exponent:")
    println(coverage_bayes_10_2)
        
    p3 = groupedbar(nam, coverage_bayes_10_3, group = ctg, title = "Std", xlabel = "CV of data", ylabel = "Coverage by VP",
        bar_width = 0.67,
        left_margin = 7Plots.mm, right_margin = 1Plots.mm,bottom_margin = 1Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,0.75), color = [my_gray_1 my_gray_2 my_gray_3])
    println("Std:")
    println(coverage_bayes_10_3)
    
    plot(p1,p2,p3, layout=grid(3, 1,heights=0.3.*ones(3),widths=0.9.*ones(3)),
        size=(300,700))
    
    savefig(parent_folder*"/parameters_90CPI_coverage.png")
    savefig(parent_folder*"/parameters_90CPI_coverage.svg")
    
end


# Plot the standard deviations of the variational posteriors
function plot_rhos_for_parameters(parent_folder)

    CVs = ["CV03", "CV07"]
    sets = ["1", "2", "3"]
    colors = [my_gray, my_green, my_pink]

    data_names = ["TN1", "TN2", "TN3"]
    ndatas = length(data_names)
    conf_names = ["0.3", "0.7"]
    nconf = length(conf_names)

    ctg = repeat(data_names, inner = nconf)
    nam = repeat(conf_names, outer = ndatas)

    average_rhos_10_ = zeros(nconf, ndatas)
    average_rhos_10_half_sat = zeros(nconf, ndatas)
    average_rhos_10_shape = zeros(nconf, ndatas)
    average_rhos_10_std = zeros(nconf, ndatas)

    for i in 1:2
        CV = CVs[i]
            for j in 1:3
                set = sets[j]
                # ten guilds:

                I = readdlm(parent_folder*"/dataset"*set*"/I.txt",Int64)
                fr_half_sat = readdlm(parent_folder*"/dataset"*set*"/B0.txt",Float64)
                fr_shape = readdlm(parent_folder*"/dataset"*set*"/q.txt",Float64)

                I_ind = findall(x->(x .> 0),I)
                n_spec = size(I,1)
                nLinks = length(I_ind)
                
                param_bayes_10 = readdlm(parent_folder*"/dataset"*set*"_results_with_noise_"*CV*"_100offspring_50_iterations/EO_parameter_estimates/parameter_estimates.txt")
                
                rhos = [FoodWebUtils.rho_transform(param_bayes_10[2*nLinks+n_spec+i])[1] 
                       for i=1:2*nLinks]

                average_rhos_10_[i,j] = Statistics.mean(rhos)

                rhos_half_sat = [FoodWebUtils.rho_transform(param_bayes_10[2*nLinks+n_spec+i])[1] 
                       for i=1:nLinks]

                average_rhos_10_half_sat[i,j] = Statistics.mean(rhos_half_sat)

                rhos_shape = [FoodWebUtils.rho_transform(param_bayes_10[2*nLinks+n_spec+i])[1] 
                for i=nLinks+1:2*nLinks]

                average_rhos_10_shape[i,j] = Statistics.mean(rhos_shape)

                rhos_std = [FoodWebUtils.rho_transform(param_bayes_10[2*nLinks+n_spec+i])[1] 
                for i=2*nLinks+1:2*nLinks+n_spec]

                average_rhos_10_std[i,j] = Statistics.mean(rhos_std)

            end
    end


    p1 = groupedbar(nam, average_rhos_10_half_sat, group = ctg, xlabel = "CV of data", ylabel = "Mean std of VP",
        title = "A. Half-saturation", bar_width = 0.67,
        left_margin = 7Plots.mm, right_margin = 1Plots.mm,bottom_margin = 1Plots.mm,
        lw = 0, framestyle = :box, legend=:topleft, ylim=(0,1), color = [my_gray_1 my_gray_2 my_gray_3],fontfamily=:"Computer Modern")

    p2 = groupedbar(nam, average_rhos_10_shape, group = ctg, xlabel = "CV of data", ylabel = "Mean std of VP",
        title = "B. FR exponent", bar_width = 0.67,
        left_margin = 7Plots.mm, right_margin = 1Plots.mm,bottom_margin = 1Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,1), color = [my_gray_1 my_gray_2 my_gray_3],fontfamily=:"Computer Modern")

    p3 = groupedbar(nam, average_rhos_10_std, group = ctg, xlabel = "CV of data", ylabel = "Mean std of VP",
        title = "C. Std", bar_width = 0.67,
        left_margin = 7Plots.mm, right_margin = 1Plots.mm,bottom_margin = 1Plots.mm,
        lw = 0, framestyle = :box, legend=:none, ylim=(0,1), color = [my_gray_1 my_gray_2 my_gray_3],fontfamily=:"Computer Modern")

    plot(p1,p2,p3, layout=grid(3, 1,heights=0.3.*ones(3),widths=0.9.*ones(3)),
    size=(300,700))

    savefig(parent_folder*"/rhos.png")
    savefig(parent_folder*"/rhos.svg")


end


end # module


#FoodWebPlots.plot_parameter_estimates("my_folder",1000) 

#FoodWebPlots.plot_90CPI_coverage_for_parameters("my_folder")

#FoodWebPlots.plot_iterations_for_minimum_loss("my_folder")

#FoodWebPlots.plot_parameter_estimate_errors("my_folder") 

#FoodWebPlots.plot_rhos_for_parameters("my_folder") 

#FoodWebPlots.plot_posterior_and_prior_predictive_errors("my_folder")

#FoodWebPlots.plot_90_CPI_coverage_for_posterior_predictives("my_folder",30000) 

#FoodWebPlots.plot_abundance_estimates("my_folder",30000) 

#FoodWebPlots.plot_execution_times("my_folder") 
