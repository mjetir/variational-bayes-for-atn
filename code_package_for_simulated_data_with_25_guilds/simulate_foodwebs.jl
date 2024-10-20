
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


# Simulates proper foodwebs, uses the BioEnergeticFoodWebs package 
# to generate the feeding links
module SimulateFoodWebs

using StatsPlots
using LinearAlgebra
using Random, Distributions, Statistics
using CSV, DataFrames
using DelimitedFiles
using BioEnergeticFoodWebs
using DifferentialEquations

include("set_foodweb_parameters.jl")
include("foodweb_model_ODE.jl")
include("plot_results_for_foodwebs_public.jl")
include("foodweb_utils.jl")

function generate_proper_foodwebs(nSpec,nFoodWebs,q,K)

    Random.seed!(313)

    folder_name = "ATN_simulated_webs"
    if ~isdir(folder_name) 
        mkdir(folder_name)
    end

    subfolder_name = string(nSpec)*"_guilds_data"
    if ~isdir(subfolder_name) 
        mkpath(folder_name*"/"*subfolder_name)
    end

    for n=1:nFoodWebs

        is_proper_foodweb=0

        I::Matrix{Int64} = zeros(nSpec,nSpec) #initialize
        shortest_p = BioEnergeticFoodWebs.distance_to_producer(I)
        fw=0    
        ground_truth = 0 

        while(is_proper_foodweb==0)
            println("Simulating a food web.")
            is_proper_foodweb = 1 
    
            ### generate random networks
            # feeding link matrix, the desired connectance set to 0.15
            try 
                I=BioEnergeticFoodWebs.nichemodel(nSpec,0.15)
                for i=1:nSpec
                    # require that every node is connected to the web
                    # would be enough to check the connectance of producers 
                    # (see the next requirement)
                    if(sum(I[i,:])<1.0 && sum(I[:,i])<1.0)
                        is_proper_foodweb=0
                        println("Not connected.")
                    end
                end

                # shortest path to a basal producer
                shortest_p = BioEnergeticFoodWebs.distance_to_producer(I)
                # require that, from every consumer, there is a path to 
                # a basal producer
                if(sum(shortest_p.<1.0)>0)
                        is_proper_foodweb=0
                        println("Not connected to basal species.")
                end
            catch
                println("Niche model generation failed.")
                is_proper_foodweb=0
            end
            
            if(is_proper_foodweb==1)

                # set u0.=1
                # sample fr_half_sat 
                fw = SetFoodWebParameters.initialize_generic_foodweb("test_data", I,
                            ones(nSpec), 0.05 .+ 0.95.*rand(nSpec,nSpec),
                            q.*ones(nSpec,nSpec), K, zeros(nSpec)) 
                fw.time_grid = fw.tspan[1]:1.0:fw.tspan[2] # for simulation of data

                ex_time = @elapsed ground_truth =  FoodWebModel.foodweb_model(fw)

                println("Execution time:")
                println(ex_time)

                ground_truth_array = Array(ground_truth)

                if(length(ground_truth.t)!=length(fw.time_grid))
                    is_proper_foodweb=0
                    println("Too few or too many values saved.")
                elseif(sum(abs,ground_truth.t.-fw.time_grid)>0.01)
                    is_proper_foodweb=0
                    println("The saved values are unexact.")
                elseif(sum(ground_truth_array.<=fw.extinction_threshold)>0)
                    is_proper_foodweb=0
                    println("Extinction.")
                elseif(sum(Statistics.std(ground_truth_array[:,end-100:end],dims=2)./
                        Statistics.mean(ground_truth_array[:,end-100:end],dims=2) .> 0.001) 
                        > 0)
                    is_proper_foodweb=0
                    println("Too much oscillation.")
                elseif(sum(abs.(Statistics.mean(ground_truth_array[:,end-200:end-100],dims=2).-
                        Statistics.mean(ground_truth_array[:,end-100:end],dims=2)) .> 0.001) 
                        > 0)
                    is_proper_foodweb=0
                    println("Trend.")
        end

            end

        end

        # consider bistability:
        fw_ini2 = SetFoodWebParameters.copy_foodweb(fw)
        fw_ini2.u0 .= 0.1.*ones(fw_ini2.nSpec)

        model_ini2 =  FoodWebModel.foodweb_model_extinct(fw_ini2)

        fw_ini3 = SetFoodWebParameters.copy_foodweb(fw)
        fw_ini3.u0 .= 0.3.*ones(fw_ini3.nSpec)

        model_ini3 =  FoodWebModel.foodweb_model_extinct(fw_ini3)

        fw_ini4 = SetFoodWebParameters.copy_foodweb(fw)
        fw_ini4.u0 .= 0.5.*ones(fw_ini4.nSpec)

        model_ini4 =  FoodWebModel.foodweb_model_extinct(fw_ini4)

        fw_ini5 = SetFoodWebParameters.copy_foodweb(fw)
        fw_ini5.u0 .= 0.7.*ones(fw_ini5.nSpec)

        model_ini5 =  FoodWebModel.foodweb_model_extinct(fw_ini5)

        subsubfolder_name = "dataset"*string(n)
        write_to = folder_name*"/"*subfolder_name*"/"*subsubfolder_name
        if ~isdir(write_to) 
            mkpath(write_to)
        end

        open(write_to*"/I.txt","w") do io 
            writedlm(io,I)
        end
        open(write_to*"/B0.txt","w") do io 
            writedlm(io,fw.fr_half_sat)
        end 
        open(write_to*"/q.txt","w") do io 
            writedlm(io,fw.fr_shape)
        end
        open(write_to*"/K.txt","w") do io 
            writedlm(io,fw.K)
        end

        FoodWebPlots.plot_model(ground_truth,
        model_ini2,model_ini3,model_ini4,model_ini5,
        fw,write_to*"/plot_model",)

    end
  
end 

function simulate_noisy_biomass_data(parent_folder_name, cv_rand)

    data_folder_names = ["dataset"*string(i) for i=1]
    
    cd(parent_folder_name)

    if(cv_rand==0.3)
        suffix="CV03"
    elseif(cv_rand==0.7)
        suffix="CV07"
    else
        suffix=""
    end

    for i=1:length(data_folder_names)

        I = 0
        fr_half_sat = 0
        fr_shape = 0
        K = 0
        try
            I = readdlm(data_folder_names[i]*"/I.txt",Int64)
            fr_half_sat = readdlm(data_folder_names[i]*"/B0.txt",Float64)
            fr_shape = readdlm(data_folder_names[i]*"/q.txt",Float64)
            K = readdlm(data_folder_names[i]*"/K.txt",Float64)[1]
        catch
            println("No appropriate data files in the folder "*data_folder_names[i]*"!")
            return 0
        end

        I_ind = findall(x->(x .> 0),I)
        n_spec = size(I,1)

        fw = SetFoodWebParameters.initialize_generic_foodweb(data_folder_names[i],I,ones(n_spec),
                fr_half_sat,fr_shape,K,
                zeros(n_spec))
        
        ground_truth =  FoodWebModel.foodweb_model(fw)
        ground_truth_array = Array(ground_truth)

        ### Create training data

        Random.seed!(465438)

        # Normal noise:
        deviations_from_mean_curve = zeros(fw.nSpec)
        for i = 1:fw.nSpec
            deviations_from_mean_curve[i] = cv_rand*Statistics.mean(ground_truth_array[i,:])
        end

        scatter_distributions = MvNormal(zeros(fw.nSpec),deviations_from_mean_curve)

        training_data_with_noise = ground_truth_array .+ rand(scatter_distributions,length(fw.time_grid))
        training_data_with_noise[findall(x->(x<0.0),training_data_with_noise)].=0.0

        ### Create data for testing, remove biomass from consumer species

        fw_exploited = SetFoodWebParameters.copy_foodweb(fw)
        fw_exploited.tspan = (0.0,2030.0)
        fw_exploited.time_grid = fw.tspan[end]+1.0:1.0:fw_exploited.tspan[2]
        
        ground_truth_exploited =  FoodWebModel.foodweb_model_exploited(fw_exploited)
        ground_truth_exploited_array = zeros(fw_exploited.nSpec,length(fw_exploited.time_grid))
        for i in 1:length(fw_exploited.time_grid)
            ground_truth_exploited_array[:,i]=ground_truth_exploited[
                FoodWebUtils.closest_index(ground_truth_exploited.t,fw_exploited.time_grid[i])] 
        end

        # Normal noise:
        deviations_from_mean_curve_exploited = zeros(fw_exploited.nSpec)
        for i = 1:fw_exploited.nSpec
            deviations_from_mean_curve_exploited[i] = cv_rand*Statistics.mean(ground_truth_exploited_array[i,:])
        end
        
        scatter_distributions_exploited = MvNormal(zeros(fw_exploited.nSpec),deviations_from_mean_curve_exploited)

        test_data_with_noise = ground_truth_exploited_array .+ rand(scatter_distributions_exploited,length(fw_exploited.time_grid))
        test_data_with_noise[findall(x->(x<0.0),test_data_with_noise)].=0.0

        FoodWebUtils.write_data_to_file(fw_exploited.time_grid, test_data_with_noise,data_folder_names[i],"test_data_"*suffix)
        FoodWebUtils.write_data_to_file(fw.time_grid, training_data_with_noise,data_folder_names[i],"training_data_"*suffix)
    end
    cd("../")
    cd("../")

end

end #module

#SimulateFoodWebs.generate_proper_foodwebs(25,1,0.5,10.0)


#SimulateFoodWebs.simulate_noisy_biomass_data("ATN_simulated_webs/25_guilds_data",0.3)

