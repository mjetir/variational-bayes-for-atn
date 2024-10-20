
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


# Some useful functions
module FoodWebUtils

using DelimitedFiles

# FIND THE CLOSEST VALUE
# return its index
@views function closest_index(x, val)
    ibest = 1
    dxbest = abs(x[ibest]-val)
    for I in eachindex(x)
        dx = abs(x[I]-val)
        if dx < dxbest
            dxbest = dx
            ibest = I
        end
    end
    return ibest
end 

# ACTIVATION FUNCTION  FOR THE FOODWEB PARAMETERS IN THE NEURAL NETWORK  
# transform parameters in order to keep their values in predefined intervals
function activation1(par,par_min,par_max)
    if(sum(par_min .>= par_max)>0)
        println("Error in the parameters bounds!")
        return -Inf
    end
    return par_min .+ (par_max.-par_min)./(1.0 .+ exp.(-par))
end

# INVERSE OF THE ACTIVATION FUNCTION FOR THE FOODWEB PARAMETERS
function inverse_activation1(par,par_min,par_max)
    if(sum(par_min .>= par_max)>0)
        println("Error in the parameters bounds!")
        return -Inf
    end
    return -log.((par_max.-par_min)./(par.-par_min) .- 1.0)
end

# ACTIVATION FUNCTION FOR THE VARIANCE PARAMETERS 
# OF THE VARIATIONAL DISTRIBUTIONS
function rho_transform(x, param_bounds...)
    return [if (x[i] < 500.0) log(1.0 + exp(x[i]))[1]
                else x[i] end for i=1:length(x)]
end

# INVERSE ACTIVATION FOR THE VARIANCE PARAMETERS
function inverse_rho_transform(x, param_bounds...)
    return [if (x[i] < 500.0) log(exp(x[i]) - 1.0) # log(exp(500)+1) ~ 500
            else x[i] end for i=1:length(x)] 
end

# WRITE EXECUTION TIME TO FILE
function write_ex_time_to_file(ex_time,folder_name)
    println("Writing execution time to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    # execution time
    open(folder_name*"/execution_time.txt","w") do io
        writedlm(io,ex_time)
    end    
end

# WRITE PARAMETERS TO FILE
function write_parameters_to_file(results,folder_name)
    println("Writing parameter estimates to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    # the best candidate
    open(folder_name*"/parameter_estimates.txt","w") do io
        writedlm(io,results.param_final)
    end    
    # the population where the best candidate belongs to
    open(folder_name*"/population_of_parameter_estimates.txt","w") do io
        writedlm(io,results.pop_final)
    end    
    # best candidate in each iteration step
    open(folder_name*"/parameters_in_iteration.txt","w") do io
        writedlm(io,results.param)
    end        
end

# WRITE LOSS FUNCTION VALUES TO FILE
function write_losses_to_file(results,folder_name)
    println("Writing loss function values to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    open(folder_name*"/losses.txt","w") do io
        writedlm(io,results.loss)
    end    
end

# WRITE LOSS FUNCTION VALUES TO FILE
function write_data_to_file(years, biomasses,folder_name,file_name)
    println("Writing data to "*folder_name)

    if ~ispath(folder_name)
        mkpath(folder_name)
    end

    open(folder_name*"/"*file_name*".txt","w") do f
        print(f,"Year;")
        for i in 1:size(biomasses,1)-1
            print(f,"Guild_"*string(i)*";")
        end
        println(f,"Guild_"*string(size(biomasses,1)))

        for j in 1:size(biomasses,2)
            print(f,string(years[j])*";")
            for i in 1:size(biomasses,1)-1
                print(f,string(biomasses[i,j])*";")
            end
            println(f,string(biomasses[end,j]))
        end
    end
end


end # module
