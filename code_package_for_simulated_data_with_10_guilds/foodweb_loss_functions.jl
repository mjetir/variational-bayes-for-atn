
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


module LossFunctions

using Random, Distributions, Statistics

include("foodweb_utils.jl")

# LOSS FUNCTION FOR OLS
# par_fit: temporary values of the parameters to be fitted
# fw_tmp: the parameter values of the foodweb model
# ols_opt: OLSOptions struct
# ols_cont: OLSContainers struct (for avoiding temporary allocations)
@views function loss_OLS(par_fit,fw_tmp,ols_opt,ols_cont)
    fw_tmp.fr_half_sat[fw_tmp.I_ind] .= par_fit[1:fw_tmp.nLinks]

    fw_tmp.fr_shape[fw_tmp.I_ind] .= par_fit[fw_tmp.nLinks+1:2*fw_tmp.nLinks]
    
    model_tmp =  ols_opt.model(fw_tmp)

    if maximum(model_tmp.t) >= fw_tmp.tspan[end]
        for i in 1:length(fw_tmp.time_grid)
            ols_cont.prediction[:,i]=model_tmp[FoodWebUtils.closest_index(model_tmp.t,fw_tmp.time_grid[i])] 
        end
        # check that there is not too much oscillation
        if(sum(Statistics.std(ols_cont.prediction,dims=2)./
                    Statistics.mean(ols_cont.prediction,dims=2) .> ols_opt.max_oscillation) 
                    > 0) 
#            println("Too much oscillation.")           
            return Inf  
        else
            return sum(abs2,skipmissing(ols_cont.prediction .- ols_opt.training_data))
        end
    else
        return Inf
    end
end


# LOSS FUNCTION FOR BAYES BY BACKPROPAGATION
# (Blundell et al. (2015): Weight Uncertainty in Neural Networks)
# var_par_fit: temporary values of the parameters to be fitted (variational parameters)
# fw_tmp: the parameter values of the foodweb model
# b_opt: BayesOptions struct
# b_cont: BayesContainers struct (for avoiding temporary allocations)
function loss_BBP(var_par_fit,fw_tmp,b_opt,b_cont)
    b_cont.epsilons = rand(b_opt.epsilon_dist)
    
    b_cont.sampled_parameters .= var_par_fit[1:b_opt.nPar] .+ 
      FoodWebUtils.rho_transform(var_par_fit[b_opt.nPar+1:end]).*b_cont.epsilons 

    # there are two layers the activation functions of which correspond to 
    # 1 the transformations and 
    # 2 the numerical solution of the ODE
    ### the first activation:
    fw_tmp.fr_half_sat[fw_tmp.I_ind] .= FoodWebUtils.activation1(b_cont.sampled_parameters[1:fw_tmp.nLinks],
                                                             b_opt.param_min[1:fw_tmp.nLinks],
                                                             b_opt.param_max[1:fw_tmp.nLinks])

    fw_tmp.fr_shape[fw_tmp.I_ind] .= FoodWebUtils.activation1(b_cont.sampled_parameters[fw_tmp.nLinks+1:2*fw_tmp.nLinks],
                                                          b_opt.param_min[fw_tmp.nLinks+1:2*fw_tmp.nLinks],
                                                          b_opt.param_max[fw_tmp.nLinks+1:2*fw_tmp.nLinks])

    fw_tmp.std .= FoodWebUtils.activation1(b_cont.sampled_parameters[2*fw_tmp.nLinks + 1:2*fw_tmp.nLinks + fw_tmp.nSpec],
                                       b_opt.param_min[2*fw_tmp.nLinks + 1:2*fw_tmp.nLinks + fw_tmp.nSpec],
                                       b_opt.param_max[2*fw_tmp.nLinks + 1:2*fw_tmp.nLinks + fw_tmp.nSpec])

    ### the second activation:
    model_tmp =  b_opt.model(fw_tmp)

    if maximum(model_tmp.t) >= fw_tmp.tspan[end]

        for i in 1:length(fw_tmp.time_grid)
            b_cont.prediction[:,i]=model_tmp[FoodWebUtils.closest_index(model_tmp.t,fw_tmp.time_grid[i])] 
        end

        # check that there is not too much internal oscillation
        if(sum(Statistics.std(b_cont.prediction,dims=2)./
                    Statistics.mean(b_cont.prediction,dims=2) .> b_opt.max_oscillation) 
                    > 0)
#            println("Too much oscillation.")           
            return Inf  
        else

            # compute the log-likelihood of the variational distribution (log[q(w|theta)])
            b_cont.log_variational = sum([(-(b_cont.sampled_parameters[i] - 
                                                var_par_fit[i])^2)/2.0/
                                                (FoodWebUtils.rho_transform(var_par_fit[b_opt.nPar+i])[1])^2 +
                            log(1.0/sqrt(2.0*pi)/FoodWebUtils.rho_transform(var_par_fit[b_opt.nPar+i])[1]) 
                                                                            for i=1:b_opt.nPar]) 
            # compute the log-likelihood (log[P(D|w)])
            ### normal noise:
            b_cont.log_likelihood = sum([logpdf(
                                Normal(b_cont.prediction[i,j],fw_tmp.std[i]), 
                                b_opt.training_data[i,j]) for i=1:fw_tmp.nSpec,j=1:length(fw_tmp.time_grid)])
            
            # compute the log priors (log[P(w)])
            b_cont.log_prior = sum([logpdf(b_opt.prior_dist[i],b_cont.sampled_parameters[i]) 
                        for i=1:b_opt.nPar])        
            
            # define here the cost function
                # log[q(w|theta)]        - log[P(D|w)]            - log[P(w)]   
            return  b_cont.log_variational - b_cont.log_likelihood - b_cont.log_prior 

        end
    else
        return Inf
    end
end

end # module


