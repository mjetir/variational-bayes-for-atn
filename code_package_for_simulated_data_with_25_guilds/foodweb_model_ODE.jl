
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


# Implements the ODE model for the foodweb.
module FoodWebModel

using DifferentialEquations
using LinearAlgebra
using Random, Distributions, Statistics
using CSV, DataFrames


# ODE MODEL FOR FOODWEB following Williams (2008)
@views function foodweb!(du,u,p,t)

### FUNCTIONAL RESPONSES
    # interspecific feeding interference ignored
    # multispecies functional response, eqn. (4) by Williams (2008) 
    # (Effects of network and dynamical model structure on species
    # persistence in large model food webs, published in Theoretical Ecology) 
    p.F.=0.0
    p.fr_denom_sum.=0.0
    @inbounds for i in p.cons_ind
        @inbounds for l in 1:(p.nSpec) # iterate over resources
            if in(CartesianIndex(i,l),p.I_ind)
                p.fr_denom_sum[i]+=(max(u[l],0.0)/p.fr_half_sat[i,l])^(p.fr_shape_const[i,l] + p.fr_shape[i,l])
            end
        end
    end
    @inbounds for i in p.cons_ind
        @inbounds for j in 1:(p.nSpec)
            if in(CartesianIndex(i,j),p.I_ind)
                p.F[i,j]=(max(u[j],0.0)/p.fr_half_sat[i,j])^(p.fr_shape_const[i,j] + p.fr_shape[i,j])/
                (1.0 + p.fr_denom_sum[i])				
            end
        end 
    end

    ### the differential equations   
    
    # producers:	
    @inbounds for i in p.prod_ind
        if(u[i]>0.0)
            # logistic growth function for producers:
            p.G[i]= 0.0
            @inbounds for j in p.prod_ind
                p.G[i] -= p.c[i,j]*max(u[j],0.0)/p.K
            end
            p.G[i] += 1.0
            p.producers_loss_to_consumers[i] = 0.0
            @inbounds for j in p.cons_ind
                if in(CartesianIndex(j,i),p.I_ind)
                    p.producers_loss_to_consumers[i] += p.x[j]*p.y[j,i]*max(u[j],0.0)*p.F[j,i]/p.ef[j,i] 
                end
            end
            du[i] = p.r[i]*u[i]*p.G[i]-p.producers_loss_to_consumers[i]
            else 
                du[i]=0.0
            end
        end
        
    # consumers: 
    @inbounds for i in p.cons_ind
        if(u[i]>0.0)
            p.consumers_gain_from_resources[i] = 0.0
            @inbounds for j in 1:p.nSpec # cannibalism allowed
                if in(CartesianIndex(i,j),p.I_ind)
                    p.consumers_gain_from_resources[i] += p.y[i,j]*p.F[i,j]
                end
            end
            p.consumers_loss_to_consumers[i] = 0.0
            @inbounds for j in p.cons_ind # cannibalism allowed
                if in(CartesianIndex(j,i),p.I_ind)
                    p.consumers_loss_to_consumers[i] += p.x[j]*p.y[j,i]*max(u[j],0.0)*p.F[j,i]/p.ef[j,i]
                end
            end
            du[i] = -p.x[i]*u[i] + p.x[i]*u[i]*p.consumers_gain_from_resources[i] - 
                    p.consumers_loss_to_consumers[i]
        else
            du[i]=0.0
        end
    end     

    nothing
end
 

# SOLVE THE ODE NUMERICALLY AND RETURN THE SOLUTION
# use a stopping criterion for extinction that executes the process
function foodweb_model(param)

    ### callbacks

    function extinction_condition(out, u, t, integrator)
        out .= u .- param.extinction_threshold # if population size is less than the extinction treshold,
        # the population is extinct 
    end

    function stop_integration!(integrator,event_index)
        terminate!(integrator)
    end

    cb_extinction_stop = VectorContinuousCallback(extinction_condition,stop_integration!,param.nSpec,interp_points=20)
    
    underlying_prob = ODEProblem(foodweb!,param.u0,param.tspan,param)

    return solve(underlying_prob,Tsit5(),saveat=param.time_grid,callback=cb_extinction_stop)

end


# SOLVE THE ODE NUMERICALLY AND RETURN THE SOLUTION
# use a stopping criterion for extinction that truncates values to zero after extinction
function foodweb_model_extinct(param)

    ### callbacks

    function extinction_condition(out, u, t, integrator)
        out .= u .- param.extinction_threshold # if population size is less than the extinction treshold,
        # the population is extinct 
    end

    function set_to_zero!(integrator,event_index)
        integrator.u[event_index]=0.0 # if population is extinct it is set to 0
    end
    
    cb_extinction_truncate = VectorContinuousCallback(extinction_condition,set_to_zero!,param.nSpec,interp_points=20)
    
    underlying_prob = ODEProblem(foodweb!,param.u0,param.tspan,param)

    return solve(underlying_prob,Tsit5(),saveat=param.time_grid,callback=cb_extinction_truncate)

end


# FOR TESTING 
# SOLVE THE ODE NUMERICALLY AND RETURN THE SOLUTION
# includes exploitation
# use a stopping criterion for extinction that truncates values to zero after extinction
function foodweb_model_exploited(param)

    ### callbacks

    function extinction_condition(out, u, t, integrator)
        out .= u .- param.extinction_threshold # if population size is less than the extinction treshold,
        # the population is extinct 
    end

    function set_to_zero!(integrator,event_index)
        integrator.u[event_index]=0.0 # if population is extinct it's set to 0
        # consider detritus...
    end

    function exploitation!(integrator)
        for i in param.exploited_pop_ind
            if integrator.u[i] > 0
                integrator.u[i] -= param.catches_prop[i]*integrator.u[i]
            end
        end
    end

    cb_extinction_truncate = VectorContinuousCallback(extinction_condition,set_to_zero!,
                            param.nSpec,interp_points=20)
    cb_exploitation = PresetTimeCallback(param.exploitation_times,exploitation!)
    cbs = CallbackSet(cb_extinction_truncate,cb_exploitation)
    
    underlying_prob = ODEProblem(foodweb!,param.u0,param.tspan,param)

    return solve(underlying_prob,Tsit5(),saveat=param.time_grid,callback=cbs)

end

end

