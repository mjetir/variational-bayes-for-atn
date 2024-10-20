
# Maria Tirronen 2024
# https://orcid.org/0000-0001-6052-3186


# Set food web parameters.
module SetFoodWebParameters

using StatsPlots
using LinearAlgebra
using Random, Distributions, Statistics
using CSV, DataFrames
using DelimitedFiles
using BioEnergeticFoodWebs

# FOODWEB STRUCT 
mutable struct FoodWeb

        data_name::String

        tspan::Tuple{Float64,Float64}
        time_grid::Vector{Float64}

        exploitation_times::Vector{Float64}

        ### the initial structure of the foodweb:
        # the number of species (guilds)
        nSpec::Int64

        # the number of feeding links
        nLinks::Int64

        # the number of producers
        nProd::Int64

        # the number of consumers
        nCons::Int64

        # the indeces of feeding links in the feeding matrix
        I_ind::Vector{CartesianIndex{2}}

        # the indices of producers and consumers
        prod_ind::Vector{Int64}
        cons_ind::Vector{Int64}

        # the indices of exploited populations
        exploited_pop_ind::Vector{Int64}

        ### initial biomasses
        u0::Vector{Float64}

        ### propotional catches (% of the biomass)
        catches_prop::Vector{Float64}

        ### functional response parameters
        # half saturation constants        
        fr_half_sat::Matrix{Float64} # to be estimated

        # the shape parameters are given as a+q, a>1, q>=0
        # a:
        fr_shape_const::Matrix{Float64}

        # the qs, to be estimated
        # initialized for avoiding temporary allocations
        fr_shape::Matrix{Float64}

        # feeding preferences
        # pref[i,j]: the relative preference of species i for species j, normalized so that
        # the sum over prey species is one for each predator
        pref::Matrix{Float64}

        ### other parameters
		
        # the intrinsic growth rates of producers
        r::Vector{Float64}

        # producer competition coefficients
        c::Matrix{Float64}

        # the mass-specific metabolic rates of consumers
        x::Vector{Float64} 

        # the maximum consumption rates of species feeding on each other,
        # relative to the metabolic rates of consumers
        # y[i,j]: the rate when species i feeds on species j, relative to the
        # metabolic rate of i
        y::Matrix{Float64}

        # the assimilation efficiences
        # e[i,j]: the assimilation efficiency when species i feeds on species j
        ef::Matrix{Float64}

        # the system-wide carrying capacity 
        K::Float64 

        ### threshold for extinction
        extinction_threshold::Float64

        ### normal noise around the mean abundances: the sigma parameters,
        # assumed constant in time
        std::Vector{Float64} 

        ### for avoiding temporary allocations:
        # functional responses
        F::Matrix{Float64}

        # producer growth function
        G::Vector{Float64}

        # for avoiding temporary allocations, used in the model formulation
        # initialized for all species but all values are not used
        producers_loss_to_consumers::Vector{Float64}
        consumers_gain_from_resources::Vector{Float64}
        consumers_loss_to_consumers::Vector{Float64}
        fr_denom_sum::Matrix{Float64}
end


# INITIALIZE A GENERIC FOODWEB STRUCT
# I : give a proper feeding link matrix
# u0 : initial biomasses
# fr_half_sat : half saturation constants in functional responses
# fr_shape : q
# shape parameters in functional responses are given as a+q, a>1, q>=0
# std: the amount of scatter around the mean curve (normal or lognormal noise)
function initialize_generic_foodweb(data_name,I::Matrix{Int64},u0,fr_half_sat,fr_shape,std)

        ### the time range of simulations
        tspan = (0.0,2000.0) 
        sampling = 1.0
        time_grid = tspan[2]-30.0:sampling:tspan[2] # when fitting the model to data, 
                                                #save the solution in the last 30 time steps
#        time_grid = tspan[1]:sampling:tspan[2] # for simulation of data

        exploitation_times = [2003.0]

        ### the initial structure of the food web 

        I_ind=findall(x->(x.>0.0),I)

        # the indices of producers and consumers
        prod_ind_=findall(i->i==0,sum(I,dims=2))
        cons_ind_=findall(i->(i>0),sum(I,dims=2))

        prod_ind=[prod_ind_[i][1] for i=1:length(prod_ind_)]
        cons_ind=[cons_ind_[i][1] for i=1:length(cons_ind_)]

        exploited_pop_ind = cons_ind

        # the number of species
        nSpec = size(I,1)
        nLinks = length(I_ind)
        nProd = length(prod_ind)
        nCons = length(cons_ind)

        # proportional catches
        catches_prop = zeros(nSpec)
        catches_prop[cons_ind] .= 0.5

        ### other functional response parameters
        # the functional response shape parameters are given as a+q (a constant, q varies population 
        # by population)
        fr_shape_const = ones(nSpec,nSpec) # can be set higher

        # feeding preferences
        # not set
        pref = zeros(nSpec,nSpec)

        ### metabolic rates
        shortest_p = BioEnergeticFoodWebs.distance_to_producer(I) 

        # the intrinsic growth rates of producers are all set to one:
        r = ones(nSpec)

        # producer competition coefficients are all set to one:
        c = ones(nSpec,nSpec)

        # the mass-specific metabolic rates of consumers
        x =  [0.314*(100^(shortest_p[i]-1))^(-0.15) for i=1:nSpec]
        x[prod_ind].=0.0 # exclude producers by setting their growth rates to zero

        ### other parameters
        # the relative maximum consumption rates of species feeding on each other
        # y[i,j]: the rate when species i feeds on species j, relative to the metabolic rate of i
        y = 8*ones(nSpec, nSpec) # 4/8 (ectotherm vertebrates/invertebrate predators)
        y[prod_ind,:] .= 0

        # the assimilation efficiences
        # e[i,j]: the assimilation efficiency when species i feeds on species j
        # initialize: 
        ef = 0.85.*ones(nSpec,nSpec) 
        ef[:,prod_ind].=0.45
        ef[prod_ind,:].=0.0
        
        # the system-wide carrying capacity 
        K = 1.0
        
        extinction_threshold = 0.000001 # 10^(-6)

        ### for avoiding temporary allocations:
            
        # functional response
        F = zeros(nSpec,nSpec)
            
        # for avoiding temporary allocations, used in the model formulation
        # initialized for all species but all values are not used
        G = zeros(nSpec)
        producers_loss_to_consumers = zeros(nSpec)
        consumers_gain_from_resources = zeros(nSpec)
        consumers_loss_to_consumers = zeros(nSpec)
        fr_denom_sum = zeros(nSpec,nSpec)    

        return FoodWeb(data_name,
        tspan,
        time_grid, exploitation_times,
        nSpec,
        nLinks, nProd, nCons,
        I_ind,
        prod_ind, cons_ind, exploited_pop_ind,
        u0, catches_prop,
        fr_half_sat,
        fr_shape_const,
        fr_shape,
        pref,    
        r, c,
        x, y, ef,  
        K,extinction_threshold, 
        std,
        F, G,
        producers_loss_to_consumers, consumers_loss_to_consumers,
        consumers_gain_from_resources, 
        fr_denom_sum)
end



function copy_foodweb(fw_old)
    data_name = fw_old.data_name

    tspan = fw_old.tspan
    time_grid = copy(fw_old.time_grid)
    exploitation_times = copy(fw_old.exploitation_times)

    nSpec = copy(fw_old.nSpec)

    nLinks = copy(fw_old.nLinks)

    nProd = copy(fw_old.nProd)

    nCons = copy(fw_old.nCons)

    I_ind = copy(fw_old.I_ind)

    prod_ind = copy(fw_old.prod_ind)
    cons_ind = copy(fw_old.cons_ind)
    exploited_pop_ind = copy(fw_old.exploited_pop_ind)

    u0 = copy(fw_old.u0)
    catches_prop = copy(fw_old.catches_prop)

    fr_half_sat = copy(fw_old.fr_half_sat)

    fr_shape_const = copy(fw_old.fr_shape_const)
    fr_shape = copy(fw_old.fr_shape)

    pref = copy(fw_old.pref)

    r = copy(fw_old.r)

    c = copy(fw_old.c)
    
    x = copy(fw_old.x)

    y = copy(fw_old.y)

    ef = copy(fw_old.ef)

    K = copy(fw_old.K)

    extinction_threshold = copy(fw_old.extinction_threshold)

    std = copy(fw_old.std)

    F = copy(fw_old.F)

    G = copy(fw_old.G)

    producers_loss_to_consumers = copy(fw_old.producers_loss_to_consumers)
    consumers_gain_from_resources = copy(fw_old.consumers_gain_from_resources)
    consumers_loss_to_consumers = copy(fw_old.consumers_loss_to_consumers)
    fr_denom_sum = copy(fw_old.fr_denom_sum)

    return FoodWeb(data_name,
        tspan,
        time_grid, exploitation_times,
        nSpec,
        nLinks,nProd,nCons,
        I_ind,
        prod_ind, cons_ind, exploited_pop_ind,
        u0, catches_prop,
        fr_half_sat,
        fr_shape_const,
        fr_shape,
        pref,  
        r,c, x, y, ef,  
        K,extinction_threshold, 
        std,
        F, G,
        producers_loss_to_consumers, consumers_loss_to_consumers,
        consumers_gain_from_resources, 
        fr_denom_sum)
end

end #module

