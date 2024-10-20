# variational-bayes-for-atn
Parameter estimation for allometric trophic network models using a variational Bayesian inverse problem approach.

These codes use Julia version 1.7.0 and dependences from atn/Project.toml

Includes separate code packages for synthetic food webs with ten and 25 guilds.

Generate synthetic data using code_package_for_simulated_data_*/simulate_foodwebs.jl

Fit models to data using code_package_for_simulated_data_*/method_validation.jl

Visualize and report results using code_package_for_simulated_data_*/report_results_for_foodwebs_public.jl
