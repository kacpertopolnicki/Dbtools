#!/usr/bin/env julia

using Pkg
project_dir = joinpath(@__DIR__ , "..")
Pkg.activate(project_dir)

using Enzyme , Distributions , Random
using Dbtools , LinearAlgebra
using ArgParse

# potential energy function for Hamiltonian Monte Carlo
function u(parameters , vec)
	return (parameters[1]^2 + parameters[2]^2) * 0.5 + log(2*pi)
end

# gradient of the potential energy function for Hamiltonian Monte Carlo
function nablau(parameters, vector)
	return ([parameters[1] , parameters[2]],)
end


if abspath(PROGRAM_FILE) == @__FILE__
	
	# parse command line arguments ...
	#
	s = ArgParseSettings()
	@add_arg_table! s begin
		"configuration_file"
			help = "Path to the configuration file"
			required = true
		"chain_path"
			help = "Path for file with chain"
			required = true
	end

	parsed_args = parse_args(ARGS , s)

	# initial values ...
	
	initial_q = [0.0 , 0.0]
	initial_p = [0.0 , 0.0]

	# run ...

	sp = ParamsHMC(parsed_args["configuration_file"] , u , nablau) 

	println("RUNNING HMC ...")
	link = Link(sp ,
				initial_q,
				initial_p)
	#println(link)

	#newlink = IterateLeapFrog(sp , link)
	#println(newlink)

	chain = iterateHMC(sp , link)
	println("... DONE")

	println("SAVING CHAIN ...")
	open(parsed_args["chain_path"] , "w") do file
		for i = 1:length(chain)
			write(file , string(chain[i]) * "\n")
		end
	end
	println("...DONE")
end

