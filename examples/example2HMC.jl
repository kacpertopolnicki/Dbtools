#!/usr/bin/env julia

using Pkg
project_dir = joinpath(@__DIR__ , "..")
Pkg.activate(project_dir)

using Enzyme , Distributions , Random
using Dbtools , LinearAlgebra
using ArgParse

# potential energy function for Hamiltonian Monte Carlo
function u(parameters , vec)
	return (parameters[1]^2 + log(2*pi))/2.
end

# gradient of the potential energy function for Hamiltonian Monte Carlo
function nablau(parameters, vector)
	return ([parameters[1]],)
end


if abspath(PROGRAM_FILE) == @__FILE__
	
	# parse command line arguments ...
	#
	s = ArgParseSettings()
	@add_arg_table! s begin
		"configuration_file_hmc"
			help = "Path to the configuration file for hamiltonian monte carlo. For generating gaussian process. Used for gaussian process."
			required = true
		"configuration_file_hmc1"
			help = "Path to the configuration file for hamiltonian monte carlo. For generating gaussian process. Using gaussian process."
			required = true
		"configuration_file_gp"
			help = "Path to the configuration file for gaussian processes."
			required = true
		"chain_path"
			help = "Path for file with chain"
			required = true
	end

	parsed_args = parse_args(ARGS , s)

	# initial values ...
	
	initial_q = [0.0]
	initial_p = [0.0]

	# run ...

	println("RUNNING HMC ...")
	
	sp = ParamsHMC(parsed_args["configuration_file_hmc"] , u , nablau) 
	link = Link(sp ,
				initial_q,
				initial_p)

	chain = iterateHMC(sp , link)
	
	println("... DONE")
	
	println("CALCULATING GAUSSIAN PROCESS PARAMETERS ...")
	qvals = [l.q for l in chain]

	vals = vcat(
				[l.u for l in chain] ,
				[l.nu[1] for l in chain] )

	gp = GPParams(parsed_args["configuration_file_gp"] , qvals , vals , [[1.0]])
	println("... DONE")
	
	println("CALCULATING GAUSSIAN PROCESS PARAMETERS AGAIN ...")
	gp1 = GPParams(parsed_args["configuration_file_gp"] , qvals , vals , [[0.5]])
	spl1 = GPSample!(gp1 , n = 10)
	println("spl1 : ")
	display(spl1)
	gp1 = GPParams!(gp1 , [[0.5]])
	spl2 = GPSample!(gp1 , n = 10)
	println("spl2 : ")
	display(spl2)
	println("... DONE")
	
	gpNabla = GPParams(parsed_args["configuration_file_gp"] , qvals , vals , [[0.5]])

	function nablauGP(parameters, vector)
		#global gpNabla
		#gpNabla = GPParams!(gpNabla , [parameters])
		#spl = GPSample!(gpNabla , n = 1)
		#return ([spl[2,1]],)
		global gpNabla
		gpNabla = GPParams!(gpNabla , [parameters])
		return ([gpNabla.mu[2]],)
	end

	println("RUNNING HMC WITH GAUSSIAN PROCESS...")
	
	spGP = ParamsHMC(parsed_args["configuration_file_hmc1"] , u , nablauGP) 
	linkGP = Link(spGP ,
				initial_q,
				initial_p)

	chainGP = iterateHMC(spGP , linkGP)

	println("... DONE")
	
	println("SAVING CHAIN ...")
	open(parsed_args["chain_path"] , "w") do file
		for i = 1:length(chainGP)
			write(file , string(chainGP[i]) * "\n")
		end
	end
	println("...DONE")
end

