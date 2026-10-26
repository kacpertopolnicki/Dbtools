#!/usr/bin/env julia

using Pkg
project_dir = joinpath(@__DIR__ , "..")
Pkg.activate(project_dir)

using Dbtools
using ArgParse
using Enzyme , Plots

if abspath(PROGRAM_FILE) == @__FILE__

	s = ArgParseSettings()
	@add_arg_table! s begin
		"configuration_file"
			help = "Path to the configuration file"
			required = true
		"output_file"
			help = "Output file for plots"
			required = true
	end

	parsed_args = parse_args(ARGS , s)

	# function and gradient
	function fun(x , y)
		return sin(0.1 * (x^2 + y^2)) / (x^2 + y^2)
	end

	function gradfun(x , y)
		return gradient(Reverse , fun , x , y)
	end

	# range for known values of x , y
	# NOTE the number of points should be an even number
	# NOTE to avoid dividing by 0
	r = range(-4 , 4 , 10)

	# inference will be for y = r[ny]
	ny = 5

	# known values
	xy = vec([[x , y] for x in r , y in r])
	xyval = vcat(
				 [fun(r...) for r in xy] , 
				 [gradfun(r...)[1] for r in xy] , 
				 [gradfun(r...)[2] for r in xy])

	# unknown values
	# NOTE the number of points should be an even number
	# NOTE to avoid dividing by 0
	xyxy = [[x , r[ny]] for x in range(-4 , 4 , 100)]
	# NOTE check if this a problem here too?

	# calculate Gaussian process data
	println("calculating Gaussian process data ...")
	gp = GPParams(parsed_args["configuration_file"] , xy , xyval , xyxy)
	println("... done")

	# sample
	spl = GPSample!(gp , n = 100)

	# plot

	ENV["GKSwstype"]="nul"

	pl = plot(
		 [v[1] for v in xyxy] , 
		 spl[1:100 , :] , 
		 xlabel = "x" , ylabel = "f" ,
		 legend = false , color = :gray , alpha = 0.1)

	scatter!( pl ,
			 [v for v in r] , 
			 [fun(v , r[ny]) for v in r] , 
			 color = :black , legend = false)

	pl1 = plot(
		 [v[1] for v in xyxy] , 
		 spl[101:200 , :] , 
		 xlabel = "x" , ylabel = "d(1)f" ,
		 legend = false , color = :gray , alpha = 0.1)

	scatter!( pl1 ,
			 [v for v in r] , 
			 [gradfun(v , r[ny])[1] for v in r] , 
			 color = :black , legend = false)

	pl2 = plot(
		 [v[1] for v in xyxy] , 
		 spl[201:300 , :] , 
		 xlabel = "x" , ylabel = "d(2)f" ,
		 legend = false , color = :gray , alpha = 0.1)

	scatter!( pl2 ,
			 [v for v in r] , 
			 [gradfun(v , r[ny])[2] for v in r] , 
			 color = :black , legend = false)

	savefig(plot(pl , pl1 , pl2 , layout = (3 , 1)) , parsed_args["output_file"])

end
