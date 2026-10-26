module HamiltonianMonteCarlo

export TuneEpsilon , Link , iterateHMC , ParamsHMC , IterateLeapFrog

using Enzyme
using Distributions , Random , Plots 
using LinearAlgebra 
using TOML
using ProgressBars

"""
	ParamsHMC

	Parameters for the Hamiltonian Monte Carlo simulation. 
	This structure holds all the necessary parameters and settings for running 
	the HMC algorithm, including the number of parameters (n), 
	leap frog settings (epsilon ,max_epsilon , tune_iter , tune_f , length), 
	number of iterations (all), mass parameter (mass),
	distribution for momenta (distp), data for inference (data), accuracy goal for tuning (accuracy), 
	random number generator (rng),
	and functions for potential energy and its gradient (u_function , nablau_function).

"""
Base.@kwdef struct ParamsHMC
	# all simulation parameters in a neat little structure
	n :: Int							# the number of parameters
	epsilon :: Float64					# time step for leap frog
	max_epsilon :: Float64				# time step for leap frog
	tune_iter :: Int					# time step for leap frog
	tune_f :: Float64					# time step for leap frog
	length :: Float64					# number of iterations for leap frog
	#initial_q :: Vector{Float64}		# initial q (positions) 
	#initial_p :: Vector{Float64}		# initial q (momenta) 
	all :: Int							# number of iterations for Hamiltonian Monte Carlo
	mass :: Float64						# mass parameter for Hamiltonian Monte Carlo
	distp								# distribution for momenta
	datalength							# number of data points
	data								# data generated from the true distribution
	accuracy :: Float64					# accuracy goal for tuning epsilon
	rng									# random number generator
	u_function							# potential energy function
	nablau_function						# gradient of the potential energy function
end

"""
	ParamsHMC

	Constructor of ParamsHMC from a configuration file.
	Accepts the potential function (u) and its gradient (nablau) as arguments, 
	along with initial conditions (init_q , init_p).
	Data can be Nothing or a matrix of observed data points for inference.
"""
function ParamsHMC(config_file , u , nablau; data = Nothing)
	c = TOML.parsefile(config_file)
	n = c["general"]["n"]
	seed = c["general"]["seed"]
	random_number_generator = Xoshiro(seed)
	dta_copy = Nothing
	if data != Nothing
		dta_copy = copy(data)
	end
	dta_length = Nothing
	if data != Nothing
		dta_length = size(dta_copy)[2]
	end
	ParamsHMC(n = c["general"]["n"] ,
		   #dimensions = dim ,
		   epsilon = c["leap_frog"]["epsilon"] ,
		   max_epsilon = c["leap_frog"]["max_epsilon"] ,
		   tune_iter = c["leap_frog"]["tune_iter"] ,
		   tune_f = c["leap_frog"]["tune_f"] ,
		   length = c["leap_frog"]["length"] ,
		   #initial_q = copy(init_q) ,
		   #initial_p = copy(init_p) ,
		   all = c["hamiltonian_monte_carlo"]["iterations"] ,
		   mass = c["hamiltonian_monte_carlo"]["mass"] ,
		   distp = MvNormal(c["hamiltonian_monte_carlo"]["mass"] .* Matrix(1.0I , n , n)) ,
		   #true_parameters = tp ,
		   #true_sigma = ts ,
		   #true_distribution = td ,
		   datalength = dta_length ,
		   data = dta_copy ,
		   accuracy = c["leap_frog"]["accuracy"],
		   rng = random_number_generator ,
		   u_function = u ,
		   nablau_function = nablau
		   )
end

"""
	Link

A single link in the Hamiltonian Monte Carlo chain, 
representing a state of the system with position (q), momentum (p), 
potential energy (u), and gradient of potential energy (nu).
"""
struct Link
	q :: Vector{Float64}			# position
	p :: Vector{Float64}			# momentum
	u :: Float64					# potential -ln(pdf)
	nu :: Vector{Float64}			# gradient of potential
end

"""
	Link(sp , q , p)

Creates a Link object from the given parameters (sp), momentum (p), and position (q).
Calculates the potential energy and its gradient based on the provided functions in sp.
"""
function Link(sp , q , p)
	q = copy(q)
	p = copy(p)

	uu = 0.0
	if sp.datalength == Nothing
		uu = sp.u_function(q , Nothing)
	else
		for i = 1 : sp.datalength
			uu += sp.u_function(q , sp.data[: , i])
		end
	end

	gu = zeros(Float64 , sp.n)
	if sp.datalength == Nothing
		gu = sp.nablau_function(q , Nothing)[1]
	else
		for i = 1:sp.datalength
			gu += sp.nablau_function(q , sp.data[: , i])[1]
		end
	end

	return Link(q , p , uu , gu)
end

Base.copy(lnk :: Link) = Link(copy(lnk.q) , copy(lnk.p) , copy(lnk.u) , copy(lnk.nu))
function Base.show(io :: IO , lnk :: Link)
	qstr = join(map(string , lnk.q) , " ")
	pstr = join(map(string , lnk.p) , " ")
	ustr = string(lnk.u)
	nustr = join(map(string , lnk.nu) , " ")
	print(io , qstr * " " * pstr * " " * ustr * " " * nustr)
end

"""
	LeapFrog

Runs a single leapfrog step in the Hamiltonian Monte Carlo algorithm, 
updating the position and momentum based on the potential energy and its gradient.
The argument is a ParamsHMC object (sp) and a Link object (lnk),
representing the current state of the system. The optional argument 
(eps) allows for overriding the time step defined in sp.
"""
function LeapFrog(sp , lnk :: Link ; eps = nothing)
	epsilon = sp.epsilon
	if eps != nothing
		epsilon = eps
	end

	gu = lnk.nu

	pp = lnk.p .- (0.5 * epsilon) .* gu
	qqq = lnk.q .+ (epsilon / sp.mass) .* pp

	ggu = zeros(Float64 , sp.n)
	if sp.datalength == Nothing
		ggu = sp.nablau_function(qqq , Nothing)[1]
	else
		for i = 1:sp.datalength
			ggu += sp.nablau_function(qqq , sp.data[: , i])[1]
		end
	end

	ppp = pp .- (0.5 * epsilon) .* ggu

	uuu = 0.0
	if sp.datalength == Nothing
		uuu = sp.u_function(qqq , Nothing)
	else
		for i = 1 : sp.datalength
			uuu += sp.u_function(qqq , sp.data[: , i])
		end
	end

	return Link(qqq , ppp , uuu , ggu)
end

"""
	IterateLeapFrog

Runs multiple leapfrog steps in the Hamiltonian Monte Carlo algorithm,
updating the position and momentum based on the potential energy and its gradient.
The argument is a ParamsHMC object (sp) and a Link object (lnk),
representing the current state of the system. The optional argument
(eps) allows for overriding the time step defined in sp.
"""
function IterateLeapFrog(sp , lnk :: Link ; eps = nothing)
	lc = copy(lnk)
	iterations = round(sp.length / sp.epsilon)
	if eps != nothing
		iterations = round(sp.length / eps)
	end
	for i = 1 : iterations
		lc = LeapFrog(sp , lc ; eps = eps)
	end
	return lc
	# WITH THE FOLLOWING WE HAVE JUST METROPOLIS HASTINGS:
	#=lq = lc.q
	lp = lc.p
	add = 0.3 * (2.0 * rand(sp.rng) - 1.0)
	newlq = lq .+ add
	return Link(sp , newlq , lp)=#
end

"""
	TuneEpsilon

	Currently experimental and does not work very well.

	Adjusts the time step (epsilon) for the leapfrog integrator in the Hamiltonian Monte Carlo algorithm.
	The argument is a ParamsHMC object (sp) and initial position (initial_q) and momentum (initial_p).
	The function iteratively adjusts epsilon based on the difference between the final and initial states
	after a series of leapfrog steps, aiming to achieve a specified accuracy.
"""
function TuneEpsilon(sp , initial_q , initial_p)
	# TODO this is not a great approach

	it = sp.tune_iter
	f = sp.tune_f

	current_epsilon = sp.max_epsilon
	link = Link(sp , 
			  initial_q,
			  initial_p
			)
	fin = IterateLeapFrog(sp , link ; eps = current_epsilon)

	ok = true
	for i in 1:it
		current_epsilon = f * current_epsilon
		link = Link(sp , 
				  sp.initial_q,
				  sp.initial_p
				)
		nif = IterateLeapFrog(sp , link ; eps = current_epsilon)

		# TODO use this?
		#diff = sqrt(norm(nif.q - fin.q)^2 + norm(nif.p - fin.p)^2)
		
		diff = norm(nif.q - fin.q)
		println(i , " current_epsilon : " , current_epsilon , " diff : " , diff)

		if diff < sp.accuracy
			return current_epsilon
		end

		fin = nif
	end

	return sp.epsilon
end

"""
	HMC

	Runs a single iteration of the Hamiltonian Monte Carlo algorithm,
	updating the position and momentum based on the potential energy and its gradient.
	The argument is a ParamsHMC object (sp) and a Link object (lnk),
	representing the current state of the system. The optional argument
	(eps) allows for overriding the time step defined in sp.
"""
function HMC(sp , lnk :: Link ; eps = nothing)
	newp = rand(sp.rng , sp.distp)
	fin = IterateLeapFrog(sp , 
						  Link(lnk.q , newp , lnk.u , lnk.nu) ; 
						  eps = eps)
	acc = min(
		1.0 , 
		exp(
			lnk.u - fin.u
			#+ (sum(lnk.p .* lnk.p) / (2.0 * sp.mass))
			+ (sum(newp  .* newp)  / (2.0 * sp.mass))
			- (sum(fin.p .* fin.p) / (2.0 * sp.mass))
			)
		)
	# WITH THE FOLLOWING WE HAVE JUST METROPOLIS HASTINGS:
	#=acc = min(
		1.0 , 
		exp(
			lnk.u - fin.u
			))=#
	if rand(sp.rng) <= acc
		return fin , 1
	end
	return lnk , 0
end

"""
	iterateHMC

	Runs multiple iterations of the Hamiltonian Monte Carlo algorithm,
	updating the position and momentum.
	The argument is a ParamsHMC object (sp) and a Link object (lnk),
	representing the current state of the system. The optional argument
	(eps) allows for overriding the time step defined in sp.
"""
function iterateHMC(sp , lnk ; eps = nothing)
	accepted = 0
	current_link = copy(lnk)
	chain = [current_link]
	for i = ProgressBar(1 : sp.all)
		current_link , to_add = HMC(sp , current_link ; eps = eps)
		accepted += to_add
		push!(chain , current_link)
	end
	println("accepted % : " * string(100. * accepted / sp.all))
	return chain
end

end 
