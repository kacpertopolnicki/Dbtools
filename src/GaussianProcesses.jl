module GaussianProcesses


export GPParams , GPParams! , GPSample!

using Distributions , Random , Enzyme
using LinearAlgebra
using TOML

"""
	simple_kernel(xp , xq , parameters)

Calculates the covariance between two points xp and xq using a simple gaussian kernel function.
"""
function simple_kernel(xp :: Vector{Float64} , xq :: Vector{Float64} , parameters :: Vector{Float64})
	dsq = transpose(xp - xq) *(xp - xq)
	exp(-dsq)
end

"""
	KERNEL_DICTIONARY

A dictionary that maps kernel names to their corresponding kernel functions. 
Currently, it only contains the "simple_kernel".
"""
KERNEL_DICTIONARY = Dict("simple_kernel" => simple_kernel)

"""
	covq(kernel , xp , xq , parameters)

Calculates the derivatives of the kernel with respect to the second argument and the kernel.
"""
function covq(kernel :: Function , xp :: Vector{Float64} , xq :: Vector{Float64} , parameters :: Vector{Float64})
	gradient(Reverse , kernel , Const(xp) , xq , Const(parameters))[2]
end

"""
	covp(kernel , xp , xq , parameters)

Calculates the derivatives of the kernel with respect to the first argument and the kernel.
"""
function covp(kernel :: Function , xp :: Vector{Float64} , xq :: Vector{Float64} , parameters :: Vector{Float64})
	gradient(Reverse , kernel , xp , Const(xq) , Const(parameters))[1]
end

"""
	covpq(kernel , xp , xq , parameters)

Calculates the second derivatives of the kernel.
"""
function covpq(kernel :: Function , xp :: Vector{Float64} , xq :: Vector{Float64} , parameters :: Vector{Float64})
	jacobian(Forward , covq , Const(kernel) , xp , Const(xq) , Const(parameters))[2]
end

"""
	correctPositiveDefinite(A)

Corrects a matrix A to ensure it is positive definite by symmetrizing it and adding a small 
multiple of the identity matrix based on the machine epsilon and the norm of A.
"""
function correctPositiveDefinite(A :: Matrix{Float64})

	# this is a suggestion from LLMs, the only source they give
	# is a github repository where this method is used:
	# https://github.com/higham/modified-cholesky/blob/master/modchol_ldlt.m?utm_source=chatgpt.com
	#
	B = 0.5 * (A + transpose(A))
	mul = sqrt(eps(Float64)) * norm(B)
	return B + mul * I
end
	
"""
	sigma(kernel , xx , x , D , parameters)

Calculates the covariance matrix for the Gaussian Process given a kernel function, 
a set of input points xx, a set of training points x, the dimensionality D, and kernel parameters.

This function assumes that both values and derivatives are included in the training data, 
and it constructs the covariance matrix accordingly. The size of the resulting matrix
is (N_xx * (1 + D)) x (N_x * (1 + D)), where N_xx is the number of points in xx and
N_x is the number of points in x.
"""
function sigma(kernel :: Function , xx :: Vector{Vector{Float64}} , x :: Vector{Vector{Float64}} , D :: Int , parameters :: Vector{Float64})
	N_xx = length(xx)
	N_x = length(x)

	sigma11 = zeros(Float64 , N_xx , N_x)
	for p = 1 : N_xx
		for q = 1 : N_x
			xp = xx[p]
			xq = x[q]
			sigma11[p , q] = kernel(xp , xq , parameters)
		end
	end

	sigma22 = zeros(Float64 , N_xx * D , N_x * D)
	for p = 1 : N_xx
		for q = 1 : N_x
			h = covpq(kernel , xx[p] , x[q] , parameters)
			for dp = 1 : D
				for dq = 1 : D
					row = (dp - 1) * N_xx + p
					col = (dq - 1) * N_x + q
					sigma22[row , col] = h[dp , dq]
				end
			end
		end
	end

	sigma12 = zeros(Float64 , N_xx , N_x * D)
	for p = 1 : N_xx
		for q = 1 : N_x
			c = covq(kernel , xx[p] , x[q] , parameters)
			for dq = 1 : D
				row = p
				col = (dq - 1) * N_x + q
				sigma12[row , col] = c[dq]
			end
		end
	end

	sigma21 = zeros(Float64 , N_xx * D , N_x)
	for p = 1 : N_xx
		for q = 1 : N_x
			c = covp(kernel , xx[p] , x[q] , parameters)
			for dp = 1 : D
				row = (dp - 1) * N_xx + p
				col = q
				sigma21[row , col] = c[dp]
			end
		end
	end
	
	[sigma11 sigma12;
	sigma21 sigma22]
end

"""
	GPParams
	
A structure that holds the parameters of a Gaussian Process, 
including the mean vector (mu), covariance matrix (sigma), 
distribution (dist), and random number generator (rng).
"""
####################################################################
# TODO                                                             #
# TODO remove mutable, change GPParams! to new version of GPParams #
# TODO                                                             #
# ##################################################################
Base.@kwdef mutable struct GPParams
	mu :: Vector{Float64}
	sigma :: Matrix{Float64}
	dist :: Distribution
	rng :: AbstractRNG	
	sgm_x_x :: Matrix{Float64}
	inv_sgm_x_x :: Matrix{Float64}
	D :: Int
	kernel 
	parameters :: Vector{Float64}
	x :: Vector{Vector{Float64}}
	y :: Vector{Float64}
end

function GPParams(
		x :: Vector{Vector{Float64}} , 
		y :: Vector{Float64} , 
		xx :: Vector{Vector{Float64}} , 
		D :: Int ; 
		seed = 1234 ,
		kernel = simple_kernel ,
		parameters = Float64[]) #:: GPParams

	# for the details of the procedure see for example: 
	# https://bookdown.org/rbg/surrogates/chap5.html#chap5gpprior
	
	sgm_x_x = sigma(kernel , x , x , D , parameters)
	sgm_x_x = correctPositiveDefinite(sgm_x_x)

	inv_sgm_x_x = inv(sgm_x_x)

	sgm_xx_xx = sigma(kernel , xx , xx , D , parameters)
	sgm_xx_xx = correctPositiveDefinite(sgm_xx_xx)

	sgm_xx_x = sigma(kernel , xx , x , D , parameters)

	mun = sgm_xx_x * inv_sgm_x_x * y 

	sigman = sgm_xx_xx - sgm_xx_x * inv_sgm_x_x * transpose(sgm_xx_x)
	sigman = correctPositiveDefinite(sigman)
	
	dist = MvNormal(mun , sigman)
	rng = Xoshiro(seed)
	GPParams(mun , sigman , dist , rng , sgm_x_x , inv_sgm_x_x , D , kernel , copy(parameters) , copy(x) , copy(y))
end

"""
	GPParams(config_file , x , y , xx)
Creates a GPParams object by reading configuration parameters from a TOML file,
and using the provided training data (x, y) and prediction points (xx).
Note that, since derivatives of the funtion are taken into account in the inference, 
the size of the training data y is expected to be (N_x * (1 + D)), 
where N_x is the number of training points and D is the dimensionality of the input space.

An example configuration file `config_gp` is available in the configs directory.
"""
function GPParams(
		config_file ,
		x :: Vector{Vector{Float64}} , 
		y :: Vector{Float64} , 
		xx :: Vector{Vector{Float64}} , 
		) 
	c = TOML.parsefile(config_file)
	dim = c["general"]["dimensions"]
	seed = c["general"]["seed"]
	kernel_name = c["kernel"]["kernel_name"]
	kernel_parameters = Vector{Float64}(c["kernel"]["kernel_parameters"])
	kernel = KERNEL_DICTIONARY[kernel_name]
	return GPParams(
			x , 
			y , 
			xx , 
			dim ; 
			seed = seed ,
			kernel = kernel ,
			parameters = kernel_parameters) 
end

function GPParams!(
		gp :: GPParams ,
		xx :: Vector{Vector{Float64}}
		) #:: GPParams

	# for the details of the procedure see for example: 
	# https://bookdown.org/rbg/surrogates/chap5.html#chap5gpprior
	
	D = gp.D
	kernel = gp.kernel
	parameters = gp.parameters
	
	sgm_x_x = gp.sgm_x_x
	inv_sgm_x_x = gp.inv_sgm_x_x

	x = gp.x
	y = gp.y

	#

	sgm_xx_xx = sigma(kernel , xx , xx , D , parameters)
	sgm_xx_xx = correctPositiveDefinite(sgm_xx_xx)

	sgm_xx_x = sigma(kernel , xx , x , D , parameters)

	mun = sgm_xx_x * inv_sgm_x_x * y 

	sigman = sgm_xx_xx - sgm_xx_x * inv_sgm_x_x * transpose(sgm_xx_x)
	sigman = correctPositiveDefinite(sigman)
	
	gp.mu = mun
	gp.sigma = sigman
	gp.dist = MvNormal(mun , sigman)

	return gp
end

"""
	GPSample!(gp ; n = 1)
Draws n samples from the Gaussian Process defined by the GPParams object gp.
Note that this function is marked ! since the random number generator is modified.
"""
function GPSample!(gp :: GPParams ; n = 1)
	return rand(gp.rng , gp.dist , n)
end

end
