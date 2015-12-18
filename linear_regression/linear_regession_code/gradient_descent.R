library(dplyr)

# gradient descent for linear regression

# [GradientDescent] :
# @data          : The whole data frame type data.   
# @target        : Takes a character stating column name that serves as the output variable.  
# @learning_rate : Learning rate for the gradient descent algorithm. 
# @iteration     : Halting criterion : maximum iteration allowed for training the gradient descent algorithm.
# @epsilon       : Halting criterion : If the trained parameter's difference between the two iteration is smaller than this value then the algorithm will halt.
# @normalize     : Boolean value indicating whether to performing z-score normalization for the input variables. Default to TRUE.
# @method        : Specify either "batch" or "stochastic" for the gradient descent method. Use batch for now, this will be explained later.

GradientDescent <- function( data, target, learning_rate, iteration, 
	                         epsilon = .001, normalize = TRUE, method  )	                         
{
	# separate the input and output variables 
	input  <- data %>% select( -one_of(target) ) %>% as.matrix()
	output <- data %>% select( one_of(target) ) %>% as.matrix()

	# normalize the input variables if specified
	# record the mean and standard deviation  
	if(normalize)
	{
		input <- scale(input)
		input_mean <- attr( input, "scaled:center" )
		input_sd   <- attr( input, "scaled:scale" )
	}

	# implementation trick, after the normalizing the original input column
	# add a new column of all 1's to the first column, this serves as X0
	input <- cbind( theta0 = 1, input )

	# theta_new : initialize the theta value as all 1s
	# theta_old : a random number whose absolute difference between new one is 
	#             larger than than epsilon 
	theta_new <- matrix( 1, ncol = ncol(input) )
	theta_old <- matrix( 2, ncol = ncol(input) )

	# cost function 
	costs <- function( input, output, theta )
	{
		sum( ( input %*% t(theta) - output )^2 ) / ( 2 * nrow(output) )
	}

	# records the theta and cost value for visualization ; add the inital guess 
	theta_trace <- vector( mode = "list", length = iteration ) 
	theta_trace[[1]] <- theta_new
	costs_trace <- numeric( length = iteration )
	costs_trace[1] <- costs( input, output, theta_old )

	# first derivative of the cost function 
	if( method == "batch" )
	{				
		derivative <- function( input, output, theta, step )
		{
			error <- ( input %*% t(theta) ) - output 
			descent <- ( t(input) %*% error ) / nrow(output)
			return( t(descent) )
		}		
	}else # stochastic gradient descent, using one training sample per update 
	{
		derivative <- function( input, output, theta, step )
		{
			r <- step %% nrow(input) + 1
			error <- input[ r, ] %*% t(theta) - output[ r, ]
			descent <- input[ r, ] * error
			return(descent)
		}
	}	

	# keep updating as long as any of the theta difference is still larger than epsilon
	# or exceeds the maximum iteration allowed
	step <- 1 
	while( any( abs(theta_new - theta_old) > epsilon ) & step <= iteration )
	{
		step <- step + 1

		# gradient descent 
		theta_old <- theta_new
		theta_new <- theta_old - learning_rate * derivative( input, output, theta_old, step )

		# record keeping 
		theta_trace[[step]] <- theta_new
		costs_trace[step]   <- costs( input, output, theta_new )
	}
	
	# returns the noramalized mean and standard deviation for each input column
	# and the cost, theta record 
	costs <- data.frame( costs = costs_trace )
	theta <- data.frame( do.call( rbind, theta_trace ), row.names = NULL )
	norm  <- data.frame( input_mean = input_mean, input_sd = input_sd )

	return( list( costs = costs, theta = theta, norm = norm ) )
}




