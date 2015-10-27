# gradient descent in r 
# http://www.r-bloggers.com/gradient-descent-in-r/

# -----------------------------------------------------------------------------------
# drawing function formula after giving the x coordinates 
library(grid)
library(ggplot2)

# original formula 
Formula <- function(x) 1.2 * (x-2)^2 + 3.2

# first derivative of the formula
Derivative <- function(x) 2 * 1.2 * (x-2) 

# visualize the function, and the optimal solution
ggplot( data.frame( x = c( 0, 4 ) ), aes( x ) ) + 
stat_function( fun = Formula ) + 
geom_point( data = data.frame( x = 2, y = Formula(2) ), aes( x, y ), 
	        color = "blue", size = 3 ) + 
ggtitle( expression( 1.2 * (x-2)^2 + 3.2 ) )


# ------------------------------------------------------------------------------------
# gradient descent implementation : 
# keep updating the x value until the difference between this iteration and the last 
# one, is smaller than epsilon (a given small value) or the process count of updating the 
# x value surpass user-specified iteration

# x_new : initial guess for the x value
# x_old : assign a random value to start for the first iteration 
x_new <- .1 
x_old <- 0
# manually assign a fix learning rate 
learning_rate <- .6
# other paramaters : manually assign epilson value, maximum iteration allowed 
epsilon <- .05
step <- 1
iteration <- 10

# records the x and y value for visualization ; add the inital guess 
xtrace <- list() ; ytrace <- list()
xtrace[[1]] <- x_new ; ytrace[[1]] <- Formula(x_new)

while( abs( x_new - x_old ) > epsilon & step <= iteration )
{
	# update iteration count 
	step <- step + 1	
	
	# gradient descent
	x_old <- x_new
	x_new <- x_old - learning_rate * Derivative(x_old)
	
	# record keeping 
	xtrace[[step]] <- x_new
	ytrace[[step]] <- Formula(x_new)	
}

# create a data points' coordinates
record <- data.frame( x = do.call( rbind, xtrace ), y = do.call( rbind, ytrace ) )

# create the segment between each points (gradient steps)
segment <- data.frame( x = double(), y = double(), xend = double(), yend = double() )
for( i in 1:( nrow(record)-1 ) )
{
	segment[ i, ] <- cbind( record[ i, ], record[ i+1, ] )	
}

# visualize the gradient descent's value 
ggplot( data.frame( x = c( 0, 4 ) ), aes( x ) ) + 
stat_function( fun = Formula ) + 
ggtitle( expression( 1.2 * (x-2)^2 + 3.2 ) ) + 
geom_point( data = record, aes( x, y ), color = "red", size = 3, alpha = .8, shape = 2 ) +
geom_segment( data = segment , aes( x = x, y = y, xend = xend, yend = yend ), 
              color = "blue", alpha = .8, arrow = arrow( length = unit( 0.25, "cm" ) ) )


##################
# start from here 

library(dplyr)

# their code do not work 

# http://digitheadslabnotebook.blogspot.tw/2012/07/linear-regression-by-gradient-descent.html
# http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/
# http://www.moneyscience.com/pg/blog/StatAlgo/read/361152/stanford-ml-3-multivariate-regression-gradient-descent-and-the-normal-equation

# housing data 
setwd("/Users/ethen/machine-learning/linear regression")
housing <- read.csv("housing.csv")
model <- lm( price ~. , data = housing )


# normalize the vector to be between values of 0 and 1 [0,1]
Normalize <- function(x) ( x - min(x) ) / ( max(x) - min(x) )

GradientDescent <- function( target, data, learning_rate = .0001, 
	                         epsilon = .001, iteration = 1000, normalize = TRUE )	                         
{
	# separate the input and output variables 
	input  <- data %>% select( -one_of(target) ) %>% as.matrix()
	output <- data %>% select( one_of(target) ) %>% as.matrix()

	# normalize the input variables if specified 
	if(normalize)
	{
		input <- apply( input, 2, Normalize )
	}

	# add a new column of all 1's to the first column, this serves as X0
	input <- cbind( theta0 = 1, input )

	# theta_new : initialize the theta value as all 0s
	# theta_old : a random number whose absolute difference between new is 
	#             larger than than epsilon 
	theta_new <- matrix( c( 7000, 100, -9000 ), nrow = 1 )
	theta_old <- matrix( rep( 2, ncol(input) ), nrow = 1 )

	# cost function 
	costs <- function( input, output, theta )
	{
		sum( ( input %*% t(theta) - output )^2 ) / ( 2 * length(output) )
	}

	# records the theta and cost value for visualization ; add the inital guess 
	theta_trace <- list() ; costs_trace <- list()
	theta_trace[[1]] <- theta_new
	costs_trace[[1]] <- costs( input, output, theta_old )

	# first derivative of the cost function 
	derivative <- function( input, output, theta )
	{
		error <- ( input %*% t(theta) ) - output 
		descent <- ( t(input) %*% error ) / nrow(output)
		return( t(descent) )
	}

	# keep updating as long as any of the theta difference is still larger than epsilon
	# or exceeds the maximum iteration allowed
	step <- 1 
	while( any( abs(theta_new - theta_old) > epsilon ) & step <= iteration )
	{
		step <- step + 1

		# gradient descent 
		theta_old <- theta_new
		theta_new <- theta_old - learning_rate * derivative( input, output, theta_old )

		# record keeping 
		theta_trace[[step]] <- theta_new
		costs_trace[[step]] <- costs( input, output, theta_new )
	}

	costs <- data.frame( costs = do.call( rbind, costs_trace ) )
	theta <- data.frame( do.call( rbind, theta_trace ), row.names = NULL )

	return( list( costs = costs, theta = theta ) )
}


trace <- GradientDescent( target = "price", data = housing, learning_rate = .001 )


solve( t(input) %*% input ) %*% t(input) %*% output




