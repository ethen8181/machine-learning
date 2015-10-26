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



# start from here 


# --------------------------------------------------------------------------------------------
# http://digitheadslabnotebook.blogspot.tw/2012/07/linear-regression-by-gradient-descent.html

# example data 
x <- runif( 100, -5, 5 )
y <- x + rnorm(100)
data <- data.frame( x = x, y = y )

# visualize 
ggplot( data, aes( x, y ) ) +
geom_point() +
geom_smooth( method = "lm", se = FALSE )

model1 <- lm( y ~ x, data = data )


###################
x0 <- c(1,1,1,1,1) # column of 1's
x1 <- c(1,2,3,4,5) # original x-values
y  <- c(3,7,5,11,14) # create the y-matrix of dependent variables
 
# create the x-matrix of explanatory variables
library(dplyr)

data <- data.frame( x0 = x0, x1 = x1, y = y )

Normalize <- function()

GradientDescent <- function( target = y, data = data )
{
	features <- data %>% select( -target )



	return(features)
}
GradientDescent( y, data )

m <- nrow(y)

grad <- function(x, y, theta) {
  gradient <- (1/m)* (t(x) %*% ((x %*% t(theta)) - y))
  return(t(gradient))
}
 
# define gradient descent update algorithm
grad.descent <- function(x, maxit){
    theta <- matrix(c(0, 0), nrow=1) # Initialize the parameters
 
    alpha = .05 # set learning rate
    for (i in 1:maxit) {
      theta <- theta - alpha  * grad(x, y, theta)   
    }
 return(theta)
}

https://github.com/cran/gettingtothebottom
http://www.statisticsviews.com/details/feature/5722691/Getting-to-the-Bottom-of-Regression-with-Gradient-Descent.html
http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/

http://www.moneyscience.com/pg/blog/StatAlgo/read/361152/stanford-ml-3-multivariate-regression-gradient-descent-and-the-normal-equation

# housing data 
setwd("/Users/ethen/machine-learning/linear regression")
housing <- read.csv("housing.csv")










