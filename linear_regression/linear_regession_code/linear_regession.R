# linear regression
library(grid)
library(dplyr)
library(scales)
library(ggplot2)
setwd("/Users/ethen/machine-learning/linear_regression")


# original formula 
Formula <- function(x) 1.2 * (x-2)^2 + 3.2

# visualize the function, and the optimal solution
# drawing function formula after giving the x coordinates 
ggplot( data.frame( x = c( 0, 4 ) ), aes( x ) ) + 
stat_function( fun = Formula ) + 
geom_point( data = data.frame( x = 2, y = Formula(2) ), aes( x, y ), 
	        color = "blue", size = 3 ) + 
ggtitle( expression( 1.2 * (x-2)^2 + 3.2 ) )


# ------------------------------------------------------------------------------------
# gradient descent toy example : 
# keep updating the x value until the difference between this iteration and the last 
# one, is smaller than epsilon (a given small value) or the process count of updating the 
# x value surpass user-specified iteration

# first derivative of the formula
Derivative <- function(x) 2 * 1.2 * (x-2) 

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


# --------------------------------------------------------------------------------
# housing data

housing <- read.table( "housing.txt", header = TRUE, sep = "," )

# example :
# suppose you've already calculated that the difference of 
# the two rows are 100 and 200 respectively, then 
# using the first two rows of the input variables
housing[ c( 1, 2 ), -3 ]

# multiply 100 with row 1 
( row1 <- 100 * housing[ 1, -3 ] )

# multuply 200 with row 2
( row2 <- 200 * housing[ 1, -3 ] )

# sum each row up
list( area = sum( row1[1] + row2[1] ), bedrooms = sum( row1[2] + row2[2] ) )


# z-score normalize 
Normalize <- function(x) ( x - mean(x) ) / sd(x)


# --------------------------------------------------------------------------------------------
# gradient descent 
source("linear_regession_code/gradient_descent.R")

trace_b <- GradientDescent( data = housing, target = "price",  
	                        learning_rate = 0.05, iteration = 500, method = "batch" )
# final parameters 
parameters_b <- trace_b$theta[ nrow(trace_b$theta), ]

# linear regression 
normed <- apply( housing[ , -3 ], 2, scale )
normed_data <- data.frame( cbind( normed, price = housing$price ) )
model <- lm( price ~ ., data = normed_data )


# visualize cost
costs_df <- data.frame( iteration = 1:nrow(trace_b$cost), 
	                    costs = trace_b$cost / 1e+8 )
ggplot( costs_df, aes( iteration, costs ) ) + geom_line()


# ---------------------------------------------------------------------------
# appendix : summary of the linear model 

summary(model)

# residuals 
summary( model$residuals )
summary( normed_data$price - model$fitted.values )


# t-value
library(broom)
( coefficient <- tidy(model) )
coefficient$estimate / coefficient$std.error


# p-value
summary(model)$df
( df <- nrow(normed_data) - nrow(coefficient) )
pt( abs(coefficient$statistic), df = df, lower.tail = FALSE ) * 2


# r squared

# @y  : original output value
# @py : predicted ouput value
RSquared <- function( y , py )
{
	rss <- sum( ( y - py )^2 )
	tss <- sum( ( y - mean(y) )^2 )
	return( 1 - rss / tss )
}

summary(model)$r.squared
RSquared( normed_data$price, model$fitted.values )


# adjusted r square 
k  <- nrow(coefficient) - 1

# @y  : original output value
# @py : predicted ouput value
# @k  : number of the model's coefficient, excluding the intercept
AdjustedRSquared <- function( y, py, k )
{	 
	n  <- length(y)
	r2 <- RSquared( y, py )	
	return( 1 - ( 1 - r2 ) * ( n - 1 ) / ( n - k - 1 ) )
}
summary(model)$r.squared * df / nrow(normed_data)
summary(model)$adj.r.squared
AdjustedRSquared( normed_data$price, model$fitted.values, k )


# linear regression plot 
source("/Users/ethen/machine-learning/linear_regression/linear_regession_code/LMPlot.R")
lm_plot <- LMPlot( model = model, actual = normed_data$price )
grid.draw(lm_plot$plot)
lm_plot$outlier


# variance inflation score 
library(car)
car::vif(model)

# area calculation 
area_model <- lm( area ~ .-price, data = normed_data )
area_r2 <- RSquared( y = normed_data$area, py = area_model$fitted.values )
1 / ( 1 - area_r2 )

# ----------------------------------------------------------------------------
# test code 
# normal equation
solve( t(input) %*% input ) %*% t(input) %*% output


# stochastic approach 
trace_s <- GradientDescent( target = "price", data = housing, 
	                        learning_rate = 0.05, iteration = 3000, method = "stochastic" )
parameters_s <- trace_s$theta[ nrow(trace$theta), ]

library(microbenchmark)
runtime <- microbenchmark( 

	batch = GradientDescent( target = "price", data = housing, 
	                         learning_rate = 0.05, iteration = 500, method = "batch" ),
	stochastic = GradientDescent( target = "price", data = housing, 
	                              learning_rate = 0.05, iteration = 521, method = "stochastic" )

)

