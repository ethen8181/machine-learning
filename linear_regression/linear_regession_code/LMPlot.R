library(grid)
library(scales)
library(ggplot2)
library(gridExtra)
library(data.table)

# [LMPlot] : 
# visualizations that works with linear regression
# @model  : linear regression model object
# @actual : your data's actual (original) output value
# returns : 1. plot    : returns the four plot in one side by side plot
#   		2. outlier : observation index of the possible outliers, if none return NULL

LMPlot <- function( model, actual )
{
	if( class(model) != "lm" )
		stop( "Must be a linear model" )

	cooks_distance <- cooks.distance(model)
	plot_data <- data.table( actual 		= actual, 
					 		 predicted 		= model$fitted.values,
					 		 residuals 		= model$residuals,
					 		 cooks_distance = cooks_distance )

	# cooks distance > 1 or > 4 / number of data is considered a possible outlier 
	boolean <- ( cooks_distance > 1 ) | ( cooks_distance > 4 / length(actual) )
	outlier <- which(boolean)
	
	if( length(outlier) > 0 )
		plot_data[ , boolean := boolean ]

	# -- plot -- 
	# defines the basic layout,
	# if there's outlier, then color the outlier plots,
	# if not then simply add the points to the aesthetic 

	# 1. cooks distance v.s. predicted value 
	cooks <- ggplot( plot_data, aes( predicted, cooks_distance ) ) + 
			 scale_x_continuous( labels = comma ) + 
			 ggtitle( "Cooks Distance of Predicted Value" )

	# 2. predicted value versus actual value :
	# if the model is considered to a good estimate of the outcome, 
	# there should be strong correlation between the model’s predictions and its actual results.
	pred <- ggplot( plot_data, aes( predicted, actual ) ) + 
			ggtitle( "Predicted Value v.s. Actual Value" ) + 
			scale_x_continuous( labels = comma ) + 
			scale_y_continuous( labels = comma )

	# 3. residual plot :
	# Ideally your plot of the residuals should be symmetrically distributed around the lower 
	# digits of the y-axis, with no clear patterns what so ever.		
	resid <- ggplot( plot_data, aes( predicted, residuals ) ) +
			 ggtitle("Residuals of the Predicted Value") +			 
			 scale_x_continuous( labels = comma ) + 
			 scale_y_continuous( labels = comma )

	# 4. QQ-plot of the residuals :
	# The plot will be very close to the y = x straight line if the residuals 
	# is a close approximation to a normal distribution.
	QQPlot <- function( plot_data )
	{
		# qqline draws the line between the 25% and 75% quantile by default 
		y <- quantile( plot_data$residuals, c(0.25, 0.75) )
		x <- qnorm( c(0.25, 0.75) )
		
		# y = slope * x + intercept
		slope <- diff(y) / diff(x)
		intercept <- y[1] - slope * x[1]

		qqplot <- ggplot( plot_data, aes( sample = residuals ) ) + 
		  		  scale_y_continuous( labels = comma ) +
			 	  geom_abline( slope = slope, intercept = intercept, color = "blue" ) + 
			 	  ggtitle( "Residual's QQ Plot " )
		return(qqplot)
	}	
	qqplot <- QQPlot( plot_data = plot_data )

	# color the plot to distinguish outlier and normal data point if there is in fact one 
	if( length(outlier) > 0 )
	{		
		cooks <- cooks + geom_point( aes( color = boolean ), size = 2, shape = 1 ) + 
				 		 guides( color = FALSE )

		pred <- pred + geom_point( aes( color = boolean ), size = 2, shape = 1 ) +
					   geom_smooth( method = "lm" ) + 				 		 
				 	   guides( color = FALSE )

		resid <- resid + geom_point( aes( color = boolean ), size = 2, shape = 1 ) +
						 geom_smooth( aes( x = predicted, y = residuals ) ) +						  				 		  
				 		 guides( color = FALSE )

		qqplot <- qqplot + stat_qq( aes( color = boolean ), size = 2, shape = 1 ) + 
				  		   guides( color = FALSE )

		plot <- arrangeGrob( pred, cooks, resid, qqplot )
		return( list( plot = plot, outlier = outlier ) )
	}else
	{
		cooks <- cooks + geom_point( size = 2, shape = 1 )

		pred <- pred + geom_point( size = 2, shape = 1 ) + 
					   geom_smooth( method = "lm" )

		resid <- resid + geom_point( size = 2, shape = 1 ) + 
						 geom_smooth( aes( x = predicted, y = residuals ) )
		
		qqplot <- qqplot + stat_qq( size = 2, shape = 1 )

		plot <- arrangeGrob( pred, cooks, resid, qqplot )
		return( list( plot = plot, outlier = NULL ) )	
	}		
}

