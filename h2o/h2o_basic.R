
# -----------------------------------------------------------------------------
# Basics 
# http://h2o-release.s3.amazonaws.com/h2o/rel-tibshirani/8/docs-website/h2o-docs/booklets/R_Vignette.pdf
# -----------------------------------------------------------------------------
library(h2o)
library(data.table)
# @nthread : -1 uses all CPUs on the host
# @max_mem_size : e.g. "1g", this will set a limit of the maximum memory allowed 
# 				  note that the user is limited by the total amount
# 				  of memory allocated to H2o
h2o.init( nthread = -1 )
h2o.clusterInfo()

# this converts a H2o data frame into a R data.frame 
# though R will crash if it exceeds the amount of 
# data supported by R 
as.data.frame()

# transfers data from to the H2o instance 
as.h2o()

# -----------------------------------------------------------------------------
# example 1 
airlines_url <- "https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv"

# Imports the files into an H2O cloud
# can take zip files 
# @path : file path 
# @destination_frame : key associated with this dataset, like an identifier
airlines_hex <- h2o.importFile( path = airlines_url, 
								destination_frame = "airlines_hex" )
# basic summary 
summary(airlines_hex)

# view quantiles and histogram 
quantile( airlines_hex$ArrDelay, na.rm = TRUE )
h2o.hist(airlines_hex$ArrDelay)

# group by using the h2o way
# @gb.control : list of dealing with na values 
# 				and col.names allows you to specify the column names

# find the number of flights per origin  
origin <- h2o.group_by( data = airlines_hex, 
						by = "Origin", 
			  			nrow("Origin"), 
			  			gb.control = list( na.methods = "rm" ) )

# convert back to R data frame after grouping the result
as.data.frame(origin)

# ----------------------------------------------------------------------
# Task 1 : find the months with the highest cancellation ratio

# obtain the number of flights each month
flights_by_month <- h2o.group_by( data = airlines_hex, 
								  by = "Month", 
								  nrow("Month"), 
								  gb.control = list( na.methods = "rm" ) )

# obtain the cancellation count for each month
cancellations_by_month <- h2o.group_by( data = airlines_hex, 
										by = "Month", 
										sum("Cancelled"),
										gb.control = list( na.methods = "rm" ) )

cancellation_rate <- cancellations_by_month$sum_Cancelled

# column bind the data 
rates_table <- h2o.cbind( flights_by_month$Month, cancellation_rate )
rates_table_R <- as.data.frame(rates_table)


# construct test and train sets using sampling
airlines_split <- h2o.splitFrame( data = airlines_hex, ratios = 0.85 )
airlines_train <- airlines_split[[1]]
airlines_test  <- airlines_split[[2]]

# table counts
h2o.table(airlines_train$Cancelled)
h2o.table(airlines_test$Cancelled)



# set the column names of the response and predictors
Y <- "IsDepDelayed"
X <- c( "Origin", "Dest", "DayofMonth", "Year", "UniqueCarrier", 
		"DayOfWeek", "Month", "DepTime", "ArrTime", "Distance" )

# basic glm model 
# @training_frame for the data
# @x : for the input
# @y : for the output 
# @alpha : setting 1 becomes the lasso penalty, 0 becomes the ridge penalty 
airlines_glm <- h2o.glm( training_frame = airlines_train,
						 x = X, y = Y, 
						 family = "binomial", 
						 alpha = 0.5 )

# View model information: training statistics,
summary(airlines_glm)

# Predict using GLM model
pred <- h2o.predict( object = airlines_glm, newdata = airlines_test )


# -----------------------------
# basic manipulation 
pros_path <- system.file( "extdata", "prostate.csv", package = "h2o" )
prostate_hex <- h2o.importFile( path = pros_path )

# it is possbile to change the key to the h2o object ( more readable name )
h2o.ls()
prostate_hex <- h2o.assign( prostate_hex, "prostate_hex" )


prostate_q <- quantile( prostate_hex$PSA, probs = 1:10 / 10 )

# extract the top 10 or 90 percent of outlier
prostate_hex$PSA[ prostate_hex$PSA <= prostate_q["10%"] | 
				  prostate_hex$PSA >= prostate_q["90%"] ]
 
# it is possible for h2o.table to generate larger tables than
# R's capacity 				  
as.data.table( h2o.table( prostate_hex[ , "AGE" ] ) )

# obtain the model through the model_id slot 
# usefule for API calls for multiple user accessing the same model 
h2o.getModel( model_id = airlines_glm@model_id )

# same for the frame object 
h2o.getFrame()


# functions can be directly added 
airlines_glm@model$training_metrics



data(iris)
iris_hex <- as.h2o( iris, destination_frame = "iris_hex" )


# @nfolds = cross validation 
# @min_rows = Minimum number of rows to assign to teminal nodes.
iris_gbm <- h2o.gbm( training_frame = iris_hex, 
					 x = 1:4, 
					 y = 5, 
					 ntrees = 15,
					 max_depth = 5, 
					 min_rows = 2, 
					 learn_rate = 0.05, 
					 distribution = "multinomial", 
					 nfolds = 10 )


# obtaining the confusion matrix 
iris_gbm@model$training_metrics
str(iris_gbm@model$training_metrics)
as.data.frame(iris_gbm@model$training_metrics@metrics$cm$table)





