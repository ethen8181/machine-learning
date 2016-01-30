# utility functions 

FeatureEngineering <- function( data, is_train, 
								Number_Weeks_Used_impute,
								Number_Doses_Week_breaks )
{
	# --------------------------------------------------------------------------------------
	# Description :
	#     perform some feature engineering to the data,
	#     actions and the returned value for the training and test set will slightly differ
	# 	   
	# Args :
	#     @data  	= your data.table type data
	#     @is_train = TRUE indicates this is the training data and FALSE for testing data
	# 	  @Number_Weeks_Used_impute = value that will be used for imputing the missing
	#								  value of this column for the testing data, 
	#								  this value is calculated from the training data.
	#								  ( see detail in its section  )
	# 								  
	#	  @Number_Doses_Week_breaks = breaks that will be used to bin the test data's Number_Doses_Week
	#								  this value is calculated from the training data. 
	#				  				  ( see detail in its section )
	#  
	# Values :
	# 	  When is_train is TRUE, returns
	#	  1. data   	  = the training data.table after performing feature engineering
	#	  2. log_median   = value will be used to fill in testing data's missing Number_Weeks_Used 
	#	  3. doses_breaks = breaks that will be used to bin the test data's Number_Doses_Week
	#	  
	# 	  When is_train is FALSE, returns
	# 	  1. data = the testing data.table after performing feature engineering
	# --------------------------------------------------------------------------------------

	# perform log transformation to non factor columns so the counts will not be skewed 
	# plus 1 to avoid taking log(0)
	log_transform_columns <- c( "Estimated_Insects_Count", "Number_Doses_Week", 
						 	 	"Number_Weeks_Quit" )
	data[ , (log_transform_columns) := lapply( .SD, function(x)
	{
		log( x + 1 )
	}), .SDcols = log_transform_columns ]

	# record missing Number_Weeks_Used
	data[ , Number_Weeks_Used_NA := ifelse( is.na(Number_Weeks_Used),
												  "Yes", "No" )  ]
	# log transform the Number_Weeks_Used
	# and use the median of the log-transformed to fill in the missing value 
	log_week_use <- log( data$Number_Weeks_Used + 1 )
	if(is_train)
	{
		# @log_median, will be used to impute testing data's missing Number_Weeks_Used
		log_median <- median( log_week_use, na.rm = TRUE )
		data[ , Number_Weeks_Used := ifelse( is.na(Number_Weeks_Used), 
											 log_median, log_week_use ) ]

		# bin the Number_Doses_Week columns using 20 percent delimited quantile
		# @doses_breaks, used to break the Number_Doses_Week for the testing data 
		doses_breaks <- quantile( data$Number_Doses_Week, seq( 0, 1, 0.2 ) )
		data[ , Number_Doses_Week_Bins := cut( data$Number_Doses_Week, 
			 								   breaks = doses_breaks, 
			 								   include.lowest = TRUE ) ]
	}else
	{
		# for test data 
		data[ , Number_Weeks_Used := ifelse( is.na(Number_Weeks_Used), 
											 Number_Weeks_Used_impute, log_week_use ) ]

		data[ , Number_Doses_Week_Bins := cut( data$Number_Doses_Week, 
			 								   breaks = Number_Doses_Week_breaks, 
			 								   include.lowest = TRUE ) ]
	}
	# generate a new feature, using the difference between weeks used and weeks quit 
	data[ , Number_Weeks_Ratio := Number_Weeks_Used - Number_Weeks_Quit ]

	# convert categorical columns to factor variable
	factor_columns <- c( "Crop_Type", "Soil_Type", "Pesticide_Use_Category", 
						 "Season", "Number_Doses_Week_Bins", "Number_Weeks_Used_NA" )
	data[ , (factor_columns) := lapply( .SD, as.factor ), .SDcols = factor_columns ]

	if(is_train)
	{
		# convert ouput column, Crop_Damage, to factor variable
		data[ , Crop_Damage := as.factor(Crop_Damage) ]
		return( list( data 		   = data, 					   
					  log_median   = log_median,
					  doses_breaks = doses_breaks ) )
	}else
		return(data)	
}


# --------------------------------------------------------------------------
#					H2o Models
# --------------------------------------------------------------------------

Models <- function( x = x, y = y,
					training_frame = training_frame,			
					balance_classes = balance_classes,
					stopping_rounds = stopping_rounds, 
					stopping_metric = stopping_metric,
					stopping_tolerance = stopping_tolerance,
					nfolds = nfolds, path = path )
{
	# use 10 fold cross validation to determine the optimal number of trees
	# ( this prevents overfitting by using too many trees ), 
	# then train a single model using all the training data.
	# Nothing is returned, the models are directly saved to the specified path   

	# gradient boosting machine 10 fold takes about 15 minutes
	model_gbm_1 <- h2o.gbm(
		x = x, 
		y = y,
		training_frame = training_frame,
		model_id = "gbm_1",
		ntrees = 150,
		learn_rate = 0.1,
		max_depth = 10,
		sample_rate = 0.8,
		nfolds = nfolds,
		balance_classes = balance_classes,
		stopping_rounds = stopping_rounds,
		stopping_metric = stopping_metric,
		stopping_tolerance = stopping_tolerance
	)
	h2o.saveModel( model_gbm_1, path = path, force = TRUE )

	# this takes less than 1 or 2 minutes 
	model_gbm_2 <- h2o.gbm(
		x = x, 
		y = y,
		training_frame = training_frame,
		model_id = "gbm_2",
		ntrees = model_gbm_1@parameters$ntrees,
		learn_rate = 0.1,
		max_depth = 10,
		sample_rate = 0.8
	)
	h2o.saveModel( model_gbm_2, path = path, force = TRUE )
}


# --------------------------------------------------------------------------
# 						Measuring Performance 
# --------------------------------------------------------------------------

Submit <- function( predict, filename )
{
	# obtain the predicted class and write the output to a csv file 

	# the predicted class is in the first column fo the predict Frame,
	# the rest are just meeting the submission criteria, having the needed ID
	# in column 1 and Crop_Damage, the predicted class in column 2
	submit <- as.data.table( predict$predict[ , 1 ] ) 
	submit[ , ID := fread( "data/Test.csv", select = 1 ) ]
	setnames( submit, c( "Crop_Damage", "ID" ) )
	setcolorder( submit, c( "ID", "Crop_Damage" ) )
	write.csv( submit, filename, row.names = FALSE ) 	
}


SubmitEnsemble <- function( predict1, predict2, weight1, weight2, filename )
{
	# use the predicted probability of two models to obtain the final predicted class.
	# pass in weight1 and weight2 to obtain an weighted ensembled score
	# these two parameters are determined by trial and error, where
	# 0.7, 0.3 seems to be doing the best, with public leaderboard score : 0.8457
	
	ensemble1 <- as.data.table(predict1)
	ensemble2 <- as.data.table(predict2)

	# sloppy way of obtaining the weighted probability of each class
	ensemble <- data.table( 
		ensemble1[ , 2, with = FALSE ] * weight1 + ensemble2[ , 2, with = FALSE ] * weight2,
		ensemble1[ , 3, with = FALSE ] * weight1 + ensemble2[ , 3, with = FALSE ] * weight2,
		ensemble1[ , 4, with = FALSE ] * weight1 + ensemble2[ , 4, with = FALSE ] * weight2
	)
	ensemble[ , ID := fread( "data/Test.csv", select = 1 ) ]

	# obtain the class that has the maximum probability after the weighted probability
	# minus 1 since the predicted class should 0, 1, 2 and which.max obtains the 
	# column index in which it has the max probability, which are 1, 2, 3
	ensemble[ , Crop_Damage := which.max( .SD ) - 1, by = ID ]
	ensemble[ , c( "p0", "p1", "p2" ) := NULL ]
	write.csv( ensemble, filename, row.names = FALSE ) 
}

