# utility functions 

FeatureEngineering <- function( data, is_train,
								zero_variance_columns )
{
	# 1. add year, month, day column from the original date, remove that original column
	#    and convert year column to factor
	data[ , `:=`( Year  = lubridate::year(Original_Quote_Date) %>% as.factor(),
				  Month = lubridate::month(Original_Quote_Date),
				  Day   = lubridate::day(Original_Quote_Date) ) ]
	data[ , Original_Quote_Date := NULL ]

	# 2. "Field10" has commas for numbers. Should be removed and convert back to numeric
	data[ , Field10 := gsub( ",", "", Field10 ) %>% as.numeric()  ]

	# 3. names of the columns that will be dropped because of low variance.
	# 	 Also convert output variable to factor
	# 	 to be recognized as a classifcation problem. test set will not have this column
	if(is_train)
	{
		zero_variance_columns <- names(data)[ caret::nearZeroVar(data) ]
		data[ , QuoteConversion_Flag := as.factor(QuoteConversion_Flag) ]
	} 	
	data[ , ( zero_variance_columns ) := NULL ]

	# 4. names of columns that are originally numeric, but will be converted to factor
	# 	 since they contain few distinct numbers 
	convert_to_factor <- c( 
		"PropertyField35", 
		"CoverageField5A", "CoverageField5B", "CoverageField6A", "CoverageField6B",
		"SalesField3", "SalesField4", "SalesField5", "SalesField9",
		"PersonalField1", "PersonalField2", "PersonalField6"
	)
	data[ , (convert_to_factor) := lapply( .SD, as.factor ), .SDcols = convert_to_factor ]
		
	# 5. names of the columns that will log transformationed to be a bit more
	# 	 normally distributted 
	log_transform <- c( "SalesField10", "SalesField11", "SalesField12" )
	data[ , (log_transform) := lapply( .SD, function(x)
	{
		log( x + 1 )
	}), .SDcols = log_transform ]

	# 6. names of the columns that will be dropped because of too many distinct levels  
	dropped_high_levels <- c( "PersonalField16", "PersonalField17", 
							  "PersonalField18", "PersonalField19" )
	data[ , (dropped_high_levels) := NULL ]

	# return the feature engineered data,  
	if(is_train)
		return( list( data = data, zero_variance_columns = zero_variance_columns ) )
	else
		return(data)
}


# --------------------------------------------------------------------------
#					H2o Models
# --------------------------------------------------------------------------

# Note that the ensemble results below may not be reproducible since 
# h2o.deeplearning is not reproducible when using multiple cores and 
# the seeds are not set.
# each function is a wrapper specifying the model used for h2o ensemble  

# auc : 0.95049
h2o.deeplearning.1 <- function( ..., hidden = c( 50, 50 ), 
								epochs = 100,
								activation = "RectifierWithDropout", 
								balance_classes = TRUE, 
								stopping_rounds = 5, 
								stopping_metric = "AUC",
								stopping_tolerance = 0.1,
								overwrite_with_best_model = TRUE )
{
	h2o.deeplearning.wrapper(
		...,
		hidden = hidden,
		epochs = epochs,
		activation = activation,
		balance_classes = balance_classes,
		stopping_rounds = stopping_rounds,
		stopping_metric = stopping_metric,
		stopping_tolerance = stopping_tolerance,
		overwrite_with_best_model = overwrite_with_best_model
	)
}

# auc : 0.93191
h2o.deeplearning.2 <- function( ..., hidden = c( 100, 100, 100 ), 
								epochs = 65,
								activation = "Rectifier", 
								balance_classes = TRUE, 
								stopping_rounds = 5, 
								stopping_metric = "AUC",
								stopping_tolerance = 0.1,
								overwrite_with_best_model = TRUE )
{
	h2o.deeplearning.wrapper(
		...,
		hidden = hidden,
		epochs = epochs,
		activation = activation,
		balance_classes = balance_classes,
		stopping_rounds = stopping_rounds,
		stopping_metric = stopping_metric,
		stopping_tolerance = stopping_tolerance,
		overwrite_with_best_model = overwrite_with_best_model
	)
}

# auc : 0.949
h2o.randomForest.1 <- function( ..., ntrees = 250,
								max_depth = 14, 
								balance_classes = TRUE,
								stopping_rounds = 5, 
								stopping_metric = "AUC",
								stopping_tolerance = 0.1 )
{
	h2o.randomForest.wrapper(
	 	..., 
	 	ntrees = ntrees,
	 	max_depth = max_depth, 
	 	balance_classes = balance_classes,
		stopping_rounds = stopping_rounds,
		stopping_metric = stopping_metric,
		stopping_tolerance = stopping_tolerance
	)
}

# auc : 0.962418
h2o.gbm.1 <- function( ..., ntrees = 54, 
					   learn_rate = 0.1,
					   max_depth = 14, 
					   sample_rate = 0.8,
					   balance_classes = TRUE,
					   stopping_rounds = 5, 
					   stopping_metric = "AUC",
					   stopping_tolerance = 0.1 )
{
	h2o.gbm.wrapper(
		...,
		ntrees = ntrees,
		learn_rate = learn_rate,
		max_depth = max_depth,
		sample_rate = sample_rate,
		balance_classes = balance_classes,
		stopping_rounds = stopping_rounds,
		stopping_metric = stopping_metric,
		stopping_tolerance = stopping_tolerance
	)
}

