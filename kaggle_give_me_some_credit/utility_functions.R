# utility functions

FeatureEngineering <- function( data, is_train, income_medians )
{
	# --------------------------------------------------------------------------------------
	# Description :
	#     perform some feature engineering to the data,
	#     actions and the returned value on the training and test set will slightly differ
	# 	   
	# Args :
	#     @data  	      = your data.table type data
	#     @is_train       = TRUE indicates this is the training data and FALSE for testing data
	# 	  @income_medians = pass in income_medians for the testing data, this value is calculated 
	#				  		from the training data and is used will in the missing value for 
	#				  		the MonthlyIncome column ( see detail in that function section )
	#  
	# Values :
	# 	  When is_train = TRUE, returns
	#	  1. data   = the training data.table after performing feature engineering
	#	  2. log_median = breaks that will used to bin the test data's MonthlyIncome column
	#	  When is_train = FALSE, returns
	# 	  1. data = the testing data.table after performing feature engineering
	# --------------------------------------------------------------------------------------

	# replacing columns names that contains "-" with "To"
	columns_containing_hyphen <- grep( "-", colnames(data), value = TRUE )
	setnames( data, columns_containing_hyphen, gsub( "-", "To", columns_containing_hyphen ) )	

	# add up the number of times past due regardless of how many days
	data[ , SumPastDueNotWorse := NumberOfTime30To59DaysPastDueNotWorse + 
								  NumberOfTime60To89DaysPastDueNotWorse +
								  NumberOfTimes90DaysLate ]

	# record rows where both columns are NAs  
	# since whenever NumberOfDependents is missing MonthlyIncome is also missing 
	# so we won't record it 
	data[ , bothNA := ifelse( is.na(NumberOfDependents) & is.na(MonthlyIncome),
							  "Yes", "NO" ) %>% as.factor() ]
	
	# replace missing NumberOfDependents with 0, 
	# since the mean is 0 and the median is less than 1 
	data[ is.na(NumberOfDependents), NumberOfDependents := 0 ]

	# record the monthly income that are NA
	data[ , MonthlyIncomeNA := ifelse( is.na(MonthlyIncome), "Yes", "NO" ) ]

	# log transform the monthly income, plus 1 to avoid taking log 0
	# then fill in the missing income with the median of the logged income 
	# missing income on the test set are filled in with the median of the logged income 
	# calculated from the training set 
	log_income <- log10( data$MonthlyIncome + 1 )
	if(is_train)
	{		
		log_median <- median( log_income, na.rm = TRUE )
		data[ , MonthlyIncome := ifelse( is.na(MonthlyIncome), log_median, log_income ) ]
	}else
		data[ , MonthlyIncome := ifelse( is.na(MonthlyIncome), income_medians, log_income ) ]

	# log transform extremely skew-distributed columns, 
	# plus 1 to avoid taking log 0
	# @columns are columns that will not be transformed 
	columns <- c( "SeriousDlqin2yrs", "age", "MonthlyIncome", "NumberOfDependents",
				  "bothNA", "MonthlyIncomeNA" )
	columns_log_transform <- setdiff( names(data), columns )
	data[ , (columns_log_transform) := lapply( .SD, function(x)
	{
		log10( x + 1 )
	}), .SDcols = columns_log_transform ]

	# return 
	# also convert output column to factor, so it'll be recognize as a classification problem
	# by the machine learning library, note that testing data does not include this column
	if(is_train)
	{
		data[ , SeriousDlqin2yrs := as.factor(SeriousDlqin2yrs) ]
		return( list( data = data, log_median = log_median ) )	
	}else
		return(data)	
}

# --------------------------------------------------------------------------
#					H2o Models
# --------------------------------------------------------------------------

# Note that the ensemble results above are not reproducible since 
# h2o.deeplearning is not reproducible when using multiple cores.
# each function is a wrapper specifying the model used for h2o ensemble  

h2o.deeplearning.1 <- function( ..., hidden = c( 50, 50 ), 
								epochs = 100,
								activation = "RectifierWithDropout", 
								balance_classes = TRUE, 
								stopping_rounds = 5, 
								stopping_metric = "AUC",
								stopping_tolerance = 0.05,
								variable_importances = TRUE )
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
		variable_importances = variable_importances
	)
}

h2o.deeplearning.2 <- function( ..., hidden = c( 100, 100, 100 ), 
								epochs = 60,
								activation = "Rectifier", 
								balance_classes = TRUE, 
								stopping_rounds = 5, 
								stopping_metric = "AUC",
								stopping_tolerance = 0.05,
								variable_importances = TRUE,
								l1 = 0.02 )
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
		variable_importances = variable_importances,
		l1 = l1
	)
}

h2o.randomForest.1 <- function( ..., ntrees = 250,
								max_depth = 6, 
								balance_classes = TRUE,
								stopping_rounds = 5, 
								stopping_metric = "AUC",
								stopping_tolerance = 0.05 )
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

h2o.gbm.1 <- function( ..., ntrees = 200, 
					   learn_rate = 0.05,
					   max_depth = 6,
					   sample_rate = 0.9,
					   balance_classes = TRUE,
					   stopping_rounds = 5, 
					   stopping_metric = "AUC",
					   stopping_tolerance = 0.05 )
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

# h2o.glm
# @alpha : 1 = lasso regression
# lambda_search is not currently supported in conjunction with N-fold cross-validation

