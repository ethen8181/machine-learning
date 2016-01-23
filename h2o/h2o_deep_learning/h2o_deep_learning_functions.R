library(dplyr)
library(data.table)

# General h2o function description 

# h2o.getModel : obtain the h2o model using model id, stored in the @model_id slot 
# 				 for single model, stored in the @model_ids slot for h2o.grid 
# h2o.performance : obtain the various performance information.
# 					default measure is "accuracy"
# 					additional argument valid = TRUE to 
# 					return the performance of the validation set instead of the training set 

BestGridSearch <- function( ids )
{
	# ------------------------------------------------------------------
	# Description : 
	#     Pass in a list of h2o model id obtained by h2o.grid
	# 	  and returns the best model its classification error
	# 
	# Args :
	# 	  @ids = a list indicating the unique model id of the model to retrieve
	#
	# Values :
	# 	  A list containing the slot best_model and best_error 
	# 	  
	# ------------------------------------------------------------------

	# obtain the confusion matrix and return the last error for every id 
	error <- vapply( ids, function(x)
	{
		cm <- h2o.getModel(ids[[1]]) %>%
			  h2o.performance( valid = TRUE ) %>%
			  h2o.confusionMatrix()

		return( cm$Error[ length(cm$Error) ] )
	}, numeric(1) )

	# a data.table with two columns :
	# the model and its corresponding error
	# sorted ascendingly by error 
	scores <- data.table( 

		model = unlist(ids),
		error = error 
	
	)[ order(error) ]	

	# different slots for obtaining its parameters and all its parameters
	# best_model@parameters
	# best_model@allparameters
	
	return( list( best_model = h2o.getModel( scores$model[1] ),
				  best_error = scores$error[1] ) )
}









