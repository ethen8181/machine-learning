# [ Metrics ] : 
# Obtain binary classification's
# performance metrics for different cutoff value 

# name the prediction's column by "pred_.", where . equals the model name
# 
Metrics <- function( data, predict, actual, threshold = 0.5 )
{
	# relevel so table will be a standard confusion matrix 
	cm <- table(
		
		predict = relevel( as.factor( ifelse( data[[predict]] > threshold, 1, 0 ) ), "1" ),
		actual  = relevel( data[[actual]], "1" )
	)

	model 		<- gsub( "pred_(.*)", "\\1", predict ) # extract the model's name 
	prevalence  <- mean( data[[actual]] == 1 )
	accuracy 	<- sum( diag(cm) ) / sum(cm)
	precision   <- cm[ 1, 1 ] / sum( cm[ 1, ] )
	recall 		<- cm[ 1, 1 ] / sum( cm[ , 1 ] ) # sensitivity
	specificity <- cm[ 2, 2 ] / sum( cm[ , 2 ] ) # true negative rate 

	# @prevalence : percentage of positive class  
	metric <- data.table( model 	  = model,
						  prevalence  = prevalence,
						  recall 	  = recall, 
						  accuracy 	  = accuracy, 
						  precision   = precision,						  
						  specificity = specificity )
	# convert to long format 
	return( melt( metrices, id.vars = "model" ) )
}

