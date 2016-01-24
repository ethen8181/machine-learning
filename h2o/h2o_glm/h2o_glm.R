
library(h2o)
library(ggplot2)
# initialize the cluster with all the threads available
# and clean it just in case 
h2o.init( nthreads = -1 )
h2o.removeAll()

setwd("/Users/ethen/machine-learning/h2o")

df <- h2o.importFile( path = normalizePath("covtype.full.csv" ) )

# h2o.summary(df)

y <- "Cover_Type"
x <- setdiff( names(df), y )

# binomial data
df_binomial <- df[ df$Cover_Type %in% c( "class_1", "class_2" ), ]
h2o.setLevels( df_binomial$Cover_Type, c( "class_1","class_2" ) )

# split to train / test / validation
# use smaller dataset for testing 
data_binomial <- h2o.splitFrame( df_binomial, ratios = c( .2, .1 ) ) 

hyper_parameters <- list( alpha = c( 0, .5, 1 ) )
model_glm_grid <- h2o.grid(

	algorithm = "glm", 
	hyper_params = hyper_parameters,
	training_frame = data_binomial[[1]], 
	validation_frame = data_binomial[[2]], 
	x = x, 
	y = y,
	lambda_search = TRUE,
	family = "binomial"
)

m1 <- h2o.getModel(model_glm_grid@model_ids[[1]])

# for binomial output, h2o will choose the cutoff threshold by 
# maximizing the f1 score.
h2o.confusionMatrix( m1, valid = TRUE )

# look at different cutoff values for different criteria to optimize
m1@model$training_metrics@metrics$max_criteria_and_metric_scores

# area under the curve
h2o.auc( m1, valid = TRUE )

# coefficients ( standardized and non-standardized )
m1@model$coefficients
m1@model$standardized_coefficient_magnitudes

# obtain the regularization, alpha and lambda 
m1@model$model_summary$regularization

# obtain the false positive rate 
fpr <- m1@model$validation_metrics@metrics$thresholds_and_metric_scores$fpr
tpr <- m1@model$training_metrics@metrics$thresholds_and_metric_scores$tpr

# easier way
fpr <- h2o.fpr( h2o.performance( m1, valid = TRUE ) )
tpr <- h2o.tpr( h2o.performance( m1, valid = TRUE ) )
ggplot( data.frame( fpr = fpr, tpr = tpr ), aes( fpr, tpr ) ) + 
geom_line()


# ------------------------------------------------------------------------
# obtain a copy of each model 

grid_models <- lapply( model_glm_grid@model_ids, function(model_id)
{
	model <- h2o.getModel(model_id)
})
for( i in 1:length(grid_models) ) 
{
	# print format - left aligned ?
	print( sprintf( "regularization: %-50s auc: %f",
	grid_models[[i]]@model$model_summary$regularization, h2o.auc( grid_models[[i]] ) ) )
}

grid_models[[1]]@model$coefficients
grid_models[[1]]@model$standardized_coefficient_magnitudes

