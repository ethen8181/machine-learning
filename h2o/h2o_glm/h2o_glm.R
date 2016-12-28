# generalized linear model
library(h2o)
library(ggplot2)
library(data.table)
setwd("/Users/ethen/machine-learning/h2o")

# initialize the cluster with all the threads available
h2o.init(nthreads = -1)

# disable progress bar so it doesn't clutter up the document
h2o.no_progress()

# import and convert to binomial data
df <- h2o.importFile( path = normalizePath("covtype.full.csv") )
y <- "Cover_Type"
x <- setdiff( names(df), y )
df_binomial <- df[ df$Cover_Type %in% c("class_1", "class_2"), ]
h2o.setLevels( df_binomial$Cover_Type, c("class_1","class_2") )

# split to train / test / validation
# use smaller dataset for testing 
data_binomial <- h2o.splitFrame( df_binomial, ratios = c(.6, 0.15) )
names(data_binomial) <- c('train', 'valid', 'test')
data_binomial$train

# perform grid search, it's best to give the model
# a id so retrieving information on them will be easier later
grid_id <- 'glm_grid'
hyper_parameters <- list( alpha = c(0, .5, 1) )
model_glm_grid <- h2o.grid(
	algorithm = "glm", 
	grid_id = grid_id,
	hyper_params = hyper_parameters,
	training_frame = data_binomial$train, 
	validation_frame = data_binomial$valid, 
	x = x, 
	y = y,
	lambda_search = TRUE,
	family = "binomial"
)

# sort the model by the specified evaluation metric
# and obtain the top one (the best model)
stopping_metric <- 'accuracy'
sorted_models <- h2o.getGrid(
	grid_id = grid_id, 
	sort_by = stopping_metric,
	decreasing = TRUE
)
best_model <- h2o.getModel(sorted_models@model_ids[[1]])

# for binomial output, h2o will choose the cutoff threshold by 
# maximizing the f1 score by default, we can change the metric
# to change that behavior
h2o.confusionMatrix(best_model, valid = TRUE, metrics = 'accuracy')

# coefficients (standardized and non-standardized)
# or we can use the short-cut below
# h2o.coef(best_model)
# h2o.coef_norm(best_model)
best_model@model$coefficients

# obtain the regularization, alpha and lambda 
best_model@model$model_summary$regularization

# area under the curve
auc <- h2o.auc(best_model, valid = TRUE)
fpr <- h2o.fpr( h2o.performance(best_model, valid = TRUE) )[['fpr']]
tpr <- h2o.tpr( h2o.performance(best_model, valid = TRUE) )[['tpr']]
ggplot( data.table(fpr = fpr, tpr = tpr), aes(fpr, tpr) ) + 
geom_line() + theme_bw() + ggtitle( sprintf('AUC: %f', auc) )

# remember to shutdown the cluster once we're done
h2o.shutdown(prompt = FALSE)

