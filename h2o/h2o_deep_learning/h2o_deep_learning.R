
library(h2o)
library(dplyr)
library(data.table)
setwd("/Users/ethen/machine-learning/h2o")

# Start up a 1-node H2O server on your local machine 
# and allow it to use all CPU cores ( nthreads = -1 )
# you can specify the @max_mem_size = "2G" to make it use no more than 2GB of memory
 
h2o.init( nthreads = -1 )

# h2o.importFile : imports the file to the h2o cluster
# normalizePath( "../data/covtype.full.csv" ) 

# detailed dataset description
# # https://archive.ics.uci.edu/ml/datasets/Covertype
df <- h2o.importFile( path = "covtype.full.csv" ) 
list( dim(df), df )

# h2o.splitFrame : Split an existing H2O data set according to user-specified ratios.
# h2o.assign : makes a copy of / rename the dataset 
splits <- h2o.splitFrame( df, c( 0.6, 0.2 ), seed = 1234 )
train  <- h2o.assign( splits[[1]], "train.hex" ) # 60%
valid  <- h2o.assign( splits[[2]], "valid.hex" ) # 20%
test   <- h2o.assign( splits[[3]], "test.hex" )  # 20%

# a more scalable way of doing scatter plot 
plot( h2o.tabulate(df, "Elevation", "Cover_Type" ) )
plot( h2o.tabulate( df, "Soil_Type", "Cover_Type") )

# first try of deep learning 
output <- "Cover_Type"
input  <- setdiff( names(df), output )
input

# @overwrite_with_best_model : overwrite the final model with the best model found during training. 
# 							   Default to TRUE. Combined with a validation set, it can be used
# 							   for early stopping
# @score_validation_samples : Can be randomly sampled or stratified if balance classes 
# 							  is set to TRUE and score validation sampling is "Stratified". 
# 							  To select the entire validation dataset, specify 0, which is the default.
# @score_validation_sampling : Specifies the method used to sample the validation dataset for scoring. 
# 							   The options are "Uniform" and "Stratified". The default is Uniform. 
# @balance_classes : default = FALSE. For imbalanced data, setting to TRUE 
# 					 can result in improved predictive accuracy
# @l1 : L1 regularization, improves generalization and prevents overfitting

# first model 
model_dl_1 <- h2o.deeplearning(

	model_id = "dl_1", # (optional) assign a user-specified id to the model
	training_frame = train, 
	validation_frame = valid, # validation dataset: used for scoring and early stopping
	x = input,
	y = output,
	#activation = "Rectifier", # default
	hidden = c( 50, 50 ),    # default = 2 hidden layers with 200 neurons each
	epochs = 1, # How many times the dataset should be iterated
	variable_importances = TRUE # allows obtaining the variable importance, not enabled by default
)

# obtain the variable importance 
head( as.data.frame( h2o.varimp(model_dl_1) ) )

# second model 
model_dl_2 <- h2o.deeplearning(

	model_id = "dl_2", 
	training_frame = train, 
	validation_frame = valid,
	x = input,
	y = output,
	hidden = c( 32, 32, 32 ), # smaller network, runs faster
	epochs = 100, # hopefully converges earlier...
	score_validation_samples = 10000, # sample the validation dataset (faster)
	stopping_rounds = 5,
	stopping_metric = "misclassification", # could be "MSE","logloss","r2", "AUC"
	stopping_tolerance = 0.01
)

pred1 <- h2o.predict( model_dl_1, test )
pred2 <- h2o.predict( model_dl_2, test )
list( model_dl_1 = mean( pred1$predict == test$Cover_Type ), 
      model_dl_2 = mean( pred2$predict == test$Cover_Type ) )

# append the prediction column back to the original dataset if you prefer 
# test$Accuracy <- pred1$predict == test$Cover_Type


# ---------------------------------------------------------------------------------------
#								Hyparameters Tuning 
# ---------------------------------------------------------------------------------------

# grid search 

# train samples of the training data for speed 
sampled_train <- train[ 1:10000, ]

# specify the list of paramters 
hyper_params <- list(

	hidden = list( c( 32, 32, 32 ), c( 64, 64 ) ) ,
	input_dropout_ratio = c( 0, 0.05 ),
	l1 = c( 1e-4, 1e-3 )
)

# performs the grid search 
model_dl_grid <- h2o.grid(

	algorithm = "deeplearning", # name of the algorithm 
	model_id = "dl_grid", 
	training_frame = sampled_train,
	validation_frame = valid, 
	x = input, 
	y = output,
	epochs = 10,
	stopping_metric = "misclassification",
	stopping_tolerance = 1e-2, # stop when logloss does not improve by >=1% for 2 scoring events
	stopping_rounds = 2,
	score_validation_samples = 10000, # downsample validation set for faster scoring
	hyper_params = hyper_params
)

# Find the best model and its full set of parameters

source("h2o_deep_learning_functions.R") 
ids <- model_dl_grid@model_ids
model_best_grid <- BestGridSearch( ids = ids )

# use it obtain the prediction 
pred3 <- h2o.predict( model_best_grid$best_model, test )
mean( pred3$predict == test$Cover_Type )

# summary(model_dl_1)
# plot(model_dl_1)
# plot(model_dl_2)


# ---------------------------------------------------------------------------------------
#								Binary Classification 
# ---------------------------------------------------------------------------------------

# convert to binary classification 
train$bin_output <- ifelse( train[ , output ] == "class_1", 0, 1 ) %>% as.factor()

model_dl_binary <- h2o.deeplearning(

	x = input,
	y = "bin_output", # specify the character name of the column
	training_frame = train,
	hidden = c( 10, 10 ),
	epochs = 1,
	l1 = 1e-5
	#balance_classes = TRUE enable this for high class imbalance
)

# Now the model metrics contain AUC for binary classification
summary(model_dl_binary) 
h2o.auc(model_dl_binary)
plot( h2o.performance(model_dl_binary) ) # ROC plot 


# storing and loading the model 
# path <- h2o.saveModel( model, path = "mybest_deeplearning_covtype_model", force = TRUE )
# print(path)
# loaded <- h2o.loadModel(path)

# 
h2o.shutdown( prompt = FALSE )

# --------------------------------------
# start from here 



# http://www.h2o.ai/resources/
# http://learn.h2o.ai/content/tutorials/deeplearning/index.html



