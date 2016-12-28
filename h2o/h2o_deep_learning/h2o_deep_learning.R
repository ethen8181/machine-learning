# deep learning
library(h2o)
library(dplyr)
library(data.table)
setwd("/Users/ethen/machine-learning/h2o")

# Start up a 1-node H2O server on your local machine 
# and allow it to use all CPU cores using nthreads = -1
# you can specify the parameter, max_mem_size = "2G" to make it use 
# no more than 2GB of memory
 h2o.init(nthreads = -1)

# detailed dataset description
# https://archive.ics.uci.edu/ml/datasets/Covertype
df <- h2o.importFile(path = "covtype.full.csv")
list( dimension = dim(df), head = df )

# specify input and output features
output <- "Cover_Type"
input  <- setdiff( names(df), output )

# h2o.splitFrame : Split an existing H2O data set according to user-specified ratios
# h2o.assign : makes a copy of / rename the dataset 
split <- h2o.splitFrame( df, c(0.6, 0.2), seed = 1234 )
train <- h2o.assign( split[[1]], "train" ) # 60%
valid <- h2o.assign( split[[2]], "valid" ) # 20%
test  <- h2o.assign( split[[3]], "test" )  # 20%

# first model 
model_dl_1 <- h2o.deeplearning(
	model_id = "dl_1", # (optional) assign a user-specified id to the model
	training_frame = train, 
	validation_frame = valid, # validation dataset: used for scoring and early stopping
	x = input,
	y = output,
	# activation = "Rectifier", # default (a.k.a Relu)
	# hidden = c(200, 200),    # default = 2 hidden layers with 200 neurons each
	epochs = 1, # How many times the dataset should be iterated
	variable_importances = TRUE # allows obtaining the variable importance, not enabled by default
)

# h2o.varimp : obtaining the variable importance
head( as.data.table( h2o.varimp(model_dl_1) ) )

# validation accuracy
h2o.hit_ratio_table(model_dl_1, valid = TRUE)[1, 2]


# second model 
model_dl_2 <- h2o.deeplearning(
	model_id = "dl_2", 
	training_frame = train, 
	validation_frame = valid,
	x = input,
	y = output,
	hidden = c(32, 32, 32), # smaller network, runs faster
	epochs = 100, # hopefully converges earlier...
	score_validation_samples = 10000, # sample the validation dataset (faster)
	stopping_rounds = 5,
	stopping_metric = "misclassification",
	stopping_tolerance = 0.01
)

# evaluate the two models on the test set
pred1 <- h2o.predict(model_dl_1, test)
pred2 <- h2o.predict(model_dl_2, test)
list( model_dl_1 = mean(pred1$predict == test$Cover_Type), 
      model_dl_2 = mean(pred2$predict == test$Cover_Type) )


# ---------------------------------------------------------------------------------------
#								Hyparameters Tuning 
# ---------------------------------------------------------------------------------------

# train samples of the training data for speed 
sampled_train <- train[1:10000, ]

# specify the list of paramters 
hyper_params <- list(
	hidden = list( c(32, 32, 32), c(64, 64) ),
	input_dropout_ratio = c(0, 0.05),
	l1 = c(1e-4, 1e-3)
)

# performs the grid search
grid_id <- "dl_grid"
model_dl_grid <- h2o.grid(
	algorithm = "deeplearning", # name of the algorithm 
	grid_id = grid_id, 
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

# find the best model and evaluate its performance
stopping_metric <- 'accuracy'
sorted_models <- h2o.getGrid(
	grid_id = grid_id, 
	sort_by = stopping_metric,
	decreasing = TRUE
)
best_model <- h2o.getModel(sorted_models@model_ids[[1]])
pred3 <- h2o.predict(best_model, test)
mean(pred3$predict == test$Cover_Type)


# storing and loading the model 
# path <- h2o.saveModel( model, path = "mybest_deeplearning_covtype_model", force = TRUE )
# print(path)
# loaded <- h2o.loadModel(path)
h2o.shutdown(prompt = FALSE)

