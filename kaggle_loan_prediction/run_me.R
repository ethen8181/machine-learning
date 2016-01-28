
# 1. kaggle competition webpage
# https://www.kaggle.com/c/GiveMeSomeCredit
# 2. sample documentation on the competition
# http://www.slideshare.net/pragativbora/predicting-delinquencygive-me-some-credit


library(h2o)
library(dplyr)
library(data.table)
library(h2oEnsemble)
setwd("/Users/ethen/machine-learning/kaggle_loan_prediction")
data_train <- fread( "data/cs-training.csv", select = 2:12 )

# source in functions including :
# [FeatureEngineering]
# [h2o.deeplearning.1]
# [h2o.deeplearning.2]
# [h2o.randomForest.1]
# [h2o.gbm.1]
source("loan_prediction_functions.R")

# perform feature engineering on training data 
train_info <- FeatureEngineering( data = data_train, is_train = TRUE )
# train_info$data
# sapply( train_info$data, class )


# -------------------------------------------------------------
# Model 

# initialize the h2o cluster with all possible cores
h2o.init( nthreads = -1 )

# define the input and output variable 
output <- "SeriousDlqin2yrs"
input  <- setdiff( colnames(train_info$data), output )

# convert the data to h2o's cloud 
train <- as.h2o( train_info$data )

# define the base learner and metalearner for stacking 
learner <- c(
	"h2o.gbm.1",
	"h2o.deeplearning.1", 
	"h2o.deeplearning.2",
	"h2o.randomForest.1"
)
metalearner <- "h2o.glm.wrapper"

# train and save the model 
# this takes about an hour or two 
model_1 <- h2o.ensemble(
	
	x = input, 
	y = output, 
	training_frame = train,
	model_id = "model_1",
	family = "binomial", 
	learner = learner, 
	metalearner = metalearner,
	cvControl = list( V = 10 )
)
h2o.save_ensemble( model1, path = paste0( getwd(), "/model1" ), force = TRUE )
# model1 <- h2o.load_ensemble( "model1" )


# --------------------------------------------------------------------
# measuring performance 

# read in the test data; perform feature engineering and load into a h2o cloud 
# do not read the id and output variable column
data_test <- fread( "data/cs-test.csv", select = 3:12 )
data_test <- FeatureEngineering( data = data_test, is_train = FALSE, 
								 income_medians = train_info$log_median )
test  <- as.h2o(data_test)

# make the prediction and output the file 
pred <- predict( model1, test )

# the predicted probability is in the third column fo the pred Frame
submit <- as.data.table( pred$pred[ , 3 ] )
submit[ , Id := 1:nrow(test) ]
setnames( submit, c( "Probability", "Id" ) )
setcolorder( submit, c( "Id", "Probability" ) )

write.csv( submit, "submission.csv", row.names = FALSE )

# shutting down the h2o cluster 
# h2o.shutdown( prompt = FALSE )

