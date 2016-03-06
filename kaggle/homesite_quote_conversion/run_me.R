# https://www.kaggle.com/c/homesite-quote-conversion

library(h2o)
library(caret)
library(dplyr)
library(lubridate)
library(data.table)
library(h2oEnsemble)
setwd("/Users/ethen/Desktop/homesite_quote_conversion")
data_train <- fread( "data/train.csv", stringsAsFactors = TRUE, select = 2:299 )

# source in functions including 
# [FeatureEngineering]
source("utility_functions.R")
train_info <- FeatureEngineering( data = data_train, is_train = TRUE )


# -------------------------------------------------------------
# 							H2o Model
# -------------------------------------------------------------

# initialize the h2o cluster with all possible cores
h2o.init( nthreads = -1 )

# define the input and output variable 
y <- "QuoteConversion_Flag"
x <- setdiff( colnames(train_info$data), y )

# load the data to h2o's cloud 
training_frame <- as.h2o(train_info$data)

# define the base learner and metalearner for stacking 
learner <- c(
	"h2o.deeplearning.1", 
	"h2o.deeplearning.2",
	"h2o.gbm.1",
	"h2o.randomForest.1"	
)
metalearner <- "h2o.glm.wrapper"

# takes about 2 and a half hour 
model_1 <- h2o.ensemble(
	x = x, 
	y = y, 
	training_frame = training_frame,
	model_id = "model_1",
	family = "binomial", 
	learner = learner, 
	metalearner = metalearner,
	cvControl = list( V = 10 )
)
h2o.save_ensemble( model_1, path = paste0( getwd(), "/model_1" ), force = TRUE )
# model_1 <- h2o.load_ensemble( "model_1" )


# --------------------------------------------------------------------
# 							H2o measuring performance 
# --------------------------------------------------------------------

# read in the test data; perform feature engineering and load into a h2o cloud 
# do not read the id and output variable column
data_test <- fread( "data/test.csv", stringsAsFactors = TRUE, select = 2:298 )
data_test <- FeatureEngineering( data = data_test, is_train = FALSE,
								 zero_variance_columns = train_info$zero_variance_columns )
test <- as.h2o(data_test)

# h2o ensemble submission, make the prediction and output the file, 
# the predicted probability is in the third column fo the pred Frame
# and the submission column order is "QuoteNumber", "QuoteConversion_Flag"
pred <- predict( model_1, test )
submit <- as.data.table( pred$pred[ , 3 ] )
submit[ , QuoteNumber := fread( "data/test.csv", select = 1 ) ]
setnames( submit, c( "QuoteConversion_Flag", "QuoteNumber" ) )
setcolorder( submit, c( "QuoteNumber", "QuoteConversion_Flag" ) )
write.csv( submit, "submission1.csv", row.names = FALSE ) # 0.96297
# h2o.shutdown( prompt = FALSE )


