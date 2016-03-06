# http://datahack.analyticsvidhya.com/contest/last-man-standing

# top3 solutions links
# http://discuss.analyticsvidhya.com/t/last-man-standing-reveal-your-approach/7208
# http://www.analyticsvidhya.com/blog/2016/02/secrets-winners-signature-hackathon-last-man-standing/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

library(h2o)
library(data.table)
setwd("/Users/ethen/machine-learning/kaggle/last_man_standing")

# source in helper functions, including :
# [FeatureEngineering]
# [Models]
# [Submit]
# [SubmitEnsemble]
source("utility_functions.R")

# read in training data and perform feature engineering 
# don't read in the id column
data_train <- fread( "data/Train.csv", select = 2:10 )
train_info <- FeatureEngineering( data = data_train, is_train = TRUE )
# train_info$data


# --------------------------------------------------------------------------
#					H2o Models
# --------------------------------------------------------------------------

# initialize the h2o cluster with all possible cores
h2o.init( nthreads = -1 )

# convert the data to h2o's cloud 
training_frame <- as.h2o( train_info$data )

# shared parameters of the Models function 
y <- "Crop_Damage"
x <- setdiff( colnames(train_info$data), y ) 
balance_classes <- TRUE
stopping_rounds <- 5 
stopping_metric <- "r2"
stopping_tolerance <- 0.01
nfolds <- 10
path <- paste0( getwd(), "/models" ) # path to store the models

# this takes about 15 ~ 20 minutes 
Models()


# --------------------------------------------------------------------------
# 						Measuring Performance 
# --------------------------------------------------------------------------

# read in the test data; perform feature engineering and load into a h2o cloud 
# do not read the id column, append them back after obtaining the result for submission
data_test <- fread( "data/Test.csv", select = 2:9 )
data_test <- FeatureEngineering( data = data_test, is_train = FALSE, 
								 Number_Weeks_Used_impute = train_info$log_median,
								 Number_Doses_Week_breaks = train_info$doses_breaks )
test <- as.h2o(data_test)


# load the h2o model, make the prediction and output the file

# gbm1 : 10 fold
# public leaderboard score : 0.84565
model_gbm_1 <- h2o.loadModel("models/gbm_1")
pred_gbm_1  <- predict( model_gbm_1, test )
Submit( predict = pred_gbm_1, filename = "submission_gbm_1.csv" )


# gbm2 : using all the training data
# public leaderboard score : 0.84341
model_gbm_2 <- h2o.loadModel("models/gbm_2")
pred_gbm_2  <- predict( model_gbm_2, test )
Submit( predict = pred_gbm_2, filename = "submission_gbm_2.csv" )


# submit a weighted ensemble using gbm1 and gbm2 
SubmitEnsemble( predict1 = pred_gbm_1, predict2 = pred_gbm_2, 
				weight1 = 0.7, weight2 = 0.3, filename = "submission_ensemble.csv" )

