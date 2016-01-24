
# kaggle ensemble guide
# http://mlwave.com/kaggle-ensembling-guide/

# super learner 

# installing for the first time 
# library(devtools)
# install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
library(ROCR)
library(h2oEnsemble)

h2o.init( nthreads = -1 )

setwd("/Users/ethen/machine-learning/h2o/h2o_super_learner")

train <- h2o.importFile( path = normalizePath( "higgs_10k.csv" ) )
test  <- h2o.importFile( path = normalizePath( "higgs_test_5k.csv" ) )

# the column C1 is where the output 
y <- "C1"
x <- setdiff( names(train), y )

train[ , y ] <- as.factor( train[ , y ] )  
test[ , y ]  <- as.factor( test[ , y ] )

# the base learner and meta learner for h2o ensemble
# note that you need a .wrapper for the current version 
learner <- c( "h2o.glm.wrapper", "h2o.randomForest.wrapper", 
			  "h2o.gbm.wrapper", "h2o.deeplearning.wrapper" )

# glm tends to be a better meta learner 
metalearner <- "h2o.glm.wrapper"

fit <- h2o.ensemble(

	x = x, 
	y = y, 
	training_frame = train, 
	family = "binomial", 
	learner = learner, 
	metalearner = metalearner,
	cvControl = list( V = 5 ) # 5-fold on each of the learner 
)

pred <- predict( fit, test )

# pred is a list that consists of pred and basepred
# where basepred is the prediction obtained by each of the base-learner 

# convert back to R to calculate the performance for now 
# ( future version will provide performance measure )
predictions <- as.data.frame(pred$pred)[ , 3 ]  # third column is P( Y==1 )
labels <- as.data.frame( test[ , y ] )[ , 1 ]




# obtain the ensemble auc 
p <- prediction( predictions, labels )
perf <- performance( p, "auc" )
perf@y.values

# obtaining the auc for the various base learners 
auc <- sapply( seq( length(learner) ), function(l)
{
	p <- prediction( as.data.frame(pred$basepred)[ , l ], labels )
	auc <- performance( p, "auc" )@y.values[[1]]
	return( auc )
})
data.frame( learner, auc )


# ------------------------------------------------------------------------------
# customizing your own learner
# just some random models with different parameters  

h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

learner <- c(
	"h2o.glm.wrapper",
	"h2o.randomForest.1", 
	"h2o.randomForest.2",
	"h2o.gbm.1", 
	"h2o.gbm.6", 
	"h2o.gbm.8",
	"h2o.deeplearning.1", 
	"h2o.deeplearning.6", 
	"h2o.deeplearning.7"
)

fit <- h2o.ensemble(
	
	x = x, 
	y = y, 
	training_frame = train,
	family = "binomial", 
	learner = learner, 
	metalearner = metalearner,
	cvControl = list( V = 5 )
)

# 
pred <- predict(fit, test)
predictions <- as.data.frame(pred$pred)[ , 3 ]
labels <- as.data.frame( test[ , y ] )[ , 1 ]

p <- prediction( predictions, labels )
perf <- performance( p, "auc" )
perf@y.values

# to save the model use h2o.save_ensemble
# save it in a separate folder since it will store all the records for 
# each base learner and meta learner 
h2o.save_ensemble( fit, path = paste0( getwd(), "/model" ), force = TRUE )
fit <- h2o.load_ensemble( "model" )






