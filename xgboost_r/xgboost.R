# walking though xgboost tutorial 
# https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/xgboostPresentation.Rmd

# it is widely used for its 
# 1. efficiency : it's built to manage huge dataset very efficiently. 
#    and includes automatic parallel computation on a single machine, 
# 2. accuracy : good result for most data sets
# 3. feasibilty : customized objective and evaluation, tunable parameters 

library(xgboost)


# example dataset; includes information, stored in sparseMatrix,
# for different kind of mushrooms.

# load the example data from xgboost to R
data(agaricus.train, package = "xgboost" )
data(agaricus.test , package = "xgboost" )

train <- agaricus.train
test  <- agaricus.test
# train and test both contain :
# @data  : the data 
# @label : the outcome of our dataset.
# 		   binary value indicating whether the mushroom is poisonous or not 


# xgboost offers a way to group the data and label in a xgb.DMatrix,
# which will be useful for other advanced features we will discover later
dtrain <- xgb.DMatrix( data = train$data, label = train$label )

# xgboost's basic parameters : 
# @objective : regression use "reg:linear",
#			   binary classification use "binary:logistic"
# @nround    : number of iteration or number of trees added to the model 
# @verbose   : 1 prints evaluation metric, 0 will not print out any messages
# 			   this helps you view the learning progress
# @eta       : step size of each boosting step
# @max.depth : maximum depth of the tree
bst <- xgboost( 
	data = dtrain, 
	max.depth = 2, 
	eta = 1, 
	nround = 2, 
	objective = "binary:logistic"
)
pred <- predict( bst, test$data )

# the prediction returns the probability of each observation being classified into
# each of the binary class ( same for muticlass prediction ), thus you'll have to 
# specify the cutoff value to convert it into actual labels 

# choose a random threshold 0.5 and observe the prediction error 
error <- mean( as.numeric( pred > 0.5 ) != test$label )


# ---------------------------------------------------------------------------------------
# 								Advanced 
# ---------------------------------------------------------------------------------------

# xgb.train is the capacity to follow the progress of the learning after each round. 
# Because of the way boosting works, there is a time when having too many rounds lead to 
# overfitting the training data.
# One way to measure progress in learning of a model is to provide to XGBoost a 
# second dataset already classified. Therefore it can learn on the first dataset and 
# test its model on the second one. Some metrics are measured after 
# each round during the learning process.

# the parameter for this functionality : 
# @watchlist : a list of xgb.DMatrix
dtrain <- xgb.DMatrix( data = train$data, label = train$label )
dtest  <- xgb.DMatrix( data = test$data, label = test$label )

bst <- xgb.train( 
	data = dtrain, 
	max.depth = 2, 
	eta = 1, 
	nround = 2, 
	objective = "binary:logistic",
	watchlist = list( train = dtrain, test = dtest )
)
# the number of nround is set to 2, that's why there will be 2 lines 
# now apart from the training information from training data, it will
# also print out information on the validation data 

# you can also supply different and multiple evaluation metric through 
# @eval.metric : "error" = binary classifaction error
bst <- xgb.train( 
	data = dtrain, 
	max.depth = 2, 
	eta = 1, 
	nround = 2, 
	objective = "binary:logistic",
	watchlist = list( train = dtrain, test = dtest ),
	eval.metric = "auc",
	eval.metric = "error"
)
# saving and loading the model 
# xgb.save function should return TRUE if everything goes well and crashes otherwise.

# xgb.save( bst, "xgboost.model")
# bst <- xgb.load("xgboost.model")

# --------------------------------------------------------------------------------
# cross validation using xgb.cv

# specify the list of parameters 
xgb_params_1 <- list(
	objective = "binary:logistic", # for binary classification
	eta = 0.01,					   # learning rate
	max.depth = 3,				   # max tree depth
	eval_metric = "auc"			   # evaluation / loss metric
)

xgb_cv_1 <- xgb.cv(
	params = xgb_params_1,
	data = dtrain,
	nrounds = 20, 
	nfold = 10,
	# prediction = TRUE, # return the prediction using the final model 
	# showsd = TRUE, # standard deviation of loss across folds
	stratified = TRUE, # sample is unbalanced; use stratified sampling
	# verbose = TRUE,
	# print.every.n = 1, # associate with verbose  
	early.stop.round = 5
)


# random sample means that each member of the population has the same chance 
# of getting selected for the sample.
# Where as for strata random sample, you divide your population into 
# different stratas based on a certain criteria, e.g. demographics.
# thus when you conduct a proportional stratified sample. The size of each 
# stratum in the sample is proportionate to the size of the stratum in the population.

# Other reference for ways of hypertuning parameters : 
# http://datascience.stackexchange.com/questions/9364/hypertuning-xgboost-parameters
# http://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees


# --------------------------------------------------------------------
# 							xgboost  
# --------------------------------------------------------------------

# template of splitting into training and validaion, did not work well .... 
library(Matrix)
library(xgboost)

# xgboost only takes numeric features,
# 1. use sparse.model.matrix to one hot encode the factor variables 
# 	 the formula takes the : output variable ~ .-1, where the -1 is used
# 	 to remove the intercept column that is added by the function
# 2. convert the output column for factor variable back to numeric 
set.seed(1234)
train_info$data[ , QuoteConversion_Flag := QuoteConversion_Flag %>% 
										   as.character() %>% as.numeric() ]
index <- createDataPartition( train_info$data[["QuoteConversion_Flag"]], p = 0.8, list = FALSE )
train <- train_info$data[ index, ]
valid <- train_info$data[ -index, ]

formula <- as.formula( paste( "QuoteConversion_Flag", "~ .-1" ) )
xgb_train <- xgb.DMatrix( data  = sparse.model.matrix( formula, data = train ), 
						  label = train[["QuoteConversion_Flag"]] )
xgb_valid <- xgb.DMatrix( data  = sparse.model.matrix( formula, data = valid ), 
						  label = valid[["QuoteConversion_Flag"]] )
xgb_all   <- xgb.DMatrix( data  = sparse.model.matrix( formula, data = train_info$data ), 
						  label = train_info$data[["QuoteConversion_Flag"]] )

model_xgb_1 <- xgb.train(
	data = xgb_train,
	objective = "binary:logistic",
	nrounds = 400,
	eta = 0.001,
	max.depth = 8,
	colsample_bytree = 0.8,
	verbose = 1,
	print.every.n = 5,
	watchlist = list( valid = xgb_valid ),
	eval_metric = "auc",	
	early.stop.round = 5
)
model_xgb_2 <- xgb.train( 
	data = xgb_all,
	objective = "binary:logistic",
	nrounds = model_xgb_1$bestInd,
	eta = 0.001,
	max.depth = 8,
	colsample_bytree = 0.8,
	verbose = 0
)
pred_xgb <- predict( model_xgb_2, sparse.model.matrix( ~ .-1, data = data_test ) )
submit <- data.table( fread( "data/test.csv", select = 1 ),
					  QuoteConversion_Flag = pred_xgb )
write.csv( submit, "submission2.csv", row.names = FALSE )

