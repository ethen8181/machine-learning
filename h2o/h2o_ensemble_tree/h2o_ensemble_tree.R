# random forest and gradient boosting 
library(h2o)
setwd("/Users/ethen/machine-learning/h2o")

# -1: use all available threads and allocate memory to the cluster,
# the cluster size should be about 4 times larger than your dataset 
h2o.init(nthreads = -1, max_mem_size = '2G')

# disable progress bar so it doesn't clutter up the document
h2o.no_progress()

# import the file and perform train/test split
df <- h2o.importFile( path = normalizePath("covtype.full.csv") )
splits <- h2o.splitFrame( df, c(0.6, 0.2), seed = 1234 )

# assign a h2o id name train to the training data 
train <- h2o.assign( splits[[1]], "train" )  
valid <- h2o.assign( splits[[2]], "valid" )
test  <- h2o.assign( splits[[3]], "test" )

# use a subset of the training data for demonstration speed
train <- train[1:100000, ]

# run our first predictive model
rf1 <- h2o.randomForest(
	training_frame = train,
	validation_frame = valid,
	x = 1:12, 
	y = 13, 
	model_id = "rf_covType_v1",
	ntrees = 200,
	stopping_rounds = 2,
	seed = 1000000
)

# hit ratio table tells you if you give the model n number of shots at guessing the output
# variable's class, how likely is it going to get it correct. Thus, 
# the first row of the hit_ratop table is basically the accuracy of the classification
h2o.hit_ratio_table(rf1, valid = TRUE)[1, 2]

# the variable importance shows you that about 52 percent of the model 
# is captured by Elevation and Soil_Type 
h2o.varimp(rf1)


# ---------------------------------------------------------------------------
# GBM

# Use all default settings and then make some changes.
gbm1 <- h2o.gbm(
	training_frame = train,
	validation_frame = valid,
	x = 1:12,
	y = 13,
	model_id = "gbm_covType1",
	seed = 2000000
)
h2o.hit_ratio_table(gbm1, valid = TRUE)[1, 2]


# This default GBM is much worse than our original random forest
# because it's is far from converging and there are three primary knobs to adjust.
# 1: Adding trees will help. The default is 50.
# 2: Increasing the learning rate will also help. The contribution of each
#    tree will be stronger, so the model will move further away from the
# 	 overall mean.
# 3: Increasing the depth will help. This is the parameter that is the least
#	 straightforward. Tuning trees and learning rate both have direct impact
#  	 that is easy to understand. Changing the depth means you are adjusting
#  	 the "weakness" of each learner. Adding depth makes each tree fit the data closer. 
#
# The first configuration will attack depth the most, since we've seen the
# random forest focus on a continuous variable (elevation) and 40-class factor
# (soil type) the most.

# Also we will take a look at how to review a model while it is running.
gbm2 <- h2o.gbm(
	training_frame = train,
	validation_frame = valid,
	model_id = "gbm_covType2",
	x = 1:12,
	y = 13,
	ntrees = 20, 
	learn_rate = 0.2, # increase the learning rate (from 0.1)
	max_depth = 10, # increase the depth (from 5)
	stopping_rounds = 2, 
	stopping_tolerance = 0.01, 
	seed = 2000000
)

# review the new model's accuracy
h2o.hit_ratio_table(gbm2, valid = TRUE)[1, 2]

# so even though we ran fewer trees, you can see that by adding the depth, making 
# each tree have a greater impact gave us a net gain in the overall accuracy 

# This has moved us in the right direction, but still lower accuracy than the random forest.
# and it still has not converged, so we can make it more aggressive.
# We can now add some of the nature of random forest into the GBM
# using some of the new settings. This will help generalize the model's performance 
gbm3 <- h2o.gbm(
	training_frame = train,
	validation_frame = valid,
	x = 1:12,
	y = 13,
	ntrees = 30, # add a few trees (from 20, though default is 50)
	learn_rate = 0.3, # increase the learning rate even further
	max_depth = 10,
	sample_rate = 0.7, # use a random 70% of the rows to fit each tree
	col_sample_rate = 0.7, # use 70% of the columns to fit each tree
	stopping_rounds = 2,
	stopping_tolerance = 0.01,
	model_id = "gbm_covType3",
	seed = 2000000
)
# review the newest model's accuracy
h2o.hit_ratio_table(gbm3, valid = TRUE )[1, 2]


# Now the GBM is close to the initial random forest.
# However, we used a default random forest. 
# And while there are only a few parameters to tune, we can 
# experiment with those to see if it will make a difference.
# The main parameters to tune are the tree depth and the mtries, which
# is the number of predictors to use. The default depth of trees is 20. 
# Note that the default mtries depends on whether classification or regression
# is being run. The default for classification is one-third of the columns, while 
# the default for regression is the square root of the number of columns.
rf2 <- h2o.randomForest( 
	training_frame = train,
	validation_frame = valid,
	x = 1:12,
	y = 13,
	model_id = "rf_covType2",
	ntrees = 200,
	max_depth = 30, # Increase depth, from 20
	stopping_rounds = 2,
	stopping_tolerance = 1e-2,
	score_each_iteration = TRUE,
	seed = 3000000 
)

# newest random forest accuracy
h2o.hit_ratio_table(rf2, valid = TRUE)[1, 2]


# create predictions using our latest RF model against the test set.
rf2_pred <- h2o.predict(rf2, newdata = test)

# Glance at what that prediction set looks like
# We see a final prediction in the "predict" column,
# and then the predicted probabilities per class.
rf2_pred

# test set accuracy
mean(rf2_pred$predict == test$Cover_Type) 

# shutting down the h2o cluster 
h2o.shutdown(prompt = FALSE)

