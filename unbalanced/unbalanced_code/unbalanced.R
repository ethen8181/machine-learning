# logistic regression

# environment setting 
library(ROCR)
library(grid)
library(broom)
library(caret)
library(tidyr)
library(dplyr)
library(scales)
library(ggplot2)
library(ggthemr) 
library(ggthemes)
library(gridExtra)
library(data.table)
setwd("/Users/ethen/machine-learning/logistic_regression")

# read in HR dataset 
data <- fread( list.files( "data", full.names = TRUE )[2] )
str(data)

# using summary to check if columns contain missing values like NAs 
summary(data)

# find correlations to exclude from the model 
findCorrelation( cor(data), cutoff = .75, names = TRUE )

# from this probability table we can see that 16 percent of 
# your emplyees have left
prop.table( table(data$left) )


# -------------------------------------------------------------------------
#						Model Training 
# -------------------------------------------------------------------------

# convert the newborn to factor variables
data[ , Newborn := as.factor(Newborn) ]

# split the dataset into two parts. 80 percent of the dataset will be used to actually 
# train the model, while the rest will be used to evaluate the accuracy of this model, 
# i.e. out of sample error
set.seed(4321)
test <- createDataPartition( data$left, p = .2, list = FALSE )
data_train <- data[ -test, ]
data_test  <- data[ test, ]
rm(data)

model_glm <- glm( left ~ . , data = data_train, family = binomial(logit) )
summary_glm <- summary(model_glm)

# p-value and pseudo r squared 
list( model_glm_sum$coefficient, 
	  1- ( model_glm_sum$deviance / model_glm_sum$null.deviance ) )
# all the p value of the coefficients indicates significance 


# -------------------------------------------------------------------------
#						Predicting and Assessing the Model 
# -------------------------------------------------------------------------

# obtain the predicted value that a employee will leave in the future on the train
# and test set, after that we'll perform a quick evaluation by using the double density plot
data_train$prediction <- predict( model_glm, newdata = data_train, type = "response" )
data_test$prediction  <- predict( model_glm, newdata = data_test , type = "response" )

# given that our model's final objective is to classify new instances 
# into one of two categories, whether the employee will leave or not
# we will want the model to give high scores to positive
# instances ( 1: employee left ) and low scores ( 0 : employee stayed ) otherwise. 

# distribution of the prediction score grouped by known outcome
ggplot( data_train, aes( prediction, color = as.factor(left) ) ) + 
geom_density( size = 1 ) +
ggtitle( "Training Set's Predicted Score" ) + 
scale_color_economist( name = "data", labels = c( "negative", "positive" ) ) + 
theme_economist()

# Ideally you want the distribution of scores to be separated, 
# with the score of the negative instances to be on the left and the score of the
# positive instance to be on the right.
# In the current case, both distributions are slight skewed to the left. 
# Not only is the predicted probability for the negative outcomes low, but 
# the probability for the positive outcomes are also lower than it should be. 
# The reason for this is because our dataset only consists of 16 percent of positive 
# instances ( employees that left ). Thus our predicted scores sort of gets pulled 
# towards a lower number because of the majority of the data being negative instances.

# A slight digression, when developing models for prediction, we all know that we want the model to be
# as accurate as possible, or in other words, to do a good job in 
# predicting the target variable on out of sample observations.

# Our plot, however, can actually tell us a very important thing :
# Accuracy will not be a suitable measurement for this model 

# We'll show why below :

# Since the prediction of a logistic regression model is a 
# probability, in order to use it as a classifier, we'll have a choose a cutoff value,
# or you can say its a threshold. Where scores above this value will classified as 
# positive, those below as negative. We'll be using the term cutoff for the rest of 
# the documentation

# Here we'll use a function to loop through several cutoff values and 
# compute the model's accuracy on both training and testing set
source("logistic_regression_code/logistic_functions.R")

accuracy_info <- AccuracyCutoffInfo( train = data_train, test = data_test, 
									 predict = "prediction", actual = "left" )
# define the theme for the next plot
ggthemr("light")
accuracy_info$plot


# from the output, you can see that starting from the cutoff value of .6
# our accuracy for both training and testing set grows higher and higher showing 
# no sign of decreasing at all 
# we'll visualize the confusion matrix of the test set to see what's causing this
cm_info <- ConfusionMatrixInfo( data = data_test, predict = "prediction", 
					 			actual = "left", cutoff = .6 )
ggthemr("flat")
cm_info$plot

# wiki : https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Worked_example
# The above plot depicts the tradeoff we face upon choosing a reasonable cutoff. 

# if we increase the cutoff value, 
# the number of true negative (TN) increases and the number of true positive (TP) decreases.
# Or you can say, If we increase the cutoff's value, the number of false positive (FP) is lowered, 
# while the number of false negative (FN) rises. 
# Here, because we have very few positive instances, thus our model will be 
# less likely to make a false negative mistake, so if we keep on adding 
# the cutoff value, we'll actually increase our model's accuracy, since 
# we have a higher chance of turning the false positive into true negative. 

# predict all the test set's outcome as 0
prop.table( table( data_test$left ) )

# Section conclusion : 
# Accuracy is not the suitable indicator for the model 
# for unbalanced distribution or costs

# -------------------------------------------------------------------------
#						Choosing the Suitable Cutoff Value 
# -------------------------------------------------------------------------

# use the roc curve to determine the cutoff
# it plots the false positive rate (FPR) on the x-axis and the true positive rate (TPR) on the y-axis
print(cm_info$data)

ggthemr_reset()
# different cost for false negative and false positive 
cost_fp <- 100
cost_fn <- 200
roc_info <- ROCInfo( data = cm_info$data, predict = "predict", 
					 actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )

# reset to default ggplot theme 
grid.draw(roc_info$plot)


# re plot the confusion matrix plot 
cm_info <- ConfusionMatrixInfo( data = data_test, predict = "prediction", 
                                actual = "left", cutoff = roc_info$cutoff )
ggthemr("flat")
cm_info$plot


# -------------------------------------------------------------------------
#						Interpretation 
# -------------------------------------------------------------------------

# tidy from the broom package
coefficient <- tidy(model_glm)[ , c( "term", "estimate", "statistic" ) ]

coefficient$estimate <- exp( coefficient$estimate )

# one unit increase in statisfaction, the odds of leaving the company 
# (versus not leaving) increase by a factor of
coefficient[ coefficient$term == "S", "estimate" ]

# use the model to predict a unknown outcome data "HR_unknown.csv"
# specify the column's class 
col_class <- sapply( data_test, class )[1:6]
data <- read.csv( list.files( "data", full.names = TRUE )[1], colClasses = col_class )
data$prediction <- predict( model_glm, newdata = data, type = "response" )

# cutoff
data <- data[ data$prediction >= roc_info$cutoff, ]

# time spent in the company 
median_tic <- data %>% group_by(TIC) %>% 
					   summarise( prediction = median(prediction), count = n() )
ggthemr("fresh")
ggplot( median_tic, aes( TIC, prediction, size = count ) ) + 
geom_point() + theme( legend.position = "none" ) +
labs( title = "Time and Employee Attrition", y = "Attrition Probability", 
	  x = "Time Spent in the Company" ) 

# last project evaluation 
data$LPECUT <- cut( data$LPE, breaks = quantile(data$LPE), include.lowest = TRUE )
median_lpe <- data %>% group_by(LPECUT) %>% 
					   summarise( prediction = median(prediction), count = n() )

ggplot( median_lpe, aes( LPECUT, prediction ) ) + 
geom_point( aes( size = count ), color = "royalblue3" ) + 
theme( legend.position = "none" ) +
labs( title = "Last Project's Evaluation and Employee Attrition", 
	  y = "Attrition Probability", x = "Last Project's Evaluation by Client" )

# This is probabily an indication that it'll be worth trying out other classification 
# algorithms. Since logistic regressions assumes monotonic relationships ( either entirely increasing or decreasing )
# between the input paramters and the outcome ( also true for linear regression ). Meaning the   
# if more of a quantity is good, then much more of the quantity is better. This is often not the case in the real world

# given this probability we can prioritize our actions by adding back how much 
# do we wish to retain these employees. Recall that from our dataset, we have the performance
# information of the employee ( last project evaluation ). 
# given this table, we can easily create a visualization to tell the story
ggplot( data, aes( prediction, LPE ) ) + 
geom_point() + 
ggtitle( "Performace v.s. Probability to Leave" )

# we first have the employees that are underperforming, we probably should 
# improve their performance or you can say you can't wait for them to leave....
# for employees that are not likely to leave, we should manage them as usual
# then on the short run, we should focus on those with a good performance, but
# also has a high probability to leave.

# the next thing we can do, is to quantify our priority by 
# multiplying the probablity to leave with the performance.
# we'll also use row names of the data.frame to 
# to serve as imaginery employee ids.
# Then we will obtain a priority score. Where the score will be high for 
# the employees we wish to act upon as soon as possible, and low for the other ones
result <- data %>% 
		  mutate( priority = prediction * LPE ) %>%
		  mutate( id = rownames(data) ) %>%
		  arrange( desc(priority) )

# after obtaining this result, we can schedule a face to face interview with employees 
# at the top of the list.

# using classification in this example enabled us to detect events that will 
# happen in the future. That is which employees are more likely to leave the company.
# Based on this information, we can come up with a more efficient strategy to cope
# with matter at hand.


# ----------------------------------------------------------------------
# document later, strange statistic test 
# http://www.r-bloggers.com/evaluating-logistic-regression-models/
		  	
