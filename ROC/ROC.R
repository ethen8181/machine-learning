# ROC curve with logistic regression(binary data)
# Clarify which threshold should be used to determine whether the 
# prediction is to be classified as positive or negative
# compare the idea using the prediction accuracy and cost function with the ROC curve 

# further reimplementation of the following link
# http://www.r-bloggers.com/illustrated-guide-to-roc-and-auc/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+RBloggers+%28R+bloggers%29

library(pROC)
library(dplyr)
library(ggplot2)
library(gridExtra)

setwd("/Users/ethen/machine-learning/ROC")
# download files
if( !file.exists("titanic.csv") )
{
    url <- "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv"
    download.file( url, destfile = "titanic.csv" )    
}
# read in file
data <- read.csv( "titanic.csv", stringsAsFactors = FALSE )


# ----------------------------------------------------------------------------------------
# preprocessing
# @column : only use these as the columns as the predictor
# @col : will be convertd into factor variables
columns <- c( "survived", "pclass", "sex", "age", "sibsp", "parch" )
col  <- which ( names(data) %in% c( "pclass", "survived", "sex" ) )
# convert integer columns to factor and exclude the NA data
df <- data[ , columns ] %>% modifyList( lapply( data[ , col ], as.factor ) ) %>%
                            filter( !is.na( data$age ) )
# preprocessed dataset
head(df)


# ----------------------------------------------------------------------------------------
# cross validation leave ten out, the training sample will leave 10 samples out randomly.
# but each sample has to be used at least once in the training set
# generate parameters for the leave ten out training :
# size : leave n out
N <- nrow(df)
size <- 10
# randomize the dataset
set.seed(1234)
df <- df[ sample(N), ]
# num : times training the model    
num  <- floor( N / size )
# rest : number that can't be fully divided ( remainder )
rest <- N - num * size
# number of the cumulative value 
ncv  <- cumsum( c( rep( size, num ), rest ) )
# create a new data.frame prediction, to store the probability
# don't add the pred column to the original column, the model will include that, thus returning error
predictions <- data.frame( survived = df$survived, pred = NA )
# testing : sample_frac( df, 0.9 ) 
for( n in ncv ) 
{
    v <- rep( TRUE, N )
    v[ ( n-size+1 ):n ] <- FALSE
    # logistic regression model using the training dataset
    # skip whether this is a good model or not 
    lr <- glm( survived ~ ., data = df[ v, ], family = binomial )
    # predict the testing data value 
    predictions[ !v, "pred" ] <- predict( lr, newdata = df[ !v, ], type = "response" )
}
df$pred <- predictions$pred
head(df)


# ----------------------------------------------------------------------------------------
# exploratory analysis
# threshold, if the probability is bigger than this value then the person will survive

threshold <- 0.7
# relevel making 1 appears on the left in the graph, as in left of the confusion matrix
df$survived <- relevel( df$survived, "1" ) 
# the outcome of whether the person survived
outcome <- as.factor( ifelse( df$pred > threshold, 1, 0 ) ) %>% relevel("1")
# confusion matrix, the later value in the table is the column
table <- table( outcome, df$survived )

# as far as I know, the confusionMatrix from the caret package is based on a 50% probability cutoff for binary data
# compare it with the user specified threshold
prediction  <- as.factor( ifelse( df$pred > 0.5, 1, 0 ) ) %>% relevel("1") 
confusion   <- table( prediction, df$survived )

# list the two confusion matrix
list( threshold_.7 = table, threshold_.5 = confusion ) 
# accuracy rate between the two threshold, disregarding which one is better, proceed on with the idea of evaluating cost
cbind( threshold_.7 = ( table[ 1, 1 ] + table[ 2, 2 ] ) / sum(table), 
       threshold_.5 = ( confusion[ 1, 1 ] + confusion[ 2, 2 ] ) / sum(confusion) )

# generate the confusion matrix plot
# testing
# v <- rep( NA, nrow(df) )
# v <- ifelse( predictions$pred >= threshold & predictions$survived == 1, "TP", v )

# confusion matrix link : 
# https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Worked_example
# caculating each pred falls into which category for the confusion matrix
df$type <- with( predictions, 
                 ifelse( pred >= threshold & survived == 1, "TP",
                 ifelse( pred >= threshold & survived == 0, "FP", 
                 ifelse( pred <  threshold & survived == 1, "FN", "TN" ) ) ) )

ggplot( df, aes( survived, pred, color = type ) ) + geom_point()

# jittering, makes most sense when visualizing a large number of individual observations
# representing each system.
# Without jittering, we would essentially see two vertical lines. 
# With jittering, we can spread the points along the x axis 
ggplot( df, aes( survived, pred, color = type ) ) + geom_jitter( shape = 1 ) + 
geom_hline( yintercept = threshold, color = "blue", alpha = 0.6 ) + 
labs ( title = sprintf( "Confusion Matrix with Threshold at %.2f", threshold ) )

# The above plot illustrates the tradeoff we face upon choosing a reasonable threshold. 
# If we increase the threshold, the number of false positive (FP) results is lowered, 
# while the number of false negative (FN) results increases.


# ----------------------------------------------------------------------------------------
# [calculate_roc] :
# calculate the x and y value for the ROC curve for each specified threshold
# x = false positive rate, y = true positive rate

CalculateROC <- function( df, cost_of_fp, cost_of_fn, n = 100 ) 
{
    # true positive rate for the specified threshold
    tpr  <- function( df, threshold ) 
    {
        with( df, sum( pred >= threshold & survived == 1 ) / sum( survived == 1 ) )
    }
    # false positive rate
    fpr  <- function( df, threshold ) 
    {
        with( df, sum( pred >= threshold & survived == 0 ) / sum( survived == 0 ) )
    }
    # cost 
    cost <- function( df, threshold, cost_of_fp, cost_of_fn ) 
    {
        with( df, sum( pred >= threshold & survived == 0 ) * cost_of_fp + 
                  sum( pred <  threshold & survived == 1 ) * cost_of_fn )
    }

    # generate n threshold between 0 and 1
    roc <- data.frame( threshold = seq( 0, 1, length.out = n ) )

    # calculate the tpr, fpr and cost for each of the specified threshold
    roc$tpr  <- sapply( roc$threshold, function(th) tpr ( df, th ) )
    roc$fpr  <- sapply( roc$threshold, function(th) fpr ( df, th ) )
    roc$cost <- sapply( roc$threshold, function(th) cost( df, th, cost_of_fp, cost_of_fn ) )
    return(roc)
}

# specify the cost for false positive and negative, usually false negative costs tend to be higher
fpcost <- 1
fncost <- 2
# return the data frame that has the cost, tp and fp rate for each threshold
# ( generate 100 from 0~1 ), n specifies the number of the amount of generated threshold
roc <- CalculateROC( predictions, fpcost, fncost, n = 100 )
head(roc)
# threshold for the lowest cost
roc[ which.min(roc$cost), "threshold" ]


# ----------------------------------------------------------------------------------------
# function that does the ROC and cost plotting
PlotROC <- function( roc, threshold, cost_of_fp, cost_of_fn ) 
{
    # function that normalize the vector
    norm_vec <- function(v) ( v - min(v) )/ diff( range(v) )

    # the generated theshold from 0 ~ 100, which one is closest to the specified threshold 
    index <- which.min( abs( roc$threshold - threshold ) )
    
    # create color from a palette to assign to the 100 generated threshold between 0 ~ 1
    # then normalize each cost and assign colors to it, the higher the blacker
    # don't times it by 100, there will be 0 in the vector
    col_ramp <- colorRampPalette( c( "green", "orange", "red", "black" ) )(100)   
    col_by_cost <- col_ramp[ ceiling( norm_vec(roc$cost) * 99 ) + 1 ]

    # calculate the area under the curve
    area <- auc( df$survived, df$pred )

    # ---------------------------------------
    # ROC curve, wuth crossed point of two dashed line approximately the point for our threshold    
    p_roc <- ggplot( roc, aes( fpr,tpr ) ) + 
             geom_line( color = rgb( 0, 0, 1, alpha = 0.3 ) ) +
             geom_point( color = col_by_cost, size = 4, alpha = 0.5 ) +
             geom_line( aes( threshold, threshold ), color = "blue", alpha = 0.5 ) +
             labs( title = "ROC", x = "False Postive Rate", y = "True Positive Rate" ) +
             geom_hline( yintercept = roc[ index, "tpr" ], alpha = 0.5, linetype = "dashed" ) +
             geom_vline( xintercept = roc[ index, "fpr" ], alpha = 0.5, linetype = "dashed" ) +
             coord_fixed( ratio = 1 ) # equal ratio of x and y

    # plot the calculated cost for each threshold
    p_cost <- ggplot( roc, aes( threshold, cost ) ) +
              geom_line( color = "blue", alpha = 0.5 ) +
              geom_point( color = col_by_cost, size = 4, alpha = 0.5 ) +
              ggtitle("cost function") +
              geom_vline( xintercept = threshold, alpha = 0.5, linetype = "dashed" )
    
    # the main title for the two arranged plot
    sub_title <- sprintf( "Threshold at %.2f - Cost of FP = %d, Cost of FN = %d\nAUC = %.3f", 
                          threshold, cost_of_fp, cost_of_fn, area )
    # return the arranged plot
    grid.arrange( p_roc, p_cost, ncol = 2, 
                  main = textGrob( sub_title, gp = gpar( fontsize = 16, fontface = "bold" ) ) )
}

PlotROC( roc, 0.7, fpcost, fncost )




