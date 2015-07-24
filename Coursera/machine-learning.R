library(tree)
library(caret)
# set the working directory
setwd("C:/Users/ASUS/machine-learning/Coursera")
# download the file into data folder
if( !file.exists("data") )
{
    dir.create("data")
    url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    link <- c( url1, url2 )
    file <- c( "data/training.csv", "data/testing.csv" )
    for( i in 1:length(link) )
        download.file( link[i], file[i] )
}    
# read in both training and testing files
dataset <- lapply( list.files( "data", full.names = TRUE ), read.csv, na.strings = c( "NA", "" ) )   


# -----------------------------------------------------------------------
# preprocessing 
# exclude the column that consists of NA values, and the first five column.
# which consists of the row number, username and timestamp  
boolean <- which( apply( dataset[[1]], 2, function(x) {sum( is.na(x) )} ) != 0 )

preprocessed <- lapply( dataset, function(x)
{
    x[ , -c( 1:5, boolean ) ]
})

# extract the preprocessed training data
training <- preprocessed[[2]]


# -----------------------------------------------------------------------
# build regression tree and use cross validation to determine the size

set.seed(1234)

inTrain <- createDataPartition( training$classe, p = .75, list = FALSE )
train <- training[  inTrain, ]
test  <- training[ -inTrain, ]


# -----------------------------------------------------------------------
# using the tree package
train_tree <- tree( classe ~., data = train )

# pruning for tree
cross <- cv.tree( train_tree, K = 10 )
# evaluate the difference of the deviation
diff(cross$dev)
# choose the size that has a significant drop 
final_train_tree <- prune.tree( train_tree, best = cross$size[7] )

# prediction
# add the type class for predict or it will return the probability of it being in each class
result_tree <- predict( train_tree, newdata = test, type = "class" )
matrix_tree <- confusionMatrix( test$classe, result_tree )
list( confusionMatrix = matrix_tree$table, modelAccuracy = matrix_tree$overall["Accuracy"] )


# -----------------------------------------------------------------------
# using the rpart package
library(rpart)
library(rpart.plot)
# specify method = class for classification tree, "anova" for regression
train_rpart <- rpart( classe ~., data = train, method = "class" )

# pruning for rpart
# select a tree size that minimizes the cross-validated error
# the xerror column printed by cptable of the rpart model
# then you can prune it by selecting the complexity parameter associated with minimum error
train_rpart$cptable
minCP <- which.min(train_rpart$cptable[,"xerror"])
# you have to specify a complexity parameter that is bigger than this to perform pruning
# you can multiply the corresponding CP and the one above it then sqrt it, e.g.
train_rpart$cptable[ minCP, "CP" ]
cp <- sqrt( train_rpart$cptable[minCP] * train_rpart$cptable[minCP-1] )
prune( train_rpart, cp = cp )
# root node error * rel error is the error rate computed on training sample depending on the complexity parameter (first column)
# root node error * xerror is the cross-validated error rate (using 10-fold CV, see xval in rpart.control() )
printcp(train_rpart)

# prediction
result_rpart <- predict( train_rpart, newdata = test, type = "class" )
matrix_rpart <- confusionMatrix( test$classe, result_rpart )
list( confusionMatrix = matrix_rpart$table, modelAccuracy = matrix_rpart$overall["Accuracy"] )


# --------------------------------------------------
# plot
par( mfrow = c( 1, 2 ) )
rpart.plot(train_rpart)
plot(final_train_tree)
text(final_train_tree, digits = 2 )


# --------------------------------------------------
# predict the actual test set
testing <- preprocessed[[1]]
testing_result <- predict( train_rpart, newdata = testing, type = "class" )

pml_write_files <- function(x)
{
    n <- length(x)
    for( i in 1:n )
    {
        # paste0, the default separtor will be ""
        filename <- paste0(" problem_id_", i, ".txt" )
        # quote = FALSE, prevent double quotes with character or factor columns
        write.table( x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE )
    }
}
pml_write_files(testing_result)
