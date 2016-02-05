# https://cran.r-project.org/web/packages/xgboost/vignettes/discoverYourData.html

library(vcd)
library(caret)
library(Matrix)
library(xgboost)
library(data.table)

# data from the vcd package 
data(Arthritis)
df <- data.table(Arthritis)

str(df)

# group age by every ten years 
df[ , AgeDiscrete := as.factor( round( Age / 10, 0 ) ) ]

# further simplification of ages with an arbitrary split at age 30 

df[ , AgeCat := as.factor( ifelse( Age > 30, "Old", "Young" ) ) ]

# these new added features are highly correlated with the original age feature
# since they're simple transformation of the original feature 
# for many machine learning algorithms, including correlated features may
# be a bad idea, since it may make predictions less acurrate and make it
# harder to interpret 

df[ , ID := NULL ]

levels( df[ , Treatment ] )

# -1 is here to remove the first column which is full of 1
# Improved is the output variable, it will not be transformed  
sparse_matrix <- sparse.model.matrix( Improved ~ .-1, data = df )

# store the output variable 
output_vector <- df[ , Improved ] == "Marked"

# @eta : step size of each boosting step
# @max.depth : maximum depth of the tree, can prevent overfitting 
bst <- xgboost( data = sparse_matrix, label = output_vector, 
			    max.depth = 4,
                eta = 1, 
                nround = 10, 
                objective = "binary:logistic" )

# the number decreases then starts to increase, sign of overfitting 

# variable importance
# extract the dimnames from the sparse matrix object  
importance <- xgb.importance( sparse_matrix@Dimnames[[2]], model = bst )
head(importance)

# the column Gain provides the information we're looking for 
# Gain is the improvement in accuracy brought by a 
# feature to the branches it is on


# interpretation
importance_raw <- xgb.importance( sparse_matrix@Dimnames[[2]], 
								  model = bst, 
								  data  = sparse_matrix, 
								  label = output_vector )

importance_clean <- importance_raw[ , `:=`( Cover = NULL, Frequence = NULL ) ]

# Feature have automatically been divided in 2 clusters: the interesting features… and the others
# Depending of the dataset and the learning parameters you may have more than two clusters. Default value is to limit them to 10
xgb.plot.importance( importance_matrix = importance_raw )


