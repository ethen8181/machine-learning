# explore 


library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(data.table)
setwd("/Users/ethen/Desktop/loan_prediction/data")

# objective :
# Predict the probability that somebody will experience financial distress in the next two years.
# the goal is to provide each applicant with a credit score

# https://www.creditkarma.com/article/what-is-a-delinquent-account

# variable
# @SeriousDlqin2yrs : 
# 	Person experienced 90 days past due delinquency or worse ( Y / N )
# @RevolvingUtilizationOfUnsecuredLines : 
# 	Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits
# @age :
#	Age of borrower in years
# @NumberOfTime30-59DaysPastDueNotWorse :
# 	Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
# @DebtRatio :
# 	Monthly debt payments, alimony,living costs divided by monthy gross income
# @MonthlyIncome :
# 	Monthly income
# 

# do not read in the first column, they're meaningless id numbers for the model 
data_train <- fread( "cs-training.csv", select = 2:12 )
data_test  <- fread( "cs-test.csv" )
prop.table( table(data_train$SeriousDlqin2yrs) )

# check the probability of the output, missing values 
summary(data_train)

# @MonthlyIncome and @NumberOfDependents contains missing values
# and @age seems odd 

findCorrelation( cor(data_train), cutoff = 0.8 )
zero_var_columns <- colnames(data_train)[ nearZeroVar( data_train ) ]
table(data_train$NumberOfTimes90DaysLate)
table(data_train$NumberOfTime60To89DaysPastDueNotWorse)


# visualize all the input variables 
output <- "SeriousDlqin2yrs"
input  <- setdiff( colnames(data_train), output )

VisualizeDistribution <- function( data_train, input )
{
	plot_density <- lapply( input, function(column)
	{
		ggplot( data_train[ , input, with = FALSE ], aes_string( column ) ) + 
		geom_density()
	})
	do.call( grid.arrange, plot_density )	
}
VisualizeDistribution( data_train, input )
 

