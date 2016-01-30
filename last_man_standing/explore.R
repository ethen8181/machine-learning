# exploring the data

library(ggplot2)
library(gridExtra)
library(data.table)
setwd("/Users/ethen/machine-learning/last_man_standing/data")
data_train <- fread( "Train.csv", select = 2:10 )
data_test  <- fread( "Test.csv", select = 2:9 )

output <- "Crop_Damage"
input  <- setdiff( colnames(data_train), output )

VisualizeDistribution <- function( data_train, input )
{
	plot_density <- lapply( input, function(column)
	{
		ggplot( data_train[ , input, with = FALSE ], aes_string( column ) ) + 
		geom_histogram()
	})
	do.call( grid.arrange, plot_density )	
}
VisualizeDistribution( data_train, input )
VisualizeDistribution( data_test, input )


# unbalanced output class
prop.table( table(data_train$Crop_Damage) )

quantile(data_train$Number_Weeks_Quit)
quantile(data_train$Number_Doses_Week)
quantile(data_train$Number_Weeks_Used)

# dosage unit is 5 each 
table(data_train$Number_Doses_Week)
table(data_test$Number_Doses_Week)
quantile(data_train$Number_Doses_Week, seq( 0, 1, 0.2 ) )
quantile(data_test$Number_Doses_Week, seq( 0, 1, 0.2 ) )

# log-transform 
ggplot( data_train, aes( log(Estimated_Insects_Count) ) ) + 
geom_histogram()

ggplot( data_train, aes( log(Number_Doses_Week) ) ) + 
geom_histogram()

ggplot( data_train, aes( log(Number_Weeks_Quit+1) ) ) + 
geom_histogram()





