# exploring the data

library(ggplot2)
library(gridExtra)
library(data.table)
setwd("/Users/ethen/machine-learning/kaggle/last_man_standing/data")
data_train <- fread( "Train.csv", select = 2:10 )
data_test  <- fread( "Test.csv", select = 2:9 )

# unbalanced output class
prop.table( table(data_train$Crop_Damage) )

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
# similar distribution on training and test 
VisualizeDistribution( data_train, input )
VisualizeDistribution( data_test, input )

# dosage unit is 5 each 
table(data_train$Number_Doses_Week)
table(data_test$Number_Doses_Week)

# log-transform makes sense
ggplot( data_train, aes( log( Estimated_Insects_Count + 1 ) ) ) + 
geom_histogram()
ggplot( data_train, aes( log( Number_Doses_Week + 1 ) ) ) + 
geom_histogram()
ggplot( data_train, aes( log( Number_Weeks_Quit + 1 ) ) ) + 
geom_histogram()

