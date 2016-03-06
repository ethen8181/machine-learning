# exploring around 
library(caret)
library(ggplot2)
library(gridExtra)
library(data.table)

setwd("/Users/ethen/Desktop/homesite_quote_conversion")
data_train <- fread( "data/train.csv", stringsAsFactors = TRUE, select = 2:299 )
dim(data_train) 
# 260753 observations 
# 299 features, the first two being ID and Dates 

prop.table( table(data_train$QuoteConversion_Flag) )
# output class percentage 81 / 19

y <- "QuoteConversion_Flag"

VisualizeDistribution <- function( data, input )
{
	# takes the @input specifying the column name for the @data and visualize the 
	# density plot for factor variable and histogram for numeric ( all in one plot )
	plot_density <- lapply( input, function(column)
	{
		# define basic layout
		p <- ggplot( data[ , input, with = FALSE ], aes_string( column ) )
		
		# add aesthetic according to class type 	
		if( class(data[[column]]) == "factor" )
			p <- p + geom_density()		 
		else
			p <- p + geom_histogram()
		return(p)
	})
	do.call( grid.arrange, plot_density )	
}

# use nearZeroVar function to check for near zero variance columns
# , they'll be removed from the input column. The following visualization 
# is also a sanity check to see if these columns are in fact dominated by a single value 
zero_variance_columns <- caret::nearZeroVar(data_train)
names(data_train)[zero_variance_columns]

# the index is used to differentiate different types of situation


# Field Column
x <- grep( "^Field", names(data_train), value = TRUE )
VisualizeDistribution( data_train, x ) 
# "Field10" has commas for numbers. Should be removed and convert back to numeric 


# CoverageField column
x <- grep( "^CoverageField", names(data_train), value = TRUE )
VisualizeDistribution( data_train, x )
# 1. "CoverageField5A", "CoverageField5B", "CoverageField6A", "CoverageField6B"
#	 contains only a few distinct value. Convert to factor variable


# SalesField column
x <- grep( "^SalesField", names(data_train), value = TRUE )
VisualizeDistribution( data_train, x )
# 1. "SalesField3", "SalesField4", "SalesField5", "SalesField9"
#    contains only a few distinct value. Convert to factor variable
# 2. "SalesField10", "SalesField11", "SalesField12" left skewed. 
#    Use log( x + 1 ) transformation
# 3. "SalesField13" dominated by a single value


# PersonalField column
x <- grep( "^PersonalField", names(data_train), value = TRUE )
# length(x)
# contains 83 columns, visualize separately, 16 at a time 

VisualizeDistribution( data_train, x[1:16] )
# 1. "PersonalField1", "PersonalField2", "PersonalField6",
#    contains only a few distinct value. Convert to factor variable
# 3. "PersonalField7", "PersonalField8", "PersonalField11", "PersonalField13" 
#	 "PersonalField9", "PersonalField12", "PersonalField14" are dominated by a single value.

VisualizeDistribution( data_train, x[17:32] )
# 3. "PersonalField22", "PersonalField23", "PersonalField24", "PersonalField25"   
#    "PersonalField26" are dominated by a single value
# 4. "PersonalField16", "PersonalField17", "PersonalField18", "PersonalField19"
# 	 high level column. 


VisualizeDistribution( data_train, x[33:48] )
# 3. "PersonalField34" to "PersonalField43" and "PersonalField49" 
#    are dominated by a single value

VisualizeDistribution( data_train, x[49:64] )
# 3. "PersonalField50" to "PersonalField65" are dominated by a single value

VisualizeDistribution( data_train, x[65:length(x)] )
# 3. "PersonalField66" to "PersonalField73" and 
#    "PersonalField79" to "PersonalField84" are dominated by a single value 


# PropertyField
x <- grep( "^PropertyField", names(data_train), value = TRUE )
# length(x) 
# contains 47 columns, visualize separately, 16 at a time 

VisualizeDistribution( data_train, x[1:16] )
# 3. "PropertyField2A", "PropertyField5", "PropertyField6",
#    "PropertyField9", "PropertyField10", "PropertyField11A", "PropertyField11B"
# 	 are dominated by a single value
table(data_train$PropertyField8)

VisualizeDistribution( data_train, x[17:32] )
# 3. "PropertyField20", "PropertyField22" are columns that are dominated by a single value

VisualizeDistribution( data_train, x[33:length(x)] )
# 3. "PropertyField28", "PropertyField29", "PropertyField36", "PropertyField38"
# 	 are dominated by a single value. 29 is dominated by NA 
# 1. "PropertyField35" contains only a few distinct value. Convert to factor variable


x <- grep( "^GeographicField", names(data_train), value = TRUE )
# length(x) 
# contains 126 columns, visualize separately, 16 at a time 

VisualizeDistribution( data_train, x[1:16] )
# 3. "GeographicField5A" dominated by a single value

VisualizeDistribution( data_train, x[17:32] )
# 3. "GeographicField10A", "GeographicField10B", "GeographicField14A"
#    are dominated by a single value

VisualizeDistribution( data_train, x[33:48] )
# 3. "GeographicField18A" "GeographicField21A" "GeographicField22A" "GeographicField23A"
# 	 are dominated by a single value

VisualizeDistribution( data_train, x[49:64] )
# looks normal

VisualizeDistribution( data_train, x[65:80] )
# looks normal

VisualizeDistribution( data_train, x[81:96] )
# looks normal

VisualizeDistribution( data_train, x[97:112] )
# 3. "GeographicField56A" dominated by a single value

VisualizeDistribution( data_train, x[113:length(x)] )
# 3. "GeographicField60A" "GeographicField61A" "GeographicField62A"
# 	 "GeographicField63" are dominated by a single value 



# important features 
SalesField5
PersonalField10A
PropertyField37

