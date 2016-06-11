
# https://www.kaggle.com/solegalli/d/crowdflower/twitter-airline-sentiment/airline-sentiment-part-1/comments
# https://cran.r-project.org/web/packages/cowplot/vignettes/introduction.html
library(dplyr)
library(ggplot2)
library(cowplot)
library(data.table)
setwd('/Users/ethen/machine-learning/sentiment/twitter_sentiment')


# The dataset contains 14640 tweets and 15 variables (columns)
# which we'll explore
tweets <- fread( 'Tweets.csv', na.strings = c( '', ' ', 'NA' ) )
dim(tweets)

# check which columns contain NA and how many
# Airline_sentiment_gold and nevative_reason_gold are mostly empty columns, 
# i.e., they contain almost no information
apply( tweets, 2, function(x) sum( is.na(x) ) )


# Proportion of tweets with each sentiment
# We can see from the bar plot that most tweets contain negative sentiment
sentiment_freq <- data.table( prop.table( table(tweets$airline_sentiment) ) )
setnames( sentiment_freq, c( 'sentiment', 'percentage' ) )

# plot
sentiment_color <- c( "indianred1", "deepskyblue", "chartreuse3" )
ggplot( sentiment_freq, aes( sentiment, percentage, fill = sentiment ) ) + 
geom_bar( stat = "identity" ) + guides( fill = FALSE ) + 
ggtitle('Overall Sentiment Distribution') + 
scale_fill_manual( values = sentiment_color ) + 
theme( plot.title = element_text( size = 14, face = 'bold' ) )


# Proportion of sentiment tweets per airline
sentiment_per_airline <- with( tweets, table( airline_sentiment, airline ) ) %>%
						 prop.table() %>%
						 data.table()
setnames( sentiment_per_airline, c( 'sentiment', 'airline', 'percentage' ) )

ggplot( sentiment_per_airline, aes( airline, percentage, fill = sentiment ) ) + 
geom_bar( stat = "identity", position = 'fill' ) + 
ggtitle('Proportion of Tweets per Airline') +
scale_fill_manual( values = sentiment_color ) + 
theme( plot.title = element_text( size = 14, face = 'bold', hjust = 0.5 ), 
	   axis.title.x = element_text( vjust = -1 ) )
# The filled bar chart allows us to grasp the proportion of negative sentiment tweets 
# per airline. We see that American, United and US Airways directed tweets 
# are mostly negative. On the contrary, tweets directed towards Delta, 
# Southwest and Virgin contain a good proportion of neutral and positive sentiment tweets.


# Reasons for negative sentiment per airline
negativereason_per_airline <- with( tweets, table( airline, negativereason ) ) %>%
						 	  prop.table() %>%
						 	  data.table()
# some of the negative reasons have been left blank we'll change them to not specified
negativereason_per_airline[ negativereason == '', negativereason := 'not specified' ]

ggplot( negativereason_per_airline, aes( negativereason, N, fill = airline ) ) + 
geom_bar( stat = "identity" ) + guides( fill = FALSE ) + 
facet_wrap( ~ airline ) + 
theme( plot.title = element_text( size = 14, face = 'bold', vjust = 1 ), 
	   axis.text.x = element_text( angle = 90, size = 10, vjust = 1 ) )


# re-tweet
table(tweets$retweet_count)
tweets[ retweet_count == 44, .(text) ]


# location of the tweet
location <- tweets$tweet_coord
location <- location[ !is.na(location) ]

# add a count column filled with 1s
# remove duplicate locations and count the times they appeared
location <- data.table( count = 1, location = location )
location <- location[ , .( count = sum(count) ), by = location ][ order(-count) ]

location[ , location := gsub( '\\[(.*)\\]', '\\1', location ) ]
location[ , c( 'lat', 'long') := tstrsplit( location, ',' ) ]
location[ , location := NULL ]
location[ , `:=`( long = as.numeric(long), lat = as.numeric(lat) ) ]

# removes row containing coords [0,0] which are probably wrong
location <- location[ !( lat == 0 & long == 0 ), ]


world_map <- map_data("world")
ggplot() + 
geom_polygon( data = world_map, aes( long, lat, group = group ), 
			  color = "black", fill = 'lightblue' ) + 
geom_point( data = location, aes( long, lat, size = count ), color = "coral1" ) +
ggtitle("Location of tweets across the World") + 
ylim( c( -50, 80 ) ) + scale_size( name = "Total Tweets" )

# https://www.kaggle.com/solegalli/d/crowdflower/twitter-airline-sentiment/airline-sentiment-part-2/comments
# https://rpubs.com/gaston/dendrograms
# http://www.sthda.com/english/wiki/beautiful-dendrogram-visualizations-in-r-5-must-known-methods-unsupervised-machine-learning