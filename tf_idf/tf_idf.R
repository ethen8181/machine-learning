library(tm)
library(proxy)
library(dplyr)

doc <- c( "The sky is blue.", "The sun is bright today.",
		  "The sun in the sky is bright.", "We can see the shining sun, the bright sun." )

# -----------------------------------------------------------------------------------
#                                    TF-IDF
# -----------------------------------------------------------------------------------

# create corpus
# stop words list 
# stopwords("english")
doc_corpus <- Corpus( VectorSource( doc ) )
control_list <- list( removePunctuation = TRUE, stopwords = TRUE, tolower = TRUE )
tdm <- TermDocumentMatrix( doc_corpus, control = control_list )
# inspect(tdm_train)

# tf
tf <- as.matrix(tdm)

# idf
( idf <- log( ncol(tf) / ( 1 + rowSums( tf != 0 ) ) ) )

# diagonal matrix
( idf <- diag(idf) )

# remember to transpose the original tf matrix
# equivalent to t(tf) %*% idf, but crossprod is faster 
tf_idf <- crossprod( tf, idf )
colnames(tf_idf) <- rownames(tf)
tf_idf

# normalize 
tf_idf / sqrt( rowSums( tf_idf^2 ) )


# -----------------------------------------------------------------------------------
#                                    Text Clustering
# -----------------------------------------------------------------------------------

# cosine example 
a <- c( 3, 4 )
b <- c( 5, 6 )

# print cos and degree 
l <- list( numerator = sum( a * b ), denominator = sqrt( sum( a^2 ) ) * sqrt( sum( b^2 ) ) )
list( cosine = l$numerator / l$denominator, 
      degree = acos( l$numerator / l$denominator ) * 180 / pi )


# news data
setwd("/Users/ethen/machine-learning/tf_idf")
news <- read.csv( "news.csv", stringsAsFactors = FALSE )

# [TFIDF] :
# @vector = pass in a vector of documents  
TFIDF <- function( vector )
{
	# tf 
	news_corpus  <- Corpus( VectorSource(vector) )
	control_list <- list( removePunctuation = TRUE, stopwords = TRUE, tolower = TRUE )
	tf <- TermDocumentMatrix( news_corpus, control = control_list ) %>% as.matrix()

	# idf
	idf <- log( ncol(tf) / ( 1 + rowSums( tf != 0 ) ) ) %>% diag()

	return( crossprod( tf, idf ) )
}

# tf-idf matrix using news' title 
news_tf_idf <- TFIDF(news$title)


# [Cosine] :
# distance between two vectors 
Cosine <- function( x, y )
{
	similarity <- sum( x * y ) / ( sqrt( sum( y^2 ) ) * sqrt( sum( x^2 ) ) )

	# given the cosine value, use acos to convert back to degrees
	# acos returns the radian, multiply it by 180 and divide by pi to obtain degrees
	return( acos(similarity) * 180 / pi )
}

# calculate pair-wise distance matrix 
pr_DB$set_entry( FUN = Cosine, names = c("Cosine") )
d1 <- dist( news_tf_idf, method = "Cosine" )
pr_DB$delete_entry( "Cosine" )

# equivalent to the built in "cosine" distance 
# d1 <- dist( news_tf_idf, method = "cosine" )

# heirachical clustering 
cluster1 <- hclust( d1, method = "ward.D" )
plot(cluster1)
rect.hclust( cluster1, 17 )
groups1 <- cutree( cluster1, 17 )
# table(groups1)

news$title[ groups1 == 2 ]
news$title[ groups1 == 7 ]
news$title[ groups1 == 17 ]

# -----------------------------------------------------------------------------------
# topic model compare results 

library(topicmodels)

rect.hclust( cluster1, 8 )
groups2 <- cutree( cluster1, 8 )

lapply( 1:length( unique(groups2) ), function(i) news$title[ groups2 == i ] )

LDACaculation <- function(vector)
{
	news_corpus  <- Corpus( VectorSource(vector) )
	control_list <- list( removePunctuation = TRUE, stopwords = TRUE, tolower = TRUE )
	dtm <- DocumentTermMatrix( news_corpus, control = control_list )
	lda <- LDA( dtm, k = 8, method = "Gibbs", 
	   		    control = list( seed = 1234, 
	   		    				burnin = 1000, 
	   		    				thin = 100, 
	   		    				iter = 1000 ) )
	return(lda)	
}

lda <- LDACaculation(news$title)


topics(lda)
table( topics(lda) )
lapply( 1:length( unique( topics(lda) ) ), function(i) news$title[ topics(lda) == i ] )


terms( lda, 6 )

lda@gamma
lda@alpha
posterior(lda)$documents

best_topics <- data.frame( best = apply( posterior(lda)$topics, 1, max ) )

library(ggplot2)
ggplot( best_topics, aes( best ) ) + 
geom_histogram()

