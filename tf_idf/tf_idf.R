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
control_list <- list( removePunctuation = TRUE, stopwords = TRUE )
tdm <- TermDocumentMatrix( doc_corpus, control = control_list )
# inspect(tdm_train)

# tf
tf <- as.matrix(tdm)

# idf
( idf <- log( ncol(tf) / ( 1 + rowSums( tf != 0 ) ) ) )

# diagonal matrix
( idf <- diag(idf) )

# remember to transpose the original tf matrix
tf_idf <- t(tf) %*% idf
colnames(tf_idf) <- rownames(tf)
tf_idf

# normalize 
tf_idf / sqrt( rowSums( tf_idf^2 ) )


# -----------------------------------------------------------------------------------
#                                    Text Clustering
# -----------------------------------------------------------------------------------

setwd("/Users/ethen/machine-learning/tf_idf")
news <- read.csv( "news.csv", stringsAsFactors = FALSE )

# [TFIDF] :
# @vector = pass in a vector of documents  
TFIDF <- function( vector )
{
	# tf 
	news_corpus  <- Corpus( VectorSource(vector) )
	control_list <- list( removePunctuation = TRUE, stopwords = TRUE )
	tf <- TermDocumentMatrix( news_corpus, control = control_list ) %>% as.matrix()

	# idf
	idf <- log( ncol(tf) / ( 1 + rowSums( tf != 0 ) ) ) %>% diag()

	return( t(tf) %*% idf )
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
d <- dist( news_tf_idf, method = "Cosine" )
pr_DB$delete_entry( "Cosine" )

# heirachical clustering 
cluster <- hclust( d, method = "ward.D" )
plot(cluster)

# manually examine some cluster 
list( news$title[ c( 8, 9, 22, 36, 69 ) ], news$title[ c( 55, 57, 66 ) ] )


