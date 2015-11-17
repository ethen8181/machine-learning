# http://dsnotes.com/blog/2015/01/02/locality-sensitive-hashing-in-r-part-1/

# -----------------------------------------------------------------------------------------------------
# Documentation code 
library(dplyr)
library(proxy)
library(data.table)
setwd("/Users/ethen/machine-learning/text_similarity/data")

doc <- lapply( list.files(), readLines )

# remove punctuation mark and convert to lower cases
# and extra white space as a single blank
doc1 <- lapply( doc, function(x)
{
	text <- gsub( "[[:punct:]]", "", x ) %>% tolower()
	text <- gsub( "\\s+", " ", text )	
	word <- strsplit( text, " " ) %>% unlist()
	return(word)
})


# ------------------------------------------
# test code, not implemented
# doc1 <- lapply( doc, function(x)
# {
# 	 text <- gsub( "[[:punct:]]|\\s+", "", x ) %>% tolower()
#	 return(text)
# })

Shingling <- function( document, k )
{
	shingles <- character( length = nchar(document) - k + 1 )	

	for( i in 1:nchar(document) )
	{
		if( i + k - 1 > nchar(document) )
			break
		shingles[i] <- substring( document, i, i + k - 1 )
	}
	return( unique(shingles) )	
}


# ---------------------------------------------------------------------------------
#                  Shingling 
# ---------------------------------------------------------------------------------

Shingling <- function( document, k )
{
	shingles <- character( length = ( length(document) - k + 1 ) )

	for( i in 1:( length(document) - k + 1 ) )
	{
		shingles[i] <- paste( document[ i:( i + k - 1 ) ], collapse = " " )
	}
	return(shingles)	
}

doc1 <- lapply( doc1, function(x)
{
	Shingling( x, k = 3 )
})
doc1[[1]]

# ---------------------------------------------------------------------------------
#                 Jaccard Similarity  
# ---------------------------------------------------------------------------------

# unique sets on shingles across all documents
doc_dict <- unlist(doc1) %>% unique()

# convert to boolean matrices, where 
# rows    = elements of the universal set (every possible combinations across all documents )
# columns = one column per document
# thus the matrix has one in row i and column j if and only if document j contains the term i 
M <- lapply( doc1, function( set, dict )
{
	as.integer( dict %in% set )
}, dict = doc_dict ) %>% data.frame() 

# set the names for both rows and columns
setnames( M, paste( "doc", 1:length(doc), sep = "_" ) )
rownames(M) <- doc_dict
M

# How similar is two given document, jaccard similarity 
JaccardSimilarity <- function( x, y )
{
	non_zero <- which( x | y )
	set_intersect <- sum( x[non_zero] & y[non_zero] )
	set_union <- length(non_zero)
	return( set_intersect / set_union ) 
}

# create a new entry in the registry
pr_DB$set_entry( FUN = JaccardSimilarity, names = c("JaccardSimilarity") )

# cosine degree distance matrix 
d1 <- dist( t(M), method = "JaccardSimilarity" )

# delete entry
pr_DB$delete_entry( "JaccardSimilarity" )
d1
doc


# ---------------------------------------------------------------------------------
#                 MinHash   
# ---------------------------------------------------------------------------------

# random permutation
# number of hash functions (signature number )
signature_num <- 8

# prime number
prime <- 17
set.seed(12345)
coeff_a <- sample( nrow(M), signature_num )
coeff_b <- sample( nrow(M), signature_num )

# check that it does permute 
permute <- lapply( 1:signature_num, function(s)
{
	hash <- numeric( length = length(nrow(M)) )
	for( i in 1:nrow(M) )
		hash[i] <- ( coeff_a[s] * i + coeff_b[s] ) %% prime
	
	return(hash)
})
# # convert to data frame 
permute_df <- structure( permute, names = paste0( "hash_", 1:length(permute) ) ) %>%
              data.frame()


# -----------------------------------------------------------------------------
# bind with the original characteristic matrix, using the first two sig 
M1 <- cbind( M, permute_df[1:2] )
rownames(M1) <- 1:nrow(M1)
M1

# calculate signatures 

# obtain the non zero rows' index
non_zero_rows <- lapply( 1:ncol(M), function(j)
{
	return( which( M[ , j ] != 0 ) )
})

# initialize signature matrix
SM <- matrix( data = NA, nrow = signature_num, ncol = ncol(M) )

# for each column (document)
for( i in 1:ncol(M) )
{
	# for each hash function (signature)'s value 
	for( s in 1:signature_num )
		SM[ s, i ] <- min( permute_df[ , s ][ non_zero_rows[[i]] ] )
}
SM	

# signature similarity 
SigSimilarity <- function( x, y ) mean( x == y )

pr_DB$set_entry( FUN = SigSimilarity, names = c("SigSimilarity") )
d2 <- dist( t(SM), method = "SigSimilarity" )
pr_DB$delete_entry( "SigSimilarity" )
d2



# explains  
# http://www.bogotobogo.com/Algorithms/minHash_Jaccard_Similarity_Locality_sensitive_hashing_LSH.php


# the # of shingle sets are large, use minhash to convert large sets into short signatures
# while still preserving similarity


# python implement 
# https://github.com/chrisjmccormick/MinHash


# more basic version ?
# https://github.com/rahularora/MinHash


# blog post
# http://matthewcasperson.blogspot.tw/2013/11/minhash-for-dummies.html


# good blog 
# http://okomestudio.net/biboroku/?p=2065



