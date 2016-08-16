# Latent Dirichlet Allocation using gibbs sampling 

# @docs  	 : document that have be converted to token ids
# @vocab 	 : unique token for all the document collection 
# @K     	 : Number of topic groups
# @alpha 	 : parameter for the document - topic distribution
# @eta 	 	 : parameter for the word - topic distribution 
# @iteration : Number of iterations to run gibbs sampling to train our model
# returns 	 : the "unnormalized" distribution matrix 
#			   1. wt : word-topic matrix
# 			   2. dt : document-topic matrix
	
LDA1 <- function( docs, vocab, K, alpha, eta, iterations )
{
	# initialize count matrices 
	# @wt : word-topic matrix 
	wt <- matrix( 0, K, length(vocab) )
	colnames(wt) <- vocab

	# @ta : topic assignment list
	ta <- lapply( docs, function(x) rep( 0, length(x) ) ) 
	names(ta) <- paste0( "doc", 1:length(docs) )

	# @dt : counts correspond to the number of words assigned to each topic for each document
	dt <- matrix( 0, length(docs), K )

	for( d in 1:length(docs) )
	{ 
		# randomly assign topic to word w
		for( w in 1:length( docs[[d]] ) )
		{		
			ta[[d]][w] <- sample( 1:K, 1 ) 

			# extract the topic index, word id and update the corresponding cell 
			# in the word-topic count matrix  
			ti <- ta[[d]][w]
			wi <- docs[[d]][w]
			wt[ ti, wi ] <- wt[ ti, wi ] + 1    
		}

		# count words in document d assigned to each topic t
		for( t in 1:K )  
			dt[ d, t ] <- sum( ta[[d]] == t ) 
	}

	# for each pass through the corpus
	for( i in 1:iterations ) 
	{
		# for each document
		for( d in 1:length(docs) )
		{
			# for each word
			for( w in 1:length( docs[[d]] ) )
			{
				t0  <- ta[[d]][w]
				wid <- docs[[d]][w]
				
				dt[ d, t0 ]   <- dt[ d, t0 ] - 1
				wt[ t0, wid ] <- wt[ t0, wid ] - 1 
				
				left  <- ( wt[ , wid ] + eta ) / ( rowSums(wt) + length(vocab) * eta )
				right <- ( dt[ d, ] + alpha ) / ( sum( dt[ d, ] ) + K * alpha )

				t1 <- sample( 1:K, 1, prob = left * right )
				
				# update topic assignment list with newly sampled topic for token w.	
				# and re-increment word-topic and document-topic count matrices with 
				# the new sampled topic for token w.
				ta[[d]][w] <- t1 
				dt[ d, t1 ]   <- dt[ d, t1 ] + 1  
				wt[ t1, wid ] <- wt[ t1, wid ] + 1
	    		
	    		# examine when topic assignments change
				# if( t0 != t1 ) 
				#	 print( paste0( "doc:", d, " token:" , w, " topic:", t0, "=>", t1 ) ) 
			}
		}
	}

	return( list( wt = wt, dt = dt ) )
}

