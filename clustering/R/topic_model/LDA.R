# Latent Dirichlet Allocation
# conditioned on a dirichlet distribution 
# for two class = binomial distribution
# for K class = multinomial distribution
# the dirichlet distribution allows us model 
# the random selection from a multinomial distribution with K classes 
# For the symmetric distribution, a high alpha-value means that each document is 
# likely to contain a mixture of most of the topics, and not any single topic specifically

# ----------------------------------------------------------------------------------------
#								Prepare Example
# ----------------------------------------------------------------------------------------

# toy example 
rawdocs <- c(

	"eat turkey on turkey day holiday",
	"i like to eat cake on holiday",
	"turkey trot race on thanksgiving holiday",
	"snail race the turtle",
	"time travel space race",
	"movie on thanksgiving",
	"movie at air and space museum is cool movie",
	"aspiring movie star"
)
docs <- strsplit( rawdocs, split = " " )

# unique words
vocab <- unique( unlist(docs) )

# replace words in documents with wordIDs
for( i in 1:length(docs) )
	docs[[i]] <- match( docs[[i]], vocab )

# number of topics
K <- 2 

# initialize count matrices 
# @wt : word-topic matrix 
wt <- matrix( 0, K, length(vocab) )
colnames(wt) <- vocab

# @ta : topic assignment list
ta <- lapply( docs, function(x) rep( 0, length(x) ) ) 
names(ta) <- paste0( "doc", 1:length(docs) )

# @dt : counts correspond to the number of words assigned to each topic for each document
dt <- matrix( 0, length(docs), K )

set.seed(1234)
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

# the count of each word being assigned to each topic 
# topic assignment list
print(ta)
print(wt)
print(dt)


# ----------------------------------------------------------------------------------------
#								Gibbs sampling one iteration 
# ----------------------------------------------------------------------------------------

# hyperparameters
alpha <- 1
eta <- 1

# initial topics assigned to the first word of the first document
# and its corresponding word id 
t0  <- ta[[1]][1]
wid <- docs[[1]][1]

# z_-i means that we do not include token w in our word-topic and document-topic 
# count matrix when sampling for token w, 
# only leave the topic assignments of all other tokens for document 1
dt[ 1, t0 ]   <- dt[ 1, t0 ] - 1 
wt[ t0, wid ] <- wt[ t0, wid ] - 1

# Calculate left side and right side of equal sign
left  <- ( wt[ , wid ] + eta ) / ( rowSums(wt) + length(vocab) * eta )
right <- ( dt[ 1, ] + alpha ) / ( sum( dt[ 1, ] ) + K * alpha )

# draw new topic for the first word in the first document 
t1 <- sample( 1:K, 1, prob = left * right )
t1

# refresh the dt and wt with the newly assigned topic 
ta[[1]][1] <- t1 
dt[ 1, t1 ]   <- dt[ 1, t1 ] + 1  
wt[ t1, wid ] <- wt[ t1, wid ] + 1


# ----------------------------------------------------------------------------------------
#								Gibbs sampling ; topicmodels library 
# ----------------------------------------------------------------------------------------

# define parameters
K <- 2
alpha <- 1
eta <- .001
iterations <- 1000

source("/Users/ethen/machine-learning/lda_1/lda_1_functions.R")
set.seed(4321)
lda1 <- LDA1( docs = docs, vocab = vocab, 
			  K = K, alpha = alpha, eta = eta, iterations = iterations )


# posterior probability 
# topic probability of every word 
phi <- ( lda1$wt + eta ) / ( rowSums(lda1$wt) + length(vocab) * eta )

# topic probability of every document
theta <- ( lda1$dt + alpha ) / ( rowSums(lda1$dt) + K * alpha )

# topic assigned to each document, the one with the highest probability 
topic <- apply( theta, 1, which.max )

# possible words under each topic 
# sort the probability and obtain the user-specified number
Terms <- function( phi, n )
{
	term <- matrix( 0, n, K )
	for( p in 1:nrow(phi) )
		term[ , p ] <- names( sort( phi[ p, ], decreasing = TRUE )[1:n] )

	return(term)
}
term <- Terms( phi = phi, n = 3 )

list( original_text = rawdocs[ topic == 1 ], words = term[ , 1 ] )
list( original_text = rawdocs[ topic == 2 ], words = term[ , 2 ] )


# compare 
library(tm)
library(topicmodels)

# @burning : number of omitted Gibbs iterations at beginning
# @thin : number of omitted in-between Gibbs iterations
docs1 <- Corpus( VectorSource(rawdocs) )
dtm <- DocumentTermMatrix(docs1)
lda <- LDA( dtm, k = 2, method = "Gibbs", 
	   		control = list( seed = 1234, burnin = 500, thin = 100, iter = 4000 ) )

list( original_text = rawdocs[ topics(lda) == 1 ], words = terms( lda, 3 )[ , 1 ] )
list( original_text = rawdocs[ topics(lda) == 2 ], words = terms( lda, 3 )[ , 2 ] )

# ----------------------------------------------------------------------------------------
#										Reference
# ----------------------------------------------------------------------------------------

# why tagging matters
# http://cyber.law.harvard.edu/wg_home/uploads/507/07-WhyTaggingMatters.pdf

# math notations
# https://www.cl.cam.ac.uk/teaching/1213/L101/clark_lectures/lect7.pdf

# hyperparameters explanation 
# http://stats.stackexchange.com/questions/37405/natural-interpretation-for-lda-hyperparameters/37444#37444

# Reimplementation R code
# http://brooksandrew.github.io/simpleblog/articles/latent-dirichlet-allocation-under-the-hood/
