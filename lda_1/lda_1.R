# Latent Dirichlet Allocation

# https://sumidiot.wordpress.com/2012/06/13/lda-from-scratch/
# http://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/

# http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf

# https://www.cl.cam.ac.uk/teaching/1213/L101/clark_lectures/lect7.pdf
# we want to find themes (or topics) in documents
# and we need a approach that automatically teases out topics
# note that we do not know the topics beforehand, thus this 
# will essentially become a clustering problem, where we're clustering
# both the word and the document.

# the key assumptions behind lda
# each given documents exhibit multiple topics. 
# a topic is a distibution over a fixed vocabulary.
# the number of topics has to be specified a-priori

# ----------------------------------------------------------------------------------------
#										Reference
# ----------------------------------------------------------------------------------------

# why tagging matters
# http://cyber.law.harvard.edu/wg_home/uploads/507/07-WhyTaggingMatters.pdf

# ----------------------------------------------------------------------------------------
#										Reimplementation R code
# http://brooksandrew.github.io/simpleblog/articles/latent-dirichlet-allocation-under-the-hood/
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
	{	  
		dt[ d, t ] <- sum( ta[[d]] == t ) 
	}
}

# the count of each word being assigned to each topic 
# topic assignment list
print(ta)
print(wt)
print(dt)


# ----------------------------------------------------------------------------------------
#										Gibbs sampling one iteration 
# ----------------------------------------------------------------------------------------

alpha <- 1 # hyperparameter. single value indicates symmetric dirichlet prior. higher=>scatters document clusters
eta <- .001 # hyperparameter
iterations <- 3 # iterations for collapsed gibbs sampling.  This should be a lot higher than 3 in practice.


# initial topics assigned to the first word of the first document
# and its corresponding word id 
t0  <- ta[[1]][3]
wid <- docs[[1]][3]

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

ta[[1]][1] <- t1 
dt[ 1, t1 ]   <- dt[ 1, t1 ] + 1  
wt[ t1, wid ] <- wt[ t1, wid ] + 1

# ----------------------------------------------------------------------------------------
#										Gibbs sampling iteration 
# ----------------------------------------------------------------------------------------

# define parameters
K <- 2 
alpha <- 1 
eta <- .001 
iterations <- 100

source("/Users/ethen/machine-learning/lda_1/lda_1_functions.R")
lda1 <- LDA1( docs = docs, vocab = vocab, 
			  K = K, alpha = 1, eta = .001, iterations = iterations )







