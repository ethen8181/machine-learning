# Functions for evaluating clustering  

# test example data for all the functions, remove row.names or else
# will produce warnings when using it on bootstrapping (complaining about 
# duplicated row.names )

# mtcars_scaled <- scale(mtcars)
# row.names(mtcars_scaled) <- NULL

# ------------------------------------------------------------------------------------
#### Choosing the right k for clustering
library(dplyr)
library(tidyr)
library(ggplot2)

## method 1. WSS :compute the total within sum square error, this measures how close
#  are the points in a cluster to each other 

# [Distance] : calculates the sum squared distance of a given cluster of points,
#              note that "sum squared distance" is used here for measuring variance 
Distance <- function(cluster)
{
	# the center of the cluster, mean of all the points
	center <- colMeans(cluster)
	
	# calculate the summed squared error between every point and 
	# the center of that cluster 
	distance <- apply( cluster, 1, function(row)
	{
		sum( ( row - center )^2 )
	}) %>% sum()

	return(distance)
}

# calculate the within sum squared error manually for hierarchical clustering 
# [WSS] : pass in the dataset, and the resulting groups(cluster)
WSS <- function( data, groups )
{
	k <- max(groups)

	# loop through each groups (clusters) and obtain its 
	# within sum squared error 
	total <- lapply( 1:k, function(k)
	{
		# extract the data point within the cluster
		cluster <- subset( data, groups == k )

		distance <- Distance(cluster)
		return(distance)
	}) %>% unlist()

	return( sum(total) )
}

# testing 
# sum_squared_error <- WSS( data = mtcars_scaled, groups =  groups )

# this value will will decrease as the number of clusters increases, 
# because each cluster will be smaller and tighter.
# And the rate of the decrease will slow down after the optimal cluster number


## method 2 : Calinski-Harabasz index, ratio of the between cluster variance
#			  to the total within cluster variance
# http://www.mathworks.com/help/stats/clustering.evaluation.calinskiharabaszevaluation-class.html 

# TSS (total sum of square) : the squared distance of all the data points from 
# the dataset's centroid 

# BSS (between sum of square) = TSS - WSS, measures how far apart are the clusters
# from each other 
# !! a good clustering has a small WSS and a high BSS

# CHIndex = B / W, the ratio should be maximized at the optimal k
# B = BSS(k) / (k-1) ; k = # of cluster
# W = WSS(k) / (n-k) ; n = # of data points

# [CHCriterion] : calculates both Calinski-Harabasz index and within sum squared error
# @kmax          = maximum cluster number, caculates the CH index from 2 cluster to kmax
# @clustermethod = "kmeanspp", "hclust"

CHCriterion <- function( data, kmax, clustermethod, ...  )
{
	if( !clustermethod %in% c( "kmeanspp", "hclust" ) )
		stop( "method must be one of 'kmeanspp' or 'hclust'" )

	# total sum squared error (independent with the number of cluster k)
	tss <- Distance( cluster = data )

	# initialize a numeric vector storing the score
	wss <- numeric(kmax)

	# k starts from 2, cluster 1 is meaningless
	if( clustermethod == "kmeanspp" )
	{
		for( k in 2:kmax )
		{
			results <- Kmeanspp( data, k, ... )
			wss[k]  <- results$tot.withinss 
		}		
	}else # "hclust"
	{
		d <- dist( data, method = "euclidean" )
		clustering <- hclust( d, ... )
		for( k in 2:kmax )
		{
			groups <- cutree( clustering, k )
			wss[k] <- WSS( data = data, groups =  groups )
		}
	}		
	
	# between sum of square
	bss <- tss - wss[-1]

	# cluster count start from 2! 
	numerator <- bss / ( 1:(kmax-1) )
	denominator <- wss[-1] / ( nrow(data) - 2:kmax )

	criteria <- data.frame( k = 2:kmax,
	                        CHIndex = numerator / denominator,
							wss = wss[-1] )

	# convert to long format for plotting 
	criteria_long <- gather( criteria, "index", "value", -1 )

	plot <- ggplot( criteria_long, aes( k, value, color = index ) ) + 
			geom_line() + geom_point( aes( shape = index ), size = 3 ) +
			facet_wrap( ~ index, scale = "free_y" ) + 
			guides( color = FALSE, shape = FALSE )

	return( list( data = criteria, 
				  plot = plot ) )
}


# testing 
# criteria <- CHCriterion( data = mtcars_scaled, kmax = 10, 
#	                       clustermethod = "hclust", method = "ward.D" )

# criteria <- CHCriterion( data = mtcars_scaled, kmax = 10, 
#	                       clustermethod = "kmeanspp", nstart = 10, iter.max = 100 )


# for CHindex, the local maximum should be your ideal cluster number
# for WWS, the "elbow" center should be your ideal cluster number

# for the mtcars_scaled example, CHindex shows local optimal at cluster 4
# seems reasonable when looking at the dendogram 

# test <- hclust( dist(mtcars_scaled), method = "ward.D" )
# plot(test)
# rect.hclust( test, k = 4 )


# ----------------------------------------------------------------------------------------------
#### boostrap evaluation of a cluster result 
# clustering algorithms will often produce several clusters that represents 
# actual cluster of the data, and then one or two clusters that represents "others".
# which means that they're made up of data points that have no relationship with each other
# they just don't fit anywhere else.

# use boostrap resampling to evaluate the stability of the cluster, steps :
# 1. Cluster the original data.
# 2. Draw a new dataset of the same size as the original by resampling the original
#    dataset with replacement, therefore some data point may show up more than once
#    while others not at all. Cluster this new data.
# 3. For every cluster in the original cluster, find the most similar cluster in the 
#    new clustering, which is the one with the max Jaccard Similarity. Then if this 
#    Jaccard Similarity is less than .5 than the original cluster is considered to 
#    be "dissolved", that is it did not show up in the new cluster. A cluster that
#    dissolved too often is most likely not a real cluster.
#    Jaccard Similarity of two vectors = intersect / union


# ----------------------------------------------------------------------------------------------
# [ClusterMethod] : supports heirarchical clustering and kmeans++ bootstrap

# kmeans++ explanation and source code
# source in function Kmeanspp
source("/Users/ethen/machine-learning/clustering_old/clustering/kmeanspp.R")

# @data          = data frame type data, matrix also works 
# @k             = specify the number of clusters
# @clustermethod = "hclust" for heirarchical clustering, and "kmeanspp" for kmeans++
# @noise.cut     = if specified, the points of the resulting cluster whose number is smaller
#                  than it will be considered as noise, and all of these noise cluster will be
#                  grouped together as one whole cluster
# @...           = pass in other parameters for hclust or kmeans++ (same as kmeans)

ClusterMethod <- function( data, k, noise.cut = 0, clustermethod, ... )
{
	if( !clustermethod %in% c( "kmeanspp", "hclust" ) )
		stop( "method must be one of 'kmeanspp' or 'hclust'" )

	# hierarchical clustering 
	if( clustermethod == "hclust" )
	{
		cluster   <- hclust( dist(data), ... )
		partition <- cutree( cluster, k )

	}else # kmeanspp
	{
		cluster   <- Kmeanspp( data = data, k = k, ... )
		partition <- cluster$cluster
	}	

	# equivalent to k
	cluster_num <- max(partition) 

	# calculate each cluster's size 
	cluster_size <- numeric(cluster_num)
	for( i in 1:cluster_num )
		cluster_size[i] <- sum( partition == i )

	# if there're cluster size smaller than the specified noise.cut, do :
	not_noise_num <- sum( cluster_size > noise.cut )

	if( cluster_num > not_noise_num )
	{
		# extract the cluster whose size is larger than noise.cut
		cluster_new <- (1:cluster_num)[ cluster_size > noise.cut ]
		
		# all the data points whose original cluster is smaller than the noise.cut
		# will be assigned to the same new cluster
		cluster_num <- not_noise_num + 1

		# new clustering number, assign the noise cluster's number first
		# then adjust the original cluster's number
		new <- rep( cluster_num, nrow(data) )

		for( i in 1:not_noise_num )
			new[ ( partition == cluster_new[i] ) ] <- i

		partition <- new
	}
	
	# boolean vector indicating which data point belongs to which cluster
	cluster_list <- lapply( 1:cluster_num, function(x)
	{
		return( partition == x )
	})

	cluster_result <- list( result      = cluster,	                        
	                        partition   = partition,
	                        clusternum  = cluster_num,
	                        clusterlist = cluster_list )
	return(cluster_result)
}

# cluster_result <- ClusterMethod( data = mtcars_scaled, k = 5, clustermethod = "hclust" )


# ----------------------------------------------------------------------------------------------
# [ClusterBootstrap] : 
# @bootstrap : number of boostrap iteraion
# @dissolve  : if the jaccard similarity is smaller than this number, then it is considered
#              to be "dissolved"
# Returns    : 1. result        : the original clustering object.
#              2. bootmean      : mean of the Jaccard Similarity for specified bootstrap time.
#              3. partition     : the original clustering result, a vector specifying which 
#                                 group does the data point belong.
#              4. clusternum    : final cluster count, 
#                                 if you specified noise.cut then it might be different from k.
#              5. bootdissolved : number of times each cluster's jaccard similarity is smaller than
#                                 the dissolve value.

ClusterBootstrap <- function( data, k, noise.cut = 0, bootstrap = 100, 
	                          dissolve = .5, clustermethod, ... )
{
	# step 1
	cluster_result <- ClusterMethod( data = data, k = k, noise.cut = noise.cut, 
		                             clustermethod = clustermethod, ... )

	cluster_num  <- cluster_result$clusternum
	boot_jaccard <- matrix( 0, nrow = bootstrap, ncol = cluster_num )
	
	# pass in two vectors containing TRUE and FALSE
	# ( do not use built in intersect or union ! )
	jaccardSimilarity <- function( x, y )
	{
		jaccard <- sum( x & y ) / ( sum(x) + sum(y) - sum( x & y ) )
		return(jaccard)
	}

	n <- nrow(data)
	for( i in 1:bootstrap )
	{
		# step 2, cluster the new sampled data 
		sampling  <- sample( n, n, replace = TRUE )
		boot_data <- data[ sampling, ]

		boot_result <- ClusterMethod( data = boot_data, k = k, noise.cut = noise.cut, 
			                          clustermethod = clustermethod, ... )
		boot_num <- boot_result$clusternum

		# step 3
		for( j in 1:cluster_num )
		{
			# compare the original cluster with every other bootstrapped cluster
			similarity <- lapply( 1:boot_num, function(k)
			{
				jaccard <- jaccardSimilarity( x = cluster_result$clusterlist[[j]][sampling],
				                              y = boot_result$clusterlist[[k]] )
			}) %>% unlist()

			# return the largest jaccard similarity
			boot_jaccard[ i, j ] <- max(similarity)
		}	
	}

	# cluster's stability, mean of all the boostrapped jaccard similarity 
	boot_mean <- colMeans(boot_jaccard)

	# how many times are each cluster's jaccard similarity below the 
	# specified "dissolved" value  
	boot_dissolved <- apply( boot_jaccard, 2, function(x)
	{
		sum( x < dissolve, na.rm = TRUE )
	})

	boot_result <- list( result        = cluster_result$result,
	                     bootmean      = boot_mean,
	                     partition     = cluster_result$partition,
	                     clusternum    = cluster_num,                     
	                     bootdissolved = boot_dissolved )
	return(boot_result)
}

# test
# set.seed(1234)
# boot_clust <- ClusterBootstrap( data = mtcars_scaled, k = 4, clustermethod = "kmeanspp",
#                                 nstart = 10, iter.max = 100 )

# ... parameters for hclust
# method = "ward.D"

# boot_clust

# rule of thumb, values below 0.6 should be considered unstable 
# boot_clust$bootmean

# clusters that have a low bootmean or high bootdissolved
# has the characteristics of what we’ve been calling the “other” cluster.


