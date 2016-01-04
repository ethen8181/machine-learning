# kmeans++, generating a better k initial random center for kmeans.

# workflow for the kmeans++ algorithm
# 1. choose a data point at random from the dataset, this serves as the first center point. 
# 2. compute the distance of all other data points to the randomly chosen center point.
# 3. to generate the second center point, each data point is chosen with the prob (weight) of 
#    its distance to the chosen center divided by the the total distance ( in R, sample function's
#    probability are already weighted, do not need to add up to one ).
# 4. next recompute the weight of each data point as the minimum of the distance between it and
#    all the centers that are already generated ( e.g. for the second iteration, compare the 
#    distance of the data point between the first and second center and choose the smaller one ).
# 5. repeat step 3 and 4 until having k centers. 


# test example data

# remove the species column
# iris_data <- iris[ , -5 ]

# normalize the dataset 
# iris_data <- sapply( iris_data, function(x) 
# {
#	( x - min(x) ) / ( max(x) - min(x) )
# })

# ---------------------------------------------------------------------------------
# [Kmeanspp] :
# @k = cluster number 
# returns the kmeans result 

Kmeanspp <- function( data, k, ... )
{
	if( is.matrix(data) )
		data <- data.frame(data)
	
	# used with bootstrapped data. Avoid duplicates, or kmeans will warn about 
	# identical cluster center
	dataset <- unique(data)	
		
	n <- nrow(dataset)	
	center_ids <- sample.int( n, 1 )

	for( i in 1:( k-1 ) )
	{		
		dists <- apply( dataset[ center_ids[i], ], 1, function(center)
		{
			sqrt( rowSums( ( dataset - center )^2 ) )
		})
	   
		if( i == 1 )
		{
			distance <- dists

			# somehow distance between the same rows are not zero 
			# exclude the probability of the choosing the data point that are already chosen as centers
			dists[center_ids] <- 0

			center_ids[ i+1 ] <- sample.int( n, 1, prob = dists )

		}else
		{
			distance <- cbind( distance, dists )
			
			probs <- apply( distance, 1, min )			
			probs[center_ids] <- 0
			center_ids[ i+1 ] <- sample.int( n, 1, prob = probs )
		}					
	}

	# cluster the whole "data", using the center_ids generated from "dataset"
	results <- kmeans( data, centers = dataset[ center_ids, ], ... )
	return(results)	
}

# results <- kmeanspp( data.frame(iris_data) )

# results$size
# table( iris$Species, results$cluster )

# questionable results on the iris dataset


