library(data.table)

Kmeanspp <- function( data, k, ... )
{
	# kmeans++, generating a better k initial random center for kmeans. Workflow:
	# 1. choose a data point at random from the dataset, this serves as the first center point. 
	# 2. compute the SQUARED distance of all other data points to the randomly chosen center point.
	# 3. to generate the next center point, each data point is chosen with the prob (weight) of 
	#    its squared distance to the chosen center of this round divided by the the 
	#    total squared distance (in R, sample function's probability are already weighted, 
	#    do not need to tune them to add up to one).
	# 4. next recompute the weight of each data point as the minimum of the distance between it and
	#    ALL the centers that are already generated ( e.g. for the second iteration, compare the 
	#    distance of the data point between the first and second center and choose the smaller one ).
	# 5. repeat step 3 and 4 until having k centers. 
	#
	# Parameters
	# ----------
	# data : data.frame, data.table, matrix data
	#
	# k : int 
	#     number of clusters
	# 
	# ... : 
	#     all other parameters that can be passed into R's kmeans except for the data and center
	#     , see ?kmeans for more detail
	#
	# Returns
	# -------
	# result : list
	#     R's kmeans original output
	#
	# Reference
	# ---------
	# https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

	if( !is.data.table(data) )
		data <- data.table(data)
	
	# used with bootstrapped data. so unique the data
	# to avoid duplicates, or kmeans will warn about 
	# identical cluster center
	unique_data <- unique(data)

	# generate the first center randomly
	n <- nrow(unique_data)
	center_ids <- integer(k)
	center_ids[1] <- sample.int( n, 1 )

	for( i in 1:( k - 1 ) ){		
		
		# calculate the squared distance between the center and 
		# all the data points
		center <- unique_data[ center_ids[i], ]
		dists <- apply( unique_data, 1, function(datapoint){
			sum( ( datapoint - center )^2 )
		})

		# sample the next center using the squared distance as the weighted probability,
		# starting from the second center, the measure "squared distance" for each data point
		# is the min distance between each data point and each center that has already been
		# generated
		if( i == 1 ){		
			distance <- dists
		}else{
			distance <- cbind( distance, dists )
			distance <- apply( distance, 1, min )
		}
		center_ids[ i + 1 ] <- sample.int( n, 1, prob = distance )					
	}

	# cluster the whole "data", using the center_ids generated using kmeanspp
	results <- kmeans( data, centers = unique_data[ center_ids, ], ... )
	return(results)	
}


test <- function(){
	# test example data
	# the example code is wrapped in the string below
	"
	# remove the species column
	iris_data <- iris[ , -5 ]

	# normalize the dataset
	iris_data <- data.table( scale(iris_data) )
	results <- Kmeanspp( data = iris_data, k = 3 )

	# example output, the generated center, size of each cluster
	# and confusion matrix of the original cluster and clustered result
	results$center
	results$size
	table( iris$Species, results$cluster )

	iris_data[ , `:=`( Species = iris$Species, cluster = results$cluster ) ]
	split( iris_data, iris_data$cluster )
	"
	print('testing')
}



