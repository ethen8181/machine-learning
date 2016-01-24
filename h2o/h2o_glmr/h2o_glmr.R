
# https://www.youtube.com/watch?v=gEZtZRANeLc
library(h2o)
h2o.init( nthreads = -1 )

setwd("/Users/ethen/machine-learning/h2o/h2o_glmr")

df <- h2o.importFile( normalizePath("subject01_walk1.csv" ) )



df_glrm <- h2o.glrm(

	training_frame = df, 
	cols = 2:ncol(df), # skip the first column ( time index )
	k = 10, # rank
	loss = "Quadratic", 
	regularization_x = "None", 
	regularization_y = "None", 
	max_iterations = 1000 # it is an iterative training algorithm, so it usually converges faster
)

# plot to see the algorithm converged 
plot(df_glrm)



