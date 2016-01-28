# -----------------------------------------------------------------------------------------------
# anomaly detection used on kaggle give me some credit 
# http://amunategui.github.io/anomaly-detection-h2o/

anomaly_model <- h2o.deeplearning(

	x = input, 
	training_frame = train,
	autoencoder = TRUE,
	hidden = c( 50, 50, 50 ), 
	epochs = 50,
	l1 = 1e-4
)

anomaly <- h2o.anomaly( anomaly_model, train, per_feature = FALSE )
error <- quantile( anomaly$Reconstruction.MSE, 0.8 )

train1 <- train[ anomaly$Reconstruction.MSE < error, ]
train2 <- train[ anomaly$Reconstruction.MSE >= error, ]


model_train1 <- h2o.ensemble(
	
	x = input, 
	y = output, 
	training_frame = train1,
	model_id = "model_1",
	family = "binomial", 
	learner = learner, 
	metalearner = metalearner,
	cvControl = list( V = 10 )
)
h2o.save_ensemble( model_train1, path = paste0( getwd(), "/model_train1" ), force = TRUE )


model_train2 <- h2o.ensemble(
	
	x = input, 
	y = output, 
	training_frame = train2,
	model_id = "model_2",
	family = "binomial", 
	learner = learner, 
	metalearner = metalearner,
	cvControl = list( V = 10 )
)
h2o.save_ensemble( model_train2, path = paste0( getwd(), "/model_train2" ), force = TRUE )

anomaly_test <- h2o.anomaly( anomaly_model, test, per_feature = FALSE )
anomaly_dt <- as.data.table(anomaly_test$Reconstruction.MSE)

index1 <- which( anomaly_dt$Reconstruction.MSE < error )
index2 <- which( anomaly_dt$Reconstruction.MSE >= error )

test1 <- test[ index1, ]
test2 <- test[ index2, ]


pred1 <- predict( model_train1, test1 )
pred2 <- predict( model_train2, test2 )

submit1 <- as.data.table( pred1$pred[ , 3 ] )
submit1[ , Id := index1 ]
submit2 <- as.data.table( pred2$pred[ , 3 ] )
submit2[ , Id := index2 ]

submit <- rbind( submit1, submit2 )[ order(Id), ]
setnames( submit, c( "Probability", "Id" ) )
setcolorder( submit, c( "Id", "Probability" ) )
write.csv( submit, "finalsubmission.csv", row.names = FALSE )