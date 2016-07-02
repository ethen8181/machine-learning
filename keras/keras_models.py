from keras.utils import np_utils
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Sequential
from sklearn.grid_search import RandomizedSearchCV
from keras.layers.advanced_activations import PReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

# make sure you get the number of feature and class correct
# and also one-hot encode the class
feature_num = X.shape[1]
class_num = np.unique(y).shape[0]
y_encode = np_utils.to_categorical( y, class_num )


def create_model( hidden_layers = [ 64, 64, 64 ], dropout_rate = 0, 
				  l2_penalty = 0.1, optimizer = 'adam' ):
	"""
	Fixed parameters in include the activation function and
	it will always uses batch normalization after the activation.
	note that feature_num and class_num are global variables that
	are not defined inside the function
	
	Parameters
	----------
	Tunable parameters are (commonly tuned)
	
	hidden_layers: list
		the number of hidden layers, and the size of each hidden layer
    
	dropout_rate: float 0 ~ 1
		if bigger than 0, there will be a dropout layer
    
	l2_penalty: float
		or so called l2 regularization
    
	optimizer: string or keras optimizer
		method to train the network
	"""   
	model = Sequential()
	
	for index, layers in enumerate(hidden_layers):       
		if not index:
			# specify the input_dim to be the number of features for the first layer
			model.add( Dense( layers, input_dim = feature_num, W_regularizer = l2(l2_penalty) ) )
		else:
			model.add( Dense( layers, W_regularizer = l2(l2_penalty) ) )
        
		model.add( PReLU() )
		model.add( BatchNormalization() )
		if dropout_rate:
			model.add( Dropout( p = dropout_rate ) )
    
	model.add( Dense(class_num) )
	model.add( Activation('softmax') )
	
	# the loss for binary and muti-class classification is different 
	loss = 'binary_crossentropy'
	if class_num > 2:
		loss = 'categorical_crossentropy'
    
	model.compile( loss = loss, optimizer = optimizer, metrics = ['accuracy'] )  
	return model



# create model, note that verbose is turned off here
model = KerasClassifier( 
	build_fn = create_model, 
	nb_epoch = 15, 
	batch_size = 64, 
	verbose = 0
)

# specify the options and store them inside the dictionary
sgd = SGD( lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True )
optimizer_opts = [ 'adam', sgd ]
dropout_rate_opts  = [ 0, 0.2, 0.5 ]
hidden_layers_opts = [ [ 64, 64, 64 ], [ 128, 32, 32, 32, 32 ] ]
l2_penalty_opts = [ 0.01, 0.1, 0.5 ]

param_dict = {
	'hidden_layers': hidden_layers_opts,
	'dropout_rate': dropout_rate_opts,  
	'l2_penalty': l2_penalty_opts
	'optimizer': optimizer_opts
}

# 1. note that for randomized search, the parameter to pass the the dictionary that
# holds the possible parameter value is `param_distributions`
# 2. `verbose` 2 will print the class info for every cross validation, kind
# of too much
keras_cv = RandomizedSearchCV( 
    estimator = model, 
    param_distributions = param_dict, 
    n_iter = 4, 
    cv = 5,
    verbose = 1 
)
keras_cv.fit( X, y_encode )

