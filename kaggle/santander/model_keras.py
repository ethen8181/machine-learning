from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import PReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import RandomizedSearchCV


def build_keras_base(hidden_layers = [64, 64, 64], dropout_rate = 0, 
                     l2_penalty = 0.1, optimizer = 'adam',
                     n_input = 100, n_class = 2):
    """
    Keras Multi-layer neural network. Fixed parameters include: 
    1. activation function (PRelu)
    2. always uses batch normalization after the activation
    3. use adam as the optimizer
    
    Parameters
    ----------
    Tunable parameters are (these are the ones that are commonly tuned)
    
    hidden_layers: list
        the number of hidden layers, and the size of each hidden layer

    dropout_rate: float 0 ~ 1
        if bigger than 0, there will be a dropout layer

    l2_penalty: float
        or so called l2 regularization

    optimizer: string or keras optimizer
        method to train the network

    Returns
    -------
    model : 
        a keras model

    Reference
    ---------
    https://keras.io/scikit-learn-api/
    """   
    model = Sequential()
    
    for index, layers in enumerate(hidden_layers):       
        if not index:
            # specify the input_dim to be the number of features for the first layer
            model.add( Dense( layers, input_dim = n_input,
                              W_regularizer = l2(l2_penalty) ) )
        else:
            model.add( Dense( layers, W_regularizer = l2(l2_penalty) ) )
    
        # insert BatchNorm layer immediately after fully connected layers
        # and before activation layer
        model.add( BatchNormalization() )
        model.add( PReLU() )
        if dropout_rate:
            model.add( Dropout(p = dropout_rate) )
    
    model.add( Dense(n_class) )
    model.add( Activation('softmax') )
    
    # the loss for binary and muti-class classification is different 
    loss = 'binary_crossentropy'
    if n_class > 2:
        loss = 'categorical_crossentropy'
    
    model.compile( loss = loss, optimizer = optimizer, metrics = ['accuracy'] )  
    return model


def build_model_keras(X_train, y_train, X_val, y_val):
    
    # one-hot encode the response variable
    y_train_encoded = to_categorical(y_train)
    y_val_encoded = to_categorical(y_val)
    n_input = X_train.shape[1]
    n_class = y_train_encoded.shape[1]
    
    # number of epochs is set to a large number, we'll let
    # early stopping terminate the training process;
    # batch size can be hyperparameters, but it is fixed
    model_keras = KerasClassifier(
        build_fn = build_keras_base,
        n_input = n_input,
        n_class = n_class,
    )

    # random search's parameter:
    dropout_rate_opts  = [0, 0.2, 0.5]
    hidden_layers_opts = [ [64, 64, 64, 64], [32, 32, 32, 32, 32], [100, 100, 100] ]
    l2_penalty_opts = [0.01, 0.1, 0.5]
    keras_params_opt = {
        'hidden_layers': hidden_layers_opts,
        'dropout_rate': dropout_rate_opts,  
        'l2_penalty': l2_penalty_opts
    }

    # set up kera's early stopping
    # and other extra parameters pass to the .fit
    callbacks = [ EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 0) ]
    keras_fit_params = {   
        'callbacks': callbacks,
        'nb_epoch': 200,
        'batch_size': 2048,
        'validation_data': (X_val, y_val_encoded),
        'verbose': 0
    }

    # `verbose` 2 will print the info for every cross validation, 
    # kind of too much
    rs_keras = RandomizedSearchCV( 
        model_keras, 
        param_distributions = keras_params_opt,
        fit_params = keras_fit_params,
        n_iter = 3, 
        cv = 5,
        n_jobs = -1,
        verbose = 1
    )
    rs_keras.fit(X_train, y_train_encoded)

    print( 'Best score obtained: {0}'.format(rs_keras.best_score_) )
    print('Parameters:')
    for param, value in rs_keras.best_params_.items():
        print( '\t{}: {}'.format(param, value) )
    
    return rs_keras


if __name__ == '__main__':
    rs_keras = build_model_keras(X_train, y_train, X_val, y_val)

