from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

def build_model_xgb(X_train, y_train, X_val, y_val):
    # should change the objective; scoring metric
    # the number of cross validation to perform
    # and number of iterations to perform the random search

    # xgboost base parameter:
    xgb_params_fixed = {
        # use 'multi:softprob' for multi-class problems
        # and 'binary:logistic' for binary classification
        'objective': 'binary:logistic',
        
        # setting it to a positive value 
        # might help when class is extremely imbalanced
        # as it makes the update more conservative
        'max_delta_step': 1,
            
        # use all possible cores for training
        'nthread': -1,
        
        # set number of estimator to a large number
        # and the learning rate to be a small number,
        # we'll let early stopping decide when to stop
        'n_estimators': 300,
        'learning_rate': 0.1,
    }
    model_xgb = XGBClassifier(**xgb_params_fixed)

    # random search's parameter:
    # scikit-learn's random search works with distributions; 
    # but it must provide a rvs method for sampling values from it,
    # such as those from scipy.stats.distributions
    # randint: discrete random variables ranging from low to high
    # uniform: uniform continuous random variable between loc and loc + scale
    xgb_params_opt = {
        'max_depth': randint(low = 3, high = 15),
        'colsample_bytree': uniform(loc = 0.7, scale = 0.3),
        'subsample': uniform(loc = 0.7, scale = 0.3) 
    }

    # xgboost's early stopping and other extra parameters
    # pass to the .fit
    eval_set = [ (X_train, y_train), (X_val, y_val) ]
    xgb_fit_params = {   
        'eval_metric': 'auc', 
        'eval_set': eval_set,
        'early_stopping_rounds': 5,
        'verbose': False
    }

    rs_xgb = RandomizedSearchCV(
        estimator = model_xgb,
        param_distributions = xgb_params_opt,
        fit_params = xgb_fit_params,
        scoring = 'roc_auc',
        cv = 10,   
        
        # number of parameter settings that are sampled
        n_iter = 10,
        n_jobs = -1,
        verbose = 1
    )
    rs_xgb.fit(X_train, y_train)
    
    print( 'Best score obtained: {0}'.format(rs_xgb.best_score_) )
    print('Parameters:')
    for param, value in rs_xgb.best_params_.items():
        print( '\t{}: {}'.format(param, value) )
    
    return rs_xgb


if __name__ == '__main__':
    rs_xgb = build_model_xgb(X_train, y_train, X_val, y_val)


