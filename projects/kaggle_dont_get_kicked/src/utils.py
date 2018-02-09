import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV


__all__ = ['clean', 'build_xgb', 'write_output']


# fit_params as a constructor argument was deprecated in version 0.19
# and will be removed in version 0.21, will worry about converting in the future
warnings.filterwarnings(
    action = 'ignore', module = 'sklearn',
    message = '.*fit_params.*', category = DeprecationWarning)


def clean(filepath, now, cat_cols, num_cols, date_cols, ids_col, label_col = None):
    """
    Clean the raw dataset, targeted for this specific problem. Details
    of the preprocessing steps are commented within the function

    Parameters
    ----------
    filepath : str
        Relative filepath of the data.

    now : str
        Date in the format of YYYY-MM-DD to compute the
        recency feature.

    cat_cols : list[str]
        Categorical features' column names.

    num_cols : list[str]
        Numeric features' column names.

    date_cols : list[str]
        Datetime features' column names.

    ids_col : str
        ID column name.

    label_col : str, default None
        Label column's name, None indicates that we're dealing with
        new data that does not have the label column.

    Returns
    -------
    data : DataFrame
        Cleaned data.
    """

    # information used when reading in the .csv file
    cat_dtypes = {col: 'category' for col in cat_cols}
    read_csv_info = {'dtype': cat_dtypes,
                     'parse_dates': date_cols,
                     'infer_datetime_format': True}
    use_cols = cat_cols + num_cols + date_cols + [ids_col]
    if label_col is not None:
        use_cols += [label_col]

    # original column name has a minor typo (Acquisiton -> Acquisition)
    rename_col = {'MMRAcquisitonRetailCleanPrice': 'MMRAcquisitionRetailCleanPrice'}
    data = (pd.
            read_csv(filepath, usecols = use_cols, **read_csv_info).
            dropna(axis = 0, how = 'any').
            rename(columns = rename_col))

    # ensure prices are greater than 0
    price_cols = ['AuctionAveragePrice', 'AuctionCleanPrice',
                  'RetailAveragePrice', 'RetailCleanPrice']
    for price_col in price_cols:
        for col in ['MMRCurrent', 'MMRAcquisition']:
            data = data[data[col + price_col] > 0]

    # VehBCost: acquisition cost paid for the vehicle at the time of purchase, we
    # will compute its ratio with the AuctionAveragePrice difference, that way this
    # number will be compared against a baseline
    # the denominator has been sanity check to be greater than 0 in previous step
    veh_cost_col = 'VehBCost'
    data['RatioVehBCost'] = (data[veh_cost_col] /
                             data['MMRAcquisitionAuctionAveragePrice'])
    data = data.drop(veh_cost_col, axis = 1)

    # transform columns into ratio (should be more indicative than the raw form)
    # compute the ratio (MRRCurrent - MRRAcquistion) / MRRAcquistion for the
    # four different price columns
    for price_col in price_cols:
        new = 'Diff' + price_col
        current = 'MMRCurrent' + price_col
        baseline = 'MMRAcquisition' + price_col
        data[new] = (data[current] - data[baseline]) / data[baseline]
        data = data.drop([current, baseline], axis = 1)

    # skewed column, log-transform to make it more normally distributed
    warranty_col = 'WarrantyCost'
    data[warranty_col] = np.log(data[warranty_col])

    # Transmission has three distinct types, but there's only 1 observation
    # for type "Manual", that record is simply dropped
    transmission_col = 'Transmission'
    data = data[data[transmission_col] != 'Manual']
    data[transmission_col] = data[transmission_col].cat.remove_unused_categories()

    # there's only 1 date column in the date_cols list,
    # use it to compute the recency
    date_col = date_cols[0]
    data[date_col] = (pd.Timestamp(now) - data[date_col]).dt.days
    return data


def build_xgb(n_iter, cv, random_state, eval_set):
    """
    Build a RandomSearchCV XGBoost model

    Parameters
    ----------
    n_iter : int
        Number of hyperparameters to try for RandomSearchCV.

    cv : int
        Number of cross validation for RandomSearchCV.

    random_state : int
        Seed used by the random number generator for random sampling
        the hyperpameter.

    eval_set : list of tuple
        List of (X, y) pairs to use as a validation set for
        XGBoost model's early-stopping.

    Returns
    -------
    xgb_tuned : sklearn's RandomSearchCV object
        Unfitted RandomSearchCV XGBoost model.
    """

    # for xgboost, set number of estimator to a large number
    # and the learning rate to be a small number, we'll simply
    # let early stopping decide when to stop training;
    xgb_params_fixed = {
        # setting it to a positive value
        # might help when class is extremely imbalanced
        # as it makes the update more conservative
        'max_delta_step': 1,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'n_jobs': -1}
    xgb = XGBClassifier(**xgb_params_fixed)

    # set up randomsearch hyperparameters:
    # subsample, colsample_bytree and max_depth are presumably the most
    # common way to control under/overfitting for tree-based models
    xgb_tuned_params = {
        'max_depth': randint(low = 3, high = 12),
        'colsample_bytree': uniform(loc = 0.8, scale = 0.2),
        'subsample': uniform(loc = 0.8, scale = 0.2)}

    xgb_fit_params = {
        'eval_metric': 'auc',
        'eval_set': eval_set,
        'early_stopping_rounds': 5,
        'verbose': False}

    # return_train_score = False
    # computing the scores on the training set can be computationally
    # expensive and is not strictly required to select the parameters
    # that yield the best generalization performance.
    xgb_tuned = RandomizedSearchCV(
        estimator = xgb,
        param_distributions = xgb_tuned_params,
        fit_params = xgb_fit_params,
        cv = cv,
        n_iter = n_iter,
        n_jobs = -1,
        verbose = 1,
        scoring = 'roc_auc',
        random_state = random_state,
        return_train_score = False)
    return xgb_tuned


def write_output(ids, ids_col, y_pred, label_col, output_path):
    """
    Output a DataFrame with the id columns and its predicted probability.

    Parameters
    ----------
    ids : 1d ndarray
        ID for each oberservation.

    ids_col : str
        ID column's name.

    y_pred : 1d ndarray
        Predicted probability for each oberservation.

    label_col : str
        Label column's name.

    output_path : str
        Relative path of the output file.
    """
    output = pd.DataFrame({
        ids_col: ids,
        label_col: y_pred
    }, columns = [ids_col, label_col])
    output.to_csv(output_path, index = False)
