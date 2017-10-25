import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from collections import defaultdict
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


__all__ = ['clean', 'build_xgb', 'write_output', 'Preprocesser']

# sklearn's LinearRegression may give harmless errors
# https://github.com/scipy/scipy/issues/5998
warnings.filterwarnings(
    action = 'ignore', module = 'scipy', message = '^internal gelsd')

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
        Relative filepath of the data

    now : str
        Date in the format of YYYY-MM-DD to compute the
        recency feature

    cat_cols : list[str]
        Categorical features' column names

    num_cols : list[str]
        Numeric features' column names

    date_cols : list[str]
        Datetime features' column names

    ids_col : str
        ID column name

    label_col : str, default None
        Label column's name, None indicates that we're dealing with
        new data that does not have the label column

    Returns
    -------
    data : DataFrame
        Cleaned data
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
    data = data[data['Transmission'] != 'Manual']

    # there's only 1 date column in the date_cols list,
    # use it to compute the recency
    date_col = date_cols[0]
    data[date_col] = (pd.Timestamp(now) - data[date_col]).dt.days
    return data


def build_xgb(n_iter, cv, eval_set):
    """
    Build a RandomSearchCV XGBoost model

    Parameters
    ----------
    n_iter : int
        Number of hyperparameters to try for RandomSearchCV

    cv : int
        Number of cross validation for RandomSearchCV

    eval_set : list of tuple
        List of (X, y) pairs to use as a validation set for
        XGBoost model's early-stopping

    Returns
    -------
    xgb_tuned : sklearn's RandomSearchCV object
        Unfitted RandomSearchCV XGBoost model
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
        return_train_score = False)
    return xgb_tuned


def write_output(ids, ids_col, y_pred, label_col, output_path):
    """
    Output a DataFrame with the id columns and its predicted probability

    Parameters
    ----------
    ids : 1d ndarray
        ID for each oberservation

    ids_col : str
        ID column's name

    y_pred : 1d ndarray
        Predicted probability for each oberservation

    label_col : str
        Label column's name

    output_path : str
        Relative path of the output file
    """
    output = pd.DataFrame({
        ids_col: ids,
        label_col: y_pred
    }, columns = [ids_col, label_col])
    output.to_csv(output_path, index = False)


class Preprocesser(BaseEstimator, TransformerMixin):
    """
    Generic data preprocessing including:
    - standardize numeric columns and remove potential
    multi-collinearity using variance inflation factor
    - one-hot encode categorical columns

    Parameters
    ----------
    num_cols : list[str], default None
        Numeric columns' name. default None means
        the input column has no numeric features.

    cat_cols : list[str], default None
        Categorical columns' name.

    threshold : float, default 5.0
        Threshold for variance inflation factor (vif).
        If there are numerical columns, identify potential multi-collinearity
        between them using vif. Conventionally, a vif score larger than 5
        should be removed.

    Attributes
    ----------
    colnames_ : str 1d ndarray
        Column name of the transformed numpy array.

    num_cols_ : str 1d ndarray or None
        Final numeric column after removing potential multi-collinearity,
        if there're no numeric input features then the value will be None.

    label_encode_dict_ : defauldict of sklearn's LabelEncoder object
        LabelEncoder that was used to encode the value
        of the categorical columns into with value between
        0 and n_classes-1. Categorical columns will go through
        this encoding process before being one-hot encoded.

    cat_encode_ : sklearn's OneHotEncoder object
        OneHotEncoder that was used to one-hot encode the
        categorical columns.

    scaler_ : sklearn's StandardScaler object
        StandardScaler that was used to standardize the numeric columns.
    """

    def __init__(self, num_cols = None, cat_cols = None, threshold = 5.0):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.threshold = threshold

    def fit(self, data, y = None):
        """
        Fit the Preprocess Transformer on the input data.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input data

        y : default None
            Ignore, argument required for constructing sklearn Pipeline

        Returns
        -------
        self
        """
        if self.num_cols is None and self.cat_cols is None:
            raise ValueError("There must be a least one input feature column")

        # Label encoding across multiple columns in scikit-learn
        # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
        if self.cat_cols is not None:
            self.label_encode_dict_ = defaultdict(LabelEncoder)
            label_encoded = (data[self.cat_cols].
                             apply(lambda x: self.label_encode_dict_[x.name].fit_transform(x)))

            self.cat_encode_ = OneHotEncoder(sparse = False)
            self.cat_encode_.fit(label_encoded)

        if self.num_cols is not None:
            self.scaler_ = StandardScaler()
            scaled = self.scaler_.fit_transform(data[self.num_cols])
            colnames = self._remove_collinearity(scaled)
            self.num_cols_ = np.array(colnames)
        else:
            colnames = []
            self.num_cols_ = None

        # store the column names (numeric columns comes before the
        # categorical columns) so we can refer to them later
        if self.cat_cols is not None:
            for col in self.cat_cols:
                cat_colnames = [col + '_' + str(classes)
                                for classes in self.label_encode_dict_[col].classes_]
                colnames += cat_colnames

        self.colnames_ = np.asarray(colnames)
        return self

    def _remove_collinearity(self, scaled):
        """
        Identify multi-collinearity between the numeric variables
        using variance inflation factor (vif)
        """
        colnames = self.num_cols.copy()
        while True:
            vif = [self._compute_vif(scaled, index)
                   for index in range(scaled.shape[1])]
            max_index = np.argmax(vif)

            if vif[max_index] >= self.threshold:
                removed = colnames[max_index]
                colnames.remove(removed)
                scaled = np.delete(scaled, max_index, axis = 1)
                self.scaler_.mean_ = np.delete(self.scaler_.mean_, max_index)
                self.scaler_.scale_ = np.delete(self.scaler_.scale_, max_index)
            else:
                break

        return colnames

    def _compute_vif(self, X, target_index):
        """
        Similar implementation as statsmodel's variance_inflation_factor
        with some enhancemants:
        1. includes the intercept by default
        2. prevents float division errors (dividing by 0)

        References
        ----------
        http://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
        """
        n_features = X.shape[1]
        X_target = X[:, target_index]
        mask = np.arange(n_features) != target_index
        X_not_target = X[:, mask]

        linear = LinearRegression()
        linear.fit(X_not_target, X_target)
        rsquared = linear.score(X_not_target, X_target)
        vif = 1. / (1. - rsquared + 1e-5)
        return vif

    def transform(self, data):
        """
        Transform the input data using Preprocess Transformer.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input data

        Returns
        -------
        X : 2d ndarray, shape [n_samples, n_features]
            Transformed input data
        """
        if self.cat_cols is not None:
            label_encoded = (data[self.cat_cols].
                             apply(lambda x: self.label_encode_dict_[x.name].transform(x)))
            cat_encoded = self.cat_encode_.transform(label_encoded)

        if self.num_cols is not None:
            scaled = self.scaler_.transform(data[self.num_cols_])

        # combine encoded categorical columns and scaled numerical
        # columns, it's the same as concatenate it along axis 1
        if self.cat_cols is not None and self.num_cols is not None:
            X = np.hstack((scaled, cat_encoded))
        elif self.num_cols is None:
            X = cat_encoded
        else:
            X = scaled

        return X
