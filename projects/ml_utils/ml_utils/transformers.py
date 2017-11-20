import warnings
import numpy as np
import numpy.ma as ma
from collections import defaultdict
from scipy.stats import mode, boxcox
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


__all__ = [
    'BoxCoxTransformer',
    'MultipleImputer',
    'ColumnExtractor',
    'Preprocesser']


# sklearn's LinearRegression may give harmless errors
# https://github.com/scipy/scipy/issues/5998
warnings.filterwarnings(
    action = 'ignore', module = 'scipy', message = '^internal gelsd')


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    BoxCox transformation on individual features. It wil be applied on
    each feature (each column of the data matrix) with lambda evaluated
    to maximise the log-likelihood.

    Parameters
    ----------
    transformed_cols : str 1d ndarray/list or "all"
        Specify what features are to be transformed:
            - "all" (default) : All features are to be transformed.
            - array of str : Array of feature names to be transformed.

    eps : float, default 1e-8
        An epsilon value can be added to the data before estimating
        the boxcox transformation.

    copy : bool, default True
        Set to False to perform inplace transformation.

    Attributes
    ----------
    transformed_cols_ : str 1d ndarray
        Names of the features to be transformed.

    lmbdas_ : float 1d ndarray [n_transformed_cols]
        The parameters of the BoxCox transform for the selected features.
        Elements in the collection corresponds to the transformed_cols_.

    n_features_ : int
        Number of features inputted during fit.

    Notes
    -----
    The Box-Cox transform is given by:
        y = (x ** lambda - 1.) / lmbda,  for lambda > 0
            log(x),                      for lambda = 0
    ``boxcox`` requires the input data to be positive.
    """

    def __init__(self, transformed_cols, eps = 1e-8, copy = True):
        self.eps = eps
        self.copy = copy
        self.transformed_cols = transformed_cols

    def fit(self, data, y = None):
        """
        Fit BoxCoxTransformer to the data.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_feature]
            Input data.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        self
        """
        self.n_features_ = data.shape[1]
        if self.transformed_cols == 'all':
            transformed_cols = data.columns.values
        else:
            # use np.copy instead of the .copy method
            # ensures it won't break if the user inputted
            # transformed_cols as a list
            transformed_cols = np.copy(self.transformed_cols)

        if np.any(data[transformed_cols] + self.eps <= 0):
            raise ValueError('BoxCox transform can only be applied on positive data')

        # TODO :
        # an embarrassingly parallelized problem, i.e.
        # each features' boxcox lambda can be estimated in parallel
        # needs to investigate if it's a bottleneck
        self.lmbdas_ = np.asarray([self._boxcox(data[i].values)
                                   for i in transformed_cols])
        self.transformed_cols_ = transformed_cols
        return self

    def _boxcox(self, x, lmbda = None):
        """Utilize scipy's boxcox transformation"""
        mask = np.isnan(x)
        x_valid = x[~mask] + self.eps
        if lmbda is None:
            _, lmbda = boxcox(x_valid, lmbda)
            return lmbda
        else:
            x[~mask] = boxcox(x_valid, lmbda)
            return x

    def transform(self, data):
        """
        Transform the data using BoxCoxTransformer.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input data.

        Returns
        -------
        data_transformed : DataFrame, shape [n_samples, n_features]
            Transformed input data.
        """
        if np.any(data[self.transformed_cols_] + self.eps <= 0):
            raise ValueError('BoxCox transform can only be applied on positive data')

        if self.copy:
            data = data.copy()

        for i, feature in enumerate(self.transformed_cols_):
            data[feature] = self._boxcox(data[feature].values, self.lmbdas_[i])

        return data


class MultipleImputer(BaseEstimator, TransformerMixin):
    """
    Extends the sklearn Imputer Transformer [1]_ by allowing users
    to specify different imputing strategies for different columns

    Parameters
    ----------
    strategies : dict of list[str]
        Keys of the dictionary should one of the three valid
        strategies {'mode', 'mean', 'median'} and the values
        are the column names whose NA values will be filled
        should its corresponding strategies.
        e.g. {'mode': fill_mode_cols, 'median': fill_median_cols}

    missing_values : float or "NaN"/np.nan, default "NaN"
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        we can either use the string value "NaN" or np.nan.

    copy : bool, default True
        Set to False to perform inplace transformation.

    Attributes
    ----------
    statistics_ : dict of 1d ndarray
        The imputation fill value for each feature, the 1d ndarray
        corresponds to the strategies argument's input order.

    References
    ----------
    .. [1] `Scikit-learn Imputer
            <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html>`_
    """

    def __init__(self, strategies, missing_values = "NaN", copy = True):
        self.copy = copy
        self.strategies = strategies
        self.missing_values = missing_values

    def fit(self, data, y = None):
        """
        Fit MultipleImputer to the input data.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input data.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        self
        """
        mode_name = 'mode'
        mean_name = 'mean'
        median_name = 'median'
        allowed_strategies = {mode_name, mean_name, median_name}
        for k in self.strategies:
            if k not in allowed_strategies:
                msg = 'Can only use these strategies: {0} got strategy={1}'
                raise ValueError(msg.format(allowed_strategies, k))

        statistics = {}
        if mean_name in self.strategies:
            mean_cols = self.strategies[mean_name]
            X_masked = self._get_masked(data, mean_cols)
            mean_masked = ma.mean(X_masked, axis = 0)
            statistics[mean_name] = mean_masked.data

        if median_name in self.strategies:
            median_cols = self.strategies[median_name]
            X_masked = self._get_masked(data, median_cols)
            median_masked = ma.median(X_masked, axis = 0)
            statistics[median_name] = median_masked.data

        # numpy MaskedArray doesn't seem to support the .mode
        # method yet, thus we roll out our own
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/maskedarray.baseclass.html#maskedarray-baseclass
        if mode_name in self.strategies:
            mode_cols = self.strategies[mode_name]
            X_masked = self._get_masked(data, mode_cols)
            mode_values = np.empty(len(mode_cols))

            # transpose to compute along each column instead of row.
            # TODO :
            # an embarrassingly parallel problem, needs to investigate
            # if this is a bottleneck
            zipped = zip(X_masked.data.T, X_masked.mask.T)
            for i, (col, col_mask) in enumerate(zipped):
                col_valid = col[~col_mask]
                values, _ = mode(col_valid)
                mode_values[i] = values[0]

            statistics[mode_name] = mode_values

        self.statistics_ = statistics
        return self

    def _get_masked(self, data, target_cols):
        """
        Utilize masked array to compute the statistics.

        Introduction to masked array for those that are not familiar
        - https://docs.scipy.org/doc/numpy-1.13.0/reference/maskedarray.generic.html
        """
        X = data[target_cols].values
        if self.missing_values == 'NaN' or np.isnan(self.missing_values):
            mask = np.isnan(X)
        else:
            mask = X == self.missing_values

        X_masked = ma.masked_array(X, mask = mask)
        return X_masked

    def transform(self, data):
        """
        Transform input data using MultipleImputer.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input data.

        Returns
        -------
        data_transformed : DataFrame, shape [n_samples, n_features]
            Transformed input data.
        """
        if self.copy:
            data = data.copy()

        values = {}
        for strategy, cols in self.strategies.items():
            stats = self.statistics_[strategy]
            value = dict(zip(cols, stats))
            values.update(value)

        data = data.fillna(values)
        return data


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts a single column for a given DataFrame, this
    is mainly used for integrating with scikit-learn pipeline

    Parameters
    ----------
    col : str
        A single column name in the given DataFrame.

    References
    ----------
    .. [2] `Custom feature selection in sklearn pipeline
            <https://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline>`_
    """

    def __init__(self, col):
        self.col = col

    def fit(self, data, y = None):
        """
        Performs no operations at fitting time.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_feature]
            Input data.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        self
        """
        return self

    def transform(self, data):
        """
        Extract the specified single column.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_feature]
            Input data.

        Returns
        -------
        column : pd.Series
            Extracted column.
        """
        column = data[self.col]
        return column


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
            Input data.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

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
            Input data.

        Returns
        -------
        X : 2d ndarray, shape [n_samples, n_features]
            Transformed input data.
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
