import warnings
import numpy as np
import pandas as pd
import numpy.ma as ma
from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.stats import mode, boxcox
from scipy.stats import chi2_contingency
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = [
    'BoxCoxTransformer',
    'MultipleImputer',
    'ColumnExtractor',
    'OneHotEncoder',
    'Preprocessor']


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
    transformed_cols : str 1d ndarray/list or 'all', default 'all'
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
        - for lmbda > 0, y = (x ** lmbda - 1.) / lmbda
        - for lmbda = 0, log(x)
    ``boxcox`` requires the input data to be positive.
    """

    def __init__(self, transformed_cols = 'all', eps = 1e-8, copy = True):
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
            # use np.copy instead of the .copy method, it
            # ensures it won't break if the user inputted
            # transformed_cols as a list
            transformed_cols = np.copy(self.transformed_cols)

        if np.any(data[transformed_cols] + self.eps <= 0):
            raise ValueError('BoxCox transform can only be applied on positive data')

        # TODO :
        # an embarrassingly parallelized problem, i.e.
        # each features' boxcox lambda can be estimated in parallel
        # needs to investigate if it's a bottleneck
        lmbdas = [self._boxcox(data[i].values) for i in transformed_cols]
        self.lmbdas_ = np.asarray(lmbdas)
        self.transformed_cols_ = transformed_cols
        return self

    def _boxcox(self, x, lmbda = None):
        """Utilize scipy's boxcox transformation"""
        mask = ~np.isnan(x)
        x_valid = x[mask] + self.eps
        if lmbda is None:
            _, lmbda = boxcox(x_valid, lmbda)
            return lmbda
        else:
            x[mask] = boxcox(x_valid, lmbda)
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

        for feature, lmbdas in zip(self.transformed_cols_, self.lmbdas_):
            data[feature] = self._boxcox(data[feature].values, lmbdas)

        return data


class MultipleImputer(BaseEstimator, TransformerMixin):
    """
    Extends the sklearn Imputer Transformer [1]_ by allowing users
    to specify different imputing strategies for different columns.

    Parameters
    ----------
    strategies : dict of list[str]
        Keys of the dictionary should one of the three valid
        strategies {'mode', 'mean', 'median'} and the values
        are the column names whose NA values will be filled
        with its corresponding strategies.
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

    def __init__(self, strategies, missing_values = 'NaN', copy = True):
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
    is mainly used for integrating with scikit-learn pipeline [2]_.

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
        data : DataFrame, shape [n_samples, n_features]
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
        data : DataFrame, shape [n_samples, n_features]
            Input data.

        Returns
        -------
        column : pd.Series
            Extracted column.
        """
        column = data[self.col]
        return column


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical integer features using a one-hot aka one-of-K scheme.
    The input to this transformer should only be a matrix of integers, denoting
    the values taken on by categorical (discrete) features.
    The output will be a sparse/dense matrix where each column corresponds to
    one possible value of one feature. It is assumed that input features take on
    values in the range [0, n_values).

    This class extends the sklearn OneHotEncoder Transformer [3]_ by allowing users
    to specify drop the first level of a categorical feature and treat it as the
    reference level, also removes the functionality of ignoring unseen categories
    during testing, i.e. it will always through an error if a category unseen during
    training is seen during testing.

    Parameters
    ----------
    drop_first : bool, default True
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level, this will not result in an information loss while being
        more memory efficient since we're not generating an extra column for
        every categorical columns.

    sparse : bool, default = True
        Will return sparse matrix if set True else will return an numpy array.

    dtype : numpy data type, default = np.float
        Desired dtype of output.

    Attributes
    ----------
    feature_indices_ : 1d ndarray, shape [n_features]
        Indices to feature ranges. Feature ``i`` in the original data
        is mapped to features from
        ``feature_indices_[i]`` to ``feature_indices_[i+1]``.

    n_values_ : 1d ndarray, shape [n_features]
        Maximum number of values/categories per feature.

    References
    ----------
    .. [3] `Scikit-learn OneHotEncoder
            <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`_
    """

    def __init__(self, drop_first = True, sparse = True, dtype = np.float):
        self.dtype = dtype
        self.sparse = sparse
        self.drop_first = drop_first

    def fit(self, X, y = None):
        """
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : 2d ndarray, shape [n_samples, n_features]
            Input array of type int.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y = None):
        """
        Fit OneHotEncoder to X, then transform X. Equivalent to
        self.fit(X).transform(X), but more convenient and more efficient.

        Parameters
        ----------
        X : 2d ndarray, shape [n_samples, n_features]
            Input array of type int.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        X_out : sparse matrix if sparse = True else a 2d ndarray
            Transformed input.
        """
        X_out, n_values, indices = self._fit_transform(X)
        self.n_values_ = n_values
        self.feature_indices_ = indices
        return X_out

    def _fit_transform(self, X, n_values = None, indices = None):
        if np.any(X < 0):
            raise ValueError('X needs to contain only non-negative integers.')

        n_samples, n_features = X.shape
        if self.drop_first:
            if n_values is None:
                n_values = np.max(X, axis = 0)
                indices = np.cumsum(np.hstack([[0], n_values]))
                n_values += 1

            # counter keeps track of the column indices' starting point
            counter = -1
            counter_values = np.hstack([[0], (n_values[:-1]) - 1])
            row_indices = []
            col_indices = []
            for col in range(n_features):
                counter += counter_values[col]

                # the first level should be values with a numerical
                # representation of 0 and they will be dropped
                mask = np.logical_not(X[:, col] == 0)
                row_indice = np.where(mask)[0]
                col_indice = X[mask, col] + counter
                row_indices.append(row_indice)
                col_indices.append(col_indice)

            row_indices = np.hstack(row_indices)
            col_indices = np.hstack(col_indices)
        else:
            if n_values is None:
                n_values = np.max(X, axis = 0) + 1
                indices = np.cumsum(np.hstack([[0], n_values]))

            col_indices = (X + indices[:-1]).ravel()
            row_indices = np.repeat(np.arange(n_samples, dtype = np.intc), n_features)

        data = np.ones_like(row_indices)
        X_out = csr_matrix((data, (row_indices, col_indices)),
                           shape = (n_samples, indices[-1]),
                           dtype = self.dtype)
        X_out = X_out if self.sparse else X_out.toarray()
        return X_out, n_values, indices

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        Parameters
        ----------
        X : 2d ndarray, shape [n_samples, n_features]
            Input array of type int.

        Returns
        -------
        X_out : sparse matrix if sparse = True else a 2-d array
            Transformed input.
        """
        n_features = X.shape[1]
        n_values = self.n_values_
        indices = self.feature_indices_
        if n_features != indices.size - 1:
            raise ValueError(
                'X has different shape than during fitting. '
                'Expected {}, got {}.'.format(indices.shape[0] - 1, n_features))

        mask = (X >= n_values).ravel()
        if np.any(mask):
            raise ValueError(
                'Unknown categorical feature present {} '
                'during transform.'.format(X.ravel()[mask]))

        X_out, _, _ = self._fit_transform(X, n_values, indices)
        return X_out


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Generic data preprocessing including:
    - standardize numeric columns and remove potential
    multi-collinearity using variance inflation factor
    - one-hot encode categorical columns

    Parameters
    ----------
    num_cols : list[str], default None
        Numerical columns' name. default None means
        the input column has no numeric features.

    cat_cols : list[str], default None
        Categorical columns' name.

    output_pandas : bool, default False
        Whether to output a pandas DataFrame or a numpy ndarray.

    use_onehot : bool, default True
        Whether to use one hot encoding for the categorical features.
        (the first-level will always be dropped when converting to
        one hot encoding scheme)

    vif_threshold : float, default 5.0
        Threshold for variance inflation factor (vif).
        If there are numerical columns, identify potential collinearity
        amongst them using vif. Conventionally, a vif score larger than 5
        should be removed. This is a positive value that has no upper bound.

    cramersv_threshold : float, default 0.8
        Threshold for Cramers' V statistics.
        If there are categorical columns, identify potential collinearity
        amongst them using Cramers' V statistics. This is a value that ranges
        between 0 and 1.

    correction : bool, default False
        Additional argument for the Cramer's V statistics.
        If True, and the degrees of freedom is 1, apply Yates’ correction for continuity.
        The effect of the correction is to adjust each observed value by 0.5 towards the
        corresponding expected value. This is set to False by defualt as the effect of
        Yates' correction is to prevent overestimation of statistical significance for small
        data. i.e. It is chiefly used when at least one cell of the table has an expected
        count smaller than 5. And most people probably aren't working with a data size that's
        that small.

    Attributes
    ----------
    colnames_ : str 1d ndarray
        Column name of the transformed data.

    num_cols_ : str 1d ndarray
        Final numerical columns after removing potential collinearity.

    cat_cols_ : str 1d ndarray
        Final categorical column safter removing potential collinearity.

    label_encode_ : dict[list]
        Categorical features will always be encoded into numerical
        representation with value between 0 and number of distinct categories - 1.
        Keys are the categorical features' name and values
        are the list of values that were seen during training time.

    cat_encode_ : sklearn's OneHotEncoder object
        OneHotEncoder that was used to one-hot encode the
        categorical columns.

    scaler_ : sklearn's StandardScaler object
        StandardScaler that was used to standardize the numerical columns.
    """

    def __init__(self, num_cols = None, cat_cols = None,
                 output_pandas = False, use_onehot = True,
                 vif_threshold = 5.0, cramersv_threshold = 0.8, correction = False):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.correction = correction
        self.use_onehot = use_onehot
        self.output_pandas = output_pandas
        self.vif_threshold = vif_threshold
        self.cramersv_threshold = cramersv_threshold

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
        self.fit_transform(data)
        return self

    def fit_transform(self, data, y = None):
        """
        Fit Preprocessor to X, then transform X. Equivalent to
        self.fit(X).transform(X), but more convenient and more efficient.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input array of type int.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        X : DataFrame if output_pandas = True else a 2d ndarray
            Transformed input.
        """
        if self.num_cols is None and self.cat_cols is None:
            raise ValueError('There must be a least one input feature column')

        # store the column names so we can refer to them later;
        # all numerical columns comes before categorical columns
        colnames = []
        scaled = None
        encoded = None
        data = data.copy()
        if self.num_cols is not None:
            # data will be converted to float64 type in StandardScaler
            # anyway, so do the conversion here to prevent raising
            # conversion warning later
            data_subset = data[self.num_cols].values.astype(np.float64)
            self.scaler_ = StandardScaler()
            scaled = self.scaler_.fit_transform(data_subset)
            colnames = self._remove_num_collinearity(scaled)
            self.num_cols_ = np.asarray(colnames)

        if self.cat_cols is not None:
            cat_cols = self._remove_cat_collinearity(data[self.cat_cols])

            # convert categorical type to numeric representation
            # using pandas category conversion is a lot faster than
            # label encoding across multiple columns in scikit-learn
            # https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn
            label_encode = {}
            encoded = np.empty((data.shape[0], len(cat_cols)))
            for col, cat_col in enumerate(cat_cols):
                data[cat_col] = data[cat_col].astype('category')
                label_encode[cat_col] = list(data[cat_col].cat.categories)
                encoded[:, col] = data[cat_col].cat.codes

            if self.use_onehot:
                cat_encode = OneHotEncoder(sparse = False)
                encoded = cat_encode.fit_transform(encoded)
                for cat_col in cat_cols:
                    classes = data[cat_col].cat.categories
                    if cat_encode.drop_first:
                        classes = data[cat_col].cat.categories[1:]

                    cat_colnames = [cat_col + '_' + str(c) for c in classes]
                    colnames += cat_colnames

                self.cat_encode_ = cat_encode
            else:
                colnames += cat_cols

            self.label_encode_ = label_encode
            self.cat_cols_ = np.asarray(cat_cols)

        self.colnames_ = np.asarray(colnames)
        X = self._combine_output(scaled, encoded, data[cat_cols])
        return X

    def _remove_num_collinearity(self, X):
        """
        Identify collinearity between the numeric variables
        using variance inflation factor (vif)
        """
        scaler = self.scaler_
        colnames = self.num_cols.copy()
        while True:
            n_features = X.shape[1]
            if n_features == 1:
                break

            vif = [self._compute_vif(X, index) for index in range(n_features)]
            max_index = np.argmax(vif)
            if vif[max_index] >= self.vif_threshold:
                removed = colnames[max_index]
                colnames.remove(removed)
                X = np.delete(X, max_index, axis = 1)
                scaler.mean_ = np.delete(scaler.mean_, max_index)
                scaler.scale_ = np.delete(scaler.scale_, max_index)
            else:
                break

        self.scaler_ = scaler
        return colnames

    def _compute_vif(self, X, target_index):
        """
        Similar implementation as statsmodel's variance_inflation_factor
        with some enhancemants:
        1. includes the intercept term
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

    def _remove_cat_collinearity(self, data):
        """
        Identify collinearity between the numeric variables
        using Cramer's V statistics.
        """
        n_features = data.shape[1]
        colnames = self.cat_cols.copy()
        if n_features > 1:
            removed = set()
            for col1, col2 in combinations(self.cat_cols, 2):
                if col1 not in removed:
                    observed = pd.crosstab(data[col1], data[col2]).values
                    cramersv = self._compute_cramersv(observed)
                    if cramersv >= self.cramersv_threshold:
                        removed.add(col1)
                        colnames.remove(col1)

        return colnames

    def _compute_cramersv(self, observed):
        """
        Expects a 2d ndarray contingency table that contains the observed
        frequencies (i.e. number of occurrences) for each category.
        """
        n_obs = observed.sum()
        n_row, n_col = observed.shape
        chi2 = chi2_contingency(observed, correction = self.correction)[0]
        cramersv = np.sqrt(chi2 / (n_obs * min(n_row - 1, n_col - 1)))
        return cramersv

    def _combine_output(self, scaled, encoded = None, data_cat = None):
        """
        Combine encoded categorical columns and scaled numerical
        columns, it's the same as concatenate it along axis 1
        """
        if self.cat_cols_ is not None and self.num_cols_ is not None:
            if self.output_pandas:
                data_num = pd.DataFrame(scaled, columns = self.num_cols_)
                if self.use_onehot:
                    cat_cols = self.colnames_[len(self.num_cols_):]
                    data_cat = pd.DataFrame(encoded, columns = cat_cols)

                data_cat = data_cat.reset_index(drop = True)
                X = pd.concat([data_num, data_cat], axis = 1)
            else:
                X = np.hstack((scaled, encoded))
        elif self.num_cols_ is None:
            if self.output_pandas:
                if self.use_onehot:
                    cat_cols = self.colnames_[len(self.num_cols_):]
                    data_cat = pd.DataFrame(encoded, columns = cat_cols)

                X = data_cat
            else:
                X = encoded
        else:
            if self.output_pandas:
                X = pd.DataFrame(scaled, columns = self.num_cols_)
            else:
                X = scaled

        return X

    def transform(self, data):
        """
        Transform the input data using Preprocess Transformer.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input data.

        Returns
        -------
        X : DataFrame if output_pandas = True else a 2d ndarray
            Transformed input.
        """
        data = data.copy()
        cat_cols = self.cat_cols_
        num_cols = self.num_cols_
        if cat_cols is not None:
            encoded = np.empty((data.shape[0], len(cat_cols)))
            for col, cat_col in enumerate(cat_cols):
                categories = self.label_encode_[cat_col]
                cat_type = CategoricalDtype(categories = categories)
                data[cat_col] = data[cat_col].astype(cat_type)
                mask = data[cat_col].isnull()
                if mask.sum():
                    raise ValueError(
                        'Unknown categorical feature present: {} during transform, '
                        'for column: {}.'.format(data.loc[mask, cat_col], cat_col))

                encoded[:, col] = data[cat_col].cat.codes

            if self.use_onehot:
                encoded = self.cat_encode_.transform(encoded)

        if num_cols is not None:
            data_subset = data[num_cols].values.astype(np.float64)
            scaled = self.scaler_.transform(data_subset)

        X = self._combine_output(scaled, encoded, data[cat_cols])
        return X
