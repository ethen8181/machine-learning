import numpy as np


class Tree:
    """
    Classification tree using information gain with entropy as impurity

    Parameters
    ----------
    max_features : int or None, default None
        The number of features to consider when looking for the best split,
        None uses all features

    min_samples_split : int, default 10
        The minimum number of samples required to split an internal node

    max_depth : int, default 3
        Maximum depth of the tree

    minimum_gain : float, default 1e-7
        Minimum information gain required for splitting
    """

    def __init__(self, max_depth = 3, max_features = None,
                 minimum_gain = 1e-7, min_samples_split = 10):

        self.max_depth = max_depth
        self.max_features = max_features
        self.minimum_gain = minimum_gain
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        """pass in the 2d-array dataset and the response column"""
        self.n_class = np.unique(y).shape[0]

        # in the case you're wondering why we have this implementation of
        # choosing the number of features to consider when looking
        # for the best split, it will become much clearer when we
        # start discussing Random Forest algorithm
        if self.max_features is None or self.max_features > X.shape[1]:
            self.max_features = X.shape[1]

        self.feature_importance = np.zeros(X.shape[1])
        self.tree = _create_decision_tree(X, y, self.max_depth,
                                          self.minimum_gain, self.max_features,
                                          self.min_samples_split, self.n_class,
                                          self.feature_importance, X.shape[0])
        self.feature_importance /= np.sum(self.feature_importance)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        pred = np.argmax(proba, axis = 1)
        return pred

    def predict_proba(self, X):
        proba = np.empty((X.shape[0], self.n_class))
        for i in range(X.shape[0]):
            proba[i] = self._predict_row(X[i, :], self.tree)

        return proba

    def _predict_row(self, row, tree):
        """Predict single row"""
        if tree['is_leaf']:
            return tree['prob']
        else:
            if row[tree['split_col']] <= tree['threshold']:
                return self._predict_row(row, tree['left'])
            else:
                return self._predict_row(row, tree['right'])


def _create_decision_tree(X, y, max_depth,
                          minimum_gain, max_features,
                          min_samples_split, n_class,
                          feature_importance, n_row):
    """recursively grow the decision tree until it reaches the stopping criteria"""
    try:
        assert max_depth > 0
        assert X.shape[0] > min_samples_split
        column, value, gain = _find_best_split(X, y, max_features)
        assert gain > minimum_gain
        feature_importance[column] += (X.shape[0] / n_row) * gain

        # split the dataset and grow left and right child
        left_X, right_X, left_y, right_y = _split(X, y, column, value)
        left_child = _create_decision_tree(left_X, left_y, max_depth - 1,
                                           minimum_gain, max_features,
                                           min_samples_split, n_class,
                                           feature_importance, n_row)
        right_child = _create_decision_tree(right_X, right_y, max_depth - 1,
                                            minimum_gain, max_features,
                                            min_samples_split, n_class,
                                            feature_importance, n_row)
    except AssertionError:
        # if criteria reached, compute the classification
        # probability and return it as a leaf node

        # note that some leaf node may only contain partial classes,
        # thus specify the minlength to class that don't appear will
        # still get assign a probability of 0
        counts = np.bincount(y, minlength = n_class)
        prob = counts / y.shape[0]
        leaf = {'is_leaf': True, 'prob': prob}
        return leaf

    node = {'is_leaf': False,
            'left': left_child,
            'right': right_child,
            'split_col': column,
            'threshold': value}
    return node


def _find_best_split(X, y, max_features):
    """Greedy algorithm to find the best feature and value for a split"""
    subset = np.random.choice(X.shape[1], max_features, replace = False)
    max_col, max_val, max_gain = None, None, None
    parent_entropy = _compute_entropy(y)

    for column in subset:
        split_values = _find_splits(X, column)
        for value in split_values:
            splits = _split(X, y, column, value, return_X = False)
            gain = parent_entropy - _compute_splits_entropy(y, splits)

            if max_gain is None or gain > max_gain:
                max_col, max_val, max_gain = column, value, gain

    return max_col, max_val, max_gain


def _compute_entropy(split):
    """entropy score using a fix log base 2"""
    _, counts = np.unique(split, return_counts = True)
    p = counts / split.shape[0]
    entropy = -np.sum(p * np.log2(p))
    return entropy


def _find_splits(X, column):
    """
    find all possible split values (threshold),
    by getting unique values in a sorted order
    and finding cutoff point (average) between every two values
    """
    X_unique = np.unique(X[:, column])
    split_values = np.empty(X_unique.shape[0] - 1)
    for i in range(1, X_unique.shape[0]):
        average = (X_unique[i - 1] + X_unique[i]) / 2
        split_values[i - 1] = average

    return split_values


def _compute_splits_entropy(y, splits):
    """compute the entropy for the splits (the two child nodes)"""
    splits_entropy = 0
    for split in splits:
        splits_entropy += (split.shape[0] / y.shape[0]) * _compute_entropy(split)

    return splits_entropy


def _split(X, y, column, value, return_X = True):
    """split the response column using the cutoff threshold"""
    left_mask = X[:, column] <= value
    right_mask = X[:, column] > value
    left_y, right_y = y[left_mask], y[right_mask]

    if not return_X:
        return left_y, right_y
    else:
        left_X, right_X = X[left_mask], X[right_mask]
        return left_X, right_X, left_y, right_y


__all__ = [Tree]
