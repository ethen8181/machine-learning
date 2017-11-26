import logging
import numpy as np
from collections import Counter
from imblearn.base import SamplerMixin
from imblearn.utils import check_target_type, hash_X_y
from sklearn.utils import check_X_y, check_random_state, safe_indexing


__all__ = ['RandomUnderSampler']


def check_ratio(ratio, y):
    """check and returns actual and valid ratio"""
    target_stats = Counter(y)
    diff_target = set(ratio.keys()) - set(target_stats.keys())

    # check to ensure all keys in ratio are also in y
    # and the ratio are all positive
    if diff_target:
        raise ValueError(
            'The {} target class is/are not present in the data.'.format(diff_target))

    if any(n_samples < 0 for n_samples in ratio.values()):
        raise ValueError(
            'The proportion of samples in a class cannot be negative. '
            'Input ratio contains some negative value: {}'.format(ratio))

    checked_ratio = {}
    for target, n_samples in ratio.items():
        target_samples = target_stats[target]
        # if it's a float then assume it's asking for a
        # proportion of the targeted sample
        if isinstance(n_samples, float):
            n_samples = int(n_samples * target_samples)

        if n_samples > target_samples:
            raise ValueError(
                'With under-sampling methods, the number of '
                'samples in a class should be less or equal '
                'to the original number of samples. '
                'Originally, there is {} samples and {} '
                'samples are asked.'.format(target_samples, n_samples))

        checked_ratio[target] = n_samples

    return checked_ratio


class BaseSampler(SamplerMixin):
    """
    Base class for sampling algorithms.

    Warning: This class should not be used directly.
    Use the derive classes instead.
    """

    def __init__(self, ratio):
        self.ratio = ratio
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):
        """
        Find the classes statistics to perform sampling.

        Parameters
        ----------
        X : 2d ndarray or scipy sparse matrix, shape [n_samples, n_features]
            Matrix containing the data which have to be sampled.

        y : 1d ndarray, shape [n_samples]
            Corresponding label for each sample in X.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, accept_sparse = ['csr', 'csc'])
        y = check_target_type(y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        self.ratio_ = check_ratio(self.ratio, y)
        return self


class RandomUnderSampler(BaseSampler):
    """
    Class to perform random under-sampling.
    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    This is an "improvement" of imbalance learn's RandomUnderSampler [1]_
    by only accepting a dictionary for the ratio argument and supports
    float value indicating the proportional sampling.

    Parameters
    ----------
    ratio : dict[(int, int/float)]
        Ratio to use for resampling the data set.
        Keys correspond to the targeted classes and the values
        correspond to the desired number/proportion of samples.
        e.g. {0: 1.0, 1: 0.5} becauses the values are float, this
        is read as we'll keep all samples from class label 0 and
        keep only 50 percent of class label 1, note that in this
        case {1: 0.5} will also work. We could also specify integer
        value for the values in the dictionary to indicate the
        actual number of samples to retain.

    replacement : bool, default False
        Whether the sample is with or without replacement.

    random_state : int, RandomState instance or None, default None
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    Attributes
    ----------
    ratio_ : dict[(int, int)]
        The actual ratio that was used for resampling the data set,
        where the class label is the key and the number of samples is the value

    X_hash_/y_hash_ : str
        Hash identifier of the input X and y. This is used for ensuring
        the X and y that was used for fitting is identical to sampling
        (resampling is only meant for the same "training" set)

    References
    ----------
    .. [1] `imbalanced-learn RandomUnderSampler
            <http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.under_sampling.RandomUnderSampler.html>`_
    """

    def __init__(self, ratio, replacement = False, random_state = None):
        super().__init__(ratio = ratio)
        self.replacement = replacement
        self.random_state = random_state

    def _sample(self, X, y):
        """resample the dataset"""
        random_state = check_random_state(self.random_state)

        sample_indices = []
        targets = np.unique(y)
        for target in targets:
            target_indices = np.flatnonzero(y == target)
            if target in self.ratio_:
                n_samples = self.ratio_[target]
                target_indices = random_state.choice(
                    target_indices, size = n_samples, replace = self.replacement)

            sample_indices.append(target_indices)

        sample_indices = np.hstack(sample_indices)
        return safe_indexing(X, sample_indices), safe_indexing(y, sample_indices)
