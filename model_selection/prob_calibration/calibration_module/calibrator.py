import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from calibration_module.utils import create_binned_data, get_bin_boundaries


__all__ = [
    'HistogramCalibrator',
    'PlattCalibrator',
    'PlattHistogramCalibrator'
]


class HistogramCalibrator(BaseEstimator):
    """
    Bins the data based on equal size interval (each bin contains approximately
    equal size of samples).

    Parameters
    ----------
    n_bins : int, default 15
        A bigger bin number requires more data. In general,
        the larger the bin size, the closer the calibration error
        will be to the true calibration error.

    Attributes
    ----------
    bins_ : 1d ndarray
        Boundaries for each bin.

    bins_score_ : 1d ndarray
        Calibration score for each bin.
    """

    def __init__(self, n_bins: int=15):
        self.n_bins = n_bins

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Learns the bin boundaries and calibration score for each bin.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        self
        """
        binned_y_true, binned_y_prob = create_binned_data(y_true, y_prob, self.n_bins)
        self.bins_ = get_bin_boundaries(binned_y_prob)
        self.bins_score_ = np.array([np.mean(value) for value in binned_y_true])
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Predicts the calibrated probability.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        indices = np.searchsorted(self.bins_, y_prob)
        return self.bins_score_[indices]


class PlattCalibrator(BaseEstimator):
    """
    Boils down to applying a Logistic Regression.

    Parameters
    ----------
    log_odds : bool, default True
        Logistic Regression assumes a linear relationship between its input
        and the log-odds of the class probabilities. Converting the probability
        to log-odds scale typically improves performance.

    Attributes
    ----------
    coef_ : ndarray of shape (1,)
        Binary logistic regresion's coefficient.

    intercept_ : ndarray of shape (1,)
        Binary logistic regression's intercept.
    """

    def __init__(self, log_odds: bool=True):
        self.log_odds = log_odds

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Learns the logistic regression weights.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        self
        """
        self.fit_predict(y_prob, y_true)
        return self

    @staticmethod
    def _convert_to_log_odds(y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_prob = np.log(y_prob / (1 - y_prob))
        return y_prob

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Predicts the calibrated probability.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        output = self._transform(y_prob)
        return output

    def _transform(self, y_prob: np.ndarray) -> np.ndarray:
        output = y_prob * self.coef_[0] + self.intercept_
        output = 1 / (1 + np.exp(-output))
        return output

    def fit_predict(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Chain the .fit and .predict step together.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability. 
        """
        if self.log_odds:
            y_prob = self._convert_to_log_odds(y_prob)

        # the class expects 2d ndarray as input features
        logistic = LogisticRegression(C=1e10, solver='lbfgs')
        logistic.fit(y_prob.reshape(-1, 1), y_true)
        self.coef_ = logistic.coef_[0]
        self.intercept_ = logistic.intercept_

        y_calibrated_prob = self._transform(y_prob)
        return y_calibrated_prob


class PlattHistogramCalibrator(PlattCalibrator):
    """
    Boils down to first applying a Logistic Regression then perform
    histogram binning.

    Parameters
    ----------
    log_odds : bool, default True
        Logistic Regression assumes a linear relationship between its input
        and the log-odds of the class probabilities. Converting the probability
        to log-odds scale typically improves performance.

    n_bins : int, default 15
        A bigger bin number requires more data. In general,
        the larger the bin size, the closer the calibration error
        will be to the true calibration error.

    Attributes
    ----------
    coef_ : ndarray of shape (1,)
        Binary logistic regresion's coefficient.

    intercept_ : ndarray of shape (1,)
        Binary logistic regression's intercept.

    bins_ : 1d ndarray
        Boundaries for each bin.

    bins_score_ : 1d ndarray
        Calibration score for each bin.
    """

    def __init__(self, log_odds: bool=True, n_bins: int=15):
        super().__init__(log_odds)
        self.n_bins = n_bins

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Learns the logistic regression weights and the
        bin boundaries and calibration score for each bin.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        self
        """
        y_prob_platt = super().fit_predict(y_prob, y_true)
        binned_y_true, binned_y_prob = create_binned_data(y_true, y_prob_platt, self.n_bins)
        self.bins_ = get_bin_boundaries(binned_y_prob)
        self.bins_score_ = np.array([np.mean(value) for value in binned_y_prob])
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Predicts the calibrated probability.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """
        y_prob_platt = super().predict(y_prob)
        indices = np.searchsorted(self.bins_, y_prob_platt)
        return self.bins_score_[indices]
