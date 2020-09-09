import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from typing import Dict, List, Tuple, Optional
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.calibration import calibration_curve


__all__ = [
    'compute_calibration_error',
    'create_binned_data',
    'get_bin_boundaries',
    'compute_binary_score',
    'compute_calibration_summary',
]


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int=15,
    round_digits: int=4) -> float:
    """
    Computes the calibration error for binary classification via binning
    data points into the specified number of bins. Samples with similar
    ``y_prob`` will be grouped into the same bin. The bin boundary is
    determined by having similar number of samples within each bin.

    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.

    y_prob : 1d ndarray
        Raw probability/score of the positive class.

    n_bins : int, default 15
        A bigger bin number requires more data. In general,
        the larger the bin size, the closer the calibration error
        will be to the true calibration error.

    round_digits : int, default 4
        Round the calibration error metric.

    Returns
    -------
    calibration_error : float
        RMSE between the average positive label and predicted probability
        within each bin.
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    binned_y_true, binned_y_prob = create_binned_data(y_true, y_prob, n_bins)

    # looping shouldn't be a source of bottleneck as n_bins should be a small number.
    bin_errors = 0.0
    for bin_y_true, bin_y_prob in zip(binned_y_true, binned_y_prob):
        avg_y_true = np.mean(bin_y_true)
        avg_y_score = np.mean(bin_y_prob)
        bin_error = (avg_y_score - avg_y_true) ** 2
        bin_errors += bin_error

    calibration_error = math.sqrt(bin_errors / n_bins)
    return round(calibration_error, round_digits)


def create_binned_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Bin ``y_true`` and ``y_prob`` by distribution of the data.
    i.e. each bin will contain approximately an equal number of
    data points. Bins are sorted based on ascending order of ``y_prob``.

    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.

    y_prob : 1d ndarray
        Raw probability/score of the positive class.

    n_bins : int, default 15
        A bigger bin number requires more data.

    Returns
    -------
    binned_y_true/binned_y_prob : 1d ndarray
        Each element in the list stores the data for that bin.
    """
    sorted_indices = np.argsort(y_prob)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_prob = y_prob[sorted_indices]
    binned_y_true = np.array_split(sorted_y_true, n_bins)
    binned_y_prob = np.array_split(sorted_y_prob, n_bins)
    return binned_y_true, binned_y_prob


def get_bin_boundaries(binned_y_prob: List[np.ndarray]) -> np.ndarray:
    """
    Given ``binned_y_prob`` from ``create_binned_data`` get the
    boundaries for each bin.

    Parameters
    ----------
    binned_y_prob : list
        Each element in the list stores the data for that bin.

    Returns
    -------
    bins : 1d ndarray
        Boundaries for each bin.
    """
    bins = []
    for i in range(len(binned_y_prob) - 1):
        last_prob = binned_y_prob[i][-1]
        next_first_prob = binned_y_prob[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)

    bins.append(1.0)
    return np.array(bins)


def compute_binary_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    round_digits: int=4) -> Dict[str, float]:
    """
    Compute various evaluation metrics for binary classification.
    Including auc, precision, recall, f1, log loss, brier score. The
    threshold for precision and recall numbers are based on the one
    that gives the best f1 score.

    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.

    y_prob : 1d ndarray
        Raw probability/score of the positive class.

    round_digits : int, default 4
        Round the evaluation metric.

    Returns
    -------
    metrics_dict : dict
        Metrics are stored in key value pair. ::

        {
            'auc': 0.82,
            'precision': 0.56,
            'recall': 0.61,
            'f1': 0.59,
            'log_loss': 0.42,
            'brier': 0.12
        }
    """
    auc = round(metrics.roc_auc_score(y_true, y_prob), round_digits)
    log_loss = round(metrics.log_loss(y_true, y_prob), round_digits)
    brier_score = round(metrics.brier_score_loss(y_true, y_prob), round_digits)

    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall)

    mask = ~np.isnan(f1)
    f1 = f1[mask]
    precision = precision[mask]
    recall = recall[mask]

    best_index = np.argmax(f1)
    precision = round(precision[best_index], round_digits)
    recall = round(recall[best_index], round_digits)
    f1 = round(f1[best_index], round_digits)
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'log_loss': log_loss,
        'brier': brier_score
    }


def compute_calibration_summary(
    eval_dict: Dict[str, pd.DataFrame],
    label_col: str='label',
    score_col: str='score',
    n_bins: int=15,
    strategy: str='quantile',
    round_digits: int=4,
    show: bool=True,
    save_plot_path: Optional[str]=None) -> pd.DataFrame:
    """
    Plots the calibration curve and computes the summary statistics for the model.

    Parameters
    ----------
    eval_dict : dict
        We can evaluate multiple calibration model's performance in one go. The key
        is the model name used to distinguish different calibration model, the value
        is the dataframe that stores the binary true targets and the predicted score
        for the positive class.

    label_col : str
        Column name for the dataframe in ``eval_dict`` that stores the binary true targets.

    score_col : str
        Column name for the dataframe in ``eval_dict`` that stores the predicted score.

    n_bins : int, default 15
        Number of bins to discretize the calibration curve plot and calibration error statistics.
        A bigger number requires more data, but will be closer to the true calibration error.

    strategy {'uniform', 'quantile'}, default 'quantile'
        Strategy used to define the boundary of the bins.

        - uniform: The bins have identical widths.
        - quantile: The bins have the same number of samples and depend on the predicted score.

    round_digits : default 4
        Round the evaluation metric.

    show : bool, default True
        Whether to show the plots on the console or jupyter notebook.

    save_plot_path : str, default None
        Path where we'll store the calibration plot. None means it will not save the plot.

    Returns
    -------
    df_metrics : pd.DataFrame
        Corresponding metrics for all the input dataframe.
    """

    fig, (ax1, ax2) = plt.subplots(2)

    # estimator_metrics stores list of dict, e.g.
    # [{'auc': 0.776, 'name': 'xgb'}]
    estimator_metrics = []
    for name, df_eval in eval_dict.items():
        prob_true, prob_pred = calibration_curve(
            df_eval[label_col],
            df_eval[score_col],
            n_bins=n_bins,
            strategy=strategy)

        calibration_error = compute_calibration_error(
            df_eval[label_col], df_eval[score_col], n_bins, round_digits)
        metrics_dict = compute_binary_score(df_eval[label_col], df_eval[score_col], round_digits)
        metrics_dict['calibration_error'] = calibration_error
        metrics_dict['name'] = name
        estimator_metrics.append(metrics_dict)

        ax1.plot(prob_pred, prob_true, 's-', label=name)
        ax2.hist(df_eval[score_col], range=(0, 1), bins=n_bins, label=name, histtype='step', lw=2)

    ax1.plot([0, 1], [0, 1], 'k:', label='perfect')

    ax1.set_xlabel('Fraction of positives (Predicted)')
    ax1.set_ylabel('Fraction of positives (Actual)')
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc='upper left', ncol=2)
    ax1.set_title('Calibration Plots (Reliability Curve)')

    ax2.set_xlabel('Predicted scores')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram of Predicted Scores')
    ax2.legend(loc='upper right', ncol=2)

    plt.tight_layout()
    if show:
        plt.show()

    if save_plot_path is not None:
        save_dir = os.path.dirname(save_plot_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_plot_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

    df_metrics = pd.DataFrame(estimator_metrics)
    return df_metrics
