import numpy as np
import matplotlib.pyplot as plt


__all__ = ['vis_importance', 'vis_coef']


def vis_importance(estimator, feature_names, threshold = 0.05, filtered_names = None):
    """
    Visualize the relative importance of predictors.

    Parameters
    ----------
    estimator : sklearn-like ensemble tree model
        A tree estimator that contains the attribute
        ``feature_importances_``.

    feature_names : str 1d array or list[str]
        Feature names that corresponds to the
        feature importance.

    threshold : float, default 0.05
        Features that have importance scores lower than this
        threshold will not be presented in the plot, this assumes
        the feature importance sum up to 1.

    filtered_names : str 1d array or list[str], default None
        Feature names that we wish to exclude from the visualization
        regardless of whether they were in the top features or not.
    """
    if not hasattr(estimator, 'feature_importances_'):
        msg = '{} does not have the feature_importances_ attribute'
        raise ValueError(msg.format(estimator.__class__.__name__))

    imp = estimator.feature_importances_
    feature_names = np.asarray(feature_names)
    if filtered_names is not None:
        keep = ~np.in1d(feature_names, filtered_names, assume_unique = True)
        mask = np.logical_and(imp > threshold, keep)
    else:
        mask = imp > threshold

    importances = imp[mask]
    idx = np.argsort(importances)
    scores = importances[idx]
    names = feature_names[mask]
    names = names[idx]

    y_pos = np.arange(1, len(scores) + 1)
    if hasattr(estimator, 'estimators_'):
        # apart from the mean feature importance, for scikit-learn we can access
        # each individual tree's feature importance and compute the standard deviation
        tree_importances = np.asarray([tree.feature_importances_
                                       for tree in estimator.estimators_])
        importances_std = np.std(tree_importances[:, mask], axis = 0)
        scores_std = importances_std[idx]
        plt.barh(y_pos, scores, align = 'center', xerr = scores_std)
    else:
        plt.barh(y_pos, scores, align = 'center')

    plt.yticks(y_pos, names)
    plt.xlabel('Importance')
    plt.title('Feature Importance Plot')


def vis_coef(estimator, feature_names, topn = 10):
    """
    Visualize the top-n most influential coefficients
    for linear models.

    Parameters
    ----------
    estimator : sklearn-like linear model
        An estimator that contains the attribute
        ``coef_``.

    feature_names : str 1d array or list[str]
        Feature names that corresponds to the coefficients.

    topn : int, default 10
        Here topn refers to the largest positive and negative coefficients,
        i.e. a topn of 10, would show the top and bottom 10, so a total of
        20 coefficient weights.
    """
    fig = plt.figure()
    n_classes = estimator.coef_.shape[0]
    feature_names = np.asarray(feature_names)
    for idx, coefs in enumerate(estimator.coef_, 1):
        sorted_coefs = np.argsort(coefs)
        positive_coefs = sorted_coefs[-topn:]
        negative_coefs = sorted_coefs[:topn]
        top_coefs = np.hstack([negative_coefs, positive_coefs])

        colors = ['#A60628' if c < 0 else '#348ABD' for c in coefs[top_coefs]]
        y_pos = np.arange(2 * topn)
        fig.add_subplot(n_classes, 1, idx)
        plt.barh(y_pos, coefs[top_coefs], color = colors, align = 'center')
        plt.yticks(y_pos, feature_names[top_coefs])
        plt.title('top {} positive/negative coefficient'.format(topn))

    plt.tight_layout()
