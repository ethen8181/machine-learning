import numpy as np
import matplotlib.pyplot as plt


__all__ = ['vis_importance']


def vis_importance(estimator, feature_names, threshold = 0.05):
    """
    Visualize the relative importance of predictors.

    Parameters
    ----------
    estimator : sklearn-like ensemble tree classifier
        A ensemble tree estimator that contains the attribute
        ``feature_importances_``.

    feature_names : str 1d array or list
        Description feature names that corresponds to the
        feature importance.

    threshold : float, default 0.05
        Features that have importance scores lower than this
        threshold will not be presented in the plot, this assumes
        the sum of the feature importance sum up to 1.
    """
    if not hasattr(estimator, 'feature_importances_'):
        msg = '{} does not have the feature_importances_ attribute'
        ValueError(msg.format(estimator.__class__.__name__))

    # apart from the mean feature importance, for scikit-learn we can access
    # each individual tree's feature importance and compute the standard deviation
    has_std = False
    if hasattr(estimator, 'estimators_'):
        has_std = True
        tree_importances = np.asarray([tree.feature_importances_
                                       for tree in estimator.estimators_])
    imp = estimator.feature_importances_
    mask = imp > threshold
    importances = imp[mask]
    names = feature_names[mask]
    if has_std:
        importances_std = np.std(tree_importances[:, mask], axis = 0)

    idx = np.argsort(importances)
    names = names[idx]
    scores = importances[idx]
    if has_std:
        scores_std = importances_std[idx]

    y_pos = np.arange(1, len(importances) + 1)
    fig, ax = plt.subplots()
    if has_std:
        plt.barh(y_pos, scores, align = 'center', xerr = scores_std)
    else:
        plt.barh(y_pos, scores, align = 'center')

    plt.yticks(y_pos, names)
    plt.xlabel('Importance')
    plt.title('Feature Importance Plot')
