import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


__all__ = ['PartialDependenceExplainer']


class PartialDependenceExplainer:
    """
    Partial Dependence explanation [1]_.

    - Currently only supports binary classification and regression classifiers.
    - Works for both numerical and categorical columns.

    Parameters
    ----------
    estimator : sklearn-like classifier
        Model that was fitted on the data.

    n_grid_points : int, default 50
        Number of grid points used in replacement
        for the original numeric data. Only used
        if the targeted column is numeric.

    Attributes
    ----------
    feature_name_ : str
        The input feature_name to the .fit unmodified, will
        be used in subsequent method.

    feature_type_ : str
        The input feature_type to the .fit unmodified, will
        be used in subsequent method.

    feature_grid_ : 1d ndarray
        Unique grid points that were used to generate the
        partial dependence result.

    References
    ----------
    .. [1] `Python partial dependence plot toolbox
            <https://github.com/SauceCat/PDPbox>`_
    """

    def __init__(self, estimator, n_grid_points = 50):
        self.estimator = estimator
        self.n_grid_points = n_grid_points

    def fit(self, data, feature_name, feature_type):
        """
        Obtain the partial dependence result.

        Parameters
        ----------
        data : DataFrame, shape [n_samples, n_features]
            Input data to the estimator/model.

        feature_name : str
            Feature's name in the data what we wish to explain.

        feature_type : str, {'num', 'cat'}
            Specify whether feature_name is a numerical or
            categorical column.

        Returns
        -------
        results : DataFrame
            Partial dependence result.
        """

        # check whether it's a classification or regression model
        estimator = self.estimator
        try:
            # TODO :
            # classes_ can be used for retrieving the number
            # of classes once we extend the functionality for
            # multi-class classification problem
            estimator.classes_
            classifier = True
            predict = estimator.predict_proba
        except AttributeError:
            classifier = False
            predict = estimator.predict

        target = data[feature_name]
        unique_target = np.unique(target)
        n_unique = unique_target.size
        if feature_type == 'num':
            if self.n_grid_points >= n_unique:
                feature_grid = unique_target
            else:
                percentile = np.percentile(target, np.linspace(0, 100, self.n_grid_points))
                feature_grid = np.unique(percentile)

            feature_cols = feature_grid
        else:
            feature_type = 'cat'
            feature_grid = unique_target
            feature_cols = np.asarray(['{}_{}'.format(feature_name, category)
                                       for category in unique_target])

        # compute prediction chunk by chunk to save memory usage
        results = []
        n_rows = data.shape[0]
        chunk_size = int(n_rows / feature_grid.size)

        # TODO :
        # embarrassingly parrallelized problem, can parallelize this if it becomes a bottleneck
        for i in range(0, n_rows, chunk_size):
            data_chunk = data[i:i + chunk_size].reset_index(drop = True)

            # generate ice (individual conditional expectation) data:
            # repeat the index and use it to slice the data to create the repeated data
            # instead of creating the repetition using the values, i.e.
            # np.repeat(data_chunk.values, repeats = feature_grid.size, axis = 0)
            # this prevents everything from getting converted to a different data type, e.g.
            # if there is 1 object type column then everything would get converted to object
            index_chunk = np.repeat(data_chunk.index.values, repeats = feature_grid.size)
            ice_data = data_chunk.iloc[index_chunk].copy()
            ice_data[feature_name] = np.tile(feature_grid, data_chunk.shape[0])

            prediction = predict(ice_data)
            if classifier:
                result = prediction[:, 1]
            else:
                result = prediction

            # reshape tiled data back to original chunk's shape
            reshaped = result.reshape((data_chunk.shape[0], feature_grid.size))
            result = pd.DataFrame(reshaped, columns = feature_cols)
            results.append(result)

        self.feature_name_ = feature_name
        self.feature_grid_ = feature_grid
        self.feature_type_ = feature_type
        results = pd.concat(results, ignore_index = True)
        return results

    def explain_plot(self, results, center = True):
        """
        Use the partial dependence result to generate
        a partial dependence plot (using matplotlib).

        Parameters
        ----------
        results : DataFrame
            Partial dependence result.

        center : bool, default True
            Center the partial dependence plot by subtacting every partial
            dependence result table's column value with the value of the first
            column, i.e. first column's value will serve as the baseline
            (centered at 0) for all other values.

        Returns
        -------
        figure
        """
        figure = GridSpec(5, 1)
        ax1 = plt.subplot(figure[0, :])
        self._plot_title(ax1)
        ax2 = plt.subplot(figure[1:, :])
        self._plot_content(ax2, results, center)
        return figure

    def _plot_title(self, ax):
        font_family = 'Arial'
        title = 'Individual Conditional Expectation Plot for {}'.format(self.feature_name_)
        subtitle = 'Number of unique grid points: {}'.format(self.feature_grid_.size)
        title_fontsize = 15
        subtitle_fontsize = 12

        ax.set_facecolor('white')
        ax.text(
            0, 0.7, title,
            fontsize = title_fontsize, fontname = font_family)
        ax.text(
            0, 0.4, subtitle, color = 'grey',
            fontsize = subtitle_fontsize, fontname = font_family)
        ax.axis('off')

    def _plot_content(self, ax, results, center):

        # pd (partial dependence)
        pd_linewidth = 2
        pd_markersize = 5
        pd_color = '#1A4E5D'
        fill_alpha = 0.2
        fill_color = '#66C2D7'
        zero_linewidth = 1.5
        zero_color = '#E75438'
        xlabel_fontsize = 10

        feature_cols = results.columns
        if self.feature_type_ == 'cat':
            # ticks = all the unique categories
            x = range(len(feature_cols))
            ax.set_xticks(x)
            ax.set_xticklabels(feature_cols)
        else:
            x = feature_cols

        # subtract all columns with the first column
        # https://stackoverflow.com/questions/43770401/subtract-pandas-columns-from-a-specified-column
        if center:
            results = results.copy()
            center_col = feature_cols[0]
            results = results[feature_cols].sub(results[center_col], axis = 0)

        pd = results.values.mean(axis = 0)
        pd_std = results.values.std(axis = 0)
        upper = pd + pd_std
        lower = pd - pd_std

        ax.plot(
            x, pd, color = pd_color, linewidth = pd_linewidth,
            marker = 'o', markersize = pd_markersize)
        ax.plot(
            x, [0] * pd.size, color = zero_color,
            linestyle = '--', linewidth = zero_linewidth)
        ax.fill_between(x, upper, lower, alpha = fill_alpha, color = fill_color)
        ax.set_xlabel(self.feature_name_, fontsize = xlabel_fontsize)
        self._modify_axis(ax)

    def _modify_axis(self, ax):
        tick_labelsize = 8
        tick_colors = '#9E9E9E'
        tick_labelcolor = '#424242'

        ax.tick_params(
            axis = 'both', which = 'major', colors = tick_colors,
            labelsize = tick_labelsize, labelcolor = tick_labelcolor)

        ax.set_facecolor('white')
        ax.get_yaxis().tick_left()
        ax.get_xaxis().tick_bottom()
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.grid(True, 'major', 'x', ls = '--', lw = .5, c = 'k', alpha = .3)
        ax.grid(True, 'major', 'y', ls = '--', lw = .5, c = 'k', alpha = .3)
