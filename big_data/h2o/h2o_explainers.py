import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class H2OPartialDependenceExplainer:
    """
    Partial Dependence explanation for binary classification H2O models.
    Works for both numerical and categorical (enum) features.

    Parameters
    ----------
    h2o_model : H2OEstimator
        H2O Model that was already fitted on the data.

    Attributes
    ----------
    feature_name_ : str
        The input feature_name to the .fit unmodified, will
        be used in subsequent method.

    is_cat_col_ : bool
        Whether the feature we're aiming to explain is a
        categorical feature or not.

    partial_dep_ : DataFrame
        A pandas dataframe that contains three columns, the
        feature's value and their corresponding mean prediction
        and standard deviation of the prediction. e.g.

        feature_name    mean_response stddev_response
        3000.000000     0.284140      0.120659
        318631.578947   0.134414      0.076054
        634263.157895   0.142961      0.083630

        The feature_name column will be the actual feature_name that
        we pass to the .fit method whereas the mean_response and
        stddev_response column will be fixed columns generated.
    """

    def __init__(self, h2o_model):
        self.h2o_model = h2o_model

    def fit(self, data, feature_name, n_bins=20):
        """
        Obtain the partial dependence result.

        Parameters
        ----------
        data : H2OFrame, shape [n_samples, n_features]
            Input data to the H2O estimator/model.

        feature_name : str
            Feature name in the data what we wish to explain.

        n_bins : int, default 20
            Number of bins used. For categorical columns, we will make sure the number
            of bins exceed the distinct level count.

        Returns
        -------
        self
        """
        self.is_cat_col_ = data[feature_name].isfactor()[0]
        if self.is_cat_col_:
            n_levels = len(data[feature_name].levels()[0])
            n_bins = max(n_levels, n_bins)

        partial_dep = self.h2o_model.partial_plot(data, cols=[feature_name],
                                                  nbins=n_bins, plot=False)
        self.feature_name_ = feature_name
        self.partial_dep_ = partial_dep[0].as_data_frame()
        return self

    def plot(self, centered=True, plot_stddev=True):
        """
        Use the partial dependence result to generate
        a partial dependence plot (using matplotlib).

        Parameters
        ----------
        centered : bool, default True
            Center the partial dependence plot by subtacting every partial
            dependence result table's column value with the value of the first
            column, i.e. first column's value will serve as the baseline
            (centered at 0) for all other values.

        plot_stddev : bool, default True
            Apart from plotting the mean partial dependence, also show the
            standard deviation as a fill between.

        Returns
        -------
        matplotlib figure
        """
        figure = GridSpec(5, 1)
        ax1 = plt.subplot(figure[0, :])
        self._plot_title(ax1)
        ax2 = plt.subplot(figure[1:, :])
        self._plot_content(ax2, centered, plot_stddev)
        return figure

    def _plot_title(self, ax):
        font_family = 'Arial'
        title = "Partial Dependence Plot for '{}' feature".format(self.feature_name_)
        subtitle = 'Number of unique grid points: {}'.format(self.partial_dep_.shape[0])
        title_fontsize = 15
        subtitle_fontsize = 12

        ax.set_facecolor('white')
        ax.text(
            0, 0.7, title,
            fontsize=title_fontsize, fontname=font_family)
        ax.text(
            0, 0.4, subtitle, color='grey',
            fontsize=subtitle_fontsize, fontname=font_family)
        ax.axis('off')

    def _plot_content(self, ax, centered, plot_stddev):
        # pd (partial dependence)
        pd_linewidth = 2
        pd_markersize = 5
        pd_color = '#1A4E5D'
        fill_alpha = 0.2
        fill_color = '#66C2D7'
        zero_linewidth = 1.5
        zero_color = '#E75438'
        xlabel_fontsize = 10

        pd_mean = self.partial_dep_['mean_response']
        if centered:
            # center the partial dependence plot by subtacting every value
            # with the value of the first column, i.e. first column's value
            # will serve as the baseline (centered at 0) for all other values
            pd_mean -= pd_mean[0]

        std = self.partial_dep_['stddev_response']
        upper = pd_mean + std
        lower = pd_mean - std
        x = self.partial_dep_[self.feature_name_]

        ax.plot(
            x, pd_mean, color=pd_color, linewidth=pd_linewidth,
            marker='o', markersize=pd_markersize)
        ax.plot(
            x, [0] * pd_mean.size, color=zero_color,
            linestyle='--', linewidth=zero_linewidth)

        if plot_stddev:
            ax.fill_between(x, upper, lower, alpha=fill_alpha, color=fill_color)

        ax.set_xlabel(self.feature_name_, fontsize=xlabel_fontsize)
        self._modify_axis(ax)

    def _modify_axis(self, ax):
        tick_labelsize = 8
        tick_colors = '#9E9E9E'
        tick_labelcolor = '#424242'

        ax.tick_params(
            axis='both', which='major', colors=tick_colors,
            labelsize=tick_labelsize, labelcolor=tick_labelcolor)

        ax.set_facecolor('white')
        ax.get_yaxis().tick_left()
        ax.get_xaxis().tick_bottom()
        for direction in ('top', 'left', 'right', 'bottom'):
            ax.spines[direction].set_visible(False)

        for axis in ('x', 'y'):
            ax.grid(True, 'major', axis, ls='--', lw=.5, c='k', alpha=.3)
