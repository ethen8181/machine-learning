import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

__all__ = ['GBTPipeline']


class GBTPipeline(BaseEstimator):
    """
    Gradient Boosted Tree Pipeline set up to do train/validation split
    and hyperparameter search.
    """

    def __init__(self, input_cols, cat_cols, label_col, weights_col,
                 model_task, model_id, model_type,
                 model_parameters, model_hyper_parameters, search_parameters):
        self.input_cols = input_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.weights_col = weights_col
        self.model_id = model_id
        self.model_type = model_type
        self.model_task = model_task
        self.model_parameters = model_parameters
        self.model_hyper_parameters = model_hyper_parameters
        self.search_parameters = search_parameters

    def fit(self, data, val_fold, fit_params=None):
        """
        Fit the pipeline to the input data.

        Parameters
        ----------
        data : pd.DataFrame
            Input training data. The data will be split into train/validation set
            by providing the val_fold (validation fold) parameter.

        val_fold : 1d ndarray
            The validation fold used for the `PredefinedSplit
            <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html>`_

        fit_params : dict
            Any additional parameters that are passed to the fit method of the model. e.g.
            `LGBMClassifier.fit
             <https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMClassifier.fit>`_

        Returns
        -------
        self
        """
        data_features = data[self.input_cols]
        label = data[self.label_col]
        sample_weights = data[self.weights_col] if self.weights_col is not None else None

        self.fit_params_ = self._create_default_fit_params(data_features, label,
                                                           val_fold, sample_weights)
        if fit_params is not None:
            self.fit_params_.update(fit_params)

        model = self._get_model()
        cv = PredefinedSplit(val_fold)
        model_tuned = RandomizedSearchCV(
            estimator=model,
            param_distributions=self.model_hyper_parameters,
            cv=cv,
            **self.search_parameters
        ).fit(data_features, label, **self.fit_params_)
        self.model_tuned_ = model_tuned
        return self

    def _get_model(self):
        if self.model_task == 'classification' and self.model_type == 'lgb':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**self.model_parameters)
        elif self.model_task == 'regression' and self.model_type == 'lgb':
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**self.model_parameters)
        else:
            raise ValueError("model_task should be regression/classification")

        return model

    def _create_default_fit_params(self, data, label, val_fold, sample_weights):
        mask = val_fold != -1
        data_train = data[~mask]
        data_val = data[mask]
        label_train = label[~mask]
        label_val = label[mask]
        fit_params = {
            'eval_set': [(data_train, label_train), (data_val, label_val)],
            'feature_name': self.input_cols,
            'categorical_feature': self.cat_cols
        }
        if sample_weights is not None:
            fit_params['sample_weights'] = sample_weights

        return fit_params

    def predict(self, data):
        """
        Prediction estimates from the best model.

        Parameters
        ----------
        data : pd.DataFrame
            Data that contains the same input_cols and cat_cols as the data that
            was used to fit the model.

        Returns
        -------
        prediction : ndarray
        """
        best = self.model_tuned_.best_estimator_
        return best.predict(data, num_iteration=best.best_iteration_)

    def get_feature_importance(self, threshold=1e-3):
        """
        Sort the feature importance based on decreasing order of the
        normalized gain.

        Parameters
        ----------
        threshold : float, default 1e-3
            Features that have a normalized gain smaller
            than the specified ``threshold`` will not be returned.
        """
        booster = self.model_tuned_.best_estimator_.booster_
        importance = booster.feature_importance(importance_type='gain')
        importance /= importance.sum()
        feature_name = np.array(booster.feature_name())

        mask = importance > threshold
        importance = importance[mask]
        feature_name = feature_name[mask]
        idx = np.argsort(importance)[::-1]
        return list(zip(feature_name[idx], np.round(importance[idx], 4)))

    def save(self, path=None):
        import os
        from joblib import dump

        model_checkpoint = self.model_id + '.pkl' if path is None else path

        # create the directory if it's not the current directory and it doesn't exist already
        model_dir = os.path.split(model_checkpoint)[0]
        if model_dir.strip() and not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        dump(self, model_checkpoint)
        return model_checkpoint

    @classmethod
    def load(cls, path):
        from joblib import load
        loaded_model = load(path)
        return loaded_model
