import os
import fasttext
import pandas as pd
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from joblib import Parallel, delayed, dump, load
from sklearn.model_selection import ParameterSampler
from fasttext_module.utils import prepend_file_name
from fasttext_module.split import train_test_split_file


__all__ = [
    'FasttextPipeline',
    'fit_and_score',
    'fit_fasttext',
    'score'
]


class FasttextPipeline:
    """
    Fasttext text classification pipeline.

    Parameters
    ----------
    model_id : str
        Unique identifier for the model, the model checkpoint will have this name.

    fasttext_params : dict
        Interpreted as fasttext.train_supervised(fasttext_params). Note that
        we do not need to specify the input text file under this parameter.

    fasttext_hyper_params : dict
        Controls which parameters and its corresponding range that will be tuned.
        e.g. {"dim": [80, 100]}

    fasttext_search_params : dict
        Controls how long to perform the hyperparameter search and what metric to optimize for.

        - n_iter (int) Number of parameter settings that are chosen fasttext_hyper_params.
        - random_state (int) Seed for sampling from fasttext_hyper_params.
        - n_jobs (int) Number of jobs to run in parallel. -1 means use all processors.
        - verbose (int) The higher the number, the more messages printed.
        - scoring (str) The metrics to use for selecting the best parameter. e.g.
          f1@1, precision@1, recall@1. The valid metrics are precision/recall/f1 followed
          by @k, where k controls the top k predictions that we'll be evaluating the prediction.

    Attributes
    ----------
    model_ : _FastText
        Fasttext model.

    df_tune_results_ : pd.DataFrame
        DataFrame that stores the hyperparameter tuning results, including the
        parameters that were tuned and its corresponding train/test score.

    best_params_ : dict
        Best hyperparameter chosen to re-fit the model on the entire dataset.
    """

    def __init__(self,
                 model_id: str,
                 fasttext_params: Dict[str, Any],
                 fasttext_hyper_params: Dict[str, List[Any]],
                 fasttext_search_params: Dict[str, Any]):
        self.model_id = model_id
        self.fasttext_params = fasttext_params
        self.fasttext_hyper_params = fasttext_hyper_params
        self.fasttext_search_params = fasttext_search_params

    def fit_file(self, fasttext_file_path: str,
                 val_size: float=0.1, split_random_state: int=1234):
        """
        Fit the pipeline to the input file path.

        Parameters
        ----------
        fasttext_file_path : str
            The text file should already be in the fasttext expected format.

        val_size: float, default 0.1
            Proportion of the dataset to include in the validation split.
            The validation set will be used to pick the best parameter from
            the hyperparameter search.

        split_random_state : int, default 1234
            Seed for the split.

        Returns
        -------
        self
        """
        self._tune_fasttext(fasttext_file_path, val_size, split_random_state,
                            **self.fasttext_search_params)
        self.model_ = fit_fasttext(fasttext_file_path, self.fasttext_params, self.best_params_)
        return self

    def _tune_fasttext(self, fasttext_file_path: str, val_size: float, split_random_state: int,
                       n_iter: int, random_state: int, n_jobs: int, verbose: int, scoring: str):
        parameter_sampler = ParameterSampler(self.fasttext_hyper_params, n_iter, random_state)

        fasttext_file_path_train = prepend_file_name(fasttext_file_path, 'train')
        fasttext_file_path_val = prepend_file_name(fasttext_file_path, 'val')
        count_train, count_val = train_test_split_file(
            fasttext_file_path, fasttext_file_path_train, fasttext_file_path_val,
            val_size, split_random_state)

        k = int(scoring.split('@')[-1])
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
        results = parallel(delayed(fit_and_score)(fasttext_file_path_train,
                                                  fasttext_file_path_val,
                                                  self.fasttext_params,
                                                  k,
                                                  param)
                           for param in parameter_sampler)

        df_tune_results = (pd.DataFrame
                           .from_dict(results)
                           .sort_values(f'test_{scoring}', ascending=False))
        self.best_params_ = df_tune_results['params'].iloc[0]
        self.df_tune_results_ = df_tune_results

        # clean up the intermediate train/test split file to prevent hogging up
        # un-needed disk space
        for file_path in [fasttext_file_path_train, fasttext_file_path_val]:
            os.remove(file_path)

        return self

    def save(self, directory: str) -> str:
        """
        Saves the pipeline.

        Parameters
        ----------
        directory : str
            The directory to save the model. Will create the directory if it
            doesn't exist.

        Returns
        -------
        model_checkpoint_dir : str
            The directory of the saved model.
        """
        model_checkpoint_dir = os.path.join(directory, self.model_id)
        if not os.path.isdir(model_checkpoint_dir):
            os.makedirs(model_checkpoint_dir, exist_ok=True)

        # some model can't be pickled and have their own way of saving it
        model = self.model_
        model_checkpoint = os.path.join(model_checkpoint_dir, 'model.fasttext')
        model.save_model(model_checkpoint)

        self.model_ = None
        pipeline_checkpoint = os.path.join(model_checkpoint_dir, 'fasttext_pipeline.pkl')
        dump(self, pipeline_checkpoint)

        self.model_ = model
        return model_checkpoint_dir

    @classmethod
    def load(cls, directory: str):
        """
        Loads the full model from file.

        Parameters
        ----------
        directory : str
            The saved directory returned by calling .save.

        Returns
        -------
        model : FasttextPipeline
        """
        pipeline_checkpoint = os.path.join(directory, 'fasttext_pipeline.pkl')
        fasttext_pipeline = load(pipeline_checkpoint)

        model_checkpoint = os.path.join(directory, 'model.fasttext')
        model = fasttext.load_model(model_checkpoint)

        fasttext_pipeline.model_ = model
        return fasttext_pipeline

    def score_str(self, fasttext_file_path: str, k: int=1, round_digits: int=3) -> str:
        """
        Computes the model evaluation score for the input data and formats
        them into a string, making it easier for logging. This method calls
        score internally.

        Parameters
        ----------
        fasttext_file_path : str
            Path to the text file in the fasttext format.

        k : int, default 1
            Ranking metrics precision/recall/f1 are evaluated for top k prediction.

        round_digits : int, default 3
            Round decimal points for the metrics returned.

        Returns
        -------
        score_str : str
            e.g. ' metric - num_records: 29740, precision@1: 0.784, recall@1: 0.243, f1@1: 0.371'
        """
        num_records, precision_at_k, recall_at_k, f1_at_k = score(
            self.model_, fasttext_file_path, k, round_digits)

        num_records = f'num_records: {num_records}'
        precision_at_k = f'precision@{k}: {precision_at_k}'
        recall_at_k = f'recall@{k}: {recall_at_k}'
        f1_at_k = f'f1@{k}: {f1_at_k}'
        return f' metric - {num_records}, {precision_at_k}, {recall_at_k}, {f1_at_k}'

    def predict(self, texts: List[str], k: int=1,
                threshold: float=0.1,
                on_unicode_error: str='strict') -> List[List[Tuple[float, str]]]:
        """
        Given a list of raw text, predict the list of labels and corresponding probabilities.
        We can use k and threshold in conjunction to control to number of labels to return for
        each text in the input list.

        Parameters
        ----------
        texts : list[str]
            A list of raw text/string.

        k : int, default 1
            Controls the number of returned labels. 1 will return the top most probable labels.

        threshold : float, default 0.1
            This filters the returned labels that are lower than the specified probability.
            e.g. if k is specified to be 2, but once the returned probable labels has a probability
            lower than this threshold, then only 1 predicted labels will be returned.

        on_unicode_error : str, default 'strict'
            Controls the behavior when the input string can't be converted according to the
            encoding rule.

        Returns
        -------
        batch_predictions : list[list[tuple[float, str]]]
            e.g. [[(0.562, '__label__label1'), (0.362, '__label__label2')]]
        """

        # fasttext's own predict method doesn't work well when k and threshold is
        # specified together for batch prediction, this is due to the size of the
        # prediction returned for each text in the batch is not equal, hence we
        # roll out our own predict method to accommodate for this.

        # appending the new line at the end of the text is needed for fasttext prediction
        # note that it should be done after the tokenization to prevent the tokenizer
        # from modifying the new line symbol
        tokenized_texts = [text + '\n' for text in texts]
        batch_predictions = self.model_.f.multilinePredict(
            tokenized_texts, k, threshold, on_unicode_error)

        return batch_predictions


def fit_and_score(fasttext_file_path_train: str,
                  fasttext_file_path_test: str,
                  fasttext_params: Dict[str, Any],
                  k: int,
                  params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fits the fasttext model and computes the score for a given train and test split
    on a set of parameters.

    Parameters
    ----------
    fasttext_file_path_train : str
         The text file should already be in the fasttext expected format.
         This is used for training the model.

    fasttext_file_path_test : str
         The text file should already be in the fasttext expected format.
         This is used for testing the model on the holdout set.

    fasttext_params : dict
        The fixed set of parameters for fastttext.

    k : int
        Ranking metrics precision/recall/f1 are evaluated for top k prediction.

    params : dict
        The parameters that are tuned. Will over-ride any parameter that
        are specified in fasttext_params.

    Returns
    -------
    result : dict
        Stores the results for the current iteration e.g.::

            {
            'params': {'epoch': 10, 'dim': 85},
            'epoch': 10,
            'dim': 85,
            'train_precision@1': 0.486,
            'train_recall@1': 0.210,
            'train_f1@1': 0.294,
            'test_precision@1': 0.407,
            'test_recall@1': 0.175,
            'test_f1@1': 0.245
            }
    """
    current_model = fit_fasttext(fasttext_file_path_train, fasttext_params, params)

    fasttext_file_path_dict = {
        'train': fasttext_file_path_train,
        'test': fasttext_file_path_test
    }

    result = {'params': params}
    result.update(params)
    for group, fasttext_file_path in fasttext_file_path_dict.items():
        num_records, precision_at_k, recall_at_k, f1_at_k = score(
            current_model, fasttext_file_path, k)
        metric = {
            f'{group}_precision@{k}': precision_at_k,
            f'{group}_recall@{k}': recall_at_k,
            f'{group}_f1@{k}': f1_at_k
        }
        result.update(metric)

    return result


def fit_fasttext(fasttext_file_path: str,
                 fasttext_params: Dict[str, Any],
                 params: Dict[str, Any]) -> fasttext.FastText._FastText:
    """
    Fits a fasttext model.

    Parameters
    ----------
    fasttext_file_path : str
         The text file should already be in the fasttext expected format.

    fasttext_params : dict
        The fixed set of parameters for fastttext.

    params : dict
        The parameters that are tuned. Will over-ride any parameter that
        are specified in fasttext_params.

    Returns
    -------
    model : _FastText
        Trained fasttext model.
    """
    current_params = deepcopy(fasttext_params)
    current_params.update(params)
    current_params['input'] = fasttext_file_path
    model = fasttext.train_supervised(**current_params)
    return model


def score(model: fasttext.FastText._FastText,
          fasttext_file_path: str,
          k: int=1,
          round_digits: int=3) -> Tuple[int, float, float, float]:
    """
    Computes the model evaluation score including precision/recall/f1 at k
    for the input file.

    Parameters
    ----------
    model : _FastText
        Trained fasttext model.

    fasttext_file_path : str
        Path to the text file in the fasttext format.

    k : int, default 1
        Ranking metrics precision/recall/f1 are evaluated for top k prediction.

    round_digits : int, default 3
        Round decimal points for the metrics returned.

    Returns
    -------
    num_records : int
        Number of records in the file.

    precision_at_k : float

    recall_at_k : float

    f1_at_k : float
    """

    num_records, precision_at_k, recall_at_k = model.test(fasttext_file_path, k)
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    precision_at_k = round(precision_at_k, round_digits)
    recall_at_k = round(recall_at_k, round_digits)
    f1_at_k = round(f1_at_k, round_digits)
    return num_records, precision_at_k, recall_at_k, f1_at_k
