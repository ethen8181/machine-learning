# Changelog

The changelog will record what content was **changed** (e.g. changed an existing paragraph to a better-worded version, re-ran the notebook using an updated version of the package, introduced new content to existing notebook), **added** (e.g. a completely new jupyter notebook).

## [2020-09]

### Changed

- Probability Calibration for classification models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/prob_calibration.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html)]
    + Massive overhaul to the content. e.g. introducing two additional calibration methods, histogram binning and Plat Scaling Binning. Bundling all helper utility function in a package structure for ease of re-use.
- MultiLabel Text Classification with Fasttext and Huggingface Tokenizers. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/fasttext.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/fasttext.html)]
    + Update Huggingface Tokenizers to 0.8.1 API.

## [2020-08]

### Changed

- Probability Calibration for classification models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/prob_calibration.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html)]
    + Fixed an error in the calibration graph.

## [2020-06]

### Added

- Approximate Nearest Neighborhood Search with Navigable Small World. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/nsw.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/nsw.html)]

## [2020-05]

### Added

- Product Quantization for Model Compression. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/product_quantization.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html)]
- Maximum Inner Product for Speeding Up Generating Recommendations. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/max_inner_product/max_inner_product.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/max_inner_product/max_inner_product.html)]

## [2020-04]

### Added

- Extremely Quick Guide to Unicode. [[markdown](https://github.com/ethen8181/machine-learning/blob/master/python/unicode.md)]
- MultiLabel Text Classification with Fasttext and Huggingface Tokenizers. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/multi_label/fasttext.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/multi_label/fasttext.html)]

### Changed

- FastAPI & Azure Kubernetes Cluster. End to end example of training a model and hosting it as a service. [[folder](https://github.com/ethen8181/machine-learning/blob/master/model_deployment/fastapi_kubernetes)]
    - Added application load testing with Apache Jmeter.

## [2020-03]

### Changed

- FastAPI & Azure Kubernetes Cluster. End to end example of training a model and hosting it as a service. [[folder](https://github.com/ethen8181/machine-learning/blob/master/model_deployment/fastapi_kubernetes)]
    - Added more best practices when specifying a deployment.

## [2020-02]

### Added

- FastAPI & Azure Kubernetes Cluster. End to end example of training a model and hosting it as a service. [[folder](https://github.com/ethen8181/machine-learning/blob/master/model_deployment/fastapi_kubernetes)]

### Changed

- Parallel programming with Python (threading, multiprocessing, concurrent.futures, joblib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/parallel.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/parallel.html)]
    - Added a short section to asynchronous programming.
- Monotonic Constraint with Boosted Tree. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/monotonic.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/monotonic.html)]
    - The original notebook uses xgboost to demonstrate the feature. Added lightgbm example.
- Logging module. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/logging.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/logging.html)]
    - Added a section that emphasizes the importance of logging the full stack trace of an exception.

## [2020-01]

### Added

- [Kaggle: Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification/): Predicting insincere questions. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_quora_insincere/)]

### Changed

- Seq2Seq for German to English Machine Translation - PyTorch. Includes quick intro to torchtext [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/1_torch_seq2seq_intro.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/1_torch_seq2seq_intro.html)]
    - Added more introduction to torchtext.

## [2019-12]

### Added

- Byte Pair Encoding (BPE) from scratch and quick walkthrough of sentencepiece. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/subword/bpe.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html)]
- Sentencepiece Subword tokenization for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_subword_tokenization.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/keras_subword_tokenization.html)]

### Changed

- Gaussian Mixture Model from scratch; AIC and BIC for choosing the number of Gaussians. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/GMM/GMM.ipynb)][[html](http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html)]
    - Fix erroneous log likelihood calculation.
    - Update deprecated function for plotting contour plots.

## [2019-11]

### Added

- Leveraging Pre-trained Word Embedding for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/keras_pretrained_embedding.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/keras_pretrained_embedding.html)]
- Monotonic Constraint with Boosted Tree. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/monotonic.ipynb)][[html](http://ethen8181.github.io/machine-learning/trees/monotonic.html)]
- Probability Calibration for classification models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/prob_calibration/prob_calibration.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html)]

## [2019-10]

### Added

- Seq2Seq with Attention for German to English Machine Translation - PyTorch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/2_torch_seq2seq_attention.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/2_torch_seq2seq_attention.html)]

## [2019-09]

### Added

- Seq2Seq with PyTorch for German to English Machine Translation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/seq2seq/1_torch_seq2seq_intro.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/seq2seq/1_torch_seq2seq_intro.html)]

## [2019-08]

### Added

- [Kaggle: Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales/): Predicting daily store sales. Also introduces deep learning for tabular data. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_rossman_store_sales/)]

### Changed

- Optimizing Pandas (e.g. reduce memory usage using category type). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pandas/pandas.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/pandas/pandas.html)]
	+ Added helper function to automatically determine optimal data type.
- Framing time series problem as supervised-learning. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/3_supervised_time_series.ipynb)][[html](http://ethen8181.github.io/machine-learning/time_series/3_supervised_time_series.html)]
	+ Added window-based features.

## [2019-06]

### Added

- Word2vec for Text Classification. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/text_classification/word2vec_text_classification.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/text_classification/word2vec_text_classification.html)]

### Changed

- Word2vec (skipgram + negative sampling) using Gensim. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/word2vec/word2vec_detailed.ipynb)][[html](http://ethen8181.github.io/machine-learning/deep_learning/word2vec/word2vec_detailed.html)]
	- Update to the more efficient file-based training.  

## [2019-04]

- Propensity Score Matching. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/causal_inference/matching.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/causal_inference/matching.html)]

## [2019-03]

### Added

- Short Walkthrough of PageRank. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/page_rank.ipynb)][[html](http://ethen8181.github.io/machine-learning/networkx/page_rank.html)]

## [2019-02]

### Added

- Quick Example of Factory Design Pattern. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/factory_pattern.ipynb)][[html](http://ethen8181.github.io/machine-learning/python/factory_pattern.html)]
- Introduction to Multi-armed Bandits. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/bandits/multi_armed_bandits.ipynb)][[html](http://ethen8181.github.io/machine-learning/bandits/multi_armed_bandits.html)]

## [2019-01]

### Added

- Quantile Regression and its application in A/B testing.
  - Quick Introduction to Quantile Regression. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/quantile_regression/quantile_regression.ipynb.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.ipynb.html)]
  - Quantile Regression's application in A/B testing. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/quantile_regression/ab_test_regression.ipynb)][[html](http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/ab_test_regression.html)]

## [2018-12]

### Added

- First Foray Into Discrete/Fast Fourier Transformation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/fft/fft.ipynb)][[html](http://ethen8181.github.io/machine-learning/time_series/fft/fft.html)]

## [2018-11]

### Added

- Introduction to BM25 (Best Match). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/search/bm25_intro.ipynb.ipynb)][[html](http://ethen8181.github.io/machine-learning/search/bm25_intro.ipynb.html)]

## [2018-10]

### Added

- Kullback-Leibler (KL) Divergence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/kl_divergence.ipynb)][[html](http://ethen8181.github.io/machine-learning/model_selection/kl_divergence.html)]
- Calibrated Recommendation for reducing bias/increasing diversity in recommendation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/calibration/calibrated_reco.ipynb)][[html](http://ethen8181.github.io/machine-learning/recsys/calibration/calibrated_reco.html)]
- Influence Maximization from scratch. Includes discussion on Independent Cascade (IC), Submodular Optimization algorithms including Greedy and Lazy Greedy, a.k.a Cost Efficient Lazy Forward (CELF) [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/max_influence/max_influence.ipynb)][[html](http://ethen8181.github.io/machine-learning/networkx/max_influence/max_influence.html)]

## [2018-09]

### Added

Introduction to Residual Networks (ResNets) and Class Activation Maps (CAM). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/resnet_cam/resnet_cam.ipynb)][[html](http://ethen8181.github.io/machine-learning/keras/resnet_cam/resnet_cam.html)]

### Changed

Hosted html-version of all jupyter notebook on github pages.

## [2018-08]

### Added

- (Text) Content-Based Recommenders. Introducing Approximate Nearest Neighborhood (ANN) - Locality Sensitive Hashing (LSH) for cosine distance from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/content_based/lsh_text.ipynb)]
- Benchmarking ANN implementations (nmslib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/ann_benchmarks/ann_benchmarks.ipynb)]

## [2018-07]

### Added

- Getting started with time series analysis with Exponential Smoothing (Holt-Winters). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/1_exponential_smoothing.ipynb)]
- Framing time series problem as supervised-learning. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/time_series/3_supervised_time_series.ipynb)]
- Tuning Spark Partitions. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/spark_partitions.ipynb)]

## [2018-06]

### Added

- Evaluation metrics for imbalanced dataset. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/imbalanced/imbalanced_metrics.ipynb)]

### Changed

- H2O API walkthrough (using GBM as an example). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/h2o/h2o_api_walkthrough.ipynb)]
    + Moved H2O notebook to its own sub-folder.
    + Added model interpretation using partial dependence plot.

## [2018-05]

### Added

- RNN, LSTM - PyTorch hello world. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/1_pytorch_rnn.ipynb)]
- Recurrent Neural Network (RNN) - language modeling basics. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/rnn_language_model_basic_keras.ipynb)]

## [2018-04]

### Added

- Long Short Term Memory (LSTM) - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/2_tensorflow_lstm.ipynb)]
- Vanilla RNN - Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/rnn/1_tensorflow_rnn.ipynb)]
- WARP (Weighted Approximate-Rank Pairwise) Loss using lightfm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/5_warp.ipynb)]

## [2018-03]

### Added

- Local Hadoop cluster installation on Mac. [[markdown](https://github.com/ethen8181/machine-learning/tree/master/big_data/local_hadoop.md)]
- Spark MLlib Binary Classification (using GBM as an example). [[raw zeppelin notebook](https://github.com/ethen8181/machine-learning/blob/master/big_data/sparkml/sparkml.json)][[Zepl](https://www.zepl.com/explore)]


## [2018-02]

### Added

- H2O API walkthrough (using GBM as an example). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/h2o/h2o_api_walkthrough.ipynb)]
- Factorization Machine from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/factorization_machine/factorization_machine.ipynb)]

### Changed

- The `spark` folder has been renamed to `big_data` to incorporate other big data tools.


## [2018-01]

### Added

- Partial Dependence Plot (PDP), model-agnostic approach for directional feature influence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/partial_dependence/partial_dependence.ipynb)]
- Parallel programming with Python (threading, multiprocessing, concurrent.futures, joblib). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/parallel.ipynb)]

## [2017-12]

### Added

- LightGBM API walkthrough and a discussion about categorical features in tree-based models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/lightgbm.ipynb)]
- Curated tips and tricks for technical and soft skills. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/tips_and_tricks/tips_and_tricks.ipynb)]
- Detecting collinearity amongst features (Variance Inflation Factor for numeric features and Cramer's V statistics for categorical features), also introduces Linear Regression from a Maximum Likelihood perspective and the R-squared evaluation metric. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/collinearity.ipynb)]

### Changed

- Random Forest from scratch and Extra Trees. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/random_forest.ipynb)]
    - Refactored code for visualizating tree's feature importance.
- Building intuition on Ridge and Lasso regularization using scikit-learn. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)]
    - Include section when there are collinear features in the dataset.
- mlutils: Machine learning utility function package [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/mlutils/)]
    - Refer to its changelog for details.
- data_science_is_software. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/data_science_is_software/notebooks/data_science_is_software.ipynb)]
    - Mention notebook extension, a project that contains various functionalities that makes jupyter notebook even more pleasant to work with.

## [2017-11]

### Added

- Introduction to Singular Value Decomposition (SVD), also known as Latent Semantic Analysis/Indexing (LSA/LSI).  [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/svd.ipynb)]

## [2017-10]

### Added

- mlutils: Machine learning utility function package [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/mlutils/)]

### Changed

- Bernoulli and Multinomial Naive Bayes from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/naive_bayes/naive_bayes.ipynb)]
    - Fixed various typos and added a more efficient implementation of Multinomial Naive Bayes.
- TF-IDF (text frequency - inverse document frequency) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/tfidf/tfidf.ipynb)]
    - Moved to its own tfidf folder.
    - Included the full tfidf implementation from scratch.

## [2017-09]

### Added

- [Kaggle challenge](https://www.kaggle.com/c/DontGetKicked): Predict if a car purchased at auction is a unfortunate purchase. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/data_challenge/)]

### Changed

- Using built-in data structure and algorithm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/1_data_structure.ipynb)]
    - Merged the content from the two notebooks `namedtuple and defaultdict` and `sorting with itemgetter and attrgetter` into this one and improved the section on priority queue.

## [2017-08]

### Added

- Understanding iterables, iterator and generators. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/iterator/iterator.ipynb)]
- Word2vec (skipgram + negative sampling) using Gensim (includes text preprocessing with spaCy). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/word2vec/word2vec_detailed.ipynb)]
- Frequentist A/B testing (includes a quick review of concepts such as p-value, confidence interval). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/frequentist_ab_test.ipynb)]
- AUC (Area under the ROC, precision/recall curve) from scratch (includes building a custom scikit-learn transformer). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/auc/auc.ipynb)]

### Changed

- Optimizing Pandas (e.g. reduce memory usage using category type). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pandas/pandas.ipynb)]
    - This is a revamp of the old content Pandas's category type.


## [2017-07]

### Added

- cohort : Cohort analysis. Visualize user retention by cohort with seaborn's heatmap and illustrating pandas's unstack. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cohort/cohort.ipynb)

### Changed

- Bayesian Personalized Ranking (BPR) from scratch & AUC evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/4_bpr.ipynb)]
    - A more efficient matrix operation using Hadamard product.
- Cython and Numba quickstart for high performance python. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cython/cython.ipynb)
    - Added Numba parallel prange.
- ALS-WR for implicit feedback data from scratch & mean average precision at k (mapk) and normalized cumulative discounted gain (ndcg) evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/2_implicit.ipynb)]
    - Included normalized cumulative discounted gain (ndcg) evaluation.
- Gradient Boosting Machine (GBM) from scratch. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/gbm/gbm.ipynb)
    - Added a made up number example on how GBM works.
- data_science_is_software. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/data_science_is_software/notebooks/data_science_is_software.ipynb)]
    - Mention `nbdime`, a tool that makes checking changes in jupyter notebook on github a lot easier.
    - Mention semantic versioning (what each number in the package version usually represents).
    - Mention `configparser`, a handy library for storing and loading configuration files.
- K-fold cross validation, grid/random search from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/model_selection.ipynb)]
    - Minor change in Kfolds educational implementation (original was passing redundant arguments to a method).
    - Minor change in random search educational implementation (did not realize scipy's .rvs method for generating random numbers returns a single element array instead of a number when you pass in size = 1).


## [2017-06]

This is the first time that the changelog file is added, thus every existing notebook will fall under the added category. Will try to group the log by month (one or two) in the future. Note that this repo will be geared towards Python3. Hence, even though the repo contains some R-related contents, they are not that well maintained and will most likely be translated to Python3. As always, any feedbacks are welcomed.

### Added

- Others (Genetic Algorithm)
- Regression (Linear, Ridge/Lasso)
- Market Basket Analysis (Apriori)
- Clustering (K-means++, Gaussian Mixture Model)
- Deep Learning (Feedforward, Convolutional Neural Nets)
- Model Selection (Cross Validation, Grid/Random Search)
- Dimensionality Reduction (Principal Component Analysis)
- Classification (Logistic, Bernoulli and Multinomial Naive Bayes)
- Text Analysis (TF-IDF, Chi-square feature selection, Latent Dirichlet Allocation)
- Tree Models (Decision Tree, Random Forest, Extra Trees, Gradient Boosting Machine)
- Recommendation System (Alternating Least Squares with Weighted Regularization, Bayesian Personalized Ranking)
- Python Programming (e.g. logging, unittest, decorators, pandas category type)
