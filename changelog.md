# Changelog

The changelog will record what content was **changed** (e.g. changed an existing paragraph to a better-explained version, re-ran the notebook using an updated version of the package), **added** (e.g. a completely new jupyter notebook).

## [2018-03]

### Added

- Local Hadoop cluster installation on Mac. [[markdown](https://github.com/ethen8181/machine-learning/tree/master/big_data/local_hadoop.md)]
- Spark MLlib Binary Classification (using GBM as an example). [[raw zeppelin notebook](https://github.com/ethen8181/machine-learning/blob/master/big_data/sparkml/sparkml.json)][[Zepl](https://www.zepl.com/explore)]


## [2018-02]

### Added

- H2O API walkthrough (using GBM as an example). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/h2o_api_walkthrough.ipynb)]
- Factorization Machine from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/factorization_machine/factorization_machine.ipynb)]

### Changed

- The `spark` folder has been renamed to `big_data` to incorporate other big data tools.


## [2018-01]

### Added

- Partial Dependece Plot (PDP), model-agnostic approach for directional feature influence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/partial_dependence/partial_dependence.ipynb)]
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
