# machine-learning

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/ethen8181/machine-learning/blob/master/LICENSE)
![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)

This is a continuously updated repository that documents personal journey on learning data science, machine learning related topics.

- **Goal:** Introduce machine learning contents in Jupyter Notebook format. The content aims to strike a good balance between mathematical notations, educational implementation from scratch (using Python's scientific stack including numpy, scipy, pandas, matplotlib etc.) and open-source library usage (scikit-learn, pyspark, gensim, keras, tensorflow).
- **Short Note:** Within each section, documentations are listed in reverse chronological order of the start date (the date when the first notebook in that folder was created, if the notebook happened to be updated, then the actual date will be at the top of each notebook). Each of them are independent of one another unless specified.


## Documentation Listings

#### projects : 2017.09.23

End to end project including data preprocessing, model building.

- [Kaggle: Don't Get Kicked](https://www.kaggle.com/c/DontGetKicked): Predict if a car purchased at auction is a unfortunate purchase. [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/kaggle_dont_get_kicked/)]
- mlutils: Machine learning utility function package [[folder](https://github.com/ethen8181/machine-learning/blob/master/projects/mlutils/)]

#### ab_tests : 2017.08.09

A/B testing, a.k.a experimental design. Includes 1) quick review of necessary statistic concepts. 2) methods and workflow/thought-process for conducting the test. 3) caveats to look out for.

- Frequentist A/B testing (includes a quick review of concepts such as p-value, confidence interval). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ab_tests/frequentist_ab_test.ipynb)]

#### model_selection : 2017.06.12

Methods for selecting, improving, evaluating models/algorithms.

- K-fold cross validation, grid/random search from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/model_selection.ipynb)]
- AUC (Area under the ROC curve and precision/recall curve) from scratch (includes the process of building a custom scikit-learn transformer). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/auc/auc.ipynb)]
- Detecting collinearity amongst features (Variance Inflation Factor for numeric features and Cramer's V statistics for categorical features), also introduces Linear Regression from a Maximum Likelihood perspective and the R-squared evaluation metric. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/collinearity.ipynb)]
- Curated tips and tricks for technical and soft skills. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/tips_and_tricks/tips_and_tricks.ipynb)]
- Partial Dependece Plot (PDP), model-agnostic approach for directional feature influence. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/model_selection/partial_dependence/partial_dependence.ipynb)]

#### big_data : 2017.06.07

Exploring big data tools, such as Spark and H2O.ai. For those interested there's also a [pyspark rdd cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_Cheat_Sheet_Python.pdf) and [pyspark dataframe cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_SQL_Cheat_Sheet_Python.pdf) that may come in handy.

- Local Hadoop cluster installation on Mac. [[markdown](https://github.com/ethen8181/machine-learning/tree/master/big_data/local_hadoop.md)]
- PySpark installation on Mac. [[markdown](https://github.com/ethen8181/machine-learning/tree/master/big_data/spark_installation.md)]
- Examples of manipulating with data (crimes data) and building a RandomForest model with PySpark MLlib. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/spark_crime.ipynb)]
- PCA with PySpark MLlib. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/spark_pca.ipynb)]
- H2O API walkthrough (using GBM as an example). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/big_data/h2o_api_walkthrough.ipynb)]

#### dim_reduct : 2017.01.02

Dimensionality reduction methods.

- Principal Component Analysis (PCA) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/PCA.ipynb)]
- Introduction to Singular Value Decomposition (SVD), also known as Latent Semantic Analysis/Indexing (LSA/LSI).  [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/svd.ipynb)]

#### recsys : 2016.12.17

Recommendation system with a focus on matrix factorization methods. Starters into the field should go through the first notebook to understand the basics of matrix factorization methods.

- Alternating Least Squares with Weighted Regularization (ALS-WR) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/1_ALSWR.ipynb)]
- ALS-WR for implicit feedback data from scratch & Mean Average Precision at k (mapk) and Normalized Cumulative Discounted Gain (ndcg) evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/2_implicit.ipynb)]
- Bayesian Personalized Ranking (BPR) from scratch & AUC evaluation. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/4_bpr.ipynb)]
- Factorization Machine from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/factorization_machine/factorization_machine.ipynb)]

#### trees : 2016.12.10

Tree-based models for both regression and classification tasks.

- Decision Tree from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/decision_tree.ipynb)]
- Random Forest from scratch and Extra Trees. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/random_forest.ipynb)]
- Gradient Boosting Machine (GBM) from scratch. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/gbm/gbm.ipynb)
- Xgboost API walkthrough (includes hyperparmeter tuning via scikit-learn like API). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/xgboost.ipynb)]
- LightGBM API walkthrough and a discussion about categorical features in tree-based models. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/lightgbm.ipynb)]

#### association_rule : 2016.09.16

Also known as market-basket analysis.

- Apriori from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/association_rule/apriori.ipynb)]
- Using R's arules package (aprori) on tabular data. [[Rmarkdown](http://ethen8181.github.io/machine-learning/association_rule/R/apriori.html)]

#### clustering : 2016.08.16

TF-IDF and Topic Modeling are techniques specifically used for text analytics.

- TF-IDF (text frequency - inverse document frequency) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/tfidf/tfidf.ipynb)]
- K-means, K-means++ from scratch; Elbow method for choosing K. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/kmeans.ipynb)]
- Gaussian Mixture Model from scratch; AIC and BIC for choosing the number of Gaussians. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/GMM/GMM.ipynb)]
- Topic Modeling with gensim's Latent Dirichlet Allocation(LDA). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/topic_model/LDA.ipynb)]

#### data_science_is_software : 2016.08.01  

Best practices for doing data science in Python.

- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/data_science_is_software/notebooks/data_science_is_software.ipynb)]

#### deep_learning : 2016.07.23

Curated notes on deep learning. [Tensorflow](https://www.tensorflow.org/) is used to implement some of the models.

- Softmax Regression from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/softmax.ipynb)]
- Softmax Regression using Tensorflow (Tensorflow hello world). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/softmax_tensorflow.ipynb)]
- Multi-layers Neural Network using Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/nn_tensorflow.ipynb)]
- Convolutional Neural Network using Tensorflow. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/cnn_image_tensorflow.ipynb)]
- Word2vec (skipgram + negative sampling) using Gensim (includes text preprocessing with spaCy). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/word2vec/word2vec_detailed.ipynb)]

#### keras : 2016.06.29

Walking through [keras](https://github.com/fchollet/keras), a high-level deep learning library. Note that this is only a API walkthrough, not a tutorial on the details of deep learning. For those interested there's also a [keras cheatsheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Keras_Cheat_Sheet_Python.pdf) that may come in handy.

- Multi-layers Neural Network (keras basics). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/nn_keras_basics.ipynb)]
- Multi-layers Neural Network hyperparameter tuning via scikit-learn like API. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/nn_keras_hyperparameter_tuning.ipynb)]
- Convolutional Neural Network (image classification). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/cnn_image_keras.ipynb)]

#### text_classification : 2016.06.15

Naive Bayes and Logistic Regression for text classification.

- Building intuition with spam classification using scikit-learn (scikit-learn hello world). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/basics/basics.ipynb)]
- Bernoulli and Multinomial Naive Bayes from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/naive_bayes/naive_bayes.ipynb)]
- Logistic Regression (stochastic gradient descent) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/logistic.ipynb)]
- Chi-square feature selection. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/chisquare.ipynb)]

#### networkx : 2016.06.13

PyCon 2016: Practical Network Analysis Made Simple. Quickstart to networkx's API. Includes some basic graph plotting and algorithms.

- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/networkx/networkx.ipynb)]

#### regularization : 2016.05.25

Building intuition on Ridge and Lasso regularization using scikit-learn.
 
- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)]

#### ga : 2016.04.25

Genetic Algorithm. Math-free explanation and code from scratch.

- Start from a simple optimization problem and extending it to traveling salesman problem (tsp).
- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ga/ga.ipynb)]

#### unbalanced : 2015.11.25

Choosing the optimal cutoff value for logistic regression using cost-sensitive mistakes (meaning when the cost of misclassification might differ between the two classes) when your dataset consists of unbalanced binary classes. e.g. Majority of the data points in the dataset have a positive outcome, while few have negative, or vice versa. The notion can be extended to any other classification algorithm that can predict class’s probability, this documentation just uses logistic regression for illustration purpose.

- Visualize two by two standard confusion matrix and ROC curve with costs using ggplot2.
- View [[Rmarkdown](http://ethen8181.github.io/machine-learning/unbalanced/unbalanced.html)]

#### clustering_old

A collection of scattered old clustering documents in R.

- 2015.12.08 | Toy sample code of the LDA algorithm (gibbs sampling) and the topicmodels library. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/topic_model/LDA.html)]
- 2015.11.19 | k-shingle, Minhash and Locality Sensitive Hashing for solving the problem of finding textually similar documents. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/text_similarity/text_similarity.html)]
- 2015.11.17 | Introducing tf-idf (term frequency-inverse document frequency), a text mining technique. Also uses it to perform text clustering via hierarchical clustering. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/tf_idf/tf_idf.html)]
- 2015.11.06 | Some useful evaluations when working with hierarchical clustering and K-means clustering (K-means++ is used here). Including Calinski-Harabasz index for determine the right K (cluster number) for clustering and boostrap evaluation of the clustering result’s stability. [[Rmarkdown](http://ethen8181.github.io/machine-learning/clustering_old/clustering/clustering.html)]

#### linear_regression : 2015.10.30

Training Linear Regression with gradient descent in R. 

- Briefly covers the interpretation and visualization of linear regression's summary output.
- View [[Rmarkdown](http://ethen8181.github.io/machine-learning/linear_regression/linear_regession.html)]


## Python Programming

- 2018.01.20 | Parallel programming with Python (threading, multiprocessing, concurrent.futures, joblib). [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/parallel.ipynb)
- 2017.08.23 | Understanding iterables, iterator and generators. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/iterator/iterator.ipynb)
- 2017.07.12 | Cohort analysis. Visualizing user retention by cohort with seaborn's heatmap and illustrating pandas's unstack. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cohort/cohort.ipynb)
- 2017.03.16 | Logging module. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/logging.ipynb)
- 2016.12.26 | Data structure, algorithms from scratch. [[folder](https://github.com/ethen8181/machine-learning/tree/master/python/algorithms)]
- 2016.12.22 | Cython and Numba quickstart for high performance Python. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cython/cython.ipynb)
- 2016.06.22 | Optimizing Pandas (e.g. reduce memory usage using category type). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pandas/pandas.ipynb)]
- 2016.06.10 | Unittest. [[Python script](https://github.com/ethen8181/machine-learning/blob/master/python/test.py)]
- 2016.04.26 | Using built-in data structure and algorithm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/1_data_structure.ipynb)]
- 2016.04.26 | Tricks with strings and text. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/2_strings_and_text.ipynb)]
- 2016.04.17 | Python's decorators (useful script for logging and timing function). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/decorators/decorators.ipynb)]
- 2016.03.18 | Pandas's pivot table. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pivot_table/pivot_table.ipynb)]
- 2016.03.02 | @classmethod, @staticmethod and @property. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/class.ipynb)]

