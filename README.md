# machine-learning

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/ethen8181/machine-learning/blob/master/LICENSE)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/ethen8181/machine-learning)

This is one of the continuously updated repositories that documents personal journey on learning data science related topics. Currently, contents are organized into two separate repositories based on the following table's description.

| Repository | Documentation Focus |
| ---------- | ------------------- |
| [machine-learning](https://github.com/ethen8181/machine-learning) | Machine learning, algorithm and programming (mainly in Python) |
| [Business-Analytics](https://github.com/ethen8181/Business-Analytics) | All other data analytic related stuffs, e.g. concepts, statistics, visualizations (R, Python) |

Within each section, documentations are listed in reverse chronological order of the start date and each of them are independent of one another unless specified.


## Documentation Listings

#### dim_reduct : 2017.01.02

Dimensionality reduction methods.

- Principal Component Analysis (PCA) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/PCA.ipynb)]
- PCA with Spark's MLlib. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/dim_reduct/PCA_spark.ipynb)]

#### recsys : 2016.12.17

Recommendation System.

- Alternating Least Squares. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/recsys/ALS.ipynb)]

#### trees : 2016.12.10

Tree-based models for both regression and classification tasks.

- Decision Tree from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/decision_tree.ipynb)]
- Random Forest from scratch and Extra Trees. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/random_forest.ipynb)]
- Gradient Boosting from scratch. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/gbm/gbm.ipynb)
- Xgboost API walkthrough (includes hyperparmeter tuning via scikit-learn like API). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/trees/xgboost.ipynb)]

#### association_rule : 2016.09.16

Also known as market-basket analysis.

- Apriori from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/association_rule/apriori.ipynb)]
- Using R's arules package (aprori) on tabular data. [[Rmarkdown](http://ethen8181.github.io/machine-learning/association_rule/R/apriori.html)]

#### clustering : 2016.08.16

TF-IDF and Topic Modeling are techniques specifically used for text analytics.

- TF-IDF (text frequency - inverse document frequency) from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/TFIDF.ipynb)]
- K-means, K-means++ from scratch; Elbow method for choosing K. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/kmeans.ipynb)]
- Gaussian Mixture Model from scratch; AIC and BIC for choosing the number of Gaussians. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/GMM/GMM.ipynb)]
- Topic Modeling with gensim's Latent Dirichlet Allocation(LDA). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/clustering/topic_model/LDA.ipynb)]

#### data_science_is_software : 2016.08.01  

SciPy 2016: Data Science is Software. Best practices for doing data science (in Python).

- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/data_science_is_software/notebooks/data_science_is_software.ipynb)]

#### deep_learning : 2016.07.23

Curated notes on deep learning. [Tensorflow](https://www.tensorflow.org/) is used to implement some of the models.

- Softmax Regression from scratch. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/softmax.ipynb)]
- Softmax Regression using Tensorflow (includes Tensorflow hello world). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/softmax_tensorflow.ipynb)]
- Multi-layers Neural Network. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/nn_tensorflow.ipynb)]
- Convolutional Neural Network. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/deep_learning/cnn_image_tensorflow.ipynb)]

#### keras : 2016.06.29

Walking through [keras](https://github.com/fchollet/keras), a deep learning library. Note that this is only a API walkthrough, NOT a tutorial on the details of deep learning.

- Multi-layers Neural Network (keras basics). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/nn_keras_basics.ipynb)]
- Multi-layers Neural Network hyperparameter tuning via scikit-learn like API. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/nn_keras_hyperparameter_tuning.ipynb)]
- Convolutional Neural Network (image classification). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/cnn_image_keras.ipynb)]
- Convolutional Neural Network and Glove word embedding (text classification). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/keras/cnn_glove/cnn_glove.ipynb)]

#### text_classification : 2016.06.15

Naive Bayes and Logistic Regression for text classification.

- Building intuition with spam classification using scikit-learn. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/basics/basics.ipynb)]
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

#### h2o : 2016.01.24

Walking through [H2O 2015 World Training GitBook](http://learn.h2o.ai/content/index.html).The walkthrough does basically zero feature engineering with the example dataset, as it is just browsing through its function calls and parameters. Apart from that, [H2o Resources](http://www.h2o.ai/resources/) also contains booklets on each of the models.

- R's API:
	- h2o’s deep learning. [[Rmarkdown](http://ethen8181.github.io/machine-learning/h2o/h2o_deep_learning/h2o_deep_learning.html)]
	- h2o’s Ensemble Tree. [[Rmarkdown](http://ethen8181.github.io/machine-learning/h2o/h2o_ensemble_tree/h2o_ensemble_tree.html)]
	- h2o’s Generalized Linear Model. [[Rmarkdown](http://ethen8181.github.io/machine-learning/h2o/h2o_glm/h2o_glm.html)]
	- h2o’s super learner. [[R code](https://github.com/ethen8181/machine-learning/blob/master/h2o/h2o_super_learner/h2o_super_learner.R)]
- Python's API:
	- h2o's deep learning, Ensemble Tree. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/h2o/h2o_python.ipynb)]

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

- 2016.12.26 | Walking through the free online tutorial, [Problem Solving with Algorithms and Data Structures](http://interactivepython.org/runestone/static/pythonds/index.html), that introduces basic data structure, algorithms from scratch. [[folder](https://github.com/ethen8181/machine-learning/tree/master/python/algorithms)]
- 2016.12.22 | Cython and Numba quickstart for high performance python. [[nbviewer]](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/cython/cython.ipynb)
- 2016.06.22 | pandas's category type. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pandas_category.ipynb)]
- 2016.06.10 | unittest. [[python code](https://github.com/ethen8181/machine-learning/blob/master/python/test.py)]
- 2016.04.26 | Some pre-implemented data structure and algorithm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/1_data_structure.ipynb)]
- 2016.04.26 | Tricks with strings and text. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/2_strings_and_text.ipynb)]
- 2016.04.17 | python's decorators (useful script for logging and timing function). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/decorators/decorators.ipynb)]
- 2016.03.18 | pandas's pivot table. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pivot_table/pivot_table.ipynb)]
- 2016.03.02 | @classmethod, @staticmethod and @property. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/class.ipynb)]
- 2016.02.22 | sorting with itemgetter and attrgetter. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/sorting_with_itemgetter.ipynb)]
- 2016.02.19 | for .. else .. statement. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/for_else.ipynb)] 
- 2016.02.18 | namedtuple and defaultdict. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/collections_module.ipynb)]

