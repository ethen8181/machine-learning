# machine-learning

This is one of the continuously updated repositories that documents personal data science journey. Currently, contents are organized into two separate repositories based on the following table's description.

| Repository | Documentation Focus |
| ---------- | ----------- |
| [machine-learning](https://github.com/ethen8181/machine-learning) | Machine learning and programming in R / python. |
| [Business-Analytics](https://github.com/ethen8181/Business-Analytics) | All other data analytic related stuffs, e.g. concepts, statistics, articles, visualizations. |

Within each section, documentations are listed in reverse chronological order of the latest complete date and each of them are independent of one another unless specified.


## Documentation Listings

**regularization : 2016.5.25**

Regularization techniques: ridge and lasso regression. 

- Building intuition using scikit-learn, it's best if you already understand linear regression and cross validation. 
- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)]


**santander : 2016.5.16**

Kaggle competition on predicting customer satisfaction. The goal is to familiarize myself with python's xgboost and H2O's API. Note that the script's auc store 0.823068 is still far off from the best score for the competition, 0.829072.

- Includes scripts for performing cross validation with xgboost; Obtaining feature importance, saving and loading the model for xgboost and H2O. Details are commented in the following notebooks. 
- run_me.ipynb: Data preprocessing and xgboost. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/santander/run_me.ipynb)]
- h2o.ipynb: H2O's randomforest and gradient boosting on the already preprocessed data. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/santander/h2o.ipynb)]


**ga : 2016.4.25**

Using Genetic Algorithm to solve a simple optimization problem.

- Math-free explanation and python code from scratch. 
- View [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/ga/ga.ipynb)]


**h2o : 2016.1.24**

Walking through H2O 2015 World Training [GitBook](http://learn.h2o.ai/content/index.html). Since H2O provides progress bar when training the model, you’ll may see a lot of them in doc. The walkthrough does basically zero feature engineering with the example dataset as it is just browsing through its function calls and parameters.

- R's API
	- h2o’s deep learning. [[R markdown](http://ethen8181.github.io/machine-learning/h2o/h2o_deep_learning/h2o_deep_learning.html)]
	- h2o’s gradient boosting and random forest. [[R markdown](http://ethen8181.github.io/machine-learning/h2o/h2o_ensemble_tree/h2o_ensemble_tree.html)]
	- h2o’s generalized linear model. [[R markdown](http://ethen8181.github.io/machine-learning/h2o/h2o_glm/h2o_glm.html)]
	- h2o’s super learner. [[R code](https://github.com/ethen8181/machine-learning/blob/master/h2o/h2o_super_learner/h2o_super_learner.R)]
- Python's API
	- h2o's deep learning, gradient boosting and random forest. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/h2o/h2o_python.ipynb)]


**lda_1 : 2015.12.8** 

Performing text clustering with Latent Dirichlet Allocation(LDA).

- Contains a toy sample code of the algorithm (gibbs sampling) and usage of the topicmodels library.
- View [[R markdown](http://ethen8181.github.io/machine-learning/lda_1/lda_1.html)]


**logistic_regression : 2015.11.25** 

Focus on choosing the optimal cutoff value for logistic regression when your dataset has unbalanced binary classes. e.g. The majority of the observations in the dataset have a positive outcome, while few have negative, or vice versa. The idea can be extended to any other classification algorithm that can predict class’s probability.

- Visualize two by two standard confusion matrix and ROC curve with costs using ggplot2.
- View [[R markdown](http://ethen8181.github.io/machine-learning/logistic_regression/logistic_regression.html)]


**text_similarity : 2015.11.19** 

Illustrates k-shingle, Minhash and Locality Sensitive Hashing for solving the problem of finding textually similar documents. 

- View [[R markdown](http://ethen8181.github.io/machine-learning/text_similarity/text_similarity.html)]


**tf_idf : 2015.11.17** 

Introducing tf-idf ( term frequency-inverse document frequency ), a text mining technique. Also uses it to perform text clustering via hierarchical clustering.
 
- View [[R markdown](http://ethen8181.github.io/machine-learning/tf_idf/tf_idf.html)]


**clustering : 2015.11.6**

Some useful evaluations when working with hierarchical clustering and k-means clustering ( k-means++ is used here ).

- Calinski-Harabasz index : Determine the right k ( cluster number ) for clustering.
- Boostrap evaluation of the clustering result’s stability.
- View [[R markdown](http://ethen8181.github.io/machine-learning/clustering/clustering.html)]


**linear_regression : 2015.10.30**

Solving linear regression with gradient descent. 

- Briefly covers the interpretation and visualization of linear regression's summary output.
- View [[R markdown](http://ethen8181.github.io/machine-learning/linear_regression/linear_regession.html)]


## General Programming

**python**

- 2016.4.26 | Some pre-implemented data structure and algorithm. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/1_data_structure.ipynb)]
- 2016.4.26 | Tricks with strings and text. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/python3_cookbook/2_strings_and_text.ipynb)]
- 2016.4.17 | python's decorators (useful script for logging and timing function). [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/decorators/decorators.ipynb)]
- 2016.3.18 | pandas's pivot table. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/pivot_table/pivot_table.ipynb)]
- 2016.3.02 | @classmethod, @staticmethod and @property. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/class.ipynb)]
- 2016.2.22 | sorting with itemgetter and attrgetter. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/sorting_with_itemgetter.ipynb)]
- 2016.2.19 | for .. else .. statement. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/for_else.ipynb)] 
- 2016.2.18 | namedtuple and defaultdict. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/python/collections_module.ipynb)]


**R**

- 2016.4.15 | data.table joining and other tricks. [[R markdown](http://ethen8181.github.io/machine-learning/R/data_table/data_table.html)]


