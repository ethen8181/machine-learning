# machine-learning

## Overall Description

This is one of the continuously updated repositories that documents personal data science journey. Currently, contents are organized into different repositories based on the following table's description.

| Repository | Documentation Focus |
| ---------- | ----------- |
| [machine-learning](https://github.com/ethen8181/machine-learning) | Machine learning in R. |
| [programming](https://github.com/ethen8181/programming) | Machine learning in python. |
| [Business-Analytics](https://github.com/ethen8181/Business-Analytics) | All the other data analytic related stuffs that are not solely about machine-learning. |
| [database](https://github.com/ethen8181/database) | Database (MySQL). | 

## Documentation Listings

Listed in chronological order of the latest complete date. Each folder’s work is independent of one another. 

**h2o : 2016.1.24**

Walking through H2O 2015 World Training [GitBook](http://learn.h2o.ai/content/index.html). Since H2O provides progress bar when training the model, you’ll may see a lot of them in doc. The walkthrough does basically zero feature engineering with the example dataset as it is just browsing through its function calls and parameters.

- h2o’s deep learning. [[R markdown](http://ethen8181.github.io/machine-learning/h2o/h2o_deep_learning/h2o_deep_learning.html)]
- h2o’s gradient boosting and random forest. [[R markdown](http://ethen8181.github.io/machine-learning/h2o/h2o_ensemble_tree/h2o_ensemble_tree.html)]
- h2o’s generalized linear model. [[R markdown](http://ethen8181.github.io/machine-learning/h2o/h2o_glm/h2o_glm.html)]
- h2o's deep learning, gradient boosting and random forest. [[nbviewer](http://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/h2o/h2o_python.ipynb)]
- h2o’s super learner. [[R code](https://github.com/ethen8181/machine-learning/blob/master/h2o/h2o_super_learner/h2o_super_learner.R)]

**lda_1 : 2015.12.8** 

Performing text clustering with Latent Dirichlet Allocation(LDA).

- Contains a toy sample code of the algorithm (gibbs sampling) and compare results with the *topicmodels* library.
- View [[R markdown](http://ethen8181.github.io/machine-learning/lda_1/lda_1.html)]

**logistic_regression : 2015.11.25** 

Focus on choosing the optimal cutoff value for logistic regression when your dataset has unbalanced binary classes. e.g. The majority of the observations in the dataset have a positive outcome, while few have negative, or vice versa. The idea can be extended to any other classification algorithm that can predict class’s probability.

- Visualize two by two standard confusion matrix and ROC curve with costs using the ggplot2 library.
- View [[R markdown](http://ethen8181.github.io/machine-learning/logistic_regression/logistic_regression.html)]

**text_similarity : 2015.11.19** 

Illustrates k-shingle, Minhash and locality sensitive hashing for solving the problem of finding textually similar documents. 

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

- Includes an appendix that briefly covers the interpretation and visualization of linear regression model’s summary output.
- View [[R markdown](http://ethen8181.github.io/machine-learning/linear_regression/linear_regession.html)]


