# machine-learning

### Repository Description:

This repository focuses on documenting machine learning and data mining techniques ( Stuffs that you’ll generally see in a machine learning or data mining book ). Each folder’s work is independent of one another. 

Another repository [Business-Analytics](https://github.com/ethen8181/Business-Analytics) focuses more on documenting data science or business analytic related approaches.

### Documentation Listings: 

Listed in chronological order of the latest complete date.

**h2o : 2016.1.24**

Walking through H2O 2015 World Training [GitBook](http://learn.h2o.ai/content/index.html). Since H2O provides progress bar when training the model, you’ll may see a lot of them in the quick walkthrough. The walkthrough basically does zero feature engineering with the example dataset is just browsing through its function calls and parameters.

- View quick walkthrough of using h2o deep learning [here](http://ethen8181.github.io/machine-learning/h2o/h2o_deep_learning/h2o_deep_learning.html).
- View quick walkthrough of using h2o gradient boosting and random forest [here](http://ethen8181.github.io/machine-learning/h2o/h2o_ensemble_tree/h2o_ensemble_tree.html).
- View quick walkthrough of using h2o’s generalized linear model [here](http://ethen8181.github.io/machine-learning/h2o/h2o_glm/h2o_glm.html).
- View quick walkthrough of using h2o’s super learner in plain R code [here](https://github.com/ethen8181/machine-learning/blob/master/h2o/h2o_super_learner/h2o_super_learner.R).

**lda_1 : 2015.12.8** 

Performing text clustering with Latent Dirichlet Allocation(LDA) using gibbs sampling.

- Contains a toy sample code of the algorithm and compare results with the *topicmodels* library.
- View documentation [here](http://ethen8181.github.io/machine-learning/lda_1/lda_1.html).

**logistic_regression : 2015.11.25** 

Focus on choosing the optimal cutoff value for logistic regression when your dataset has unbalanced binary classes. e.g. The majority of the observations in the dataset have a positive outcome, while few have negative, or vice versa. The idea can be extended to any other classification algorithm that can predict class’s probability.

- Visualize two by two standard confusion matrix and ROC curve with the ggplot2 library.
- View documentation [here](http://ethen8181.github.io/machine-learning/logistic_regression/logistic_regression.html).

**text_similarity : 2015.11.19** 

Illustrates k-shingle, Minhash and locality sensitive hashing for solving the problem of finding textually similar documents. 

- View detail [here](http://ethen8181.github.io/machine-learning/text_similarity/text_similarity.html).

**tf_idf : 2015.11.17** 

Introducing tf-idf ( term frequency-inverse document frequency ), a text mining technique and using it to perform text clustering using hierarchical clustering.
 
- View documentation [here](http://ethen8181.github.io/machine-learning/tf_idf/tf_idf.html).

**clustering : 2015.11.6**

Some useful evaluations when working with hierarchical clustering and k-means clustering ( k-means++ is used here ).

- Calinski-Harabasz index : Determine the right k ( cluster number ) for clustering.
- Boostrap evaluation of the clustering result’s stability.
- View report [here](http://ethen8181.github.io/machine-learning/clustering/clustering.html).

**linear_regression : 2015.10.30**

Solving linear regression with gradient descent. 

- Includes an appendix that briefly covers the interpretation and visualization of linear regression model’s summary output.
- View report [here](http://ethen8181.github.io/machine-learning/linear_regression/linear_regession.html).

**Coursera : 2015.10.16** ( kept for personal reference )

Course project for the Practical Machine Learning on Coursera. Using R’s tree and rpart library for classification.

**ROC : 2015.10.16** ( kept for personal reference, view the logistic_regression folder instead )

Use of ROC curve and cost to determine the threshold for logistic regression on the titanic dataset.

