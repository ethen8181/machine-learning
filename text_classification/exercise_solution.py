# # Tutorial Exercise: Yelp reviews (Solution)

# ## Introduction
# 
# This exercise uses a small subset of the data from Kaggle's [Yelp Business Rating Prediction](https://www.kaggle.com/c/yelp-recsys-2013) competition.
# 
# **Description of the data:**
# 
# - **`yelp.csv`** contains the dataset. It is stored in the repository (in the **`data`** directory), so there is no need to download anything from the Kaggle website.
# - Each observation (row) in this dataset is a review of a particular business by a particular user.
# - The **stars** column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
# - The **text** column is the text of the review.
# 
# **Goal:** Predict the star rating of a review using **only** the review text.
# 
# **Tip:** After each task, I recommend that you check the shape and the contents of your objects, to confirm that they match your expectations.

# for Python 2: use print only as a function
from __future__ import print_function


# ## Task 1
# 
# Read **`yelp.csv`** into a pandas DataFrame and examine it.

# read yelp.csv using a relative path
import pandas as pd
path = 'data/yelp.csv'
yelp = pd.read_csv(path)


# examine the shape
yelp.shape


# examine the first row
yelp.head(1)


# examine the class distribution
yelp.stars.value_counts().sort_index()


# ## Task 2
# 
# Create a new DataFrame that only contains the **5-star** and **1-star** reviews.
# 
# - **Hint:** [How do I apply multiple filter criteria to a pandas DataFrame?](http://nbviewer.jupyter.org/github/justmarkham/pandas-videos/blob/master/pandas.ipynb#9.-How-do-I-apply-multiple-filter-criteria-to-a-pandas-DataFrame%3F-%28video%29) explains how to do this.

# filter the DataFrame using an OR condition
yelp_best_worst = yelp[(yelp.stars==5) | (yelp.stars==1)]

# equivalently, use the 'loc' method
yelp_best_worst = yelp.loc[(yelp.stars==5) | (yelp.stars==1), :]


# examine the shape
yelp_best_worst.shape


# ## Task 3
# 
# Define X and y from the new DataFrame, and then split X and y into training and testing sets, using the **review text** as the only feature and the **star rating** as the response.
# 
# - **Hint:** Keep in mind that X should be a pandas Series (not a DataFrame), since we will pass it to CountVectorizer in the task that follows.

# define X and y
X = yelp_best_worst.text
y = yelp_best_worst.stars


# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# examine the object shapes
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Task 4
# 
# Use CountVectorizer to create **document-term matrices** from X_train and X_test.

# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()


# fit and transform X_train into X_train_dtm
X_train_dtm = vect.fit_transform(X_train)
X_train_dtm.shape


# transform X_test into X_test_dtm
X_test_dtm = vect.transform(X_test)
X_test_dtm.shape


# ## Task 5
# 
# Use multinomial Naive Bayes to **predict the star rating** for the reviews in the testing set, and then **calculate the accuracy** and **print the confusion matrix**.
# 
# - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains how to interpret both classification accuracy and the confusion matrix.

# import and instantiate MultinomialNB
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# train the model using X_train_dtm
nb.fit(X_train_dtm, y_train)


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# ## Task 6 (Challenge)
# 
# Calculate the **null accuracy**, which is the classification accuracy that could be achieved by always predicting the most frequent class.
# 
# - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains null accuracy and demonstrates two ways to calculate it, though only one of those ways will work in this case. Alternatively, you can come up with your own method to calculate null accuracy!

# examine the class distribution of the testing set
y_test.value_counts()


# calculate null accuracy
y_test.value_counts().head(1) / y_test.shape


# calculate null accuracy manually
838 / float(838 + 184)


# ## Task 7 (Challenge)
# 
# Browse through the review text of some of the **false positives** and **false negatives**. Based on your knowledge of how Naive Bayes works, do you have any ideas about why the model is incorrectly classifying these reviews?
# 
# - **Hint:** [Evaluating a classification model](https://github.com/justmarkham/scikit-learn-videos/blob/master/09_classification_metrics.ipynb) explains the definitions of "false positives" and "false negatives".
# - **Hint:** Think about what a false positive means in this context, and what a false negative means in this context. What has scikit-learn defined as the "positive class"?

# first 10 false positives (1-star reviews incorrectly classified as 5-star reviews)
X_test[y_test < y_pred_class].head(10)


# false positive: model is reacting to the words "good", "impressive", "nice"
X_test[1781]


# false positive: model does not have enough data to work with
X_test[1919]


# first 10 false negatives (5-star reviews incorrectly classified as 1-star reviews)
X_test[y_test > y_pred_class].head(10)


# false negative: model is reacting to the words "complain", "crowds", "rushing", "pricey", "scum"
X_test[4963]


# ## Task 8 (Challenge)
# 
# Calculate which 10 tokens are the most predictive of **5-star reviews**, and which 10 tokens are the most predictive of **1-star reviews**.
# 
# - **Hint:** Naive Bayes automatically counts the number of times each token appears in each class, as well as the number of observations in each class. You can access these counts via the `feature_count_` and `class_count_` attributes of the Naive Bayes model object.

# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# first row is one-star reviews, second row is five-star reviews
nb.feature_count_.shape


# store the number of times each token appears across each class
one_star_token_count = nb.feature_count_[0, :]
five_star_token_count = nb.feature_count_[1, :]


# create a DataFrame of tokens with their separate one-star and five-star counts
tokens = pd.DataFrame({'token':X_train_tokens, 'one_star':one_star_token_count, 'five_star':five_star_token_count}).set_index('token')


# add 1 to one-star and five-star counts to avoid dividing by 0
tokens['one_star'] = tokens.one_star + 1
tokens['five_star'] = tokens.five_star + 1


# first number is one-star reviews, second number is five-star reviews
nb.class_count_


# convert the one-star and five-star counts into frequencies
tokens['one_star'] = tokens.one_star / nb.class_count_[0]
tokens['five_star'] = tokens.five_star / nb.class_count_[1]


# calculate the ratio of five-star to one-star for each token
tokens['five_star_ratio'] = tokens.five_star / tokens.one_star


# sort the DataFrame by five_star_ratio (descending order), and examine the first 10 rows
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
tokens.sort_values('five_star_ratio', ascending=False).head(10)


# sort the DataFrame by five_star_ratio (ascending order), and examine the first 10 rows
tokens.sort_values('five_star_ratio', ascending=True).head(10)


# ## Task 9 (Challenge)
# 
# Up to this point, we have framed this as a **binary classification problem** by only considering the 5-star and 1-star reviews. Now, let's repeat the model building process using all reviews, which makes this a **5-class classification problem**.
# 
# Here are the steps:
# 
# - Define X and y using the original DataFrame. (y should contain 5 different classes.)
# - Split X and y into training and testing sets.
# - Create document-term matrices using CountVectorizer.
# - Calculate the testing accuracy of a Multinomial Naive Bayes model.
# - Compare the testing accuracy with the null accuracy, and comment on the results.
# - Print the confusion matrix, and comment on the results. (This [Stack Overflow answer](http://stackoverflow.com/a/30748053/1636598) explains how to read a multi-class confusion matrix.)
# - Print the [classification report](http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report), and comment on the results. If you are unfamiliar with the terminology it uses, research the terms, and then try to figure out how to calculate these metrics manually from the confusion matrix!

# define X and y using the original DataFrame
X = yelp.text
y = yelp.stars


# check that y contains 5 different classes
y.value_counts().sort_index()


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# create document-term matrices using CountVectorizer
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# fit a Multinomial Naive Bayes model
nb.fit(X_train_dtm, y_train)


# make class predictions
y_pred_class = nb.predict(X_test_dtm)


# calculate the accuary
metrics.accuracy_score(y_test, y_pred_class)


# calculate the null accuracy
y_test.value_counts().head(1) / y_test.shape


# **Accuracy comments:** At first glance, 47% accuracy does not seem very good, given that it is not much higher than the null accuracy. However, I would consider the 47% accuracy to be quite impressive, given that humans would also have a hard time precisely identifying the star rating for many of these reviews.

# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# **Confusion matrix comments:**
# 
# - Nearly all 4-star and 5-star reviews are classified as 4 or 5 stars, but they are hard for the model to distinguish between.
# - 1-star, 2-star, and 3-star reviews are most commonly classified as 4 stars, probably because it's the predominant class in the training data.

# print the classification report
print(metrics.classification_report(y_test, y_pred_class))


# **Precision** answers the question: "When a given class is predicted, how often are those predictions correct?" To calculate the precision for class 1, for example, you divide 55 by the sum of the first column of the confusion matrix.

# manually calculate the precision for class 1
precision = 55 / float(55 + 28 + 5 + 7 + 6)
print(precision)


# **Recall** answers the question: "When a given class is the true class, how often is that class predicted?" To calculate the recall for class 1, for example, you divide 55 by the sum of the first row of the confusion matrix.

# manually calculate the recall for class 1
recall = 55 / float(55 + 14 + 24 + 65 + 27)
print(recall)


# **F1 score** is a weighted average of precision and recall.

# manually calculate the F1 score for class 1
f1 = 2 * (precision * recall) / (precision + recall)
print(f1)


# **Support** answers the question: "How many observations exist for which a given class is the true class?" To calculate the support for class 1, for example, you sum the first row of the confusion matrix.

# manually calculate the support for class 1
support = 55 + 14 + 24 + 65 + 27
print(support)


# **Classification report comments:**
# 
# - Class 1 has low recall, meaning that the model has a hard time detecting the 1-star reviews, but high precision, meaning that when the model predicts a review is 1-star, it's usually correct.
# - Class 5 has high recall and precision, probably because 5-star reviews have polarized language, and because the model has a lot of observations to learn from.
