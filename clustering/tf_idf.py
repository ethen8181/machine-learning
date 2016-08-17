from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_set = [ "The sky is blue.", "The sun is bright." ]
test_set  = [ "The sun in the sky is bright.", "We can see the shining sun, the bright sun." ]

# remove stop words 
vectorizer = CountVectorizer( stop_words = "english" )

# fit_transform tokenize and count the word occurrences for the
# training dataset 
vectorizer.fit_transform(train_set)

# print out the terms only 
# print(vectorizer.get_feature_names())

# print out the term fequency 
# print(vectorizer.vocabulary_)

# use the training dataset and count the test set 
term_matrix = vectorizer.transform(test_set)
# print(term_matrix)

# the original output is stored in a sparse format
# you can convert it into a dense format 
# print(term_matrix.toarray())

tfidf = TfidfTransformer( norm = "l2" )
tfidf_matrix = tfidf.fit_transform(term_matrix)
# print(tfidf_matrix)
# print(tfidf_matrix.toarray())


# tf-idf in one single step
documents = [
	"The sky is blue",
	"The sun is bright today",
	"The sun in the sky is bright",
	"We can see the shining sun, the bright sun"
]


tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)
# print(tfidf_matrix)


