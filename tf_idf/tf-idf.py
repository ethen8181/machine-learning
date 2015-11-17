from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_set = ["The sky is blue.", "The sun is bright."]
test_set  = [ "The sun in the sky is bright.", "We can see the shining sun, the bright sun." ]

# remove stop words 
vectorizer = CountVectorizer( stop_words = "english" )

# fit_transform tokenize and count the word occurrences for the
# training dataset 
vectorizer.fit_transform(train_set)

# print out the term fequency 
# print( vectorizer.vocabulary_ )

# print out the terms only 
# print( vectorizer.get_feature_names() )

# use the training dataset and count the test set 
term_matrix = vectorizer.transform(test_set)
# print( term_matrix )

# the original output is stored in a coordinate format
# you can convert it into a dense format 
# print( term_matrix.todense() )

tfidf = TfidfTransformer( norm = "l2" )
tfidf_matrix = tfidf.fit_transform(term_matrix)

# print(tfidf_matrix )
# print( tfidf_matrix.todense().transpose() )


# -----------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
	"The sky is blue",
	"The sun is bright today",
	"The sun in the sky is bright",
	"We can see the shining sun, the bright sun"
]


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(tfidf_matrix)


