# Reference

# setting up the twitter R package for text analytics
# http://www.r-bloggers.com/setting-up-the-twitter-r-package-for-text-analytics/

# twitter sentiment analysis tutorial
# https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107

# a list of positive and negative opinion words for English
# http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
library(dplyr)
library(ggplot2)
library(stringr)
library(twitteR)
library(data.table)
setwd('/Users/ethen/Desktop')
setup_twitter_oauth( consumer_key, consumer_secret, access_token, access_secret )

delta_tweets <- searchTwitter( '@delta', n = 1500 )

# ?status
tweet <- delta_tweets[[1]]
tweet$screenName
tweet$text
tweet$created
tweet$retweeted

delta_text <- lapply( delta_tweets, function(x) x$text )


# what: read in 'character' only
# comment.char: signals that the rest of the line should be 
# regarded as a comment and be discarded
pos_words <- scan(
	'opinion-lexicon-English/positive-words.txt',
	what = 'character', 
	comment.char = ';'
)

neg_words <- scan(
	'opinion-lexicon-English/negative-words.txt',
	what = 'character', 
	comment.char = ';'
)

sentences <- c("You're awesome and I love you19")
ScoreSentiment <- function( sentences, pos_words, neg_words ){

	scores <- vapply( sentences, function(sentence){
		# for each sentence convert to lower cases
		# split the sentence into words (regardless of the number of white spaces in between)
		# and unlist it to convert it back to vectors.
		# do not remove punctuation characters and or digits 0-9, since it'll remove hyperlinks
		# http://stackoverflow.com/questions/12193779/how-to-write-trycatch-in-r
		words <- tryCatch({
					
					words <- tolower(sentence) %>%
							 str_split('\\s+') %>% 
							 unlist()
				 
				 }, warning = function(w){
				 	return(NA)
				 }, error = function(e){
				 	return(NA)
				 })
		
		# if it returns NA, then it means the sentence contains non-english characters
		if( is.na(words) ){			
			score <- NA
		}else{
			# compare the words to the vector of positive & negative terms
			# remove the onces that are not matched and calculate the sentiment score
			# of the sentence, which is simply the number positive words - negative
			pos_matches <- !is.na( match( words, pos_words ) ) 
			neg_matches <- !is.na( match( words, neg_words ) )
			score <- sum(pos_matches) - sum(neg_matches)
		}
		
		return(score)	
	}, double(1) )

	sentence_scores <- data.table( sentences = sentences, scores = scores )
	return(sentence_scores)
}

# obtain the score of the sentiment
sentence_score <- ScoreSentiment( 
	sentences = delta_text, 
	pos_words = pos_words, 
	neg_words = neg_words
)


ggplot( sentence_score, aes(scores) ) + 
geom_histogram()
