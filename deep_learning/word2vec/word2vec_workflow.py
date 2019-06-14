"""
1) Basic preprocessing of the raw text.
2) trains a Phrase model to glue words that commonly appear next to
each other into bigrams.
3) trains the Word2vec model (skipgram + negative sampling); currently,
there are zero hyperparameter tuning.
"""
import os
import re
import logging
from joblib import cpu_count
from string import punctuation
from logzero import setup_logger
from nltk.corpus import stopwords
from gensim.models import Phrases
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from gensim.models.word2vec import LineSentence
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
logger = setup_logger(name = __name__, logfile = 'word2vec.log', level = logging.INFO)


def main():
    # -------------------------------------------------------------------------------
    # Parameters

    # the script will most likely work if we swap the TEXTS variable
    # with any iterable of text (where one element represents a document,
    # and the whole iterable is the corpus)
    newsgroups_train = fetch_20newsgroups(subset = 'train')
    TEXTS = newsgroups_train.data

    # a set of stopwords built-in to various packages
    # we can always expand this set for the
    # problem that we are working on, here we also included
    # python built-in string punctuation mark
    STOPWORDS = set(stopwords.words('english')) | set(punctuation) | set(ENGLISH_STOP_WORDS)

    # create a directory called 'model' to store all outputs in later section
    MODEL_DIR = 'model'
    UNIGRAM_PATH = os.path.join(MODEL_DIR, 'unigram.txt')
    PHRASE_MODEL_CHECKPOINT = os.path.join(MODEL_DIR, 'phrase_model')
    BIGRAM_PATH = os.path.join(MODEL_DIR, 'bigram.txt')
    WORD2VEC_CHECKPOINT = os.path.join(MODEL_DIR, 'word2vec')

    # -------------------------------------------------------------------------------
    logger.info('job started')
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if not os.path.exists(UNIGRAM_PATH):
        logger.info('preprocessing text')
        export_unigrams(UNIGRAM_PATH, texts=TEXTS, stop_words=STOPWORDS)

    if os.path.exists(PHRASE_MODEL_CHECKPOINT):
        phrase_model = Phrases.load(PHRASE_MODEL_CHECKPOINT)
    else:
        logger.info('training phrase model')
        # use LineSetence to stream text as oppose to loading it all into memory
        unigram_sentences = LineSentence(UNIGRAM_PATH)
        phrase_model = Phrases(unigram_sentences)
        phrase_model.save(PHRASE_MODEL_CHECKPOINT)

    if not os.path.exists(BIGRAM_PATH):
        logger.info('converting words to phrases')
        export_bigrams(UNIGRAM_PATH, BIGRAM_PATH, phrase_model)

    if os.path.exists(WORD2VEC_CHECKPOINT):
        word2vec = Word2Vec.load(WORD2VEC_CHECKPOINT)
    else:
        logger.info('training word2vec')
        word2vec = Word2Vec(corpus_file=BIGRAM_PATH, workers=cpu_count())
        word2vec.save(WORD2VEC_CHECKPOINT)

    logger.info('job completed')


def export_unigrams(unigram_path, texts, stop_words):
    """
    Preprocessed the raw text and export it to a .txt file,
    where each line is one document, for what sort of preprocessing
    is done, please refer to the `normalize_text` function

    Parameters
    ----------
    unigram_path : str
        output file path of the preprocessed unigram text.

    texts : iterable
        iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from
        disk/network using Gensim's Linsentence or something along
        those line.

    stop_words : set
        stopword set that will be excluded from the corpus.
    """
    with open(unigram_path, 'w', encoding='utf_8') as f:
        for text in texts:
            cleaned_text = normalize_text(text, stop_words)
            f.write(cleaned_text + '\n')


def normalize_text(text, stop_words):
    # remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)

    # lower case & tokenize text
    tokens = re.split(r'\s+', text.lower().strip())

    # filter stopwords out of text &
    # re-create text from filtered tokens
    cleaned_text = ' '.join(token for token in tokens if token not in stop_words)
    return cleaned_text


def export_bigrams(unigram_path, bigram_path, phrase_model):
    """
    Use the learned phrase model to create (potential) bigrams,
    and output the text that contains bigrams to disk

    Parameters
    ----------
    unigram_path : str
        input file path of the preprocessed unigram text

    bigram_path : str
        output file path of the transformed bigram text

    phrase_model : gensim's Phrase model object

    References
    ----------
    Gensim Phrase Detection
    - https://radimrehurek.com/gensim/models/phrases.html
    """

    # after training the Phrase model, create a performant
    # Phraser object to transform any sentence (list of
    # token strings) and glue unigrams together into bigrams
    phraser = Phraser(phrase_model)
    with open(bigram_path, 'w') as fout, open(unigram_path) as fin:
        for text in fin:
            unigram = text.split()
            bigram = phraser[unigram]
            bigram_sentence = ' '.join(bigram)
            fout.write(bigram_sentence + '\n')


if __name__ == '__main__':
    main()
