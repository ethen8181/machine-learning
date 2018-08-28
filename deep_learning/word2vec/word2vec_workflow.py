"""
1) preprocess the raw text using spaCy.
2) trains a Phrase model to glue words that commonly appear next to
each other into bigrams.
3) trains the Word2vec model (skipgram + negative sampling); currently,
there are zero hyperparameter tuning.
"""
import os
import spacy
import logging
from joblib import cpu_count
from string import punctuation
from logzero import setup_logger
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

    # spacy's english model for text preprocessing
    NLP = spacy.load('en')

    # a set of stopwords built-in to spacy, we can always
    # expand this set for the problem that we are working on,
    # here we include python built-in string punctuation mark
    STOPWORDS = spacy.en.STOP_WORDS | set(punctuation) | set(ENGLISH_STOP_WORDS)

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
        export_unigrams(UNIGRAM_PATH, texts = TEXTS, parser = NLP, stopwords = STOPWORDS)

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
        sentences = LineSentence(BIGRAM_PATH)
        word2vec = Word2Vec(sentences, workers = cpu_count())
        word2vec.save(WORD2VEC_CHECKPOINT)

    logger.info('job completed')


def export_unigrams(unigram_path, texts, parser, stopwords, batch_size = 10000, n_jobs = -1):
    """
    Preprocessed the raw text and export it to a .txt file,
    where each line is one document, for what sort of preprocessing
    is done, please refer to the `clean_corpus` function

    Parameters
    ----------
    unigram_path : str
        output file path of the preprocessed unigram text

    texts : iterable
        iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from
        disk/network using Gensim's Linsentence or something along
        those line

    parser : spacy model object
        e.g. parser = spacy.load('en')

    stopwords : set
        stopword set that will be excluded from the corpus

    batch_size : int, default 10000
        batch size for the spacy preprocessing

    n_jobs : int, default -1
        number of jobs/cores/threads to use for the spacy preprocessing
    """
    with open(unigram_path, 'w', encoding = 'utf_8') as f:
        for cleaned_text in clean_corpus(texts, parser, stopwords, batch_size, n_jobs):
            f.write(cleaned_text + '\n')


def clean_corpus(texts, parser, stopwords, batch_size, n_jobs):
    """
    Generator function using spaCy to parse reviews:
    - lemmatize the text
    - remove punctuation, whitespace and number
    - remove pronoun, e.g. 'it'
    - remove tokens that are shorter than 2
    """
    n_threads = cpu_count()
    if n_jobs > 0 and n_jobs < n_threads:
        n_threads = n_jobs

    # use the .pip to process texts as a stream;
    # this functionality supports using multi-threads
    for parsed_text in parser.pipe(texts, n_threads = n_threads, batch_size = batch_size):
        tokens = []
        for token in parsed_text:
            if valid_word(token) and token.lemma_ not in stopwords:
                tokens.append(token.lemma_)

        cleaned_text = ' '.join(tokens)
        yield cleaned_text


def valid_word(token):
    """
    Returns False if the spacy token is either
    - a punctuation, whitespace, number
    - a pronoun (indicated by the '-PRON-' flag), e.g. 'it'
    - the token is shorter than 2
    these words will be removed from the corpus
    """
    pron_flag = token.lemma_ != '-PRON-'
    word_flag = not (token.is_punct or token.is_space or token.like_num)
    word_len_flag = len(token) >= 2
    valid = pron_flag and word_flag and word_len_flag
    return valid


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
