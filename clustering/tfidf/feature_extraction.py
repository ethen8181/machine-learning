import re
import numpy as np
from collections import defaultdict
from scipy.sparse import spdiags, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


__all__ = [
    'CountVectorizer',
    'TfidfTransformer',
    'TfidfVectorizer']


class CountVectorizer(BaseEstimator):
    """
    Convert a collection of text documents to a matrix of token counts,
    this implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    The number of features will be equal to the vocabulary size found by
    analyzing all input documents and removal of stop words

    Parameters
    ----------
    analyzer : str {'word'} or callable
        Whether the feature should be made of word, if n-grams is specified,
        then the words are concatenated with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    token_pattern : str
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    stop_words : str {'english'}, collection, or None, default None
        - If 'english', a built-in stop word list for English is used.
        - If a collection, that list or set is assumed to contain stop words,
        all of which will be removed from the resulting tokens. Only applies
        if ``analyzer == 'word'``.
        - If None, no stop words will be used.

    lowercase : bool, default True
        Convert all characters to lowercase before tokenizing.

    binary : bool, default False
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models such as binomial naive bayes that model binary
        events rather than integer counts.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    """

    def __init__(self, analyzer = 'word', ngram_range = (1, 1),
                 token_pattern = r'\b\w\w+\b', stop_words = None,
                 lowercase = True, binary = False):
        self.binary = binary
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.token_pattern = token_pattern

    def fit(self, raw_documents, y = None):
        """
        Learn the vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields str

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y = None):
        """
        Learn the vocabulary dictionary and return document-term matrix.
        This is equivalent to calling fit followed by transform, but more
        efficiently implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        X : scipy sparse matrix, shape [n_samples, n_features]
            Count document-term matrix.
        """
        if isinstance(raw_documents, str):
            raise ValueError(
                'Iterable over raw text documents expected, '
                'string objected received')

        X, vocabulary = self._count_vocab(raw_documents, fixed_vocab = False)
        if self.binary:
            X.data.fill(1)

        # we can add additional filtering after we construct
        # the document-term matrix, but this is omitted for now
        self.vocabulary_ = vocabulary
        return X

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix and vocabulary if fixed_vocab = False"""
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # add new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        # lambda function to split strings into n_gramed tokens
        analyze = self._build_analyzer()

        # information to create sparse csr_matrix
        values = []
        indptr = []
        indices = []
        indptr.append(0)
        for doc in raw_documents:
            # maps feature index to count
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    # ignore out-of-vocabulary items for fixed_vocab = True
                    continue

            indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(indices))

        # disable defaultdict behaviour
        if not fixed_vocab:
            vocabulary = dict(vocabulary)

        indices = np.asarray(indices, dtype = np.intc)
        values = np.asarray(values, dtype = np.intc)
        indptr = np.asarray(indptr, dtype = np.intc)
        shape = len(indptr) - 1, len(vocabulary)
        X = csr_matrix((values, indices, indptr), shape = shape, dtype = np.intc)
        return X, vocabulary

    def _build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer
        elif self.analyzer == 'word':
            tokenize = self._build_tokenizer()
            stop_words = self._get_stop_words()
            return lambda doc: self._word_ngrams(tokenize(doc), stop_words)
        else:
            raise ValueError('{} is not a valid tokenization scheme/analyzer'.format(
                             self.analyzer))

    def _build_tokenizer(self):
        """Returns a function that splits a string into a sequence of tokens"""
        token_pattern = re.compile(self.token_pattern)
        if self.lowercase:
            return lambda doc: token_pattern.findall(doc.lower())
        else:
            return lambda doc: token_pattern.findall(doc)

    def _get_stop_words(self):
        """Build or fetch the effective stop words frozenset"""
        stop = self.stop_words
        if stop == 'english':
            return ENGLISH_STOP_WORDS
        elif stop is None:
            return None
        elif isinstance(stop, str):
            raise ValueError("Stop words not a collection")
        else:
            return frozenset(stop)

    def _word_ngrams(self, tokens, stop_words):
        """Tokenize document into a sequence of n-grams after stop words filtering"""
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n == 1:
            return tokens
        else:
            original_tokens = list(tokens)
            n_original_tokens = len(original_tokens)
            if min_n == 1:
                min_n += 1
            else:
                tokens = []

            # bind method outside of loop to reduce overhead,
            # as local variables are accessed more quickly than attribute lookups
            # https://wiki.python.org/moin/PythonSpeed
            # https://stackoverflow.com/questions/28597014/python-why-is-accessing-instance-attribute-is-slower-than-local
            tokens_append = tokens.append
            space_join = ' '.join
            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i:i + n]))

            return tokens

    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.
        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or fit_transform.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str.

        Returns
        -------
        X : scipy sparse matrix, shape [n_samples, n_features]
            Document-term matrix.
        """

        # use the same matrix-building strategy as fit_transform
        X, _ = self._count_vocab(raw_documents, fixed_vocab = True)
        if self.binary:
            X.data.fill(1)

        return X


class TfidfTransformer(BaseEstimator, TransformerMixin):
    """
    Transform a count matrix to a tf-idf representation.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, default 'l2'
        Norm used to normalize term vectors. None for no normalization.

    smooth_idf : bool, default True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool, default False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    copy : bool, default True
        Whether to copy input data and operate on the copy or perform in-place operations.
    """

    def __init__(self, norm = 'l2', smooth_idf = True, sublinear_tf = False, copy = True):
        self.norm = norm
        self.copy = copy
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y = None):
        """
        Learn the idf vector.

        Parameters
        ----------
        X : scipy sparse matrix, shape [n_samples, n_features]
            Count document-term matrix.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        doc_freq = np.bincount(X.indices, minlength = X.shape[1])

        # perform idf smoothing if required
        doc_freq += int(self.smooth_idf)
        n_samples += int(self.smooth_idf)

        # log + 1 instead of log makes sure terms with zero idf
        # don't get suppressed entirely.
        idf = np.log(float(n_samples) / doc_freq) + 1.0
        self._idf_diag = spdiags(idf, diags = 0, m = n_features, n = n_features, format = 'csr')
        return self

    def transform(self, X):
        """
        Transform a count matrix to tf-idf representation.

        Parameters
        ----------
        X : scipy sparse matrix, shape [n_samples, n_features]
            Count document-term matrix.

        Returns
        -------
        X : scipy sparse matrix, shape [n_samples, n_features]
            Tf-idf weighted document-term matrix.
        """
        if self.copy:
            X = X.copy()

        if self.sublinear_tf:
            X.data = np.log(X.data)
            X.data += 1

        # compute the tfidf matrix
        X *= self._idf_diag

        if self.norm is not None:
            X = normalize(X, norm = self.norm, copy = False)

        return X


class TfidfVectorizer(CountVectorizer):
    """
    Convert a collection of raw documents to a matrix of TF-IDF features.
    This is equivalent to CountVectorizer followed by TfidfTransformer.

    Parameters
    ----------
    analyzer : str {'word'} or callable
        Whether the feature should be made of word, if n-grams is specified,
        then the words are concatenated with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    token_pattern : str
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    stop_words : str {'english'}, collection, or None, default None
        - If 'english', a built-in stop word list for English is used.
        - If a collection, that list or set is assumed to contain stop words,
        all of which will be removed from the resulting tokens. Only applies
        if ``analyzer == 'word'``.
        - If None, no stop words will be used.

    lowercase : bool, default True
        Convert all characters to lowercase before tokenizing.

    binary : boolean, default False
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only the tf term in tf-idf will
        become binary.

    norm : 'l1', 'l2' or None, default 'l2'
        Norm used to normalize term vectors. None for no normalization.

    smooth_idf : bool, default True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool, default False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    copy : bool, default True
        Whether to copy input data and operate on the copy or perform in-place operations.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    """

    def __init__(self, analyzer = 'word', ngram_range = (1, 1), token_pattern = r'\b\w\w+\b',
                 stop_words = None, lowercase = True, binary = False, norm = 'l2',
                 smooth_idf = True, sublinear_tf = False, copy = True):
        super().__init__(
            analyzer = analyzer, ngram_range = ngram_range,
            token_pattern = token_pattern, stop_words = stop_words, lowercase = lowercase)

        self._tfidf = TfidfTransformer(
            norm = norm, smooth_idf = smooth_idf, sublinear_tf = sublinear_tf, copy = copy)

    def fit(self, raw_documents, y = None):
        """
        Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields str.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        self
        """
        X = super().fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y = None):
        """
        Learn vocabulary and idf, return term-document matrix.
        This is equivalent to calling fit followed by transform, but more
        efficiently implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields str.

        y : default None
            Ignore, argument required for constructing sklearn Pipeline.

        Returns
        -------
        X : scipy sparse matrix, shape [n_samples, n_features]
            Tf-idf weighted document-term matrix.
        """
        X = super().fit_transform(raw_documents)
        return self._tfidf.fit_transform(X)

    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies learned by fit or
        fit_transform.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields str.

        Returns
        -------
        X : scipy sparse matrix, shape [n_samples, n_features]
            Tf-idf weighted document-term matrix.
        """
        X = super().transform(raw_documents)
        return self._tfidf.transform(X)

    # broadcast the TfidfTransformer's parameters to the underlying transformer
    # instance to enable hyperparameter search and repr
    @property
    def norm(self):
        return self._tfidf.norm

    @norm.setter
    def norm(self, value):
        self._tfidf.norm = value

    @property
    def use_idf(self):
        return self._tfidf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tfidf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tfidf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tfidf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tfidf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tfidf.sublinear_tf = value
