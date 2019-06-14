import numpy as np
from tqdm import trange
from keras import layers, optimizers, Model
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from keras.preprocessing.sequence import skipgrams, make_sampling_table


class KerasWord2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Word vectors are averaged across to create the document-level vectors/features.

    Attributes
    ----------
    word2index_ : dict[str, int]
        Each distinct word in the corpus gets map to a numeric index.
        e.g. {'unk': 0, 'film': 1}

    index2word_ : list[str]
        Reverse napping of ``word2index_`` e.g. ['unk', 'film']

    vocab_size_ : int

    model_ : keras.models.Model

    """

    def __init__(self, embed_size=100, window_size=5, batch_size=64, epochs=5000,
                 learning_rate=0.05, negative_samples=0.5, min_count=2,
                 use_sampling_table=True, sort_vocab=True):
        self.min_count = min_count
        self.embed_size = embed_size
        self.sort_vocab = sort_vocab
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.use_sampling_table = use_sampling_table

    def fit(self, X, y=None):
        self.build_vocab(X)
        self.build_graph()
        indexed_texts = self.texts_to_index(X)

        sampling_table = None
        if self.sort_vocab and self.use_sampling_table:
            sampling_table = make_sampling_table(self.vocab_size_)

        for epoch in trange(self.epochs):
            (batch_center,
             batch_context,
             batch_label) = generate_batch_data(
                indexed_texts, self.batch_size, self.vocab_size_, self.window_size,
                self.negative_samples, sampling_table)
            self.model_.train_on_batch([batch_center, batch_context], batch_label)

        return self

    def transform(self, X):
        embed_in = self._get_word_vectors()
        X_embeddings = np.array([self._get_embedding(words, embed_in) for words in X])
        return X_embeddings

    def _get_word_vectors(self):
        return self.model_.get_layer('embed_in').get_weights()[0]

    def _get_embedding(self, words, embed_in):

        valid_words = [word for word in words if word in self.word2index_]
        if valid_words:
            embedding = np.zeros((len(valid_words), self.embed_size), dtype=np.float32)
            for idx, word in enumerate(valid_words):
                word_idx = self.word2index_[word]
                embedding[idx] = embed_in[word_idx]

            return np.mean(embedding, axis=0)
        else:
            return np.zeros(self.embed_size)

    def build_vocab(self, texts):

        # list[str] flatten to list of words
        words = [token for text in texts for token in text]

        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        valid_word_count = [(word, count) for word, count in word_count.items()
                            if count >= self.min_count]
        if self.sort_vocab:
            from operator import itemgetter
            valid_word_count = sorted(valid_word_count, key=itemgetter(1), reverse=True)

        index2word = ['unk']
        word2index = {'unk': 0}
        for word, _ in valid_word_count:
            word2index[word] = len(word2index)
            index2word.append(word)

        self.word2index_ = word2index
        self.index2word_ = index2word
        self.vocab_size_ = len(word2index)
        return self

    def texts_to_index(self, texts):
        """
        Returns
        -------
        texts_index : list[list[int]]
            e.g. [[0, 2], [3, 1]]
            each element in the outer list is the sentence, e.g. [0, 2]
            and each element in the inner list is each word represented in numeric index.
        """
        word2index = self.word2index_
        texts_index = []
        for text in texts:
            text_index = [word2index.get(token, 0) for token in text]
            texts_index.append(text_index)

        return texts_index

    def build_graph(self):
        input_center = layers.Input((1,))
        input_context = layers.Input((1,))

        embedding = layers.Embedding(self.vocab_size_, self.embed_size,
                                     input_length=1, name='embed_in')
        center = embedding(input_center)  # shape [seq_len, # features (1), embed_size]
        context = embedding(input_context)

        center = layers.Reshape((self.embed_size,))(center)
        context = layers.Reshape((self.embed_size,))(context)

        dot_product = layers.dot([center, context], axes=1)
        output = layers.Dense(1, activation='sigmoid')(dot_product)
        self.model_ = Model(inputs=[input_center, input_context], outputs=output)
        self.model_.compile(loss='binary_crossentropy',
                            optimizer=optimizers.RMSprop(lr=self.learning_rate))
        return self

    # def build_graph(self):
    #     """
    #     A different way of building the graph where the center word and
    #     context word each have its own embedding layer.
    #     """
    #     input_center = layers.Input((1,))
    #     input_context = layers.Input((1,))

    #     embedding_center = layers.Embedding(self.vocab_size_, self.embed_size,
    #                                         input_length=1, name='embed_in')
    #     embedding_context = layers.Embedding(self.vocab_size_, self.embed_size,
    #                                          input_length=1, name='embed_out')
    #     center = embedding_center(input_center)  # shape [seq_len, # features (1), embed_size]
    #     context = embedding_context(input_context)

    #     center = layers.Reshape((self.embed_size,))(center)
    #     context = layers.Reshape((self.embed_size,))(context)

    #     dot_product = layers.dot([center, context], axes=1)
    #     output = layers.Dense(1, activation='sigmoid')(dot_product)
    #     self.model_ = Model(inputs=[input_center, input_context], outputs=output)
    #     self.model_.compile(loss='binary_crossentropy',
    #                         optimizer=optimizers.RMSprop(lr=self.learning_rate))
    #     return self

    def most_similar(self, positive, negative=None, topn=10):

        # normalize word vectors to make the cosine distance calculation easier
        # normed_vectors = vectors / np.sqrt((word_vectors ** 2).sum(axis=-1))[..., np.newaxis]
        # ?? whether to cache the normed vector or replace the original one to speed up computation
        word_vectors = self._get_word_vectors()
        normed_vectors = normalize(word_vectors)

        # assign weight to positive and negative query words
        positive = [] if positive is None else [(word, 1.0) for word in positive]
        negative = [] if negative is None else [(word, -1.0) for word in negative]

        # compute the weighted average of all the query words
        queries = []
        all_word_index = set()
        for word, weight in positive + negative:
            word_index = self.word2index_[word]
            word_vector = normed_vectors[word_index]
            queries.append(weight * word_vector)
            all_word_index.add(word_index)

        if not queries:
            raise ValueError('cannot compute similarity with no input')

        query_vector = np.mean(queries, axis=0).reshape(1, -1)
        normed_query_vector = normalize(query_vector).ravel()

        # cosine similarity between the query vector and all the existing word vectors
        scores = np.dot(normed_vectors, normed_query_vector)

        actual_len = topn + len(all_word_index)
        sorted_index = np.argpartition(scores, -actual_len)[-actual_len:]
        best = sorted_index[np.argsort(scores[sorted_index])[::-1]]

        result = [(self.index2word_[index], scores[index])
                  for index in best if index not in all_word_index]
        return result[:topn]


def generate_batch_data(indexed_texts, batch_size, vocab_size,
                        window_size, negative_samples, sampling_table):
    batch_label = []
    batch_center = []
    batch_context = []
    while len(batch_center) < batch_size:
        # list[int]
        rand_indexed_texts = np.random.choice(indexed_texts)

        # couples: list[(str, str)], list of word pairs
        couples, labels = skipgrams(rand_indexed_texts, vocab_size,
                                    window_size=window_size,
                                    sampling_table=sampling_table,
                                    negative_samples=negative_samples)
        if couples:
            centers, contexts = zip(*couples)
            batch_center.extend(centers)
            batch_context.extend(contexts)
            batch_label.extend(labels)

    # trim to batch size at the end and convert to numpy array
    batch_center = np.array(batch_center[:batch_size], dtype=np.int)
    batch_context = np.array(batch_context[:batch_size], dtype=np.int)
    batch_label = np.array(batch_label[:batch_size], dtype=np.int)
    return batch_center, batch_context, batch_label
