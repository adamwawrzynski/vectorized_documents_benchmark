from __future__ import unicode_literals, division
import math
from collections import defaultdict

from models.tfntf import TfNtf
from models.tfcontext import TfContext
import scipy.sparse as sp
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder

vocabulary_size = None

class TfNtfVectorizer(object):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=vocabulary_size)#use_idf=False, smooth_idf=False)

    def fit(self, raw_documents, y=None):
        result = self.vectorizer.fit_transform(raw_documents)

        #self.unigram_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1), max_features=vocabulary_size)
        #self.unigram_vectorizer.fit(raw_documents)

        #self.information_gain = sp.csr_matrix(mutual_info_classif(result, y, discrete_features=True).reshape(-1, 1))

        self.onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = self.onehot_encoder.fit_transform(y.to_numpy().reshape(-1, 1))
        #
        # self.bigram_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2), max_features=vocabulary_size)
        # self.bigram_vectorizer.fit(raw_documents)
        #
        # self.pmi_matrix = sp.csr_matrix(
        #     (len(self.unigram_vectorizer.vocabulary_), len(self.unigram_vectorizer.vocabulary_)), dtype='float')
        #
        # pmi_dict = self._calculate_pmi(raw_documents)
        # for key, value in pmi_dict.items():
        #     for key2, value2 in value.items():
        #         if self.unigram_vectorizer.vocabulary_.get(key, '') and self.unigram_vectorizer.vocabulary_.get(key2, ''):
        #             self.pmi_matrix[self.unigram_vectorizer.vocabulary_[key], self.unigram_vectorizer.vocabulary_[key2]] = value2
                # self.pmi_matrix[self.vocabulary_[key]][self.vocabulary_[key2]] = value

        # return self.apply_window(result, raw_documents)
        # data = list(itertools.chain.from_iterable([doc.split(" ") for doc in raw_documents]))
        # matrix, vocab_index = self.generate_co_occurrence_matrix(data, columns=vocab_index)

        # data_matrix = []
        # for doc in raw_documents:
        #     # matrix, vocab_index = self.create_co_occurences_matrix(self.vocabulary_, [doc.split(" ")])
        #     matrix = self.create_co_occurences_matrix2([doc])
        #     data_matrix.append(matrix)

        # self.custom_model = CustomModel(self.pmi_matrix)
        # self.custom_model = TfNtf(None)
        # self.custom_model = TfNtf(None, self.information_gain)
        self.word2idx = {word: idx for idx, word in enumerate(self.vectorizer.get_feature_names())}
        self.custom_model = TfContext(self.vectorizer.vocabulary_, self.word2idx, 100, len(np.unique(y)))
        self.custom_model.train(raw_documents, result.todense(), onehot_encoded)
        return self

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents, y)
        return self.transform(raw_documents)

    def transform(self, raw_documents, copy=True):
        result = self.vectorizer.transform(raw_documents, copy=False)

        # data_matrix = []
        # for doc in raw_documents:
        #     # matrix, vocab_index = self.create_co_occurences_matrix(self.vocabulary_, [doc.split(" ")])
        #     matrix = self.create_co_occurences_matrix2([doc])
        #     data_matrix.append(matrix)

        # return self.custom_model.predict(data_matrix, self.pmi_matrix)
        # return self.custom_model.predict_generator(self._co_occurance_generator_factory, raw_documents, self.pmi_matrix)
        # return self.custom_model.predict_generator(self._co_occurance_generator_factory, raw_documents, result)
        #return self.custom_model.predict_generator(
        #    self._co_occurance_generator_factory,
        #    self._features_generator_factory,
        #    raw_documents)
        return self.custom_model.predict(raw_documents, result.todense())


    def _create_co_occurences_matrix2(self, documents):
        ngrams = self.unigram_vectorizer.transform(documents)
        ngrams[ngrams > 0] = 1
        co_occurance = (ngrams.T * ngrams)
        co_occurance.setdiag(0)
        return co_occurance

    def _co_occurance_generator(self, raw_documents):
        for doc in raw_documents:
            yield self._create_co_occurences_matrix2([doc])

    def _co_occurance_generator_factory(self, raw_documents):
        return self._co_occurance_generator(raw_documents)

    def _features_generator(self, raw_documents):
        for doc in raw_documents:
            yield self.vectorizer.transform([doc]).T

    def _features_generator_factory(self, raw_documents):
        return self._features_generator(raw_documents)


    def _calculate_pmi(self, text):
        countMatrix = self.bigram_vectorizer.transform(text)

        # all unigrams and bigrams
        feature_names = self.bigram_vectorizer.get_feature_names()

        # finding all bigrams
        featureBigrams = [item for item in self.bigram_vectorizer.get_feature_names() if len(item.split()) == 2]
        # document term matrix
        arrays = countMatrix.toarray()

        # term document matrix
        arrayTrans = arrays.transpose()

        PMIMatrix = defaultdict(dict)

        i = 0
        PMIMatrix = defaultdict(dict)
        for item in featureBigrams:
            words = item.split()
            if item in featureBigrams:
                bigramLength = len(np.where(arrayTrans[feature_names.index(item)] > 0)[0])

            if bigramLength < 2:
                continue

            if words[0] in feature_names and words[1] in feature_names:
                word0Length = len(np.where(arrayTrans[feature_names.index(words[0])] > 0)[0])
                word1Length = len(np.where(arrayTrans[feature_names.index(words[1])] > 0)[0])

            try:
                PMIMatrix[words[0]][words[1]] = (len(text) * math.log(1.0 * bigramLength, 2)) / (
                        1.0 * math.log(word0Length, 2) * math.log(1.0 * word1Length, 2))
            except:
                print(bigramLength, word0Length, word1Length)

        return PMIMatrix
