# -*- coding: utf-8 -*-
# Authors: Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Lars Buitinck
#          Robert Layton <robertlayton@gmail.com>
#          Jochen Wersdörfer <jochen@wersdoerfer.de>
#          Roman Sinayev <roman.sinayev@gmail.com>
#
# License: BSD 3 clause
"""
The :mod:`sklearn.feature_extraction.text` submodule gathers utilities to
build feature vectors from text documents.
"""
from __future__ import unicode_literals, division

import array
from collections import defaultdict
import numbers
from operator import itemgetter
import re
import unicodedata
import warnings

from models.pagerank import CustomModel
import scipy.sparse as sp
import numpy as np

from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

vocabulary_size = None


class CustomTfidfVectorizer(CountVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=vocabulary_size, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(CustomTfidfVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

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

    @property
    def idf_(self):
        return self._tfidf.idf_

    @idf_.setter
    def idf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError("idf length = %d must be equal "
                                 "to vocabulary size = %d" %
                                 (len(value), len(self.vocabulary)))
        self._tfidf.idf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)

    def fit(self, raw_documents, y=None):
        X = super(CustomTfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        result = self._tfidf.transform(X, copy=False)

        self.unigram_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1), max_features=vocabulary_size)
        self.unigram_vectorizer.fit(raw_documents)

        # self.bigram_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 2), max_features=vocabulary_size)
        # self.bigram_vectorizer.fit(raw_documents)
        # 
        # self.pmi_matrix = sp.csr_matrix(
        #     (len(self.unigram_vectorizer.vocabulary_), len(self.unigram_vectorizer.vocabulary_)), dtype='float')

        # pmi_matrix = np.zeros((len(raw_documents), len(self.vocabulary_), len(self.vocabulary_)), dtype='float')

        # pmi_dict = self._pmi(raw_documents)
        # for key, value in pmi_dict.items():
        #     for key2, value2 in value.items():
        #         if self.unigram_vectorizer.vocabulary_.get(key, '') and self.unigram_vectorizer.vocabulary_.get(key2,
        #                                                                                                         ''):
        #             self.pmi_matrix[
        #                 self.unigram_vectorizer.vocabulary_[key], self.unigram_vectorizer.vocabulary_[key2]] = value2
                # self.pmi_matrix[self.vocabulary_[w2]][self.vocabulary_[w1]] = value

        # return self.apply_window(result, raw_documents)
        # data = list(itertools.chain.from_iterable([doc.split(" ") for doc in raw_documents]))
        # matrix, vocab_index = self.generate_co_occurrence_matrix(data, columns=vocab_index)

        # data_matrix = []
        # for doc in raw_documents:
        #     # matrix, vocab_index = self.create_co_occurences_matrix(self.vocabulary_, [doc.split(" ")])
        #     matrix = self.create_co_occurences_matrix2([doc])
        #     data_matrix.append(matrix)

        self.custom_model = CustomModel(len(self.unigram_vectorizer.vocabulary_), len(np.unique(y)))
        # self.custom_model.train_generator(self._co_occurance_generator_factory, raw_documents, self.pmi_matrix, y, 5)

        self.custom_model.train_generator(self._co_occurance_generator_factory, raw_documents, result, y, 5)

        return self

    def fit_transform(self, raw_documents, y=None):
        self._check_params()
        X = super(CustomTfidfVectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        result = self._tfidf.transform(X, copy=False)

        self.fit(raw_documents, y)
        return self.transform(raw_documents)
        # return self.custom_model.predict(data_matrix, self.pmi_matrix)
        # return self.custom_model.predict_generator(self.co_occurance_generator_factory, raw_documents, self.pmi_matrix)

    def transform(self, raw_documents, copy=True):
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(CustomTfidfVectorizer, self).transform(raw_documents)
        result = self._tfidf.transform(X, copy=False)

        # data_matrix = []
        # for doc in raw_documents:
        #     # matrix, vocab_index = self.create_co_occurences_matrix(self.vocabulary_, [doc.split(" ")])
        #     matrix = self.create_co_occurences_matrix2([doc])
        #     data_matrix.append(matrix)

        # return self.custom_model.predict(data_matrix, self.pmi_matrix)
        # return self.custom_model.predict_generator(self._co_occurance_generator_factory, raw_documents, self.pmi_matrix)
        return self.custom_model.predict_generator(self._co_occurance_generator_factory, raw_documents, result)

    def _create_co_occurences_matrix2(self, documents):
        # bigrams = self.bigram_vectorizer.transform(documents)
        ngrams = self.unigram_vectorizer.transform(documents)
        # ngrams[ngrams > 0] = 1
        co_occurance = (ngrams.T * ngrams)
        co_occurance.setdiag(0)
        return co_occurance

    def _co_occurance_generator(self, raw_documents):
        for doc in raw_documents:
            yield self._create_co_occurences_matrix2([doc])

    def _co_occurance_generator_factory(self, raw_documents):
        return self._co_occurance_generator(raw_documents)


    def _pmi(self, text):
        countMatrix = self.bigram_vectorizer.transform(text)

        # all unigrams and bigrams
        feature_names = self.bigram_vectorizer.get_feature_names()

        # finding all bigrams
        featureBigrams = [item for item in self.bigram_vectorizer.get_feature_names() if len(item.split()) == 2]
        # document term matrix
        arrays = countMatrix.toarray()

        # term document matrix
        arrayTrans = arrays.transpose()

        from collections import defaultdict
        PMIMatrix = defaultdict(dict)

        import math

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
