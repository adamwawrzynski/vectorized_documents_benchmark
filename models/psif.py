import itertools
import logging
import os
from collections import Counter

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from preprocess import clean_string
from preprocess import preprocess_text, process_dataset, process_dataset_2, process_string

import warnings, random
import pandas as pd
import time,pickle, pdb
from nltk.corpus import stopwords
import numpy as np
from numpy import float32
import math
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVC
import pickle
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from models.ApproximateKSVD import ApproximateKSVD
from sklearn.decomposition import PCA


class PSIF(object):
    def __init__(
            self,
            pretrained_embedded_vector_path,
            embedding_size=100,
            num_clusters=40
    ):
        self.embedding_model = KeyedVectors.load_word2vec_format(pretrained_embedded_vector_path + '.word2vec')
        self.embedding_size = embedding_size
        self.num_clusters = num_clusters

    def fit(self, X, y=None):
        t0 = time.time()
        logging.info("Building vectorizer " + self.__class__.__name__)

        train_x = process_dataset_2(pd.Series(X))
        self.vec = CountVectorizer()
        self.vec.fit(train_x)

        self.keys = self.vec.vocabulary_

        self.word_vectors = []
        for word in self.keys:
            if word in self.embedding_model.wv:
                self.word_vectors.append(self.embedding_model[word])
            else:
                self.word_vectors.append(np.zeros(self.embedding_size))

        self.word_vectors = np.array(self.word_vectors)

        self._dictionary_KSVD()

        # Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
        self.word_centroid_map = dict(zip(self.keys, self.idx))
        # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
        # list of probabilities of cluster assignments.
        self.word_centroid_prob_map = dict(zip(self.keys, self.idx_proba))

        self.tfv = TfidfVectorizer(strip_accents='unicode', vocabulary=self.vec.vocabulary_, dtype=np.float32)
        self.tfv.fit_transform(train_x)
        self.featurenames = self.tfv.get_feature_names()
        self.idf = self.tfv._tfidf.idf_

        # Creating a dictionary with word mapped to its idf value
        a_weight = 0.01
        dictionary = {}
        for doc in train_x:
            for w in doc.split():
                if not w in dictionary:
                    dictionary[w] = 1
                else:
                    dictionary[w] = dictionary[w] + 1

        self._weight_building(dictionary, a_weight)
        self.word_idf_dict = {}
        for pair in zip(self.featurenames, self.idf):
            self.word_idf_dict[pair[0]] = pair[1]

        # Pre-computing probability word-cluster vectors.
        self._get_probability_word_vectors()

        # gwbowv is a matrix which contains normalised document vectors.
        gwbowv = np.zeros((len(X), self.num_clusters * (self.embedding_size)), dtype="float32")
        self.n_comp = self.embedding_size * self.num_clusters

        for counter, review in enumerate(X):
            words = preprocess_text(review)
            gwbowv[counter] = self._create_cluster_vector_and_gwbowv(words)

        # principal component removal
        self._pca_truncated_svd_fit(gwbowv, self.embedding_size)

        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

        return self

    def transform(self, X, y=None):
        # gwbowv is a matrix which contains normalised document vectors.
        gwbowv = np.zeros((len(X), self.num_clusters * (self.embedding_size)), dtype="float32")

        for counter, review in enumerate(X):
            words = preprocess_text(review)
            gwbowv[counter] = self._create_cluster_vector_and_gwbowv(words)

        return self._pca_truncated_svd_transform(gwbowv), y

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y), y

    def _dictionary_KSVD(self):
        # Initalize a ksvd object and use it for clustering.
        aksvd = ApproximateKSVD(n_components=self.num_clusters, transform_n_nonzero_coefs=self.num_clusters/2)
        aksvd.fit(self.word_vectors)
        self.idx_proba = aksvd.transform(self.word_vectors)
        self.idx = np.argmax(self.idx_proba, axis=1)
        # Get probabilities of cluster assignments.
        # Dump cluster assignments and probability of cluster assignments.
        joblib.dump(self.idx, 'ksvd_latestclusmodel_len2alldata.pkl')

        joblib.dump(self.idx_proba, 'ksvd_prob_latestclusmodel_len2alldata.pkl')

    def _dictionary_read_KSVD(self, idx_name, idx_proba_name):
        # Loads cluster assignments and probability of cluster assignments.
        self.idx = joblib.load(idx_name)
        self.idx_proba = joblib.load(idx_proba_name)


    def _get_probability_word_vectors(self):
        # This function computes probability word-cluster vectors.

        self.prob_wordvecs = {}

        for word in self.word_centroid_map:
            self.prob_wordvecs[word] = np.zeros(self.num_clusters * self.embedding_size, dtype="float32")
            for index in range(0, self.num_clusters):
                try:
                    self.prob_wordvecs[word][index * self.embedding_size:(index + 1) * self.embedding_size] = self.embedding_model[word] * \
                                                                    self.word_centroid_prob_map[word][index] * \
                                                                    self.word_idf_dict[word]
                except:
                    continue

    def _weight_building(self, dictionary, a_weight):
        self.weight_dict = {}
        total = 0
        for word, count in dictionary.items():
            self.weight_dict[word] = int(count)
            total = total + int(count)
        for word in self.weight_dict:
            prob = self.weight_dict[word] * 1.0 / total
            self.weight_dict[word] = a_weight * 1.0 / (a_weight * 1.0 + prob)

    def _create_cluster_vector_and_gwbowv(self, wordlist):
        # This function computes SDV feature vectors.
        bag_of_centroids = np.zeros(self.n_comp, dtype="float32" )
        for word in wordlist:
            try:
                bag_of_centroids += self.prob_wordvecs[word] * self.weight_dict[word]
            except:
                pass
        norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
        if norm != 0:
            bag_of_centroids /= norm
        return bag_of_centroids


    def _pca_truncated_svd_fit(self, X, size):
        self.sklearn_pca = PCA(n_components=size, svd_solver='full')
        self.sklearn_pca.fit_transform(X)

    def _pca_truncated_svd_transform(self, X):
        return self.sklearn_pca.transform(X)
