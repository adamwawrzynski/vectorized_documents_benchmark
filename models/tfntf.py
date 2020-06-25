import numpy as np
from scipy import sparse
import math
import os
import shutil
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.python import saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def power(x):
    return x**2

def sparse_sigmoid(x):
    A = sparse.coo_matrix(x)
    B = x.copy()
    for i, j, v in zip(A.row, A.col, A.data):
        B[i][j] = sigmoid(v)

    return B

def sparse_tanh(x):
    A = sparse.coo_matrix(x)
    B = x.copy()
    for i, j, v in zip(A.row, A.col, A.data):
        B[i][j] = math.tanh(v)

    return B


class TfNtf(object):
    def __init__(
            self,
            pmi_matrix,
            information_gain
    ):
        self.pmi_matrix = pmi_matrix
        self.information_gain = information_gain

    def _convert_to_one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).reshape(num_classes, 1)

    def _normalize_adjency_matrix(self, A):
        # create self co-ocurance in adjecency matrix
        # I = sparse.eye(A.shape[1])
        # A_hat = A + I

        # calculate degree of each vertex
        D = np.sum(A, axis=1)
        D += 1

        # create diagonal matrix with inversed vertex degree as values
        for d in range(0, D.shape[0]):
            if D[d]:
                D[d] = 1.0 / D[d]

        I = sparse.eye(A.shape[1])
        A_hat = A + I
        A_hat = A_hat.multiply(D)
        return A_hat

    def _predict_iteration(self, A, F):
        # return (A * sparse_sigmoid(A * F + F) + F).todense() # bbc: 97.30
        # return (sparse_sigmoid(A * (A * F + F) + F)).todense() # bbc: 97.30
#        return (sparse_sigmoid(A * sparse_sigmoid(A * F) + F) + F).todense() # bbc: 97.35, reuters: 74.08
        # return (((A * (A * F)) + F) + F).todense() # bbc: 97.08
        # return sparse_sigmoid(A * F).todense() # bbc: 21.97
        # return sparse_sigmoid(A * F + F).todense() # bbc: 97.30
        # return (A * F).multiply(F).todense()
        # return ((self.pmi_matrix * A) * F).todense()
        # return (A * self.information_gain + F).todense() # bbc: 85.17
        # return (sparse_tanh(A * F) + F).todense() # bbc: 97.84, reuters 73.11, ohsumed: 44.12
        return (sparse_tanh(A * self.information_gain)).todense() # bbcsport: 84.81

    def predict_generator(self, factory_adjency, factory_features, docs):
        doc_vec = []
        counter = 0
        generator_adjency = factory_adjency(docs)
        generator_features = factory_features(docs)
        for A, F in zip(generator_adjency, generator_features):
            print("Vectorization: {}".format(counter + 1), end='\r', flush=True)

            doc_vec.append(self._predict_iteration(A=self._normalize_adjency_matrix(A), F=F))
            # doc_vec += self._predict_iteration(A=A, F=F)
            counter += 1

        return np.asarray(doc_vec).reshape((counter, doc_vec[0].shape[0]))
