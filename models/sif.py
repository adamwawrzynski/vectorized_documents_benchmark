import itertools
import logging
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import PCA

from utils.preprocess import clean_string
from utils.preprocess import preprocess_text


# Implementation based on: https://github.com/peter3125/sentence2vec
# Author: https://github.com/peter3125

class SIF(object):
    def __init__(
        self,
        text,
        labels,
        pretrained_embedded_vector_path,
        embedding_size
    ):
        self.embed_size = embedding_size
        self.embedded_dir = pretrained_embedded_vector_path
        if not os.path.isfile(pretrained_embedded_vector_path+".word2vec"):
            glove2word2vec(pretrained_embedded_vector_path, pretrained_embedded_vector_path+".word2vec")
        self.embedding_model = KeyedVectors.load_word2vec_format(pretrained_embedded_vector_path+".word2vec")
        self.text = pd.Series(text)
        self.categories = pd.Series(labels)
        self.classes = self.categories.unique().tolist()
        self.word_counts = None

    def map_word_frequency(self, docs):
        return Counter(itertools.chain(*docs))

    def sentence_to_vec(
        self,
        sentence_list,
        embedding_size,
        a=1e-4
    ):
        sentence_set = []
        for sentence in sentence_list:
            vs = np.zeros(self.embedding_model.get_dimension()) # add all word2vec values into one vector for the sentence
            sentence_length = sentence.len()
            for word in sentence.word_list:
                a_value = a / (a + (self.word_counts[word.text] / self.word_counts_sum)) # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, word.vector)) # vs += sif * word_vector

            vs = np.divide(vs, sentence_length) # weighted average
            sentence_set.append(vs) # add to our existing re-calculated set of sentences

        # calculate PCA of this sentence set
        np.seterr(divide='ignore', invalid='ignore')
        pca = PCA()
        pca.fit(np.array(sentence_set))
        u = pca.components_[0] # the PCA vector
        u = np.multiply(u, np.transpose(u)) # u x uT

        # pad the vector? (occurs if we have less sentences than embeddings_size)
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                u = np.append(u, 0) # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u,vs)
            sentence_vecs.append(np.subtract(vs, sub))

        return sentence_vecs

    def preprocessing(
            self,
            text
    ):
        """Preprocessing of the text to make it more resonant for training
        """
        docs_list = []
        for doc in text:
            sentence_list = []
            sentences = clean_string(doc)
            for words in sentences.split("."):
                word_list = []
                words = preprocess_text(words)
                if words: 
                    for w in words:
                        if w in self.embedding_model.wv:
                            word_list.append(Word(w, self.embedding_model[w]))
                        else:
                            word_list.append(Word(w, np.zeros(self.embed_size)))
                while(len(word_list) < 2):
                    word_list.append(Word(w, np.zeros(self.embed_size)))
                sentence_list.append(Sentence(word_list))
            docs_list.append(sentence_list)
        return docs_list


    def average_vectors(
        self,
        docs_embeddings
    ):
        vectors = []
        for d in docs_embeddings:
            vectors.append(np.mean(d, axis=0))

        return np.array(vectors)


    def preprocess_data(
        self,
        texts,
        labels,
    ):
        docs_list = self.preprocessing(texts)
        docs_embeddings = []

        for doc in docs_list:
            docs_embeddings.append(self.sentence_to_vec(doc, self.embed_size))

        vectors = self.average_vectors(docs_embeddings)

        return vectors, labels

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer " + self.__class__.__name__)
        t0 = time.time()
        self.word_counts = self.map_word_frequency(self.preprocessing(x))
        self.word_counts_sum = sum(self.word_counts.values())
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)


class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

    def __str__(self):
        return self.text + ' : ' + str(self.vector)

    def __repr__(self):
        return self.__str__()


# a sentence, a list of words
class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    # return the length of a sentence
    def len(self) -> int:
        return len(self.word_list)

    def __str__(self):
        word_str_list = [word.text for word in self.word_list]
        return ' '.join(word_str_list)

    def __repr__(self):
        return self.__str__()