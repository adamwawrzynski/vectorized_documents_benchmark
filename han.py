import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.models import load_model
import nltk
import re
import sys
from sklearn.metrics import roc_auc_score
from nltk import tokenize
from attention_with_context import AttentionWithContext
from sklearn.utils import shuffle
import re
import time
import os
import logging
from sklearn import metrics

# Implementation based on: https://github.com/Hsankesara/DeepResearch
# Author: https://github.com/Hsankesara

class HAN(object):
    """
    HAN model is implemented here.
    """
    def __init__(
        self,
        text,
        labels,
        pretrained_embedded_vector_path,
        max_features,
        max_senten_len,
        max_senten_num,
        embedding_size,
        num_categories=None,
        validation_split=0.2, 
        verbose=1
    ):
        """Initialize the HAN module
        Keyword arguments:
        text -- list of the articles for training.
        labels -- labels corresponding the given `text`.
        pretrained_embedded_vector_path -- path of any pretrained vector
        max_features -- max features embeddeding matrix can have. To more checkout https://keras.io/layers/embeddings/
        max_senten_len -- maximum sentence length. It is recommended not to use the maximum one but the one that covers 0.95 quatile of the data.
        max_senten_num -- maximum number of sentences. It is recommended not to use the maximum one but the one that covers 0.95 quatile of the data.
        embedding_size -- size of the embedding vector
        num_categories -- total number of categories.
        validation_split -- train-test split. 
        verbose -- how much you want to see.
        """
        self.verbose = verbose
        self.max_features = max_features
        self.max_senten_len = max_senten_len
        self.max_senten_num = max_senten_num
        self.embed_size = embedding_size
        self.validation_split = validation_split
        self.embedded_dir = pretrained_embedded_vector_path
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token=True)
        # Initialize default hyperparameters
        # You can change it using `set_hyperparameters` function 
        self.hyperparameters = {
            'l2_regulizer': 1e-13,
            'dropout_regulizer' : 0.5,
            'rnn' : LSTM,
            'rnn_units' : 100,
            'dense_units': 100,
            'activation' : 'softmax',
            'optimizer' : 'adam',
            'metrics' : ['acc'],
            'loss': 'categorical_crossentropy'
        }
        self.embedding_index = self.add_glove_model()
        self.build_model(text, labels, num_categories)

    def build_model(
        self,
        text,
        labels,
        num_categories=None
    ):
        try:
            self.text = pd.Series(text)
            self.categories = pd.Series(labels)
            self.classes = self.categories.unique().tolist()
            if num_categories is not None:
                assert (num_categories == len(self.classes))
            assert (self.text.shape[0] == self.categories.shape[0])
            self.data, self.labels = self.preprocessing(self.text, self.categories)
            self.x_train, self.y_train, self.x_val, self.y_val = self.split_dataset()
            self.set_model()
        except AssertionError:
            print('Input and label data must be of same size')

    def set_hyperparameters(
        self,
        tweaked_instances
    ):
        """Set hyperparameters of HAN model.
        Keywords arguemnts:
        tweaked_instances -- dictionary of all those keys you want to change
        """
        for  key, value in tweaked_instances.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
            else:
                raise KeyError(key + ' does not exist in hyperparameters')
            self.set_model()

    def clean_string(
        self,
        string
    ):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()

    def add_dataset(
        self,
        text,
        labels
    ):
        try:
            self.text = pd.concat([self.text, pd.Series(text)])
            self.categories = pd.concat([self.categories, pd.Series(labels)])
            assert (len(self.classes) == self.categories.unique().tolist())
        except AssertionError:
            print("New class cannot be added in this manner")

    def processed_data(
        self,
        texts,
        labels,
        paras
    ):
        data = np.zeros((len(texts), self.max_senten_num,
                        self.max_senten_len), dtype='int32')
        for i, sentences in enumerate(paras):
            for j, sent in enumerate(sentences):
                if j < self.max_senten_num:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k < self.max_senten_len and word in self.tokenizer.word_index and self.tokenizer.word_index[word] < self.max_features:
                            data[i, j, k] = self.tokenizer.word_index[word]
                            k = k+1
        if self.verbose == 1:
            print('Total %s unique tokens.' % len(self.tokenizer.word_index))
        labels = pd.get_dummies(labels)
        if self.verbose == 1:
            print('Shape of data tensor:', data.shape)
            print('Shape of labels tensor:', labels.shape)
        assert (len(self.classes) == labels.shape[1])
        assert (data.shape[0] == labels.shape[0])
        return data, labels


    def preprocessing(
        self,
        text,
        labels
    ):
        """Preprocessing of the text to make it more resonant for training
        """
        paras = []
        texts = []
        for sentence in text:
            text = self.clean_string(sentence)
            texts.append(text)
            sentences = tokenize.sent_tokenize(text)
            paras.append(sentences)
        self.tokenizer.fit_on_texts(texts)
        return self.processed_data(texts, pd.Series(labels), paras)

    def split_dataset(
        self
    ):
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels.iloc[indices]
        nb_validation_samples = int(self.validation_split * self.data.shape[0])

        x_train = self.data[:-nb_validation_samples]
        y_train = self.labels[:-nb_validation_samples]
        x_val = self.data[-nb_validation_samples:]
        y_val = self.labels[-nb_validation_samples:]
        if self.verbose == 1:
            print('Number of positive and negative reviews in traing and validation set')
            print(y_train.columns.tolist())
            print(y_train.sum(axis=0).tolist())
            print(y_val.sum(axis=0).tolist())
        return x_train, y_train, x_val, y_val

    def get_model(
        self
    ):
        """
        Returns the HAN model so that it can be used as a part of pipeline
        """
        return self.model

    def load_model(
        self,
        path
    ):
        custom_objects={'AttentionWithContext': AttentionWithContext}
        self.model = load_model(path + ".h5", custom_objects=custom_objects)
        self.wordEncoder = load_model(path + "_word.h5", custom_objects=custom_objects)

    def save_model(
        self,
        path
    ):
        self.model.save(path + ".h5")
        self.wordEncoder.save(path + "_word.h5")

    def add_glove_model(
        self
    ):
        """
        Read and save Pretrained Embedding model
        """
        embeddings_index = {}
        try:
            f = open(self.embedded_dir)
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                assert (coefs.shape[0] == self.embed_size)
                embeddings_index[word] = coefs
            f.close()
        except OSError:
            print('Embedded file does not found')
            exit()
        except AssertionError:
            print("Embedding vector size does not match with given embedded size")
        return embeddings_index

    def get_embedding_matrix(
        self
    ):
        """
        Returns Embedding matrix
        """
        embedding_matrix = np.random.random((self.max_features, self.embed_size))
        absent_words = 0
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                absent_words += 1
        if self.verbose == 1:
            print('Total absent words are', absent_words, 'which is', "%0.2f" %
                (absent_words * 100 / len(self.tokenizer.word_index)), '% of total words')
        return embedding_matrix

    def get_embedding_layer(
        self
    ):
        """
        Returns Embedding layer
        """
        embedding_matrix = self.get_embedding_matrix()
        return Embedding(
            self.max_features,
            self.embed_size,
            weights=[embedding_matrix],
            input_length=self.max_senten_len,
            trainable=False,
            name="embedding_input",
            mask_zero=True)

    def set_model(
        self
    ):
        """
        Set the HAN model according to the given hyperparameters
        """
        if self.hyperparameters['l2_regulizer'] is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = regularizers.l2(self.hyperparameters['l2_regulizer'])
        if self.hyperparameters['dropout_regulizer'] is None:
            dropout_regularizer = 1
        else:
            dropout_regularizer = self.hyperparameters['dropout_regulizer']
        word_input = Input(shape=(self.max_senten_len,), dtype='float32')
        word_sequences = self.get_embedding_layer()(word_input)
        word_lstm = Bidirectional(
            self.hyperparameters['rnn'](self.hyperparameters['rnn_units'], return_sequences=True, kernel_regularizer=kernel_regularizer))(word_sequences)
        word_dense = TimeDistributed(
            Dense(self.hyperparameters['dense_units'], kernel_regularizer=kernel_regularizer))(word_lstm)
        word_att = AttentionWithContext()(word_dense)
        self.wordEncoder = Model(word_input, word_att)

        sent_input = Input(shape=(self.max_senten_num, self.max_senten_len), dtype='float32')
        sent_encoder = TimeDistributed(self.wordEncoder)(sent_input)
        sent_lstm = Bidirectional(self.hyperparameters['rnn'](
            self.hyperparameters['rnn_units'], return_sequences=True, kernel_regularizer=kernel_regularizer))(sent_encoder)
        sent_dense = TimeDistributed(
            Dense(self.hyperparameters['dense_units'], kernel_regularizer=kernel_regularizer))(sent_lstm)
        sent_att = Dropout(dropout_regularizer, name="embedding_output")(
            AttentionWithContext()(sent_dense))
        sent_att_dropout = Dropout(dropout_regularizer)(sent_att)
        preds = Dense(len(self.classes), activation=self.hyperparameters['activation'])(sent_att_dropout)
        self.model = Model(sent_input, preds)

        self.model.compile(
            loss=self.hyperparameters['loss'], optimizer=self.hyperparameters['optimizer'], metrics=self.hyperparameters['metrics'])
        
    def train(
        self,
        epochs, 
        batch_size
    ):
        """Training the model
        epochs -- Total number of epochs
        batch_size -- size of a batch
        """
        self.model.fit(
                    self.x_train,
                    self.y_train,
                    validation_data=(self.x_val, self.y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose = self.verbose,
                    shuffle=True)

    def evaluate(
        self,
        dataset, 
        y_dataset
    ):
        self.text = pd.Series(dataset)
        self.categories = pd.Series(y_dataset)
        self.classes = self.categories.unique().tolist()
        data, _ = self.preprocessing(self.text, y_dataset)

        y_pred = self.model.predict(data)
        result = metrics.accuracy_score(y_dataset, y_pred)
        logging.info("Accuracy: %.3f" % result)
        return result

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_knn.pickle") and \
            os.path.isfile(combined_path + ".h5")