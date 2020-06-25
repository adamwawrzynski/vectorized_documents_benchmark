from numpy import array
from numpy import asarray
from numpy import zeros
import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers import Embedding
from keras.models import Model
from keras.layers import multiply
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from models.attention_with_context import AttentionWithContext
from keras.layers import Embedding, Input, Dense, GRU, Bidirectional, TimeDistributed, Dropout

# bbc: 97.39
# bbcsport: 97.55

class TfContext(object):
    def __init__(
            self,
            vocabulary,
            word2idx,
            max_size,
            classes
    ):
        self.vocabulary = vocabulary
        self.word2idx = word2idx
        self.max_size = max_size
        self.classes = classes
        self.build_model()

    def tokenize_dataset(
            self,
            docs
    ):
        id_seq = [self.tokenize_data(doc) for doc in docs]
        return pad_sequences(id_seq, maxlen=self.max_size, padding='post')

    def tokenize_data(
            self,
            text
    ):
        # return [self.word2idx[w] for counter, w in enumerate(text.split()) if counter < self.max_size]
        id_seq = []
        for counter, w in enumerate(text.split()):
            if counter < self.max_size and w in self.word2idx:
                id_seq.append(self.word2idx[w])
            elif w not in self.word2idx:
                id_seq.append(0)
        return id_seq

    def build_model(self):

        model1_input = Input(shape=(len(self.vocabulary),))

        model2_input = Input(shape=(self.max_size,))
#        model2 = Embedding(len(self.vocabulary), 50, input_length=self.max_size, trainable=True)(model2_input)
#        model2 = Bidirectional(GRU(50, return_sequences=True))(model2)
#        model2 = TimeDistributed(Dense(100, activation='relu'))(model2)
#        model2 = AttentionWithContext()(model2)
#        model2 = Flatten()(model2)
#        model2 = AttentionWithContext()(model2)
        model2 = Dense(int(len(self.vocabulary) / 4), activation='relu')(model2_input)

        model2 = Dense(len(self.vocabulary), activation='sigmoid')(model2)

        mul = multiply([model1_input, model2])
        output = Dense(self.classes, activation='softmax')(mul)


        self.model = Model(inputs=[model1_input, model2_input], outputs=[output])

        # compile the model
        optimizer = Adam(lr=0.005)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        # summarize the model
        print(self.model.summary())

    def train(self, X, features, y):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        self.model.fit([features, self.tokenize_dataset(X)], y, validation_split=0.1, epochs=50, callbacks=[es])

    def predict(self, X, features):
        self.vec_model = Model(inputs=self.model.input, outputs=self.model.layers[-1].output)
        return self.vec_model.predict([features, self.tokenize_dataset(X)])
        # self.model.layers[-1].predict([features, self.tokenize_datset(X)])
