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


class CustomModel(object):
    def __init__(self, input_shape, classes):
        self.classes = classes
        self.input_shape = input_shape

        tf.compat.v1.reset_default_graph()

        # self.A1 = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape, self.input_shape],
        #                          name="adjency_matrix_placeholder")
        self.A1 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, shape=[self.input_shape, self.input_shape], name="adjency_matrix_placeholder")
        self.y_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.classes, 1], name="y_placeholder")
        # self.F1 = tf.placeholder(dtype=tf.float32, shape=[self.input_shape, self.input_shape],
        #                         name="features_matrix_placeholder")
        self.F1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.input_shape, 1], name="features_matrix_placeholder")
        # self.F1 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, shape=[self.input_shape, self.input_shape],
        #                         name="features_matrix_placeholder")

        self.W1 = tf.Variable(tf.random.normal([self.input_shape, self.input_shape]), trainable=True, name="weights1")
        self.W2 = tf.Variable(tf.random.normal([self.input_shape, 1]), trainable=True, name="weights2")
        self.W3 = tf.Variable(tf.random.normal([100, self.input_shape]), trainable=True, name="weights3")
        self.W4 = tf.Variable(tf.random.normal([self.classes, 100]), trainable=True, name="weights4")

        # self.F1_dot_1 = tf.multiply(self.F1, self.A1, name="F1_dot_1")
        # self.F1_dot_1 = tf.einsum('aij,aij->aij', self.A1, self.F1)

        self.F1_dot_1 = self.F1.__mul__(self.A1)
        # self.F1_dot_1 = self.A1.__mul__(self.F1)
        # self.F1_dot_2 = tf.einsum('aij,jk->aik', self.F1_dot_1, self.W1)

        self.F1_dot_2 = tf.sparse.sparse_dense_matmul(self.F1_dot_1, self.W1, name="F1_dot_2")

        # self.F2_dot_1 = tf.contrib.layers.dense_to_sparse(self.F1_dot_2).__mul__(self.A1)
        # self.F2_dot_1 = tf.einsum('aij,aij->aij', self.A1, self.F1_dot_2)

        self.F2_dot_1 = self.A1.__mul__(self.F1_dot_2)
        # self.F2_dot_1 = tf.multiply(self.F1_dot_2, self.A1, name="F2_dot_1")
        self.F2_dot_2 = tf.sparse.sparse_dense_matmul(self.F2_dot_1, self.W2, name="F2_dot_2")
        # self.F2_dot_2 = tf.einsum('aij,jk->aik', self.F2_dot_1, self.W2)

        # self.F3_dot = tf.einsum('ij,ajk->aik', self.W3, self.F2_dot_2)
        self.F3_dot = tf.matmul(self.W3, self.F2_dot_2)
        self.activation3 = tf.nn.elu(self.F3_dot, name="activation3")

        # self.F4_dot = tf.einsum('ij,ajk->aik', self.W4, self.activation3)
        self.F4_dot = tf.matmul(self.W4, self.F3_dot)
        self.activation4 = tf.nn.elu(self.F4_dot, name="activation4")
        self.y_pred = tf.nn.softmax(self.activation4, axis=0, name="y_pred")

        self.y_clipped = tf.clip_by_value(self.y_pred, 1e-10, 0.9999999, name="y_clipped")
        self.loss = -tf.reduce_mean(
            tf.reduce_sum(self.y_placeholder * tf.math.log(self.y_clipped) + (1 - self.y_placeholder) * tf.math.log(1 - self.y_clipped), axis=1, name="loss"))

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2, name="optimizer").minimize(self.loss)

        self.init = tf.compat.v1.global_variables_initializer()

        self.correct_prediction = tf.equal(tf.argmax(self.y_placeholder, 1), tf.argmax(self.y_pred, 1), name="correct_predictions")
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")

    def _convert_to_one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).reshape(num_classes, 1)

    def _train_iteration(self, sess, A, F, y):
        # A_hat = self._normalize_adjency_matrix(A)
        _, loss = sess.run([self.optimizer, self.loss],
                             feed_dict={self.A1: convert_sparse_matrix_to_sparse_tensor(A),
                                        # self.A1: A,
                                        # self.F1: convert_sparse_matrix_to_sparse_tensor(F),
                                        self.F1: F.reshape(-1, 1).todense(),
                                        # self.F1: [F[x].reshape(-1, 1).todense() for x in range(F.shape[0])],
                                        self.y_placeholder: y
                                        })
        return loss

    def _predict_iteration(self, sess, A, F):
        # A_hat = self._normalize_adjency_matrix(A)
        indices, values, shape = convert_sparse_matrix_to_values(A)
        vec = sess.run(["activation3:0"],
                             feed_dict={# "adjency_matrix_placeholder/shape:0": shape,
                                        "adjency_matrix_placeholder/values:0": values,
                                        "adjency_matrix_placeholder/indices:0": indices,
                                        # "adjency_matrix_placeholder:0": A,
                                        # "features_matrix_placeholder:0": convert_sparse_matrix_to_sparse_tensor(F)
                                        "features_matrix_placeholder:0": F.reshape(-1, 1).todense()
                                        # "features_matrix_placeholder:0": [F[x].reshape(-1, 1).todense() for x in range(F.shape[0])]
                                        })
        return vec[0].reshape(-1)

    def train(self, A, F, y, epochs):
        with tf.compat.v1.Session() as sess:
            sess.run(self.init)
            for epoch in range(epochs):
                avg_cost = 0
                for i in range(len(A)):
                    print("Iteration: {}/{}".format(i + 1, str(len(A))), end='\r', flush=True)

                    cost = self._train_iteration(
                        sess=sess,
                        A=A[i],
                        F=F[i],
                        y=self._convert_to_one_hot(y.to_numpy()[i], self.classes)
                    )

                    avg_cost += cost / len(A)
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            inputs = {
                "adjency_matrix_placeholder": self.A1,
                "features_matrix_placeholder": self.F1,
                "y_placeholder": self.y_placeholder
            }
            outputs = {"activation3": self.activation3}
            if os.path.isdir(os.path.join(os.path.abspath(os.getcwd()), 'custom_model')):
                shutil.rmtree(os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))

            builder = saved_model.builder.SavedModelBuilder(os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))
            signature = predict_signature_def(inputs=inputs, outputs=outputs)
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})
            builder.save()

    def train_generator(self, factory_adjency, factory_features, docs, y, epochs):
        with tf.compat.v1.Session() as sess:
            sess.run(self.init)
            for epoch in range(epochs):
                avg_cost = 0
                counter = 0
                generator_adjency = factory_adjency(docs)
                generator_features = factory_features(docs)
                for A, F in zip(generator_adjency, generator_features):
                    print("Iteration: {}".format(counter + 1), end='\r', flush=True)

                    cost = self._train_iteration(
                        sess=sess,
                        A=A,
                        F=F,
                        y=self._convert_to_one_hot(y.to_numpy()[counter], self.classes)
                    )

                    avg_cost = avg_cost + cost
                    counter += 1
                avg_cost /= counter
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
                if avg_cost < 0.01:
                    break

            inputs = {
                "adjency_matrix_placeholder": self.A1,
                "features_matrix_placeholder": self.F1,
                "y_placeholder": self.y_placeholder
            }
            outputs = {"activation3": self.activation3}
            if os.path.isdir(os.path.join(os.path.abspath(os.getcwd()), 'custom_model')):
                shutil.rmtree(os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))

            builder = saved_model.builder.SavedModelBuilder(
                os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))
            signature = predict_signature_def(inputs=inputs, outputs=outputs)
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})
            builder.save()

    def predict(self, A, F):
        with tf.compat.v1.Session() as sess:
            tf.saved_model.loader.load(sess,
                                       [tag_constants.SERVING],
                                       os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))

            doc_vec = []
            for i in range(len(A)):
                print("Vectorization: {}/{}".format(i + 1, str(len(A))), end='\r', flush=True)
                doc_vec.append(self._predict_iteration(sess=sess, A=A[i], F=F[i]))
        return doc_vec

    def predict_generator(self, factory_adjency, factory_features, docs):
        with tf.compat.v1.Session() as sess:
            tf.saved_model.loader.load(sess,
                                       [tag_constants.SERVING],
                                       os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))

            # for op in tf.get_default_graph().get_operations():
            #     print(str(op.name))

            doc_vec = []
            counter = 0
            generator_adjency = factory_adjency(docs)
            generator_features = factory_features(docs)
            for A, F in zip(generator_adjency, generator_features):
                print("Vectorization: {}".format(counter + 1), end='\r', flush=True)

                doc_vec.append(self._predict_iteration(sess=sess, A=A, F=F))
                counter += 1
        return doc_vec


def pmi(text, window=2):
    Vcount = CountVectorizer(analyzer='word', ngram_range=(1, window), stop_words='english')
    countMatrix = Vcount.fit_transform([text])

    # all unigrams and bigrams
    feature_names = Vcount.get_feature_names()

    # finding all bigrams
    featureNgrams = [item for item in Vcount.get_feature_names() if len(item.split()) == window]

    # document term matrix
    arrays = countMatrix.toarray()

    # term document matrix
    arrayTrans = arrays.transpose()

    from collections import defaultdict
    PMIMatrix = defaultdict(dict)

    import math

    i = 0
    PMIMatrix = defaultdict(dict)
    for item in featureNgrams:
        words = item.split()
        bigramLength = len(np.where(arrayTrans[feature_names.index(item)] > 0)[0])
        if bigramLength < window:
            continue
        for index1, index2 in zip(range(0, window), range(0, window)):
            if index1 == index2:
                continue
            word0Length = len(np.where(arrayTrans[feature_names.index(words[index1])] > 0)[0])
            word1Length = len(np.where(arrayTrans[feature_names.index(words[index2])] > 0)[0])
            try:
                PMIMatrix[words[index1]][words[index2]] = (len(text) * math.log(1.0 * bigramLength, 2)) / (
                        1.0 * math.log(word0Length, 2) * math.log(1.0 * word1Length, 2))
            except:
                print(bigramLength, word0Length, word1Length)

    return PMIMatrix


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.compat.v1.SparseTensorValue(indices, coo.data, coo.shape)


def convert_sparse_matrix_to_values(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return indices, coo.data, coo.shape
