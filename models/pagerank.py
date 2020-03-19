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

        tf.reset_default_graph()

        # self.A1 = tf.placeholder(dtype=tf.float32, shape=[None, self.input_shape, self.input_shape],
        #                          name="adjency_matrix_placeholder")
        self.A1 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, shape=[self.input_shape, self.input_shape], name="adjency_matrix_placeholder")
        self.y_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.classes, 1], name="y_placeholder")
        # self.F1 = tf.placeholder(dtype=tf.float32, shape=[self.input_shape, self.input_shape],
        #                         name="features_matrix_placeholder")
        self.F1 = tf.placeholder(dtype=tf.float32, shape=[self.input_shape, 1], name="features_matrix_placeholder")
        # self.F1 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, shape=[self.input_shape, self.input_shape],
        #                         name="features_matrix_placeholder")

        self.W1 = tf.Variable(tf.random_normal([self.input_shape, self.input_shape]), trainable=True, name="weights1")
        self.W2 = tf.Variable(tf.random_normal([self.input_shape, 1]), trainable=True, name="weights2")
        self.W3 = tf.Variable(tf.random_normal([100, self.input_shape]), trainable=True, name="weights3")
        self.W4 = tf.Variable(tf.random_normal([self.classes, 100]), trainable=True, name="weights4")

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
            tf.reduce_sum(self.y_placeholder * tf.log(self.y_clipped) + (1 - self.y_placeholder) * tf.log(1 - self.y_clipped), axis=1, name="loss"))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2, name="optimizer").minimize(self.loss)

        self.init = tf.global_variables_initializer()

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
        vec = sess.run(["activation3:0"],
                             feed_dict={"adjency_matrix_placeholder:0": convert_sparse_matrix_to_sparse_tensor(A),
                                        # "adjency_matrix_placeholder:0": A,
                                        # "features_matrix_placeholder:0": convert_sparse_matrix_to_sparse_tensor(F)
                                        "features_matrix_placeholder:0": F.reshape(-1, 1).todense()
                                        # "features_matrix_placeholder:0": [F[x].reshape(-1, 1).todense() for x in range(F.shape[0])]
                                        })
        return vec[0].reshape(-1)

    def train(self, A, F, y, epochs):
        with tf.Session() as sess:
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
        with tf.Session() as sess:
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
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess,
                                       [tag_constants.SERVING],
                                       os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))

            doc_vec = []
            for i in range(len(A)):
                print("Vectorization: {}/{}".format(i + 1, str(len(A))), end='\r', flush=True)
                doc_vec.append(self._predict_iteration(sess=sess, A=A[i], F=F[i]))
        return doc_vec

    def predict_generator(self, factory_adjency, factory_features, docs):
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess,
                                       [tag_constants.SERVING],
                                       os.path.join(os.path.abspath(os.getcwd()), 'custom_model'))

            doc_vec = []
            counter = 0
            generator_adjency = factory_adjency(docs)
            generator_features = factory_features(docs)
            for A, F in zip(generator_adjency, generator_features):
                print("Vectorization: {}".format(counter + 1), end='\r', flush=True)

                doc_vec.append(self._predict_iteration(sess=sess, A=A, F=F))
                counter += 1
        return doc_vec


def pagerank_train(A, F, y, classes = 5, epochs: int = 2):
    # A1 = tf.placeholder(tf.float32, shape=[A[0].shape[0], A[0].shape[1]])
    A1 = tf.compat.v1.sparse.placeholder(dtype=tf.float32, shape=[A[0].shape[0], A[0].shape[1]])
    y_placeholder = tf.placeholder(tf.float32, shape=[classes, 1])
    # F1 = tf.compat.v1.sparse.placeholder(dtype=tf.float32, shape=[F.shape[0], F.shape[1]])
    F1 = tf.placeholder(dtype=tf.float32, shape=[F.shape[0], F.shape[1]])

    W1 = tf.Variable(tf.random_normal([A[0].shape[0], A[0].shape[1]]), trainable=True, name="weights1")
    W2 = tf.Variable(tf.random_normal([A[0].shape[0], 1]), trainable=True, name="weights2")
    W3 = tf.Variable(tf.random_normal([100, A[0].shape[0]]), trainable=True, name="weights3")
    W4 = tf.Variable(tf.random_normal([classes, 100]), trainable=True, name="weights4")

    F1_dot = tf.sparse.sparse_dense_matmul(F1, A1)
    # F1_dot = tf.multiply(F1, A1)
    F1_dot = tf.sparse.sparse_dense_matmul(tf.contrib.layers.dense_to_sparse(F1_dot), W1)
    # activation1 = tf.nn.elu(F1_dot, name="relu1")

    # F2_dot = tf.sparse.sparse_dense_matmul(tf.contrib.layers.dense_to_sparse(activation1), A1)
    # F2_dot = tf.sparse.sparse_dense_matmul(tf.contrib.layers.dense_to_sparse(F1_dot), W2)
    F2_dot = tf.multiply(F1_dot, A1)
    F2_dot = tf.matmul(F2_dot, W2)
    # activation2 = tf.nn.elu(F2_dot, name="relu2")

    F3_dot = tf.matmul(W3, F2_dot)
    activation3 = tf.nn.elu(F3_dot, name="relu3")

    F4_dot = tf.matmul(W4, activation3)
    activation4 = tf.nn.elu(F4_dot, name="relu4")

    y_pred = tf.nn.softmax(activation4, axis=0)

    y_clipped = tf.clip_by_value(y_pred, 1e-10, 0.9999999)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=y_clipped)
    # loss = tf.losses.softmax_cross_entropy(y_placeholder, F4_dot)
    loss = -tf.reduce_mean(tf.reduce_sum(y_placeholder * tf.log(y_clipped) + (1 - y_placeholder) * tf.log(1 - y_clipped), axis=1))

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_placeholder, logits=F4_dot))

    optimiser = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(loss)

    init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(y_placeholder, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def one_hot(a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def train(A, F, y):
        # create self co-ocurance in adjecency matrix
        I = np.eye(A.shape[0], dtype=float)
        A_hat = A + I

        # calculate degree of each vertex
        D = np.sum(A_hat, axis=1)

        # create diagonal matrix with inversed vertex degree as values
        for d in range(0, D.shape[0]):
            if D[d]:
                D[d] = 1.0 / D[d]

        A_hat = np.multiply(A_hat, D)
        _, c = sess.run([optimiser, loss],
                             feed_dict={A1: A_hat,
                                        F1: F.todense(),
                                        y_placeholder: y
                                        })
        print(sess.run(accuracy, feed_dict={A1: A_hat, F1: F.todense()}))
        return c

    def predict(A, F):
        # create self co-ocurance in adjecency matrix
        I = np.eye(A.shape[0], dtype=float)
        A_hat = A + I

        # calculate degree of each vertex
        D = np.sum(A_hat, axis=1)

        # create diagonal matrix with inversed vertex degree as values
        for d in range(0, D.shape[0]):
            if D[d]:
                D[d] = 1.0 / D[d]

        A_hat = np.multiply(A_hat, D)
        vec = sess.run([activation4],
                             feed_dict={A1: A_hat,
                                        F1: F.todense()
                                        })
        return vec

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(len(A)):
                print("Iteration: {}/{}".format(i + 1, str(len(A))), end='\r', flush=True)

                cost = train(A[i], F, one_hot(y.to_numpy()[i], classes).reshape(classes, 1))

                avg_cost += cost / len(A)
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        doc_vec = []
        for i in range(len(A)):
            doc_vec.append(predict(A[i], F))
        return doc_vec

    # A1 = tf.placeholder(tf.float32, shape=[A_hat.shape[0], A_hat.shape[1]])
    # F1 = tf.placeholder(tf.float32, shape=[F.shape[0], F.shape[1]])
    #
    # A1_F1_dot = tf.math.multiply(A1, F1)
    #
    # init = tf.global_variables_initializer()
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     F_new = sess.run(A1_F1_dot, feed_dict={A1: A_hat, F1: F})
    #     for i in range(num_iterations):
    #         print("Iteration: {}/{}".format(i + 1, num_iterations), end='\r', flush=True)
    #         F_new = sess.run(A1_F1_dot, feed_dict={A1: A_hat, F1: F_new})
    #     return F_new

    # return output
    # A_sparse = sparse.csr_matrix(A)
    # D_hat_sparse = sparse.csr_matrix(D_hat)
    # A_hat_sparse = (A_sparse.T / D_hat_sparse).T

    doc_vec = []
    for j in range(0, len(A)):
        print("Iteration: {}/{}".format(j + 1, len(A)), end='\r', flush=True)

        I = np.eye(A[j].shape[0], dtype=float)
        A_hat = A[j] + I

        # D_hat = np.zeros((A_hat.shape[0]), dtype=float)
        D = np.sum(A_hat, axis=1)

        # create inversed diagonal matrix with vertex degree as values
        for d in range(0, D.shape[0]):
            if D[d]:
                D[d] = 1.0 / D[d]

        # A_hat = np.asmatrix(np.dual.inv(D_hat)) * A_hat
        A_hat = A_hat * D.reshape(-1, 1)

        # A_hat = (A_hat.T / D_hat).T
        # A_hat_sparse = sparse.csr_matrix(A_hat)
        # v_sparse = sparse.csr_matrix(F)
        for i in range(num_iterations):
            # print("Iteration: {}/{}".format(i+1, num_iterations), end='\r', flush=True)
            # v = np.multiply(A_hat, v)
            # A_hat = relu(F.multiply(A_hat))
            A_hat = F.multiply(A_hat)
        doc_vec.append(np.ravel(A_hat.todense().sum(axis=0)))
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
    return tf.SparseTensorValue(indices, coo.data, coo.shape)
