import logging
import time
import pickle
import os
from keras.models import load_model
from keras.models import Model
from keras import backend as K
from benchmark_model import BenchmarkModel
from sif import SIF


class SIFModel(BenchmarkModel):
    def __init__(
        self,
        text,
        labels,
        pretrained_embedded_vector_path,
        embedding_size
    ):
        super().__init__()
        self.text = text
        self.labels = labels
        self.pretrained_embedded_vector_path = pretrained_embedded_vector_path
        self.embedding_size = embedding_size

    def build_model(
        self
    ):
        super().build_model()
        self.model = SIF(
            self.text,
            self.labels,
            self.pretrained_embedded_vector_path,
            self.embedding_size)

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        t0 = time.time()
        self.model.train(x, y)
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

    def fit(
        self,
        x,
        y
    ):
        logging.info("Training kNN classifier")
        embedded_x = self.preprocess_data(x, y)
        return self.clf.fit(embedded_x ,y)

    def preprocess_data(
        self,
        dataset,
        y_dataset
    ):

        x, _ = self.model.preprocess_data(dataset, y_dataset)
        return x

    def save(
        self,
        path
    ):
        logging.info("Saving " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        pickle.dump(self.clf,
            open(combined_path + "_clf.pickle", 'wb'))

    def load(
        self,
        path
    ):
        logging.info("Loading " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.clf = pickle.load(
            open(combined_path + "_clf.pickle", 'rb'))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle")