import logging
import pickle
import os
from models.benchmark_model import BenchmarkModel
from models.psif import PSIF
import time


class PSIFModel(BenchmarkModel):
    def __init__(
        self,
        pretrained_embedded_vector_path,
        embedding_size,
        num_clusters
    ):
        super().__init__()
        self.pretrained_embedded_vector_path = pretrained_embedded_vector_path
        self.embedding_size = embedding_size
        self.num_clusters = num_clusters

    def build_model(
        self
    ):
        super().build_model()
        self.model = PSIF(
            self.pretrained_embedded_vector_path,
            self.embedding_size,
            self.num_clusters)


    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        t0 = time.time()
        self.model.fit(x, y)
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

        x, _ = self.model.transform(dataset, y_dataset)
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