import logging
import pickle
import os
from models.benchmark_model import BenchmarkModel
from models.psif import PSIF

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


class PSIFSklearnVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        X_copy = list(X)
        self.model.fit(X_copy, y)
        return self

    def transform(self, X, *_):
        X_copy = list(X)
        vectors, _ = self.model.transform(X_copy, _)
        return vectors


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
        self.psif = PSIF(
            self.pretrained_embedded_vector_path,
            self.embedding_size,
            self.num_clusters)
        self.pipeline = self.pipeline = Pipeline(steps=[
            ("vectorizer", PSIFSklearnVectorizer(self.psif)),
            ("classifier", self.clf)
        ])

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
        self.build_model()

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle")