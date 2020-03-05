import logging
import pickle
import os
from models.benchmark_model import BenchmarkModel
from models.sif import SIF

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


class SIFSklearnVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        X_copy = list(X)
        self.model.train(X_copy, y)
        return self

    def transform(self, X, *_):
        X_copy = list(X)
        vectors, _ = self.model.preprocess_data(X_copy, _)
        return vectors


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
        self.sif = SIF(
            self.text,
            self.labels,
            self.pretrained_embedded_vector_path,
            self.embedding_size)
        self.pipeline = self.pipeline = Pipeline(steps=[
            ("vectorizer", SIFSklearnVectorizer(self.sif)),
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