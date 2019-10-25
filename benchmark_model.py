import logging
import pickle
import os
from abc import ABC
from abc import abstractmethod
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


class BenchmarkModel(ABC):
    def __init__(
        self,
        n_neighbors=5,
        algorithm='brute',
        metric='cosine',
        n_jobs=1
    ):
        self.model = None   # assigned in concrete classes
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            n_jobs=n_jobs)

    @abstractmethod
    def preprocess_data(
        self,
        dataset
    ):
        raise Exception("Not implemented!")

    @abstractmethod
    def train(
        self,
        x,
        y=None
    ):
        raise Exception("Not implemented!")

    def fit(
        self,
        x,
        y
    ):
        logging.info("Training kNN classifier")
        return self.knn.fit(self.preprocess_data(x), y)

    def predict(
        self,
        x
    ):
        logging.info("Predict on kNN classifier")
        return self.knn.predict(self.preprocess_data(x))

    def evaluate(
        self,
        x,
        y
    ):
        y_pred = self.predict(x)
        result = metrics.accuracy_score(y, y_pred)
        logging.info("Accuracy: %.3f" % result)
        return result

    def save(
        self,
        name
    ):
        logging.info("Saving " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        pickle.dump(self.knn,
            open(combined_path + "_knn.pickle", 'wb'))
        pickle.dump(self.model,
            open(combined_path + "_model.pickle", 'wb'))

    def load(
        self,
        path
    ):
        logging.info("Loading " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.knn = pickle.load(
            open(combined_path + "_knn.pickle", 'rb'))
        self.model = pickle.load(
            open(combined_path + "_model.pickle", 'rb'))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_knn.pickle") and os.path.isfile(combined_path + "_model.pickle")