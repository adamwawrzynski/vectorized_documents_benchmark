import logging
import pickle
import os
from abc import ABC
from abc import abstractmethod
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class BenchmarkModel(ABC):
    def __init__(
        self,
    ):
        super().__init__()

    def build_model(
        self
    ):
        self.model = None   # assigned in concrete classes
        self.clf = SVC(
            kernel='linear',
            class_weight='balanced',
        )

    @abstractmethod
    def preprocess_data(
        self,
        dataset,
        y_dataset
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
        logging.info("Training classifier")
        return self.clf.fit(self.preprocess_data(x, y), y)

    def predict(
        self,
        x,
        y=None
    ):
        logging.info("Predict on classifier")
        return self.clf.predict(self.preprocess_data(x, y))

    def evaluate(
        self,
        x,
        y=None
    ):
        y_pred = self.predict(x, y)
        result = metrics.accuracy_score(y, y_pred)
        logging.info("Accuracy: %.3f" % result)
        return result

    def save(
        self,
        path
    ):
        logging.info("Saving " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        pickle.dump(self.clf,
            open(combined_path + "_clf.pickle", 'wb'))
        pickle.dump(self.model,
            open(combined_path + "_model.pickle", 'wb'))

    def load(
        self,
        path
    ):
        logging.info("Loading " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.clf = pickle.load(
            open(combined_path + "_clf.pickle", 'rb'))
        self.model = pickle.load(
            open(combined_path + "_model.pickle", 'rb'))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle") and \
            os.path.isfile(combined_path + "_model.pickle")