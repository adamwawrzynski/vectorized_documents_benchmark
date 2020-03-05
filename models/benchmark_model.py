import logging
import os
import pickle
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator


class BenchmarkModel(BaseEstimator):
    def __init__(
        self,
        n_neighbors=3,
        algorithm='brute',
        metric='cosine',
        n_jobs=1
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.n_jobs = n_jobs

    def build_model(
        self
    ):
        self.clf_svc = SVC(
            kernel='linear',
            class_weight='balanced')
        # CalibartedClassifierCV is used in order to use eli5 module to explain model predictions
        self.clf = CalibratedClassifierCV(self.clf_svc)
        self.pipeline = None

    def fit(
        self,
        x,
        y
    ):
        logging.info("Training classifier " + self.__class__.__name__)
        return self.pipeline.fit(x, y)

    # sklearn compliant API function
    def predict_proba(
        self,
        x
    ):
        return self.pipeline.predict_proba(x)

    def predict(
        self,
        x,
        y=None
    ):
        logging.info("Predict on classifier " + self.__class__.__name__)
        return self.pipeline.predict(x)

    def evaluate(
        self,
        x,
        y=None
    ):
        y_pred = self.pipeline.predict(x)
        result = metrics.accuracy_score(y, y_pred)
        logging.info("Accuracy: %.3f" % result)
        return result

    def score(self, x, y):
        return self.evaluate(x, y)

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