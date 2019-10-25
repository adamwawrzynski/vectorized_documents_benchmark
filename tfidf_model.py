import logging
import time
import pickle
import os
from benchmark_model import BenchmarkModel
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfModel(BenchmarkModel):
    def __init__(
        self,
        n_features=1000,
        max_df=1,
        min_df=1
    ):
        super().__init__()
        self.n_features = n_features
        self.max_df = max_df
        self.min_df = min_df

    def build_model(
        self
    ):
        super().build_model()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=self.max_df,
            decode_error="ignore",
            max_features=self.n_features,
            min_df=self.min_df,
            stop_words='english',
            use_idf=True)

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer " + self.__class__.__name__)
        t0 = time.time()
        self.tfidf_vectorizer.fit(x)
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

    def preprocess_data(
        self,
        dataset
    ):
        logging.info("Transforming dataset " + self.__class__.__name__)
        return self.tfidf_vectorizer.transform(dataset)

    def save(
        self,
        path
    ):
        logging.info("Saving " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        pickle.dump(self.knn,
            open(combined_path + "_knn.pickle", 'wb'))
        pickle.dump(self.tfidf_vectorizer.vocabulary_,
            open(combined_path + "_vec.pickle", 'wb'))
        pickle.dump(self.tfidf_vectorizer.idf_,
            open(combined_path + "_vec_idf.pickle", 'wb'))

    def load(
        self,
        path
    ):
        logging.info("Loading " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.knn = pickle.load(
            open(combined_path + "_knn.pickle", 'rb'))
        self.tfidf_vectorizer = TfidfVectorizer(
            vocabulary=pickle.load(open(combined_path + "_vec.pickle", 'rb')))
        self.tfidf_vectorizer.idf_ = pickle.load(
            open(combined_path + "_vec_idf.pickle", 'rb'))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_knn.pickle") and os.path.isfile(combined_path + "vec.pickle") and os.path.isfile(combined_path + "_vec_idf.pickle")