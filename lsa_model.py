import logging
import time
import pickle
import os
from benchmark_model import BenchmarkModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class LSAModel(BenchmarkModel):
    def __init__(
        self,
        svd_features=200,
        n_features=1000,
        n_iter=40,
        max_df=1,
        min_df=1
    ):
        super().__init__()
        self.tfidf_vectorizer = TfidfVectorizer(max_df=max_df,
            max_features=n_features,
            min_df=min_df,
            stop_words='english',
            use_idf=True)
        svd = TruncatedSVD(svd_features, n_iter=n_iter)
        self.model = make_pipeline(svd, Normalizer(copy=False))

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        t0 = time.time()
        tfidf = self.tfidf_vectorizer.fit_transform(x)
        self.model.fit(tfidf)
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

    def preprocess_data(
        self,
        dataset
    ):
        logging.info("Transforming data on " + self.__class__.__name__)
        tfidf = self.tfidf_vectorizer.transform(dataset)
        return self.model.transform(tfidf)

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
        self.model = pickle.load(
            open(combined_path + "_model.pickle", 'rb'))
        self.tfidf_vectorizer = TfidfVectorizer(
            vocabulary=pickle.load(open(combined_path + "_vec.pickle", 'rb')))
        self.tfidf_vectorizer.idf_ = pickle.load(
            open(combined_path + "_vec_idf.pickle", 'rb'))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_knn.pickle") and os.path.isfile(combined_path + "_model.pickle") and os.path.isfile(combined_path + "_vec.pickle") and os.path.isfile(combined_path + "_vec_idf.pickle")