import logging
import time
import pickle
import os
from models.benchmark_model import BenchmarkModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from preprocess import process_dataset


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
        self.svd_features = svd_features
        self.n_features = n_features
        self.n_iter = n_iter
        self.max_df = max_df
        self.min_df = min_df

    def build_model(
        self
    ):
        super().build_model()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=self.max_df,
            max_features=self.n_features,
            min_df=self.min_df,
            stop_words='english',
            use_idf=True)

        self.model = TruncatedSVD(self.svd_features, n_iter=self.n_iter)

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        t0 = time.time()
        processed_dataset = process_dataset(x)
        processed_dataset = processed_dataset.map(lambda x: ' '.join(word for word in x))
        tfidf = self.tfidf_vectorizer.fit_transform(processed_dataset.values.astype('U'))
        self.model.fit(tfidf)
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

    def preprocess_data(
        self,
        dataset,
        y_dataset
    ):
        logging.info("Transforming data on " + self.__class__.__name__)
        processed_dataset = process_dataset(dataset)
        processed_dataset = processed_dataset.map(lambda x: ' '.join(word for word in x))
        tfidf = self.tfidf_vectorizer.transform(processed_dataset.values.astype('U'))
        return self.model.transform(tfidf)

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
        self.clf = pickle.load(
            open(combined_path + "_clf.pickle", 'rb'))
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
        return os.path.isfile(combined_path + "_clf.pickle") and \
            os.path.isfile(combined_path + "_model.pickle") and \
            os.path.isfile(combined_path + "_vec.pickle") and \
            os.path.isfile(combined_path + "_vec_idf.pickle")