
import logging
import time
import pickle
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
        self.tfidf_vectorizer = TfidfVectorizer(max_df=max_df,
            decode_error="ignore",
            max_features=n_features,
            min_df=min_df,
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
        name
    ):
        logging.info("Saving " + self.__class__.__name__)
        pickle.dump(self.knn, open(name+"_knn.pickle", 'wb'))
        pickle.dump(self.tfidf_vectorizer.vocabulary_, open(name+"_vec.pickle", 'wb'))
        pickle.dump(self.tfidf_vectorizer.idf_, open(name+"_vec_idf.pickle", 'wb'))

    def load(
        self,
        name
    ):
        logging.info("Loading " + self.__class__.__name__)
        self.knn = pickle.load(open(name+"_knn.pickle", 'rb'))
        self.tfidf_vectorizer = TfidfVectorizer(vocabulary=pickle.load(open(name+"_vec.pickle", 'rb')))
        self.tfidf_vectorizer.idf_ = pickle.load(open(name+"_vec_idf.pickle", 'rb'))