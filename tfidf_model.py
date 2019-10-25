
import logging
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

    def build_vocabulary(
        self,
        dataset
    ):
        logging.info("Building vectorizer " + self.__class__.__name__)
        return self.tfidf_vectorizer.fit_transform(dataset)

    def preprocess_data(
        self,
        dataset
    ):
        logging.info("Transforming dataset " + self.__class__.__name__)
        return self.tfidf_vectorizer.transform(dataset)