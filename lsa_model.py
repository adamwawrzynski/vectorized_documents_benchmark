import logging
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
        self.lsa = make_pipeline(svd, Normalizer(copy=False))

    def build_vocabulary(
        self,
        dataset
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        tfidf = self.tfidf_vectorizer.fit_transform(dataset)
        self.lsa.fit(tfidf)

    def preprocess_data(
        self,
        dataset
    ):
        logging.info("Transforming data on " + self.__class__.__name__)
        tfidf = self.tfidf_vectorizer.transform(dataset)
        return self.lsa.transform(tfidf)