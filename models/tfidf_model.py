import logging
import time
import pickle
import os
from models.benchmark_model import BenchmarkModel
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import process_string, preprocess_text, process_dataset, TextPreprocessor
from sklearn.pipeline import Pipeline

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
        self.build_pipeline()

    def build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ("preprocess", TextPreprocessor()),
            ("vectorizer", self.tfidf_vectorizer),
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
        self.tfidf_vectorizer = TfidfVectorizer(
            vocabulary=pickle.load(open(combined_path + "_vec.pickle", 'rb')))
        self.tfidf_vectorizer.idf_ = pickle.load(
            open(combined_path + "_vec_idf.pickle", 'rb'))
        self.build_model()

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle") and \
            os.path.isfile(combined_path + "vec.pickle") and \
            os.path.isfile(combined_path + "_vec_idf.pickle")