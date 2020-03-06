import logging
import time
import pickle
import os
from models.benchmark_model import BenchmarkModel
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import process_string, preprocess_text, process_dataset, TextPreprocessor
from sklearn.pipeline import Pipeline
from sklearn import metrics


class BOWModel(BenchmarkModel):
    def __init__(
            self,
            max_features,
            max_df,
            min_df,
    ):
        super().__init__()
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.processor = TextPreprocessor()

    def build_model(
            self
    ):
        super().build_model()
        self.count_vectorizer = CountVectorizer(
            # preprocessor=process_string, # eli5 enforce use of sklearn transformation framework
            max_features=self.max_features,
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words='english')
        self.build_pipeline()

    def build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ("preprocess", TextPreprocessor()),
            ("vectorizer", self.count_vectorizer),
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
        pickle.dump(self.count_vectorizer.vocabulary_,
                    open(combined_path + "_vec.pickle", 'wb'))

    def load(
            self,
            path
    ):
        logging.info("Loading " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.clf = pickle.load(
            open(combined_path + "_clf.pickle", 'rb'))
        self.count_vectorizer = CountVectorizer(
            vocabulary=pickle.load(open(combined_path + "_vec.pickle", 'rb')))
        self.build_model()

    def can_load(
            self,
            path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle") and \
               os.path.isfile(combined_path + "_vec.pickle")
