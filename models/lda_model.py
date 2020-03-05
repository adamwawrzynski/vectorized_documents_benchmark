import logging
import time
import pickle
import os
from models.benchmark_model import BenchmarkModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import process_string, preprocess_text, process_dataset, TextPreprocessor
from sklearn.pipeline import Pipeline


class LDAModel(BenchmarkModel):
    def __init__(
        self,
        n_components,
        max_features,
        max_df,
        min_df,
        learning_method="batch",
        learning_decay=0.7,
        cores=1,
        epochs=10
    ):
        super().__init__()
        self.n_components = n_components
        self.cores = cores
        self.epochs = epochs
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.learning_method = learning_method
        self.learning_decay = learning_decay

    def build_model(
        self
    ):
        super().build_model()
        self.lda = LatentDirichletAllocation(
            n_components=self.n_components,
            learning_method=self.learning_method,
            learning_decay=self.learning_decay,
            n_jobs=self.cores,
            max_iter=self.epochs)
        self.count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words='english')
        self.build_pipeline()

    def build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ("preprocess", TextPreprocessor()),
            ("vectorizer", self.count_vectorizer),
            ("lda", self.lda),
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
        pickle.dump(self.model,
            open(combined_path + "_model.pickle", 'wb'))
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
        self.model = pickle.load(
            open(combined_path + "_model.pickle", 'rb'))
        self.count_vectorizer = CountVectorizer(
            vocabulary=pickle.load(open(combined_path + "_vec.pickle", 'rb')))
        self.build_model()

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle") and \
        os.path.isfile(combined_path + "_model.pickle") and \
        os.path.isfile(combined_path + "_vec.pickle")