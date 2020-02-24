import logging
import time
import pickle
import os
from models.benchmark_model import BenchmarkModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import process_dataset


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
        self.model = LatentDirichletAllocation(
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

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vocabulary on " + self.__class__.__name__)
        t0 = time.time()
        processed_dataset = process_dataset(x)
        processed_dataset = processed_dataset.map(lambda x: ' '.join(word for word in x))
        doc_term_matrix = self.count_vectorizer.fit_transform(processed_dataset.values.astype('U'))
        self.model.fit(doc_term_matrix)
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

    def preprocess_data(
        self,
        dataset,
        y_dataset
    ):
        logging.info("Transform data on " + self.__class__.__name__)
        processed_dataset = process_dataset(dataset)
        processed_dataset = processed_dataset.map(lambda x: ' '.join(word for word in x))
        doc_term_matrix = self.count_vectorizer.transform(processed_dataset.values.astype('U'))
        return self.model.transform(doc_term_matrix)

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

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle") and \
        os.path.isfile(combined_path + "_model.pickle") and \
        os.path.isfile(combined_path + "_vec.pickle")