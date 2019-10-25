import logging
import time
import pickle
import os
from benchmark_model import BenchmarkModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import process_dataset


class LDAModel(BenchmarkModel):
    def __init__(
        self,
        n_components,
        max_df,
        min_df,
        cores=1,
        epochs=40
    ):
        super().__init__()
        self.model = LatentDirichletAllocation(
            n_components=n_components,
            random_state=42,
            n_jobs=cores,
            max_iter=epochs)
        self.count_vectorizer = CountVectorizer(
            max_df=max_df,
            min_df=min_df,
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
        dataset
    ):
        logging.info("Transform data on " + self.__class__.__name__)
        processed_dataset = process_dataset(dataset)
        processed_dataset = processed_dataset.map(lambda x: ' '.join(word for word in x))
        doc_term_matrix = self.count_vectorizer.transform(processed_dataset.values.astype('U'))
        return self.model.transform(doc_term_matrix)

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
        pickle.dump(self.count_vectorizer.vocabulary_,
            open(combined_path + "_vec.pickle", 'wb'))

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
        self.count_vectorizer = CountVectorizer(
            vocabulary=pickle.load(open(combined_path + "_vec.pickle", 'rb')))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_knn.pickle") and os.path.isfile(combined_path + "_model.pickle") and os.path.isfile(combined_path + "_vec.pickle")