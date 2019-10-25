import logging
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
        self.lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=42,
            n_jobs=cores,
            max_iter=epochs)
        self.count_vectorizer = CountVectorizer(
            max_df=max_df,
            min_df=min_df,
            stop_words='english')

    def build_vocabulary(
        self,
        dataset
    ):
        logging.info("Building vocabulary on " + self.__class__.__name__)
        processed_dataset = process_dataset(dataset)
        processed_dataset = processed_dataset.map(lambda x: ' '.join(word for word in x))
        doc_term_matrix = self.count_vectorizer.fit_transform(processed_dataset.values.astype('U'))
        self.lda.fit(doc_term_matrix)

    def preprocess_data(
        self,
        dataset
    ):
        logging.info("Transform data on " + self.__class__.__name__)
        processed_dataset = process_dataset(dataset)
        processed_dataset = processed_dataset.map(lambda x: ' '.join(word for word in x))
        doc_term_matrix = self.count_vectorizer.transform(processed_dataset.values.astype('U'))
        return self.lda.transform(doc_term_matrix)