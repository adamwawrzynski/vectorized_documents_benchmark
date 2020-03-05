import logging
import time
from models.benchmark_model import BenchmarkModel
from abc import abstractmethod
from preprocess import process_dataset
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from preprocess import process_string, preprocess_text, process_dataset, TextPreprocessor
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


class Doc2VecSklearnVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.transformer = TextPreprocessor()
        self.model = model

    def fit(self, X, y=None):
        X_copy = X.copy()
        text = self.transformer.transform(X_copy)
        process_text = [w.split(" ") for w in text]
        documents = [TaggedDocument(doc, [tag]) for doc, tag in zip(process_text, y)]
        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        processed_dataset = self.transformer.transform(X_copy)
        vectors = [self.model.infer_vector(processed_dataset[doc_id].split(" ")) for doc_id in range(len(processed_dataset))]
        return vectors


class Doc2VecModel(BenchmarkModel):
    @abstractmethod
    def __init__(
        self, 
        negative=5,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        epochs=40
    ):
        super().__init__()

    def build_model(
        self
    ):
        super().build_model()

class Doc2VecDMModel(Doc2VecModel):
    def __init__(
        self, 
        negative=5,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        epochs=40
    ):
        super().__init__()
        self.negative = negative
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def build_model(
        self
    ):
        super().build_model()
        self.doc2vec = Doc2Vec(
            dm=1,
            negative=self.negative,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs)
        self.pipeline = Pipeline(steps=[
            ("preprocess", TextPreprocessor()),
            ("vectorizer", Doc2VecSklearnVectorizer(self.doc2vec)),
            ("classifier", self.clf)
        ])


class Doc2VecDBOWModel(Doc2VecModel):
    def __init__(
        self, 
        negative=5,
        vector_size=100,
        window=5,
        min_count=2,
        workers=1,
        epochs=40
    ):
        super().__init__()
        self.negative = negative
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs

    def build_model(
        self
    ):
        super().build_model()
        self.doc2vec = Doc2Vec(
            dm=0,
            negative=self.negative,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs)
        self.pipeline = Pipeline(steps=[
            ("preprocess", TextPreprocessor()),
            ("vectorizer", Doc2VecSklearnVectorizer(self.doc2vec)),
            ("classifier", self.clf)
        ])