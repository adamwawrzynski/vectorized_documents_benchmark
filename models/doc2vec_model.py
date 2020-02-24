import logging
import time
from models.benchmark_model import BenchmarkModel
from abc import abstractmethod
from preprocess import process_dataset
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


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

    def train(
        self,
        x,
        y
    ):
        logging.info("Training " + self.__class__.__name__)
        t0 = time.time()
        processed_x = process_dataset(x)
        documents = [TaggedDocument(doc, [tag]) for doc, tag in zip(processed_x, y)]
        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

    def preprocess_data(
        self,
        dataset,
        y_dataset
    ):
        logging.info("Transforming data on " + self.__class__.__name__)
        processed_dataset = process_dataset(dataset).tolist()
        vectors = [self.model.infer_vector(processed_dataset[doc_id]) for doc_id in range(len(processed_dataset))]
        return vectors


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
        self.model = Doc2Vec(
            dm=1,
            negative=self.negative,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs)

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
        self.model = Doc2Vec(
            dm=0,
            negative=self.negative,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs)