import logging
import time
import pickle
import os
from keras.models import load_model
from keras.models import Model
from keras import backend as K
from benchmark_model import BenchmarkModel
from han import HAN
from attention_with_context import AttentionWithContext


class HANModel(BenchmarkModel):
    def __init__(
        self,
        text,
        labels,
        pretrained_embedded_vector_path,
        max_features,
        max_senten_len,
        max_senten_num,
        embedding_size,
        num_categories=None,
        validation_split=0.2, 
        verbose=1,
        epochs=10,
        batch_size=8
    ):
        super().__init__()
        self.text = text
        self.labels = labels
        self.pretrained_embedded_vector_path = pretrained_embedded_vector_path
        self.max_features = max_features
        self.max_senten_len = max_senten_len
        self.max_senten_num = max_senten_num
        self.embedding_size = embedding_size
        self.num_categories = num_categories
        self.validation_split = validation_split
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_model = None

    def build_model(
        self
    ):
        super().build_model()
        self.model = HAN(
            self.text,
            self.labels,
            self.pretrained_embedded_vector_path,
            self.max_features,
            self.max_senten_len,
            self.max_senten_num,
            self.embedding_size,
            self.num_categories,
            self.validation_split,
            self.verbose)

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        t0 = time.time()
        self.model.train(self.epochs, self.batch_size)
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)
        self.get_embedding_model()

    def fit(
        self,
        x,
        y
    ):
        logging.info("Training kNN classifier")
        embedded_x = self.preprocess_data(x, y)
        return self.clf.fit(embedded_x ,y)

    def preprocess_data(
        self,
        dataset,
        y_dataset
    ):
        try:
            assert(self.embedding_model is not None)
        except:
            print("Embedding model was not assigned!")
        x, _ = self.model.preprocessing(dataset, y_dataset)
        embedding_matrix = self.model.get_embedding_matrix()
        self.model.wordEncoder \
            .get_layer("embedding_input") \
            .set_weights([embedding_matrix])
        self.get_embedding_model()
        return self.embedding_model.predict(x)

    def get_embedding_model(
        self
    ):
        self.embedding_model = self.model
        self.embedding_model = Model(inputs=self.embedding_model.model.input,
            outputs=self.embedding_model.model.get_layer("embedding_output").output)

    def save(
        self,
        path
    ):
        logging.info("Saving " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.model.save_model(combined_path)
        pickle.dump(self.clf,
            open(combined_path + "_clf.pickle", 'wb'))

    def load(
        self,
        path
    ):
        logging.info("Loading " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.model.load_model(combined_path)
        self.get_embedding_model()
        self.clf = pickle.load(
            open(combined_path + "_clf.pickle", 'rb'))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + ".h5")