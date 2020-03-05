import logging
import time
import pickle
import os
from keras.models import Model
from models.benchmark_model import BenchmarkModel
from models.han import HAN
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator


class HANSklearnVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,
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


    def fit(self, X, y=None):
        X_copy = X.copy()
        self.model = HAN(
            X_copy,
            y,
            self.pretrained_embedded_vector_path,
            self.max_features,
            self.max_senten_len,
            self.max_senten_num,
            self.embedding_size,
            self.num_categories,
            self.validation_split,
            self.verbose)
        self.model.train(self.epochs, self.batch_size)
        self.__get_embedding_model()

        return self

    def transform(self, X, *_):
        try:
            assert (self.embedding_model is not None)
        except:
            print("Embedding model was not assigned!")
        x, _ = self.model.preprocessing(X, _)
        embedding_matrix = self.model.get_embedding_matrix()
        self.model.wordEncoder \
            .get_layer("embedding_input") \
            .set_weights([embedding_matrix])
        self.__get_embedding_model()
        return self.embedding_model.predict(x)

    def __get_embedding_model(
            self
    ):
        self.embedding_model = self.model
        self.embedding_model = Model(inputs=self.embedding_model.model.input,
                                     outputs=self.embedding_model.model.get_layer("embedding_output").output)

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
        self.build_pipeline()

    def build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ("vectorizer", HANSklearnVectorizer(
                self.pretrained_embedded_vector_path,
                self.max_features,
                self.max_senten_len,
                self.max_senten_num,
                self.embedding_size,
                self.num_categories,
                self.validation_split,
                self.verbose,
                self.epochs,
                self.batch_size)),
            ("classifier", self.clf)
        ])

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
        self.build_model()

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + ".h5")