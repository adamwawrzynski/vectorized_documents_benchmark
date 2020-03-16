from models.benchmark_model import BenchmarkModel
from models.CustomTfIdfSklearn import CustomTfidfVectorizer
from preprocess import TextPreprocessor
from sklearn.pipeline import Pipeline

class CustomTfIdfModel(BenchmarkModel):
    def __init__(
        self
    ):
        super().__init__()

    def build_model(
        self
    ):
        super().build_model()
        self.tfidf_vectorizer = CustomTfidfVectorizer()
        self.build_pipeline()

    def build_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ("preprocess", TextPreprocessor()),
            ("vectorizer", self.tfidf_vectorizer),
            ("classifier", self.clf)
        ])
