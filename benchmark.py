import argparse
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

from utils.load_dataset import load_bbc_dataset
from utils.load_dataset import load_news_groups_dataset
from utils.load_dataset import load_yahoo_answers_dataset
from utils.load_dataset import load_ohsumed_dataset
from utils.load_dataset import load_reuters_dataset
from models.tfidf_model import TfIdfModel
from models.doc2vec_model import Doc2VecDBOWModel, Doc2VecDMModel
from models.lda_model import LDAModel
from models.lsa_model import LSAModel
from models.han_model import HANModel
from models.hwan_model import HWANModel
from models.sif_model import SIFModel
from models.bow_model import BOWModel
from models.psif_model import PSIFModel
from models.longformer_model import LongformerBERTModel

import logging
import multiprocessing
cores = multiprocessing.cpu_count() - 1


def histogram_of_dataset():
    from nltk import tokenize
    from keras.preprocessing.text import Tokenizer, text_to_word_sequence
    from utils.preprocess import clean_string
    from collections import Counter

    texts = []
    sentences = []
    paras = []
    list_of_senten_num = []
    list_of_senten_len = []
    for sentence in x_train['text']:
        text = clean_string(sentence)
        sentences = tokenize.sent_tokenize(text)
        list_of_senten_num.append(len(sentences))
    for s in sentences:
        list_of_senten_len.append(len(s))

    print("Histogram of senten_num")
    print(Counter(list_of_senten_num))

    print("Histogram of senten_len")
    print(Counter(list_of_senten_len))

def cross_validation(
    benchmark_models,
    x_train,
    y_train,
    x_test=None,
    y_test=None,
    n_splits=5
):
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=False)
    logging.info("Number of splits in sample: " + str(n_splits))
    for model in benchmark_models:
        scores = []
        trainig_times = []
        for train_text, train_target in cv.split(x_train, y_train):
            model.build_model()
            t0 = time.time()
            model.train(x_train[train_text], y_train[train_text])
            trainig_times.append((time.time() - t0))
            model.fit(x_train[train_text], y_train[train_text])
            scores.append(model.evaluate(x_train[train_target], y_train[train_target]))
            logging.info(classification_report(y_train[train_target], model.predict(x_train[train_target])))

        logging.info(model.__class__.__name__ + ": average training time: " + str(np.average(np.array(trainig_times))))
        logging.info(model.__class__.__name__ + ": training time std: " + str(np.std(np.array(trainig_times))))
        logging.info(model.__class__.__name__ + ": average score: " + str(np.average(scores)))
        logging.info(model.__class__.__name__ + ": score std: " + str(np.std(scores)))

def train_test_validator(
    benchmark_models,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=5
):
    logging.info("Number of realisations in sample: " + str(epochs))
    for model in benchmark_models:
        scores = []
        trainig_times = []
        for step in range(epochs):
            model.build_model()
            t0 = time.time()
            model.train(x_train, y_train)
            trainig_times.append(time.time() - t0)
            model.fit(x_train, y_train)
            scores.append(model.evaluate(x_test, y_test))
            logging.info(classification_report(y_test, model.predict(x_test)))

        logging.info(model.__class__.__name__ + ": average training time: " + str(np.average(np.array(trainig_times))))
        logging.info(model.__class__.__name__ + ": training time std: " + str(np.std(np.array(trainig_times))))
        logging.info(model.__class__.__name__ + ": average score: " + str(np.average(np.array(scores))))
        logging.info(model.__class__.__name__ + ": average score std: " + str(np.std(np.array(scores))))

def parse_args():
    parser =  argparse.ArgumentParser(
        description="Benchmark for documents embedding methods")

    parser.add_argument(
        "--dataset_path",
        dest="dataset_path",
        required=True,
        help="Path to dataset")

    parser.add_argument(
        "--models_path",
        dest="models_path",
        required=True,
        help="Path to models")

    parser.add_argument(
        "--pretrained_path",
        dest="pretrained_path",
        required=True,
        help="Path to pretrained embedding model")

    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        choices=['bbc','yahoo','20newsgroups', 'reuters', 'ohsumed'],
        required=True,
        help="Name of dataset")

    parser.add_argument(
        "--restore",
        dest="restore",
        required=False,
        action='store_true',
        help="Path to models")

    parser.add_argument(
        "--logging",
        dest="logging",
        required=False,
        type=str,
        default="",
        help="Path to logging file")

    parser.add_argument(
        "--hwan_features_algorithm",
        dest="hwan_features_algorithm",
        required=True,
        help="HWAN statistical features algrithm")

    parser.add_argument(
        "--hwan_features_operation",
        dest="hwan_features_operation",
        required=True,
        help="HWAN statistical features operation")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.logging:
        logging.basicConfig(
        filename=args.logging,
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)
    else:
        logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    logging.info(args.dataset_path)

    num_categories = 0
    embedding_size = 100
    load_dataset = None
    validator = None

    if args.dataset_name == "bbc":
        load_dataset = load_bbc_dataset
        num_categories = 5
        validator = cross_validation
    elif args.dataset_name == "yahoo":
        load_dataset = load_yahoo_answers_dataset
        num_categories = 10
        validator = cross_validation
    elif args.dataset_name == "20newsgroups":
        load_dataset = load_news_groups_dataset
        num_categories = 20
        validator = cross_validation
    elif args.dataset_name == "reuters":
        load_dataset = load_reuters_dataset
        num_categories = 91
        validator = train_test_validator
    elif args.dataset_name == "ohsumed":
        load_dataset = load_ohsumed_dataset
        num_categories = 23
        validator = train_test_validator

    x_train, x_test, y_train, y_test = load_dataset(args.dataset_path)

    longformer = LongformerBERTModel(
        dataset_name=args.dataset_name,
        embedding_size=embedding_size,
        num_categories=num_categories,
        epochs=10,
        batch_size=4,
    )

    hwan = HWANModel(
        text = x_train['text'],
        labels = y_train['target'],
        num_categories = num_categories,
        pretrained_embedded_vector_path = args.pretrained_path,
        max_features=5000000,
        max_senten_len=100,
        max_senten_num=30,
        embedding_size=embedding_size,
        validation_split=0.1,
        verbose=True,
        batch_size=16,
        epochs=100,
        features_algorithm=args.hwan_features_algorithm,
        features_operation=args.hwan_features_operation)

    han = HANModel(
        text=x_train['text'],
        labels=y_train['target'],
        num_categories=num_categories,
        pretrained_embedded_vector_path=args.pretrained_path,
        max_features=5000000,
        max_senten_len=100,
        max_senten_num=30,
        embedding_size=embedding_size,
        validation_split=0.1,
        verbose=True,
        batch_size=64,
        epochs=100)

    doc2vecdm = Doc2VecDMModel(
        negative=10,
        vector_size=embedding_size,
        window=5,
        workers=cores,
        min_count=1)

    doc2veccbow = Doc2VecDBOWModel(
        negative=10,
        vector_size=embedding_size,
        window=5,
        workers=cores,
        min_count=1)

    psif = PSIFModel(
        pretrained_embedded_vector_path=args.pretrained_path,
        embedding_size=embedding_size,
        num_clusters=40)

    sif = SIFModel(
        text=x_train['text'],
        labels=y_train['target'],
        pretrained_embedded_vector_path=args.pretrained_path,
        embedding_size=embedding_size)

    lda = LDAModel(
        n_components=embedding_size,
        max_features=None,
        max_df=0.95,
        min_df=1,
        epochs=10,
        cores=cores)

    lsa = LSAModel(
        svd_features=embedding_size,
        n_features=None,
        n_iter=10,
        max_df=0.95,
        min_df=1)

    tfidf = TfIdfModel(
        n_features=None,
        max_df=0.95,
        min_df=1)

    bow = BOWModel(
        max_features=None,
        max_df=0.95,
        min_df=1)

    benchmark_models = [bow, tfidf, lsa, lda, sif, psif, doc2vecdm, doc2veccbow, han, hwan, longformer]

    validator(
        benchmark_models,
        x_train['text'],
        y_train['target'],
        x_train['text'],
        y_train['target'])
