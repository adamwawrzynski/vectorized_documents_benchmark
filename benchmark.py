import argparse
import numpy as np
import time
from sklearn.model_selection import KFold

from load_dataset import load_bbc_dataset
from load_dataset import load_news_groups_dataset
from load_dataset import load_yahoo_answers_dataset
from load_dataset import load_ohsumed_dataset
from load_dataset import load_reuters_dataset
from models.tfidf_model import TfIdfModel
from models.doc2vec_model import Doc2VecDBOWModel, Doc2VecDMModel
from models.lda_model import LDAModel
from models.lsa_model import LSAModel
from models.han_model import HANModel
from models.sif_model import SIFModel
from models.bow_model import BOWModel

import logging
import multiprocessing
cores = multiprocessing.cpu_count() - 1


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
            model.fit(x_train[train_text], y_train[train_text])
            trainig_times.append((time.time() - t0))
            scores.append(model.evaluate(x_train[train_target], y_train[train_target]))

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
            model.fit(x_train, y_train)
            trainig_times.append(time.time() - t0)
            scores.append(model.evaluate(x_test, y_test))

        logging.info(model.__class__.__name__ + ": average training time: " + str(np.average(np.array(trainig_times))))
        logging.info(model.__class__.__name__ + ": training time std: " + str(np.std(np.array(trainig_times))))
        logging.info(model.__class__.__name__ + ": average score: " + str(np.average(np.array(scores))))
        logging.info(model.__class__.__name__ + ": average score std: " + str(np.std(np.array(scores))))


parser =  argparse.ArgumentParser(
    description="Benchmark for documents embedding methods")

parser.add_argument("-d",
    "--dataset_path",
    dest="dataset_path",
    required=True,
    help="Path to dataset")

parser.add_argument("-m",
    "--models_path",
    dest="models_path",
    required=True,
    help="Path to models")

parser.add_argument("-p",
    "--pretrained_path",
    dest="pretrained_path",
    required=True,
    help="Path to pretrained embedding model")

parser.add_argument("-n",
    "--dataset_name",
    dest="dataset_name",
    choices=['bbc','yahoo','20newsgroups', 'reuters', 'ohsumed'],
    required=True,
    help="Name of dataset")

parser.add_argument("-r",
    "--restore",
    dest="restore",
    required=False,
    action='store_true',
    help="Path to models")

parser.add_argument("-l",
    "--logging",
    dest="logging",
    required=False,
    help="Path to logging file")

args = parser.parse_args()

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

han = HANModel(
    text = x_train['text'],
    labels = y_train['target'],
    num_categories = num_categories,
    pretrained_embedded_vector_path = args.pretrained_path,
    max_features = 200000,
    max_senten_len = 100,
    max_senten_num = 30,
    embedding_size = 100,
    validation_split=0.2,
    verbose=1,
    batch_size=8,
    epochs=10)

doc2vecdm = Doc2VecDMModel(
    negative=10,
    vector_size=100,
    window=5,
    workers=cores,
    min_count=1)

doc2veccbow = Doc2VecDBOWModel(
    negative=10,
    vector_size=100,
    window=5,
    workers=cores,
    min_count=1)

sif = SIFModel(
    text = x_train['text'],
    labels = y_train['target'],
    pretrained_embedded_vector_path = args.pretrained_path,
    embedding_size=100)

lda = LDAModel(
    n_components=100,
    max_features=None,
    max_df=0.95,
    min_df=0,
    epochs=10,
    cores=cores)

lsa = LSAModel(
    svd_features=100,
    n_features=None,
    n_iter=10,
    max_df=0.95,
    min_df=0)

tfidf = TfIdfModel(
    n_features=None,
    max_df=0.95,
    min_df=0)

bow = BOWModel(
    max_features=None,
    max_df=0.95,
    min_df=0)

benchmark_models = [bow, tfidf, lsa, lda, sif, doc2vecdm, doc2veccbow, han]

validator(
    benchmark_models,
    x_train['text'],
    y_train['target'],
    x_test['text'],
    y_test['target'])
