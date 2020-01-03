import argparse
from glob import glob
import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from load_dataset import load_bbc_dataset
from load_dataset import load_news_groups_dataset
from load_dataset import load_yahoo_answers_dataset
from load_dataset import load_ohsumed_dataset
from load_dataset import load_reuters_dataset
from tfidf_model import TfIdfModel
from doc2vec_model import Doc2VecDBOWModel, Doc2VecDMModel
from lda_model import LDAModel
from lsa_model import LSAModel
from han import HAN
from han_model import HANModel
from sif_model import SIFModel
from cbow_model import CBOWModel

import logging
import multiprocessing
cores = multiprocessing.cpu_count()-1


def cross_validation(
    benchmark_models,
    train,
    y_train,
    n_splits=10,
):
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=False)

    for model in benchmark_models:
        scores = []
        trainig_times = []
        for train_text, train_target in cv.split(train, y_train):
            model.build_model()
            t0 = time.time()
            model.train(train[train_text], y_train[train_text])
            trainig_times.append((time.time() - t0))
            model.fit(train[train_text], y_train[train_text])
            scores.append(model.evaluate(train[train_target], y_train[train_target]))

        logging.info(model.__class__.__name__ + ": average training time: " + str(np.average(np.array(trainig_times))))
        logging.info(model.__class__.__name__ + ": average score: " + str(np.average(scores)))


parser =  argparse.ArgumentParser(
    description="Benchmark for documents embedding methods")

parser.add_argument("-d",
    "--dataset",
    dest="dataset",
    required=True,
    help="Path to dataset")

parser.add_argument("-m",
    "--models",
    dest="models_path",
    required=True,
    help="Path to models")

parser.add_argument("-p",
    "--pretrained",
    dest="pretrained_path",
    required=True,
    help="Path to pretrained embedding model")

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

logging.info(args.dataset)

# x, y = load_bbc_dataset(args.dataset)
x, y = load_yahoo_answers_dataset(args.dataset)
# x, y = load_news_groups_dataset(args.dataset)
# x, y = load_reuters_dataset(args.dataset)
# x, y = load_ohsumed_dataset(args.dataset)

train, test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.2)

train = train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

han = HANModel(
    text = train['text'],
    labels = y_train['target'],
    num_categories = 10,
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
    text = train['text'],
    labels = y_train['target'],
    pretrained_embedded_vector_path = args.pretrained_path,
    embedding_size=100)

lda = LDAModel(
    n_components=100,
    max_features=None,
    max_df=0.95,
    min_df=1,
    epochs=10,
    cores=cores)

lsa = LSAModel(
    svd_features=100,
    n_features=None,
    n_iter=10,
    max_df=0.95,
    min_df=1)

tfidf = TfIdfModel(
    n_features=None,
    max_df=0.95,
    min_df=1)

cbow = CBOWModel(
    max_features=None,
    max_df=0.95,
    min_df=1)

benchmark_models = [cbow, tfidf, lsa, lda, sif, doc2vecdm, doc2veccbow, han]
cross_validation(benchmark_models, x['text'], y['target'], n_splits=5)
