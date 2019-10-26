import argparse
from glob import glob
import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.model_selection import KFold

from tfidf_model import TfIdfModel
from doc2vec_model import Doc2VecDBOWModel, Doc2VecDMModel
from lda_model import LDAModel
from lsa_model import LSAModel
from han import HAN
from han_model import HANModel

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

import multiprocessing
cores = multiprocessing.cpu_count()-2


def cross_validation(
    benchmark_models,
    train,
    y_train,
    n_splits=10,
):
    cv = KFold(n_splits=n_splits, random_state=42, shuffle=False)

    for model in benchmark_models:
        scores = []
        for train_text, train_target in cv.split(train['text'], y_train['target']):
            model.build_model()
            model.train(train['text'][train_text], y_train['target'][train_text])
            model.fit(train['text'][train_text], y_train['target'][train_text])
            scores.append(model.evaluate(train['text'][train_target], y_train['target'][train_target]))

        print("Average: ", np.average(scores))


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
    default=False,
    type=bool,
    help="Path to models")

args = parser.parse_args()

data = load_files(args.dataset,
    encoding="us-ascii",
    decode_error="ignore")
dataset_y = pd.DataFrame(data.target, columns=['target'])
dataset = pd.DataFrame(data.data, columns=['text'])

train, test, y_train, y_test = train_test_split(
    dataset,
    dataset_y,
    train_size=0.2)

train = train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

benchmark_models = []


han = HANModel(
    text = train['text'],
    labels = y_train['target'],
    num_categories = 5,
    pretrained_embedded_vector_path = args.pretrained_path,
    max_features = 200000,
    max_senten_len = 150,
    max_senten_num = 20,
    embedding_size = 100,
    validation_split=0.2,
    verbose=1,
    epochs=1)

doc2vecdm = Doc2VecDMModel(
    negative=10,
    vector_size=100,
    window=5,
    workers=cores,
    min_count=2)

doc2veccbow = Doc2VecDBOWModel(
    negative=10,
    vector_size=100,
    window=5,
    workers=cores,
    min_count=2)

lda = LDAModel(
    n_components=10,
    max_df=0.75,
    min_df=7,
    epochs=100,
    cores=cores)

lsa = LSAModel(
    svd_features=10,
    n_features=2000, 
    n_iter=40,
    max_df=0.75,
    min_df=7)

tfidf = TfIdfModel(
    n_features=2000,
    max_df=0.75,
    min_df=7)

benchmark_models = [han, doc2vecdm, doc2veccbow, lda, lsa, tfidf]

for model in benchmark_models:
    if args.restore or not model.can_load(args.models_path):
        model.train(train['text'], y_train['target'])
        model.fit(train['text'], y_train['target'])

        model.save(args.models_path)
    else:
        model.load(args.models_path)

    model.evaluate(test['text'], y_test['target'])
