import argparse
from glob import glob
import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files

from tfidf_model import TfIdfModel
from doc2vec_model import Doc2VecDBOWModel, Doc2VecDMModel
from lda_model import LDAModel
from lsa_model import LSAModel

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

import multiprocessing
cores = multiprocessing.cpu_count()-2

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

args = parser.parse_args()

data = load_files(args.dataset,
    encoding="us-ascii",
    decode_error="ignore")
dataset_y = pd.DataFrame(data.target, columns=['target'])
dataset = pd.DataFrame(data.data, columns=['text'])

train, test, y_train, y_test = train_test_split(
    dataset,
    dataset_y,
    train_size=0.5)

model = Doc2VecDMModel(
    negative=10,
    vector_size=100,
    window=5,
    workers=cores,
    min_count=2)

if not os.path.isfile(os.path.join(args.models_path, model.__class__.__name__ + "_knn.pickle")):
    model.train(train['text'], y_train['target'])
    model.fit(train['text'], y_train['target'])

    model.save(os.path.join(args.models_path, model.__class__.__name__))
else:
    model.load(os.path.join(args.models_path, model.__class__.__name__))

model.evaluate(test['text'], y_test['target'])

model = Doc2VecDBOWModel(
    negative=10,
    vector_size=100,
    window=5,
    workers=cores,
    min_count=2)


if not os.path.isfile(os.path.join(args.models_path, model.__class__.__name__ + "_knn.pickle")):
    model.train(train['text'], y_train['target'])
    model.fit(train['text'], y_train['target'])

    model.save(os.path.join(args.models_path, model.__class__.__name__))
else:
    model.load(os.path.join(args.models_path, model.__class__.__name__))

model.evaluate(test['text'], y_test['target'])

model = LDAModel(
    n_components=10,
    max_df=0.75,
    min_df=7,
    epochs=100,
    cores=cores)

if not os.path.isfile(os.path.join(args.models_path, model.__class__.__name__ + "_knn.pickle")):
    model.train(train['text'])
    model.fit(train['text'], y_train['target'])

    model.save(os.path.join(args.models_path, model.__class__.__name__))
else:
    model.load(os.path.join(args.models_path, model.__class__.__name__))

result = model.evaluate(test['text'], y_test['target'])

model = TfIdfModel(
    n_features=2000,
    max_df=0.75,
    min_df=7)

if not os.path.isfile(os.path.join(args.models_path, model.__class__.__name__ + "_knn.pickle")):
    model.train(train['text'])
    model.fit(train['text'], y_train['target'])

    model.save(os.path.join(args.models_path, model.__class__.__name__))
else:
    model.load(os.path.join(args.models_path, model.__class__.__name__))

model.evaluate(test['text'], y_test['target'])

filename = "lsa"

model = LSAModel(
    svd_features=10,
    n_features=2000, 
    n_iter=40,
    max_df=0.75,
    min_df=7)

if not os.path.isfile(os.path.join(args.models_path, model.__class__.__name__ + "_knn.pickle")):
    model.train(train['text'])
    model.fit(train['text'], y_train['target'])

    model.save(os.path.join(args.models_path, model.__class__.__name__))
else:
    model.load(os.path.join(args.models_path, model.__class__.__name__))

model.evaluate(test['text'], y_test['target'])
