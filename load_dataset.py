import pandas as pd
import os
import numpy as np
from sklearn.datasets import load_files

def process_imbalance_dataset(data, upsampling=False):
    categories = data.groupby(by='target').size()
    min_size = categories.min()

    balanced = pd.DataFrame()

    for t in data['target'].unique():
        tmp = data['target'] == t
        samples = None
        if upsampling:
            if min_size > len(data[tmp]):
                samples = data[tmp].sample(min_size, replace=True)
            else:
                samples = data[tmp].sample(min_size, replace=False)
        else:
            samples = data[tmp].sample(min_size, replace=False)

        balanced = balanced.append(samples, ignore_index=True)

    return balanced

def load_bbc_dataset(path):
    assert os.path.isdir(path)
    data = load_files(path,
        encoding="us-ascii",
        decode_error="ignore")
    y = pd.DataFrame(data.target, columns=['target'])
    x = pd.DataFrame(data.data, columns=['text'])

    return x, y

def load_reuters_dataset(path):
    assert os.path.isdir(path)

    data_x = pd.DataFrame(columns=['text'])
    data_y = pd.DataFrame(columns=['target'], dtype=int)

    for f in ['test', 'training']:
        assert os.path.isfile(os.path.join(path, f))
        data = load_files(os.path.join(path, f),
            encoding="us-ascii",
            decode_error="ignore")
        data_y = data_y.append(
            pd.DataFrame(data.target, columns=['target']),
            ignore_index=True)
        data_x = data_x.append(
            pd.DataFrame(data.text, columns=['text']),
            ignore_index=True)

    return data_x, data_y

def load_ohsumed_dataset(path):
    assert os.path.isdir(path)

    data = load_files(path,
        encoding="utf-8",
        decode_error="ignore")

    y = pd.DataFrame(data.target, columns=['target'])
    x = pd.DataFrame(data.data, columns=['text'])

    return x, y

def load_news_groups_dataset(path, balance=False):
    assert os.path.isdir(path)

    data = load_files(path,
        encoding="utf-8",
        decode_error="ignore")

    y = pd.DataFrame(data.target, columns=['target'])
    x = pd.DataFrame(data.data, columns=['text'])

    return x, y

def load_yahoo_answers_dataset(path):
    data_x = pd.DataFrame(columns=['text'])
    data_y = pd.DataFrame(columns=['target'], dtype=int)

    for f in ['test.csv', 'train.csv']:
        assert os.path.isfile(os.path.join(path, f))
        data = pd.read_csv(os.path.join(path, f),
            names=["target", "text1", "text2", "text3"])

        data['target'] = data['target'].map(lambda x: x-1)
        data['text1'] = data['text1'].replace(np.nan, '', regex=True)
        data['text2'] = data['text2'].replace(np.nan, '', regex=True)
        data['text3'] = data['text3'].replace(np.nan, '', regex=True)

        data['text'] = data['text1'] + data['text2'] + data['text3']
        data = data.drop(['text1', 'text2', 'text3'], axis=1)

        data_y = data_y.append(
            pd.DataFrame(data.target, columns=['target']),
            ignore_index=True)
        data_x = data_x.append(
            pd.DataFrame(data.text, columns=['text']),
            ignore_index=True)

    return data_x, data_y
