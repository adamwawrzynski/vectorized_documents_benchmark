from __future__ import print_function, division
from collections import Counter
from itertools import combinations
from math import log
from pprint import pformat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from string import punctuation
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_pmi(df):
    cx = Counter()
    cxy = Counter()
    texts_tokenized = [d.split(" ") for d in df]
    for text in texts_tokenized:

        for x in text:
            cx[x] += 1

        # Count all pairs of words, even duplicate pairs.
        for x, y in map(sorted, combinations(text, 2)):
            cxy[(x, y)] += 1

    min_count = (1 / 1000) * len(df)
    max_count = (1 / 50) * len(df)
    for x in list(cx.keys()):
        if cx[x] < min_count or cx[x] > max_count:
            del cx[x]

    for x, y in list(cxy.keys()):
        if x not in cx or y not in cx:
            del cxy[(x, y)]

    x2i, i2x = {}, {}
    for i, x in enumerate(cx.keys()):
        x2i[x] = i
        i2x[i] = x

    sx = sum(cx.values())
    sxy = sum(cxy.values())

    pmi_samples = Counter()
    data, rows, cols = [], [], []
    for (x, y), n in cxy.items():
        rows.append(x2i[x])
        cols.append(x2i[y])
        data.append(log((n / sxy) / (cx[x] / sx) / (cx[y] / sx)))
        pmi_samples[(x, y)] = data[-1]
    PMI = csc_matrix((data, (rows, cols)))
    return PMI.todense()