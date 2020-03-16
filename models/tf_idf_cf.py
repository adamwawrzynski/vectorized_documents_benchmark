import pandas as pd
import math
import copy
import numpy as np
from sklearn.decomposition import TruncatedSVD


class CustomTfIdf():
    def __init__(self):
        pass

    def create_wordDict(self, doc):
        self.wordSet = set()
        for d in doc:
            self.wordSet.update(d.split(" "))

        self.wordDict = []
        for index, d in enumerate(doc):
            self.wordDict.append(dict.fromkeys(self.wordSet, 0))
            for word in d.split(" "):
                self.wordDict[index][word] += 1

    def create_tf(self):
        self.tfDict = copy.deepcopy(self.wordDict)
        for index in range(len(self.wordDict)):
            for word, count in self.wordDict[index].items():
                self.tfDict[index][word] = (count + 1)/float(len(self.wordDict[index]))


    def create_idf(self):
        self.idfDict = {}
        N = len(self.tfDict)

        self.idfDict = dict.fromkeys(self.wordSet, 0)
        for d in self.tfDict:
            for word, val in d.items():
                if val > 0:
                    self.idfDict[word] += 1

        for word, val in self.idfDict.items():
            self.idfDict[word] = math.log10((N  + 1)/ float(val))

    #
    # def create_tf_idf(self):
    #     self.tfidf = copy.deepcopy(self.wordDict)
    #     for index in range(len(doc)):
    #         for word, val in self.wordDict[index].items():
    #             self.tfidf[index][word] = val * self.idfs[word]

    #
    # def create_cf(self, labels):
    #     self.cf = []
    #     self.Ncf = []
    #     for index in range(0, max(labels) + 1):
    #         self.cf.append(dict.fromkeys(self.wordDict[0].keys(), 0))
    #
    #     for index in range(len(labels)):
    #         for (word, val) in self.wordDict[index].items():
    #             self.cf[labels[index]][word] += val
    #
    #     for index in range(0, max(labels) + 1):
    #         self.Ncf.append(labels.count(index))
    #
    #     for index, c in enumerate(self.cf):
    #         for w in c:
    #             self.cf[index][w] /= self.Ncf[index]


    def create_tf_idf(self,):
        tfidfcf = copy.deepcopy(self.wordDict)
        for index in range(len(tfidfcf)):
            for word, val in self.wordDict[index].items():
                tfidfcf[index][word] = val * self.idfDict[word] #* self.cf[labels[index]][word] / self.Ncf[labels[index]]

        self.tfidfcf = pd.DataFrame(tfidfcf).to_numpy()

    def fit(self, doc, y=None):
        self.create_wordDict(doc)
        self.create_tf()
        self.create_idf()
        # self.create_cf(labels)
        self.create_tf_idf()
        return self

    # def transform(self, doc):
    #     doc_tfidf = []
    #     for index, d in enumerate(doc):
    #         doc_tfidf.append(dict.fromkeys(self.wordDict[0].keys(), 0))
    #
    #         for word in d.split(" "):
    #             doc_tfidf[index][word] += 1
    #
    #         for i in range(len(self.wordDict) - 1):
    #             for word, val in self.wordDict[i].items():
    #                 doc_tfidf[index][word] = val * self.idfDict[word]
    #
    #         doc_tfidf[index] = [value for (key, value) in doc_tfidf[index].items()]
    #
    #     return np.array(doc_tfidf)

    def transform(self, doc, window=3):
        doc_tfidf = []
        for index, d in enumerate(doc):
            doc_tfidf.append(dict.fromkeys(self.wordSet, 0))

            for word in d.split(" "):
                doc_tfidf[index][word] += 1

            for i in range(len(self.wordDict) - 1):
                for word, val in self.wordDict[i].items():
                    doc_tfidf[index][word] = val * self.idfDict[word]

            doc_tfidf[index] = [value for (key, value) in doc_tfidf[index].items()]

        doc_tfidf = np.array(doc_tfidf)

        custom_doc_tfidf = copy.deepcopy(doc_tfidf)

        for index, d in enumerate(doc):
            document = d.split(" ")
            for i in range(0, len(document)):
                words = document[max(0, i-window):min(len(document), i + window)]
                print(words)
                for x, w in enumerate(words):
                    custom_doc_tfidf[index][i] += self.idfDict[word] / (x + 1/window)

        return custom_doc_tfidf


# doc = ["The cat sat on my face", "The cat sat on my bed cat"]
# labels = [1, 0]
#
# tfidfcf = CustomTfIdf()
#
# tfidfcf.fit(doc, labels)
# print(tfidfcf.transform(doc))
# print(tfidfcf.transform_with_window(doc))
#
# dict_word = wordDict(doc)
# tf = tf(dict_word)
# idf = idf(tf)
#
# tf_idf(dict_word, idf)
#
# cf_val, ncf_val = cf(dict_word, labels)
#
# tf_idf_cf(dict_word, idf, cf_val, ncf_val, labels)


