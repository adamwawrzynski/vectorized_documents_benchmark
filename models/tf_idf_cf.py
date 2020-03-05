import pandas as pd
import math
import copy

class TfIdfCf():
    def __init__(self):
        pass

    def fwordDict(self, doc):
        self.wordSet = set()
        for d in doc:
            self.wordSet.update(d.split(" "))

        self.wordDict = []
        for d in doc:
            self.wordDict.append(dict.fromkeys(self.wordSet, 0))

        for index, d in enumerate(doc):
            for word in d.split(" "):
                self.wordDict[index][word] += 1


    def ftf(self):
        self.tfDict = copy.deepcopy(self.wordDict)
        for index in range(len(self.wordDict)):
            for word, count in self.wordDict[index].items():
                self.tfDict[index][word] = (count + 1)/float(len(self.wordDict[index]))


    def fidf(self):
        self.idfDict = {}
        N = len(self.tfDict)

        self.idfDict = dict.fromkeys(self.tfDict[0].keys(), 0)
        for d in self.tfDict:
            for word, val in d.items():
                if val > 0:
                    self.idfDict[word] += 1

        for word, val in self.idfDict.items():
            self.idfDict[word] = math.log10((N  + 1)/ float(val))


    # def tf_idf(self):
    #     self.tfidf = copy.deepcopy(self.wordDict)
    #     for index in range(len(doc)):
    #         for word, val in self.wordDict[index].items():
    #             self.tfidf[index][word] = val * self.idfs[word]


    def fcf(self, labels):
        self.cf = []
        self.Ncf = []
        for index in range(0, max(labels) + 1):
            self.cf.append(dict.fromkeys(self.wordDict[0].keys(), 0))

        for index in range(len(labels)):
            for word, _ in self.wordDict[index].items():
                self.cf[labels[index]][word] += 1

        for index in range(0, max(labels) + 1):
            self.Ncf.append(labels.count(index))


    def ftf_idf_cf(self, labels):
        tfidfcf = copy.deepcopy(self.wordDict)
        for index in range(len(labels)):
            for word, val in self.wordDict[index].items():
                tfidfcf[index][word] = val * self.idfDict[word] * self.cf[labels[index]][word] / self.Ncf[labels[index]]

        self.tfidfcf = pd.DataFrame(tfidfcf).to_numpy()


    def fit_transform(self, doc, lables):
        self.fwordDict(doc)
        self.ftf()
        self.fidf()
        self.fcf(labels)
        self.ftf_idf_cf(labels)
        return self.tfidfcf



doc = ["The cat sat on my face", "The cat sat on my bed"]
labels = [1, 0]

tfidfcf = TfIdfCf()

print(tfidfcf.fit_transform(doc, labels))
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


