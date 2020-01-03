
# Doc2Vec methods benchmark

Framework for benchmarking different methods used for documents embeddings.

## Quick links

* [Requirements](#requirements)
* [Getting started](#getting-started)
* [Architecture](#architecture)
* [Datasets](#datasets)
* [References](#references)

## Requirements

Program was tested on Ubuntu 18.04 LTS.

To run the **benchmark.py** you need Python3.x (recommended version is
3.6) and modules listed in **requirements.txt** file.

## Getting started

### Installation (Ubuntu)

To install all dependencies you have to run the following commands:

```bash
# install Python3.x nad pip3
sudo apt-get install -y python3 python3-dev python3-pip

# install all dependencies
pip3 install -r requirements.txt
```

Or you can run *setup.sh* script with the following command:

```bash
sudo ./setup.sh
```

## Architecture

Abstract class resides inside [benchmark_model.py](https://github.com/adamwawrzynski/vectorized_documents_benchmark/blob/master/benchmark_model.py). It has 2 abstract methods:

~~~python
@abstractmethod
def preprocess_data(
	self,
	dataset,
	y_dataset
):
...
@abstractmethod
def train(
	self,
	x,
	y=None
):
...
~~~

As long as this steps may differ between different methods of document vectorization, they will be implemented in concrete classes.

The common methods are used for handling final KNeighborsClassifier and saving and loading pretrained models.

## Datasets

Reuters-21578, Ohsumed and 20 Newsgroups used in this benchmark were downloaded from [enter link description here](http://disi.unitn.it/moschitti/corpora.htm).

### BBC dataset
[BBC dataset](http://mlg.ucd.ie/datasets/bbc.html) consists of 2 datasets of news articles from BBC News:
1. BBC:	2225 articles of 5 classes(business, entertainment, politics, sport, tech)
2. BBCSport: 737 articles of 5 classes(athletics, cricket, football, rugby, tennis)

### Reuters
This is a collection of documents that appeared on Reuters newswire in 1987.

Dataset obtained from here: [Reuters(90)](http://disi.unitn.it/moschitti/corpora/Reuters21578-Apte-90Cat.tar.gz)
Dataset obtained from here: [Reuters(115)](http://disi.unitn.it/moschitti/corpora/Reuters21578-Apte-115Cat.tar.gz)

### Ohsumed
Includes medical abstracts from the _MeSH_ categories of the year 1991. The specific task was to categorize the 23 _cardiovascular diseases_ categories.

Dataset obtained from here: [Ohsumed(20,000)](http://disi.unitn.it/moschitti/corpora/ohsumed-first-20000-docs.tar.gz)
Dataset obtained from here: [Ohsumed(All)](http://disi.unitn.it/moschitti/corpora/ohsumed-all-docs.tar.gz)

### 20 Newsgroup Dataset

This data set is a collection of 20,000 messages, collected from 20 different netnews newsgroups.

Dataset obtained from here: [20 Newsgroup Dataset](http://disi.unitn.it/moschitti/corpora/20_newsgroups.tar.gz)


## References

### Hierarchical Attention Network implementation

Hierarchical Attention Network model was obtained from [https://github.com/Hsankesara/DeepResearch](https://github.com/Hsankesara/DeepResearch) and modified to suit proposed architecture.

### BBC dataset

D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006. [[PDF]](http://mlg.ucd.ie/files/publications/greene06icml.pdf)  [[BibTeX]](http://mlg.ucd.ie/files/bib/greene06icml.bib).

### News Category Dataset

[https://rishabhmisra.github.io/publications/](https://rishabhmisra.github.io/publications/)
