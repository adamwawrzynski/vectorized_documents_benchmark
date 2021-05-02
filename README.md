
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
3.7) and modules listed in **requirements.txt** file.

## Getting started

### Development Environment

This step assumes that You have conda installed.To install all dependencies
you have to run the following commands:

```bash
# create virtual environment given name "benchmark", You can change it
conda create -n benchmark python=3.7

# activate environment
conda activate benchmark
python3 -m pip install -r requirements.txt
```

## Execute

To run benchmark activate prepared conda environment and execute simillar command:
```bash
CUDA_VISIBLE_DEVICES=7 python3 benchmark.py --dataset_path datasets/bbcsport/ --models_path models/ --pretrained_path embeddings/glove.6B.100d.txt --dataset_name bbc --hwan_features_algorithm tf --hwan_features_operation mul
```
where `CUDA_VISIBLE_DEVICES=7` selects one GPU from those available at the machine,
`hwan_features_algorithm` defines algorithm used to compute statistical features
(available `bow`, `tf` and `tfidf`) and `hwan_features_operation` defines operation of
latent variables and statistical features in HWAN (available `add`, `mul` and `concat`).

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

## Pretrained embeddings

To follow proposed evaluation protocol You have to download and use GloVe embeddings. You can find them here:
[glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

## Datasets

Reuters-21578, Ohsumed and 20 Newsgroups used in this benchmark were downloaded from [http://disi.unitn.it/moschitti/corpora.htm](http://disi.unitn.it/moschitti/corpora.htm).

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
