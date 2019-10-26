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

### BBC dataset

[BBC dataset](http://mlg.ucd.ie/datasets/bbc.html) consists of 2 datasets of news articles from BBC News:
1. BBC:	2225 articles of 5 classes(business, entertainment, politics, sport, tech)
2. BBCSport: 737 articles of 5 classes(athletics, cricket, football, rugby, tennis)

### News Category Dataset

This dataset contains around 200,000 news headlines from the year 2012 to 2018 obtained from [HuffPost](https://www.huffingtonpost.com/).

Dataset obtained from here: [https://www.kaggle.com/rmisra/news-category-dataset](https://www.kaggle.com/rmisra/news-category-dataset).

### 20 Newsgroup Dataset

This data set is a collection of 20,000 messages, collected from 20 different netnews newsgroups.

Dataset obtained from here: [20 Newsgroup Dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz)


## References

### Hierarchical Attention Network implementation

Hierarchical Attention Network model was obtained from [https://github.com/Hsankesara/DeepResearch](https://github.com/Hsankesara/DeepResearch) and modified to suit proposed architecture.

### BBC dataset

D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006. [[PDF]](http://mlg.ucd.ie/files/publications/greene06icml.pdf)  [[BibTeX]](http://mlg.ucd.ie/files/bib/greene06icml.bib).

### News Category Dataset

[https://rishabhmisra.github.io/publications/](https://rishabhmisra.github.io/publications/)
