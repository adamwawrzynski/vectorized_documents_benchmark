from bs4 import BeautifulSoup 
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stem = PorterStemmer()
stops = set(stopwords.words("english"))

def clean_string(
    string
):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r"\.\.\.", ".", string)
    string = re.sub(r"\.\.", ".", string)
    string = re.sub(r"\?", ".", string)
    string = re.sub(r"\!", ".", string)
    return string.strip().lower()

def preprocess_text(
    raw,
    remove_stopwords=False,
    lemmatize=False
):
    text = re.sub("[^a-zA-Z0-9]", " ", raw)
    words = word_tokenize(text)
    if lemmatize == True:
        words = [stem.stem(w) for w in words]
    if remove_stopwords == True:
        words = [w for w in words if not w in stops]
    return words

def process_dataset(
    dataset,
    remove_stopwords=False,
    lemmatize=False
):
    return dataset.map(lambda x: preprocess_text(x, remove_stopwords,lemmatize))

def process_string(
        dataset
):
    processed_dataset = preprocess_text(dataset)
    return ' '.join(processed_dataset)

def process_dataset_2(
    dataset,
    remove_stopwords=False,
    lemmatize=False
):
    return dataset.map(lambda x: ' '.join(preprocess_text(x, remove_stopwords,lemmatize)))
