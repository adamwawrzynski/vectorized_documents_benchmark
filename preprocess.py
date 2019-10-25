from bs4 import BeautifulSoup 
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stem = PorterStemmer()
stops = set(stopwords.words("english"))


def preprocess_text(
    text,
    remove_stopwords=True
):
    text = BeautifulSoup(text, features='html.parser').get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = word_tokenize(text)
    words = [stem.stem(word) for word in words]
    if remove_stopwords == True:
        words = [w for w in words if not w in stops]
    return words

def process_dataset(
    dataset,
    remove_stopwords=True
):
    return dataset.map(lambda x: preprocess_text(x))
