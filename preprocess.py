import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import TransformerMixin, BaseEstimator
from nltk import ne_chunk, pos_tag
import contractions

stops = set(stopwords.words("english"))
wordnet_lemmatizer = WordNetLemmatizer()


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
        lemmatize=False,
        name_entity_extraction=False,
        contraction_expanding=False
):
    text_without_email = re.sub("\S*@\S*\S?|\S*@\S*\S*\S?|\S*@\S*\S*\S*\S?", " ", raw)
    text = re.sub("[^a-zA-Z']", " ", text_without_email)

    # expand contractions into full form
    if contraction_expanding:
        text = contractions.fix(text)

    # remove remaining 's, 'd, 'll and other forms
    words = word_tokenize(text)
    words = [re.sub("['][a-zA-Z]{1,2}$", "", word) for word in words]

    # remove all single quotes
    words = [re.sub("[^a-zA-Z]", "", word) for word in words]

    # filter all '' strings
    words = list(filter(None, words))

    # concatenate named entity into one string
    if name_entity_extraction:
        chunks = ne_chunk(pos_tag(words))
        words = [w[0] if isinstance(w, tuple) else " ".join(t[0] for t in w) for w in chunks]

    lowercase_words = [w.lower() for w in words]
    if remove_stopwords:
        words = [w for w in lowercase_words if w not in stops]
    else:
        words = lowercase_words
    lemma_word = []
    if lemmatize:
        for w in words:
            word1 = wordnet_lemmatizer.lemmatize(w, pos="n")
            word2 = wordnet_lemmatizer.lemmatize(word1, pos="v")
            word3 = wordnet_lemmatizer.lemmatize(word2, pos="a")
            lemma_word.append(word3)
        words = lemma_word
    return words


def process_string(
        dataset
):
    processed_dataset = preprocess_text(dataset)
    return ' '.join(processed_dataset)


def process_dataset(
        dataset,
        remove_stopwords=False,
        lemmatize=False
):
    return dataset.map(lambda x: preprocess_text(x, remove_stopwords, lemmatize))


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 remove_stopwords=False,
                 lemmatization=False,
                 name_entity_extraction=False,
                 contraction_expanding=True):
        """
        https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Stop words removal
            3. Lemmatization

        n_jobs - parallel jobs to run
        remove_stopwords - whether to remove stopwords or not
        lemmatization - wherther to lemmatize text or not
        """
        self.lemmatization = lemmatization
        self.remove_stopwords = remove_stopwords
        self.name_entity_extraction = name_entity_extraction
        self.contraction_expanding = contraction_expanding

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = list(X)

        return self._preprocess_text(X_copy)

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, doc):
        preprocessed = []
        for text in doc:
            text_normalised = self._normalize(text, self.name_entity_extraction, self.contraction_expanding)
            if self.remove_stopwords:
                words = self._remove_stop_words(text_normalised)
            else:
                words = text_normalised

            if self.lemmatization:
                process_words = self._lemmatize(words)
            else:
                process_words = words
            preprocessed.append(' '.join(process_words))
        return preprocessed

    def _normalize(self, raw, name_entity_extraction, contraction_expanding):
        text_without_email = re.sub("\S*@\S*\S?|\S*@\S*\S*\S?|\S*@\S*\S*\S*\S?", " ", raw)
        text = re.sub("[^a-zA-Z']", " ", text_without_email)

        # expand contractions into full form
        if contraction_expanding:
            text = contractions.fix(text)

        # remove remaining 's, 'd, 'll and other forms

        # trimmed_words = re.sub("[a-zA-Z]?'[a-zA-Z]{2}", " ", text)
        # trimmed_words = re.sub("'[a-zA-Z]{2}", " ", text)
        # trimmed_words = re.sub("[^a-zA-Z]", " ", text)
        words = word_tokenize(text)
        words = [re.sub("['][a-zA-Z]{1,2}$", "", word) for word in words]
        # remove all single quotes
        words = [re.sub("[^a-zA-Z]", "", word) for word in words]

        # filter all '' strings
        words = list(filter(None, words))

        # concatenate named entity into one string
        if name_entity_extraction:
            chunks = ne_chunk(pos_tag(words))
            words = [w[0] if isinstance(w, tuple) else " ".join(t[0] for t in w) for w in chunks]
        return [w.lower() for w in words if len(w) > 1]

    def _remove_stop_words(self, doc):
        return [w for w in doc if w not in stops]

    def _lemmatize(self, doc):
        lemma_word = []
        for w in doc:
            word1 = wordnet_lemmatizer.lemmatize(w, pos="n")
            word2 = wordnet_lemmatizer.lemmatize(word1, pos="v")
            word3 = wordnet_lemmatizer.lemmatize(word2, pos="a")
            lemma_word.append(word3)
        return lemma_word