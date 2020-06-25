import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk, pos_tag
import contractions

stops = set(stopwords.words("english"))
wordnet_lemmatizer = WordNetLemmatizer()


def clean_string(
        string
):
#    string = re.sub(r"\\", "", string)
#    string = re.sub(r"\'", "", string)
#    string = re.sub(r"\"", "", string)
#    string = re.sub(r"\.\.\.", ".", string)
#    string = re.sub(r"\.\.", ".", string)
#    string = re.sub(r"\?", ".", string)
#    string = re.sub(r"\!", ".", string)
#    return string.strip().lower()
    return ' '.join(preprocess_text(string, regex1="[^a-zA-Z'\.\?\!]", regex2="[^a-zA-Z\.\?\!]"))

def preprocess_text(
        raw,
        remove_stopwords=False,
        lemmatize=False,
        name_entity_extraction=False,
        contraction_expanding=False,
        regex1="[^a-zA-Z']",
        regex2="[^a-zA-Z]"
):
    # text_without_email = re.sub("\w*@\w?|\w*@\w*\.{1}\w?|\w*@\w*\.{1}\w*\.{1}\w?|\w*@\w*\.{1}\w*\.{1}\w*\.{1}\w?", " ", raw)
    text_without_email = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', raw)
    text = re.sub(regex1, " ", text_without_email)

    # expand contractions into full form
    if contraction_expanding:
        text = contractions.fix(text)

    # remove remaining 's, 'd, 'll and other forms
    words = word_tokenize(text)
    words = [re.sub("['][a-zA-Z]{1,2}$", "", word) for word in words]

    # remove all single quotes
    words = [re.sub(regex2, "", word) for word in words]

    # filter all '' strings
    words = list(filter(None, words))

    # concatenate named entity into one string
    if name_entity_extraction:
        chunks = ne_chunk(pos_tag(words))
        words = [w[0] if isinstance(w, tuple) else " ".join(t[0] for t in w) for w in chunks]

    lowercase_words = [w.lower() for w in words if len(w) > 1]
    # lowercase_words = [w for w in words if len(w) > 1]
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
        dataset,
        remove_stopwords=False,
        lemmatize=False,
        name_entity_extraction=False,
        contraction_expanding=False
):
    processed_dataset = preprocess_text(
        dataset,
        remove_stopwords,
        lemmatize,
        name_entity_extraction,
        contraction_expanding)
    return ' '.join(processed_dataset)


def process_dataset(
        dataset,
        remove_stopwords=False,
        lemmatize=False,
        name_entity_extraction=False,
        contraction_expanding=False
):
    return dataset.map(
        lambda x: preprocess_text(
            x,
            remove_stopwords,
            lemmatize,
            name_entity_extraction,
            contraction_expanding)
    )


def process_dataset_2(
    dataset,
    remove_stopwords=False,
    lemmatize=False,
    name_entity_extraction=False,
    contraction_expanding=False
):
    return dataset.map(
        lambda x: ' '.join(
            preprocess_text(
                x,
                remove_stopwords,
                lemmatize,
                name_entity_extraction,
                contraction_expanding)
        )
    )

