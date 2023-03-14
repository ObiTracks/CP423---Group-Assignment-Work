import argparse
import json
import os
from collections import defaultdict
from nltk import FreqDist, word_tokenize, download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Usage: python wikipedia_processing.py --zipf --tokenize --stopword --stemming --invertedindex
# (Only add the arguments you want to run - they're boolean flags)


def read_json_files(directory):
    """
    Reads all json files in a directory and returns a list of articles.

    Args:
        directory (str): Directory path containing json files.

    Returns:
        list: List of dictionaries containing article ID, title, and text.
    """
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                for line in f:
                    article = json.loads(line.strip())
                    articles.append(article)
    return articles


def plot_zipf(corpus):
    """
    Plots the Zipf diagram for a given corpus.

    Args:
        corpus (str): String containing text corpus.
    """
    fdist = FreqDist(word_tokenize(corpus))
    fdist.plot(50)


def tokenize(corpus, filename):
    """
    Tokenizes a given corpus using NLTK and writes output to file.

    Args:
        corpus (str): String containing text corpus.
        filename (str): Output file name.
    """
    tokens = word_tokenize(corpus)
    with open(filename, "w") as f:
        for token in tokens:
            f.write(token + "\n")


def remove_stopwords(corpus, filename):
    """
    Removes stopwords from a given corpus using NLTK and writes output to file.

    Args:
        corpus (str): String containing text corpus.
        filename (str): Output file name.
    """
    download("stopwords")
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(corpus)
    filtered_tokens = [
        token for token in tokens if token.lower() not in stop_words
    ]
    with open(filename, "w") as f:
        for token in filtered_tokens:
            f.write(token + "\n")


def stem_tokens(corpus, filename):
    """
    Applies Porter stemming to tokens in a given corpus using NLTK and writes output to file.

    Args:
        corpus (str): String containing text corpus.
        filename (str): Output file name.
    """
    ps = PorterStemmer()
    tokens = word_tokenize(corpus)
    stemmed_tokens = [ps.stem(token) for token in tokens]
    with open(filename, "w") as f:
        for token in stemmed_tokens:
            f.write(token + "\n")


def create_inverted_index(corpus):
    """
    Creates an inverted index of a given corpus.

    Args:
        corpus (str): String containing text corpus.

    Returns:
        dict: Dictionary containing inverted index of corpus.
    """
    inverted_index = defaultdict(dict)
    articles = read_json_files("data_wikipedia")
    for article in articles:
        article_id = article["id"]
        article_text = article["text"]
        tokens = word_tokenize(article_text)
        for token in tokens:
            if article_id in inverted_index[token]:
                inverted_index[token][article_id] += 1
            else:
                inverted_index[token][article_id] = 1
    return inverted_index


# Define command line arguments
parser = argparse.ArgumentParser(
    description='Wikipedia text processing using NLTK')
parser.add_argument('--zipf', action='store_true')
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--stopword', action='store_true')
parser.add_argument('--stemming', action='store_true')
parser.add_argument('--invertedindex', action='store_true')
args = parser.parse_args()

with open(os.path.join("data_wikipedia", "corpus.txt"), "r") as f:
    corpus = f.read()

    if args.zipf:
        plot_zipf(corpus)

    if args.tokenize:
        tokenize(corpus, os.path.join("data_wikipedia", "wikipedia.token"))

    if args.stopword:
        remove_stopwords(
            corpus, os.path.join("data_wikipedia", "wikipedia.token.stop"))

    if args.stemming:
        stem_tokens(corpus,
                    os.path.join("data_wikipedia", "wikipedia.token.stem"))

    if args.invertedindex:
        inverted_index = create_inverted_index(corpus)

        with open(os.path.join("data_wikipedia", "inverted_index.txt"),
                  "w") as f:
            for token in inverted_index:
                f.write(f"{token}\t{str(inverted_index[token])}\n")