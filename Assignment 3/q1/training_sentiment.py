import argparse
# import os
import re
import string
# import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix  # , plot_confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

import joblib

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

download('punkt')
download('stopwords')


def load_data(file):
    print("Loading data...")
    data = []
    with open("/sentiment_labelled_sentences/" + file, "r", encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split("\t")
            data.append((text, int(label)))
    print("Data loaded...")
    return pd.DataFrame(data, columns=["text", "label"])


def preprocess(text):
    text = re.sub(f"[{string.punctuation}]", "", text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)


def train_and_evaluate(args, X, y):
    if args.naive:
        clf = MultinomialNB()
    elif args.knn:
        clf = KNeighborsClassifier(n_neighbors=args.knn)
    elif args.svm:
        clf = SVC()
    elif args.decisiontree:
        clf = DecisionTreeClassifier()

    model = make_pipeline(TfidfVectorizer(), clf)
    y_pred = cross_val_predict(model, X, y, cv=5)

    print(f"Accuracy: {accuracy_score(y, y_pred)}")
    print(f"Precision: {precision_score(y, y_pred)}")
    print(f"Recall: {recall_score(y, y_pred)}")
    print(f"F-Measure: {f1_score(y, y_pred)}")

    cm = confusion_matrix(y, y_pred)
    disp = plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues, values_format=".0f")
    plt.show()

    model.fit(X, y)
    joblib.dump(model, "model.joblib")


def main():
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument("--imdb", action="store_true", help="Use the IMDB dataset.")
    parser.add_argument("--amazon", action="store_true", help="Use the Amazon dataset.")
    parser.add_argument("--yelp", action="store_true", help="Use the Amazon dataset.")
    parser.add_argument("--naive", action="store_true", help="Use the Naive Bayes classifier.")
    parser.add_argument("--knn", type=int, help="Use the k-NN classifier.")
    parser.add_argument("--svm", action="store_true", help="Use the SVM classifier.")
    parser.add_argument("--decisiontree", action="store_true", help="Use the Decision Tree classifier.")
    args = parser.parse_args()

    # Step 1: Load data

    train_data = None
    test_data = None

    if args.imdb:
        train_data = load_data("imdb_labelled.txt")
        test_data = load_data("imdb_labelled.txt")
    elif args.amazon:
        train_data = load_data("amazon_cells_labelled.txt")
        test_data = load_data("amazon_cells_labelled.txt")
    elif args.yelp:
        train_data = load_data("yelp_labelled.txt")
        test_data = load_data("yelp_labelled.txt")
    else:
        print("Please specify either --imdb, --amazon or --yelp.")
        return

    # Step 2: Preprocess data
    print("Preprocessing data...")
    train_data["text"] = train_data["text"].apply(preprocess)
    test_data["text"] = test_data["text"].apply(preprocess)

    # Step 3: Train and evaluate model
    print("Training and evaluating model...")
    X_train, y_train = train_data["text"], train_data["label"]
    X_test, y_test = test_data["text"], test_data["label"]

    if args.naive:
        print("Using Naive Bayes classifier.")
    elif args.knn:
        print(f"Using k-NN classifier with k={args.knn}.")
    elif args.svm:
        print("Using SVM classifier.")
    elif args.decisiontree:
        print("Using Decision Tree classifier.")
    else:
        print("Please specify a classifier.")
        return
    train_and_evaluate(args, X_train, y_train)
