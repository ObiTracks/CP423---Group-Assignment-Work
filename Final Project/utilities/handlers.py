# import os
# import hashlib
# from datetime import datetime
# import requests
# from bs4 import BeautifulSoup
# import re

# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report

from collections import defaultdict


from utils import *

def collect_new_documents():
    with open("sources.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            topic, link = line.strip().split(", ")
            content, internal_links = crawl_and_extract_content(link)
            save_content(topic, link, content)
            crawl_internal_links(topic, internal_links, link)

def index_documents():
    inverted_index = defaultdict(list)
    doc_id = 1
    mapping = {}

    for topic in ["Technology", "Psychology", "Astronomy"]:
        files = glob.glob(f"data/{topic}/*.txt")
        for file_path in files:
            file_hash = file_path.split("/")[-1].split(".")[0]
            with open(file_path, "r") as f:
                content = f.read()
                tokens = tokenize(content)
                term_freq = calculate_term_frequency(tokens)

                for term, freq in term_freq.items():
                    inverted_index[term].append((file_hash, freq))

            mapping[file_hash] = doc_id
            doc_id += 1

    with open("invertedindex.txt", "w") as f:
        for term, postings in inverted_index.items():
            f.write(f"{term}: {postings}\n")

    with open("mapping.txt", "w") as f:
        for file_hash, doc_id in mapping.items():
            f.write(f"{file_hash}: {doc_id}\n")

def search_for_query():
    query = input("Enter your search query: ")
    query_tokens = tokenize(query)
    query_term_freq = calculate_term_frequency(query_tokens)

    document_scores = defaultdict(int)
    with open("invertedindex.txt", "r") as f:
        for line in f:
            term, postings = line.strip().split(": ")
            if term in query_term_freq:
                postings = eval(postings)
                for file_hash, freq in postings:
                    document_scores[file_hash] += query_term_freq[term] * freq

    sorted_results = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 search results:")
    for file_hash, score in sorted_results:
        with open("crawl.log", "r") as log:
            for line in log:
                if file_hash in line:
                    _, url, _, _ = line.strip().split(", ")
                    print(f"{url} (Score: {score})")


def train_ml_classifier():
    texts, labels = collect_texts_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Technology", "Psychology", "Astronomy"]))

    with open("classifier.model", "wb") as f:
        pickle.dump(clf, f)

def predict_link():
    link = input("Enter a link to predict its topic: ")
    content, _ = crawl_and_extract_content(link)
    tokens = tokenize(content)

    with open("classifier.model", "rb") as f:
        clf = pickle.load(f)

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    content_counts = count_vect.fit_transform([" ".join(tokens)])
    content_tfidf = tfidf_transformer.fit_transform(content_counts)

    predicted_topic = clf.predict(content_tfidf)[0]
    print(f"Predicted topic for the link: {predicted_topic}")


def your_story():
    filename = "story.txt"

    with open(filename, "r") as f:
        content = f.read()

    print("Your story:")
    print(content)

