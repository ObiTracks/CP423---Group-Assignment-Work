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

def crawl_and_extract_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = ' '.join([p.text for p in soup.find_all('p')])
    internal_links = [a['href'] for a in soup.find_all('a', href=True) if url in a['href']]
    return content, internal_links

def save_content(topic, link, content):
    file_hash = hashlib.md5(link.encode()).hexdigest()
    file_path = os.path.join("data", topic, f"{file_hash}.txt")

    with open(file_path, "w") as f:
        f.write(content)

    with open("crawl.log", "a") as log:
        log.write(f"{topic}, {link}, {file_hash}, {datetime.now()}\n")

def crawl_internal_links(topic, internal_links, initial_url):
    for link in internal_links:
        if initial_url in link:
            content, _ = crawl_and_extract_content(link)
            save_content(topic, link, content)



# Utilities for index_documents()

def tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return words

def calculate_term_frequency(tokens):
    term_freq = defaultdict(int)
    for token in tokens:
        term_freq[token] += 1

    return term_freq


# Utilities for train_ml_classiffier()

def collect_texts_and_labels():
    texts = []
    labels = []
    for topic in ["Technology", "Psychology", "Astronomy"]:
        files = glob.glob(f"data/{topic}/*.txt")
        for file_path in files:
            with open(file_path, "r") as f:
                content = f.read()
                texts.append(content)
                labels.append(topic)

    return texts, labels