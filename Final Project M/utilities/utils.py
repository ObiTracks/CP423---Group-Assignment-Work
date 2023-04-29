import os
import re
import hashlib
from datetime import datetime
from urllib.parse import urlparse
import glob

import requests
from bs4 import BeautifulSoup
import justext
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict


# Utilities for collect_new_documents

def crawl_and_extract_content(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme.startswith('http'):
        print(f"Skipping non-HTTP URL: {url}")
        return None, []

    print("URL: ", url)
    response = requests.get(url)
    paragraphs = justext.justext(response.content, justext.get_stoplist("English"))
    content = ' '.join([paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate])
    print(content[:100])
    content = remove_stopwords(content)
    soup = BeautifulSoup(response.text, "html.parser")
    internal_links = [a['href'] for a in soup.find_all('a', href=True) if url in a['href']]
    
    return content, internal_links


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def save_content(topic, link, content):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_dir, "..", "data", topic)
    print(directory)
    if not os.path.exists(directory):
          print(f"Path does not exist. Creating directory '{directory}' ...")
          os.makedirs(directory)

    file_hash = hashlib.md5(link.encode("utf-8")).hexdigest()
    file_path = f"{directory}/{file_hash}.txt"
    
    if content == None or content == "":
      return
    print("Saving content...")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Writing content to log...")    
    log_path = os.path.join(script_dir, "..", "crawl.log")
    with open(log_path, "a") as log:
        log.write(f"{topic}, {link}, {file_hash}, {datetime.now()}\n")


def crawl_internal_links(topic, internal_links, initial_url, depth):
    if depth <= 0:
        return

    for link in internal_links:
        print(f"Collecting content from internal link: {link}")
        if initial_url in link:
            content, new_internal_links = crawl_and_extract_content(link)
            save_content(topic, link, content)
            if content is not None:
                save_content(topic, link, content)
                crawl_internal_links(topic, new_internal_links, initial_url, depth - 1)



# Utilities for index_documents()

def tokenize(text):
    words = re.findall(r'\w+', text.lower())
    return words


def calculate_term_frequency(tokens):
    term_freq = defaultdict(int)
    for token in tokens:
        term_freq[token] += 1

    return term_freq
  
def get_soundex(token):
    token = token.upper()
    soundex_code = ""

    soundex_dict = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    soundex_code += token[0]

    for char in token[1:]:
        code = soundex_dict.get(char, '0')
        if code != '0' and code != soundex_code[-1]:
            soundex_code += code

    soundex_code = soundex_code.replace('0', '')
    soundex_code = soundex_code.ljust(4, '0')

    return soundex_code[:4]

def save_inverted_index(inverted_index):
    with open("invertedindex.txt", "w", encoding="utf-8") as f:
        print("Saving inverted index...")
        for term, appearances in inverted_index.items():
            soundex = get_soundex(term)
            appearances_str = ', '.join([f'({doc_hash}, {freq})' for doc_hash, freq in appearances])
            output_str = f"{term}|{soundex}|{appearances_str}\n"
            print(output_str)
            f.write(output_str)


def save_mapping(mapping):
    with open("mapping.txt", "w") as f:
        print("Saving mapping...")
        for file_hash, doc_id in mapping.items():
            f.write(f"{file_hash} {doc_id}\n")


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