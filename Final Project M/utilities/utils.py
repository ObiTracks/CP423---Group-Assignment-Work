import hashlib
import heapq
import math
import os
import re
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List
from urllib.parse import urlparse

import justext
import requests
from bs4 import BeautifulSoup
from colorama import Fore, Style
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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

    if content is None or content == "":
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


# Utilities for index_documents

def tokenize(text: str) -> List[str]:
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words


def calculate_term_frequency(tokens):
    term_freq = defaultdict(int)
    for token in tokens:
        term_freq[token] += 1

    return term_freq


def get_soundex(token: str) -> str:
    token = token.upper()
    soundex = ""
    soundex += token[0]
    dictionary = {
        "BFPV": "1",
        "CGJKQSXZ": "2",
        "DT": "3",
        "L": "4",
        "MN": "5",
        "R": "6"
    }
    for char in token[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != soundex[-1]:
                    soundex += code
    soundex = soundex.ljust(4, "0")
    return soundex[:4]


def save_inverted_index(inverted_index):
    with open("invertedindex.txt", "w", encoding="utf-8") as f:
        for term, appearances in inverted_index.items():
            soundex = get_soundex(term)
            appearances_str = ', '.join([f'({doc_id}, {freq})' for doc_id, freq in appearances])
            output_str = f"| {term} | {soundex} | {appearances_str} |\n"
            f.write(output_str)


def save_mapping(mapping):
    with open("mapping.txt", "w") as f:
        print("Saving mapping...")
        for file_hash, doc_id in mapping.items():
            f.write(f"{file_hash} {doc_id}\n")


# Utilities for search_for_query

def read_inverted_index(filename):
    inverted_index = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            _, term, soundex_code, postings_str, _ = line.split("|", 4)

            postings = re.findall(r'\(\d+,\s*\d+\)', postings_str)

            appearances = [[int(num) for num in posting.strip('()').split(', ')] for posting in postings]

            term = term.strip()

            inverted_index[term] = (soundex_code, appearances)

    return inverted_index


def get_best_matching_terms(query, inverted_index):
    tokens = [re.sub(r'\W+', '', token.lower()) for token in tokenize(query)]

    best_matching_terms = []

    for token in tokens:
        if token in inverted_index:

            best_matching_terms.append(token)
        else:

            soundex_code = get_soundex(token)
            best_match = None
            for term in inverted_index:
                if get_soundex(term) == soundex_code:
                    best_match = term
                    break
            if best_match:
                best_matching_terms.append(best_match)

    return best_matching_terms


def find_matching_documents(terms, inverted_index):
    doc_ids = set()
    for term in terms:
        if term in inverted_index:
            appearances = inverted_index[term][1]
            doc_ids.update([appearance[0] for appearance in appearances])

    doc_id_to_path = read_mapping_file("mapping.txt")
    doc_paths = [doc_id_to_path[doc_id] for doc_id in doc_ids]

    return doc_paths


def get_document_text(doc_hash):
    with open(doc_hash, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def read_mapping_file(filename):
    doc_id_to_path = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            path, doc_id = line.strip().split()
            doc_id = int(doc_id)
            path = path.replace("\\", "/")
            doc_id_to_path[doc_id] = 'data/' + path + '.txt'
    return doc_id_to_path


def vectorize_documents(document_paths):
    doc_vectors = []
    for doc_path in document_paths:
        with open(doc_path, "r", encoding="utf-8") as file:
            text = file.read()

        tokens = tokenize(text)
        term_freq = Counter(tokens)
        doc_length = math.sqrt(sum([tf ** 2 for tf in term_freq.values()]))
        vector = {}
        for term in term_freq:
            vector[term] = term_freq[term] / doc_length
        doc_vectors.append(vector)
    return doc_vectors


def vectorize_query(query, terms):
    query_vector = {}
    query_terms = query.split()
    query_length = len(query_terms)
    term_freq = Counter(terms)
    for term in term_freq:
        query_vector[term] = term_freq[term] / query_length
    return query_vector


def display_similarity_scores(query_vector, doc_vectors, document_paths, best_matching_terms):
    sim_scores = {}
    for i in range(len(document_paths)):
        sim_scores[document_paths[i]] = cosine_similarity(query_vector, doc_vectors[i])

    results = heapq.nlargest(3, sim_scores.items(), key=lambda x: x[1])

    print("Top 3 document results")
    for file_path, score in results:
        content = get_document_text(file_path)
        highlighted_content = content
        for term in best_matching_terms:
            highlighted_content = highlighted_content.replace(term, f"{Fore.RED}{term}{Style.RESET_ALL}")

        # Get the original URL using the file_hash
        file_hash = os.path.splitext(os.path.basename(file_path))[0]

        print(f"Document: {file_path}, Similarity score: {score:.4f}\n{highlighted_content}\n\n")


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    return float(numerator) / denominator
