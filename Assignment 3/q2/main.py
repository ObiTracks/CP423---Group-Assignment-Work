import os
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score
from sklearn.decomposition import TruncatedSVD

import joblib
import re

DEBUG = True

def load_data(path):
    data = []
    labels = []
    if DEBUG: print('Loading data...')
    for root, _, files in os.walk(path):
        label = re.sub(r'\d+', '', os.path.basename(root))  # extract label from folder name
        for file in files:
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                data.append(content)
                labels.append(label)
    if DEBUG: print(f'Data loaded. {len(data)} documents found.')
    return data, labels


def preprocess_data(data):
    if DEBUG: print('Preprocessing data...')
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    if DEBUG: print('Data preprocessed.')
    return X


def reduce_data(X, n_components=100):
    if DEBUG: print('Reducing X...')
    svd = TruncatedSVD(n_components=n_components)
    X_reduced = svd.fit_transform(X)
    if DEBUG: print('X reduced.')
    return X_reduced



def cluster_data(X, n_clusters, algorithm):
    if DEBUG: print(f'Clustering using {algorithm} with {n_clusters} clusters...')
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == 'whc':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif algorithm == 'ac':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    elif algorithm == 'dbscan':
        model = DBSCAN()
    else:
        raise ValueError('Invalid algorithm')
    y_pred = model.fit_predict(X)
    return y_pred, model


def evaluate_cluster(y_true, y_pred):
    ami = adjusted_mutual_info_score(y_true, y_pred)
    ars = adjusted_rand_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    return ami, ars, comp


def save_model(model, algorithm, n_clusters):
    filename = f'{algorithm}_{n_clusters}.joblib'
    joblib.dump(model, filename)


def run_from_terminal():
    parser = argparse.ArgumentParser(description='Clustering news data')
    parser.add_argument('--ncluster', type=int, nargs='+', default=[20], help='Number of clusters')
    parser.add_argument('--kmeans', action='store_true', help='Use KMeans clustering')
    parser.add_argument('--whc', action='store_true', help='Use Ward Hierarchical Clustering')
    parser.add_argument('--ac', action='store_true', help='Use Agglomerative Clustering')
    parser.add_argument('--dbscan', action='store_true', help='Use DBSCAN Clustering')
    args = parser.parse_args()
    main(args)


def run_from_ide():
    class Args:
        def __init__(self):
            self.ncluster = [20]
            self.kmeans = True
            self.whc = True
            self.ac = True
            self.dbscan = True

        def __str__(self):
            return f'ncluster: {self.ncluster}, kmeans: {self.kmeans}, whc: {self.whc}, ac: {self.ac}, dbscan: {self.dbscan}'

    args = Args()
    main(args)

def main(args):
    data_path = '20_newsgroups'
    data, y_true = load_data(data_path)
    X = preprocess_data(data)

    for n_clusters in args.ncluster:
        if DEBUG: print(f'\nProcessing {n_clusters} clusters...')
        if args.kmeans:
            y_pred, model = cluster_data(X, n_clusters, 'kmeans')
            save_model(model, 'kmeans', n_clusters)
        if args.whc:
            X_reduced = reduce_data(X)
            y_pred, model = cluster_data(X_reduced, n_clusters, 'whc')
            save_model(model, 'whc', n_clusters)
        if args.ac:
            X_reduced = reduce_data(X)
            y_pred, model = cluster_data(X_reduced, n_clusters, 'ac')
            save_model(model, 'ac', n_clusters)
        if args.dbscan:
            y_pred, model = cluster_data(X, n_clusters, 'dbscan')
            save_model(model, 'dbscan', n_clusters)

        ami, ars, comp = evaluate_cluster(y_true, y_pred)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Number of clusters: {n_clusters}')
        print(f'Adjusted Mutual Information: {ami}')
        print(f'Adjusted Rand Score: {ars}')
        print(f'Completeness Score: {comp}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')




if __name__ == "__main__":
    # Uncomment the line below to run the script from the terminal
    # run_from_terminal()

    # Uncomment the line below to run the script from an IDE like PyCharm
    run_from_ide()