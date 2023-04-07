import os
import argparse
import joblib
import re
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import TruncatedSVD

DEBUG = False

def load_news_group_data(path):
    news_group_data = []
    news_group_labels = []
    if DEBUG: print('Loading data...')
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for d in dirs:
        label = re.sub(r'\d+', '', os.path.basename(d))
        list_of_files = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for file in list_of_files:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                news_group_data.append(content)
                news_group_labels.append(label)
    if DEBUG: print(f'Data loaded. {len(news_group_data)} documents found.')
    return news_group_data, news_group_labels


def preprocess_news_group_data(data):
    if DEBUG: print('Preprocessing data...')
    converted_data = TfidfVectorizer(stop_words='english')
    news_groups = converted_data.fit_transform(data)
    if DEBUG: print('Data preprocessed.')
    return news_groups


def reduce_news_group_data(news_groups, n_components=100):
    if DEBUG: print('Reducing X...')
    svd = TruncatedSVD(n_components=n_components)
    news_groups_reduced = svd.fit_transform(news_groups)
    if DEBUG: print('X reduced.')
    return news_groups_reduced



def cluster_news_group_data(news_groups, n_clusters, algorithm_type):
    if DEBUG: print(f'Clustering using {algorithm_type} with {n_clusters} clusters...')
    if algorithm_type == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm_type == 'whc':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif algorithm_type == 'ac':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    elif algorithm_type == 'dbscan':
        model = DBSCAN()
    else:
        raise ValueError('Invalid algorithm')
    pred = model.fit_predict(news_groups)
    return pred, model


def evaluate_clustering_performance(true_data, pred):
    ami = adjusted_mutual_info_score(true_data, pred)
    ars = adjusted_rand_score(true_data, pred)
    comp = completeness_score(true_data, pred)
    return ami, ars, comp


def save_news_group_model(model, algorithm, n_clusters):
    filename = os.path.join("models", f"{algorithm}_{n_clusters}.joblib")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)

def type_kmeans(news_groups, n_clusters):
    pred, model = cluster_news_group_data(news_groups, n_clusters, 'kmeans')
    save_news_group_model(model, 'kmeans', n_clusters)
    return pred

def type_whc(news_groups, n_clusters):
    news_groups_reduced = reduce_news_group_data(news_groups)
    pred, model = cluster_news_group_data(news_groups_reduced, n_clusters, 'whc')
    save_news_group_model(model, 'whc', n_clusters)
    return pred

def type_ac(news_groups, n_clusters):
    news_groups_reduced = reduce_news_group_data(news_groups)
    pred, model = cluster_news_group_data(news_groups_reduced, n_clusters, 'ac')
    save_news_group_model(model, 'ac', n_clusters)
    return pred

def type_dbscan(news_groups, n_clusters):
    pred, model = cluster_news_group_data(news_groups, n_clusters, 'dbscan')
    save_news_group_model(model, 'dbscan', n_clusters)
    return pred



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncluster', type=int, nargs='+', default=[20])
    parser.add_argument('--kmeans', action='store_true')
    parser.add_argument('--whc', action='store_true')
    parser.add_argument('--ac', action='store_true')
    parser.add_argument('--dbscan', action='store_true')
    args = parser.parse_args()

    data_path = '20_newsgroups'
    data, true_data = load_news_group_data(data_path)
    news_groups = preprocess_news_group_data(data)

    for n_clusters in args.ncluster:
        if DEBUG: print(f'\nProcessing {n_clusters} clusters...')
        if args.kmeans:
            pred = type_kmeans(news_groups, n_clusters)
        if args.whc:
            pred = type_whc(news_groups, n_clusters)
        if args.ac:
            pred = type_ac(news_groups, n_clusters)
        if args.dbscan:
            pred = type_dbscan(news_groups, n_clusters)

        ami, ars, comp = evaluate_clustering_performance(true_data, pred)
        print('Number of clusters: {}'.format(n_clusters))
        print('Adjusted Mutual Information: {}'.format(ami))
        print('Adjusted Rand Score: {}'.format(ars))
        print('Completeness Score: {}'.format(comp))


if __name__ == "__main__":
    main()
