import glob
import pickle

import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')

from .utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def collect_new_documents():
    with open("url_hashes.txt", "w", encoding="utf-8") as f:
        pass
    with open("sources.txt", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            topic, link = line.strip().split(", ")
            content, internal_links = crawl_and_extract_content(link)
            save_content(topic, link, content)
            crawl_internal_links(topic, internal_links, link, 1)


def index_documents():
    inverted_index = defaultdict(list)
    doc_id = 1
    mapping = {}

    for topic in ["Technology", "Psychology", "Astronomy"]:
        files = glob.glob(f"data/{topic}/*.txt")
        for file_path in files:
            file_hash = file_path.split("/")[-1].split(".")[0]
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tokens = tokenize(content)
                term_freq = calculate_term_frequency(tokens)

                for term, freq in term_freq.items():
                    inverted_index[term].append((doc_id, freq))

            mapping[file_hash] = doc_id
            doc_id += 1
    save_inverted_index(inverted_index)
    save_mapping(mapping)
    print()


def search_for_query(query):
    inverted_index = read_inverted_index("invertedindex.txt")

    # Find best matching terms
    best_matching_terms = get_best_matching_terms(query, inverted_index)

    # Find the list of documents that include the query terms
    document_paths = find_matching_documents(best_matching_terms, inverted_index)

    # Vectorize documents
    doc_vectors = vectorize_documents(document_paths)

    # Vectorize query
    query_vector = vectorize_query(query, best_matching_terms)

    # Calculate and display similarity scores
    display_similarity_scores(query_vector, doc_vectors, document_paths, best_matching_terms)


def train_ml_classifier():
    # Collect the data
    texts = []
    labels = []
    for topic in ["Technology", "Psychology", "Astronomy"]:
        files = glob.glob(f"data/{topic}/*.txt")
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(content)
                labels.append(topic)

    # Vectorize the data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Define a list of classifiers to try out
    classifiers = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        KNeighborsClassifier(),
        LogisticRegression(),
        SVC(probability=True)  # Set probability to True
    ]

    # Train and evaluate each classifier
    for clf in classifiers:
        pipeline = Pipeline([('clf', clf)])
        pipeline.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Test Accuracy: {acc}")
        print(f"Test Confusion Matrix:\n{cm}\n")

    # Select the best performing classifier
    best_clf = SVC(probability=True)  # Set probability to True
    pipeline = Pipeline([('clf', best_clf)])
    pipeline.fit(X, labels)

    # Save the best performing classifier
    with open("classifier.model", "wb") as f:
        pickle.dump(pipeline, f)

    # Save the vectorizer
    with open("vectorizer.model", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved trained model as 'classifier.model'")
    print(f"Saved trained vectorizer as 'vectorizer.model'")


def predict_link(link):
    # Crawl the content from the link
    content, internal_links = crawl_and_extract_content(link)

    if content is None or content == "":
        print("Could not fetch content from the provided link.")
        return

    # Preprocess the content
    preprocessed_content = remove_stopwords(content)

    # Load the pre-trained vectorizer and classifier
    with open("vectorizer.model", "rb") as file:
        vectorizer = pickle.load(file)

    with open("classifier.model", "rb") as file:
        classifier = pickle.load(file)

    # Vectorize the preprocessed content
    transformed_text = vectorizer.transform([preprocessed_content])

    # Predict the label of the content
    label = classifier.predict(transformed_text)[0]

    # Get the probability of the predicted label
    probabilities = classifier.predict_proba(transformed_text)
    confidence = probabilities.max() * 100

    print(f"Predicted label: {label}, Confidence: {confidence:.2f}%")


def your_story():
    filename = "story.txt"

    with open(filename, "r") as f:
        content = f.read()

    print("Your story:")
    print(content)
