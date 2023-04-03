import argparse
import re
import string
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib


def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)


def main():
    parser = argparse.ArgumentParser(description="Predict the sentiment of a given text.")
    parser.add_argument("text", type=str, help="The text to classify.")
    args = parser.parse_args()

    # Step 1: Preprocess input text
    text = preprocess(args.text)

    # Step 2: Load the saved model
    try:
        model = joblib.load("model.joblib")
    except FileNotFoundError:
        print("Error: Model file not found. Please run training_sentiment.py first to train and save the model.")
        sys.exit(1)

    # Step 3: Predict and print the label
    prediction = model.predict([text])[0]
    sentiment = "positive" if prediction == 1 else "negative"
    print(f"The predicted sentiment of the text is {sentiment}.")


if __name__ == "__main__":
    main()
