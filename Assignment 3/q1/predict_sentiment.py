# Sample usage statements:
# python predict_sentiment.py "I hate the news. But the sun is shiny. So its a good day."
# python predict_sentiment.py "I hate the news. It gives too much text to classify"

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

def predict_sentiment(text):
    # Step 1: Preprocess input text
    processed_text = preprocess(text)

    # Step 2: Load the saved model
    try:
        model = joblib.load("model.joblib")
    except FileNotFoundError:
        print("Error: Model file not found. Please run training_sentiment.py first to train and save the model.")
        sys.exit(1)

    # Step 3: Predict and print the label
    prediction = model.predict([processed_text])[0]
    sentiment = "positive" if prediction == 1 else "negative"
    print(f"The predicted sentiment of the text is {sentiment}.")
    return sentiment



def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the sentiment of input text.")
    parser.add_argument("text", help="The input text to predict sentiment for.")
    args = parser.parse_args()
    print(args.text)

    # Step 2: Predict and print the sentiment
    predict_sentiment(args.text)


if __name__ == '__main__':
    main()