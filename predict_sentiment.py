from joblib import load
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

model = load("models/logistic_model.joblib")
vectorizer= load("models/tfidf_vectorizer.joblib")


def clean_text(text):
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"[^a-zA-Z]"," ",text)
    words = text.lower().split()
    words =[w for w in words if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return" ".join(words)


def predict_sentiment():
    print("Type your review below. Type 'exit' to stop.\n")
    while True:
        review = input("Review: ")
        if review.lower()=="exit":
            print("Exiting..")
            break
        cleaned = clean_text(review)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)
        result = "Positive" if prediction[0] == 1 else "Negative"
        print("Predicted Sentiment: ", result, "\n")

if __name__ == "__main__":
    predict_sentiment()