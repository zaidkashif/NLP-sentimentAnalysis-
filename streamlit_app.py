import streamlit as st
from joblib import load
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Load NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = load("models/logistic_model.joblib")
vectorizer = load("models/tfidf_vectorizer.joblib")

# Cleaning function
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

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🔍")

st.title("🔍 Customer Review Sentiment Analyzer")
st.markdown("Enter a product review below and we'll predict whether it's **Positive** or **Negative**.")

review = st.text_area("📝 Enter your review here:", height=150)

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"**Predicted Sentiment:** {sentiment}")
