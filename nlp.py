import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm 

dataset = pd.read_csv("Reviews/Reviews.csv", nrows=50000)

print(dataset.shape)
print(dataset.columns)
print(dataset.head())

dataset = dataset[['Score', 'Text']]
print(dataset.isnull().sum())

dataset = dataset[dataset["Score"]!=3]
dataset["Sentiment"] = dataset["Score"].apply(lambda x:"positive" if x>3 else "negative")
print(dataset["Score"].value_counts())

stopwords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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

    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]"," ",text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords]
    return" ".join(words)
tqdm.pandas()
dataset["CleanedText"]=dataset["Text"].progress_apply(clean_text)

#Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X=tfidf.fit_transform(dataset["CleanedText"])
y= dataset["Sentiment"]
print(dataset["Sentiment"].value_counts())
dataset["SentimentEncoded"]=dataset["Sentiment"].map({"positive":1, "negative":0})
dataset=dataset.dropna(subset=["SentimentEncoded"])
print(dataset["SentimentEncoded"].isnull().sum())
print(dataset["SentimentEncoded"].value_counts())

#Splitting datasets
from sklearn.model_selection import train_test_split
X_features = tfidf.fit_transform(dataset['CleanedText'])
y_labels=dataset["SentimentEncoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2, random_state=42
)
print("Training set size: ", X_train.shape)
print("Test set size: ", X_test.shape)

# Model Training and Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test,y_pred))

print("\nClassification Report:\n",classification_report(y_test,y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test,y_pred))