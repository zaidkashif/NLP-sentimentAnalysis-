# 🧠 Customer Review Sentiment Analysis with Streamlit

This project is a machine learning application that performs **sentiment analysis** on customer reviews using Natural Language Processing (NLP). The goal is to automatically classify product reviews as **positive** or **negative**, helping businesses gain insights into customer feedback at scale.

---

## 📌 Features

- ✅ **Binary sentiment classification** (Positive / Negative)
- ✅ Text preprocessing pipeline: lowercasing, stopwords removal, lemmatization, etc.
- ✅ Model trained on **Amazon Food Reviews dataset** (~500k+ records)
- ✅ TF-IDF feature extraction
- ✅ Performance comparison between **Logistic Regression** and **Naive Bayes**
- ✅ Interactive **Streamlit Web App** for real-time predictions
- ✅ Model & vectorizer saved using Joblib for reuse

---

## 🧰 Tech Stack

- **Python 3.12**
- **Pandas**, **Scikit-learn**, **NLTK**
- **Streamlit** (for UI)
- **Joblib** (for model serialization)

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit scikit-learn pandas nltk joblib
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 💡 How It Works

1. User enters a product review.
2. Review is cleaned and preprocessed.
3. Text is transformed using **TF-IDF** vectorization.
4. Trained **Logistic Regression model** predicts the sentiment.
5. Output is displayed instantly in the browser.

---

## 📊 Model Performance (on full dataset)

| Metric    | Logistic Regression | Naive Bayes |
| --------- | ------------------- | ----------- |
| Accuracy  | 89.8%               | 88.2%       |
| Precision | 0.98 (Positive)     | 0.88        |
| Recall    | 0.90 (Positive)     | 1.00        |
| F1-Score  | 0.94                | 0.93        |

---

## 📁 File Structure

```
.
├── models/
│   ├── logistic_model.joblib
│   └── tfidf_vectorizer.joblib
├── streamlit_app.py      # Main UI
├── predict_sentiment.py  # CLI testing (optional)
├── reviews.csv           # Dataset (not uploaded)
└── README.md
```

---

## 📦 Future Improvements

- Multiclass emotion detection (e.g., joy, anger, sadness)
- Use of Transformer-based models (e.g., BERT)
- Model explainability using SHAP or LIME
- Deployment to cloud platforms (Streamlit Cloud, Render, etc.)

---

## 🙌 Acknowledgments

- Dataset sourced from [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- Inspired by real-world applications in eCommerce and customer feedback analytics
