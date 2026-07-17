"""
train_model.py
──────────────
Trains an ensemble sentiment classifier on the Amazon Alexa reviews dataset
and saves the model + TF-IDF vectorizer as .pkl files.

Usage:
    python train_model.py
"""

import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


# ── Text preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text


# ── Load dataset ──────────────────────────────────────────────────────────────
DATASET_PATH = "amazon_alexa.tsv"

print(f"Loading dataset from '{DATASET_PATH}' …")
df = pd.read_csv(DATASET_PATH, sep="\t")

# Ensure required columns exist
required_cols = {"verified_reviews", "feedback"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

df["cleaned_text"] = df["verified_reviews"].apply(preprocess_text)
df["sentiment"]    = df["feedback"].apply(lambda x: 1 if x == 1 else 0)

print(f"Total samples: {len(df)}  |  Positive: {df['sentiment'].sum()}  |  Negative: {(df['sentiment'] == 0).sum()}")

# ── Balance classes ───────────────────────────────────────────────────────────
df_0 = df[df["sentiment"] == 0]
df_1 = df[df["sentiment"] == 1]

# Upsample minority class to match majority
n_target = max(len(df_0), len(df_1))
df_0_up  = resample(df_0, n_samples=n_target, replace=True,  random_state=123)
df_1_up  = resample(df_1, n_samples=n_target, replace=True,  random_state=123)

df_bal = pd.concat([df_1_up, df_0_up]).reset_index(drop=True)
print(f"Balanced dataset size: {len(df_bal)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df_bal["cleaned_text"],
    df_bal["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df_bal["sentiment"],
)

# ── TF-IDF vectorisation ──────────────────────────────────────────────────────
vectorizer    = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ── Ensemble model ────────────────────────────────────────────────────────────
print("\nTraining ensemble model (this may take a minute) …")

ensemble_model = VotingClassifier(
    estimators=[
        ("naive_bayes",         MultinomialNB()),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("svm",                 SVC(probability=True, kernel="linear", random_state=42)),
        ("random_forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
    ],
    voting="soft",
)

ensemble_model.fit(X_train_tfidf, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = ensemble_model.predict(X_test_tfidf)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ── Save artefacts ────────────────────────────────────────────────────────────
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅  Saved: sentiment_model.pkl  tfidf_vectorizer.pkl")