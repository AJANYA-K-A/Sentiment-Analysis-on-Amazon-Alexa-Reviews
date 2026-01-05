import pandas as pd
import numpy as np
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
from flask import Flask, render_template, request

# Download stopwords if not already downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to clean and preprocess text
def preprocess_text(text):
    if not isinstance(text, str):  # Check if text is not a string
        text = ""  # Replace with an empty string
    
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & numbers
    text = " ".join(word for word in text.split() if word.lower() not in stop_words)  # Remove stopwords
    
    return text

# Load the dataset (TSV format)
df = pd.read_csv(r"D:\Desktop\2ND SEM PROJECTS\nagrajan sir project\amazon_alexa.tsv", sep="\t")

# Apply preprocessing (handling missing values)
df["cleaned_text"] = df["verified_reviews"].apply(preprocess_text)

# Convert positive/negative feedback into binary labels (1 = Positive, 0 = Negative)
df["sentiment"] = df["feedback"].apply(lambda x: 1 if x == 1 else 0)

# Balance the dataset
df_0 = df[df['sentiment'] == 0]
df_1 = df[df['sentiment'] == 1]
df_0_upsample = resample(df_0, n_samples=2893, replace=True, random_state=123)
df = pd.concat([df_1, df_0_upsample]).reset_index(drop=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df['sentiment'])

# Convert text data into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define Individual Models
nb_model = MultinomialNB()
logistic_model = LogisticRegression()
svm_model = SVC(probability=True, kernel="linear")
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Ensemble Model using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ("Naïve Bayes", nb_model),
        ("Logistic Regression", logistic_model),
        ("SVM", svm_model),
        ("Random Forest", random_forest_model),
    ],
    voting="soft"  # Soft voting uses probabilities for better performance
)

# Train Ensemble Model
ensemble_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = ensemble_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
with open("sentiment_model.pkl", "wb") as model_file:
    pickle.dump(ensemble_model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Ensemble model and vectorizer saved successfully!")

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)