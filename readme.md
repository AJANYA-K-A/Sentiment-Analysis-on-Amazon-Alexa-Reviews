# SentiScope 🔍 — Amazon Alexa Sentiment Analysis

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-6366f1?style=for-the-badge&logo=github)](https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NLP](https://img.shields.io/badge/NLP-VADER%20%7C%20TextBlob-10b981?style=for-the-badge)](https://github.com/cjhutto/vaderSentiment)

A full-stack & client-side NLP web application for multi-model sentiment analysis on Amazon Alexa reviews, built using Flask, SQLAlchemy, VADER, TextBlob, and a custom trained ML classifier.

> 🌐 **Live Web Application (Zero-Setup):**
> **[https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/](https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/)**

---

## 🚀 Instant Publishing & Live Demo

You can run and explore SentiScope instantly in your browser:
- **Direct GitHub Pages Link:** [https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/](https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/)
- **How to enable GitHub Pages on your repository:**
  1. Open your repository on GitHub: https://github.com/AJANYA-K-A/Sentiment-Analysis-on-Amazon-Alexa-Reviews
  2. Click **Settings** → **Pages** (under Code and automation in the left sidebar)
  3. Under **Build and deployment** → **Source**, select **Deploy from a branch**
  4. Under **Branch**, select main and folder / (root), then click **Save**
  5. Your live site will be published at https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/ within 1–2 minutes!

---

## 📋 Table of Contents

- [Live Demo & Publishing Link](#-instant-publishing--live-demo)

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [ML Model Training](#ml-model-training)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Future Improvements](#future-improvements)

---

## Overview

SentiScope is a multi-model sentiment analysis system that classifies text into **Positive**, **Negative**, or **Neutral** sentiment using an ensemble approach combining:

- **VADER** (Valence Aware Dictionary and sEntiment Reasoner) — rule-based, optimised for social media
- **TextBlob** — NLP library with polarity and subjectivity scores
- **Custom ML Model** — trained TF-IDF + Logistic Regression (or similar) classifier

The application features user authentication, dual-text comparison, batch analysis, a dashboard with charts, full history with export, and a profile management system.

---

## Features

### Core Analysis

- ✅ **Dual-text comparison** — Analyse and compare two texts side-by-side
- ✅ **3-class classification** — Positive / Neutral / Negative (improved from binary)
- ✅ **Detailed score breakdown** — VADER compound, TextBlob polarity, subjectivity, ML confidence
- ✅ **Ensemble voting** — Weighted combination of all three models
- ✅ **Voice input** — Web Speech API integration for dictating text (Chrome/Edge)

### Batch Analysis

- ✅ **Batch mode** — Analyse up to 50 texts at once (one per line)
- ✅ **Summary statistics** — Instant count of positive/negative/neutral
- ✅ **Tabular results** — Sortable table with all scores

### Dashboard & History

- ✅ **Analytics dashboard** — Doughnut chart + line chart of sentiment trends
- ✅ **Full history** — Paginated view of all past analyses
- ✅ **Filtering** — Filter history by label (Pos/Neg/Neu) and mode (single/compare/batch)
- ✅ **CSV export** — Download all history as a CSV file

### User System

- ✅ **Authentication** — Secure signup/login with hashed passwords (Werkzeug)
- ✅ **Session management** — Flask sessions with login_required decorator
- ✅ **Profile page** — Update bio, change password
- ✅ **Danger zone** — Clear analysis history
- ✅ **Case-insensitive login** — Username/email lookup is case-insensitive

### UI/UX

- ✅ **Dark/light mode** — Persisted in localStorage
- ✅ **Responsive design** — Works on mobile and desktop
- ✅ **Animations** — Smooth card reveal and result animations
- ✅ **Flash messages** — Styled feedback for all user actions
- ✅ **AJAX API endpoint** — `/api/analyse` for programmatic use

---

## Tech Stack

| Layer    | Technology                                          |
| -------- | --------------------------------------------------- |
| Backend  | Python 3.9+, Flask 3.x                              |
| Database | SQLite (dev) / PostgreSQL (prod) via SQLAlchemy     |
| Auth     | Werkzeug password hashing                           |
| NLP      | VADER (vaderSentiment), TextBlob, NLTK              |
| ML       | scikit-learn (TF-IDF + Logistic Regression), pickle |
| Charts   | Chart.js 4.4                                        |
| Fonts    | Google Fonts (Syne + DM Sans)                       |
| Frontend | Vanilla HTML/CSS/JS + Jinja2 templates              |

---

## Project Structure

```
Sentiment-Analysis-on-Amazon-Alexa-Reviews/
├── index.html                # Standalone Client-Side App & GitHub Pages Entry
├── app.py                    # Main Full-Stack Flask Application
├── sentiment_model.pkl       # Trained ML classifier
├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
├── amazon_alexa.tsv          # Amazon Alexa Dataset (3,150 customer reviews)
├── requirements.txt          # Python dependencies
├── static/                   # Static media and brand assets
│   ├── logo.svg              # SentiScope Vector Logo
│   └── image_da3f79.png      # Project app icon
├── templates/                # Server-rendered Jinja2 templates (Flask)
│   ├── base.html             # Base layout with navigation and dark/light mode
│   ├── index.html            # Dual comparison analyser
│   ├── batch.html            # Batch analysis page
│   ├── dashboard.html        # Analytics dashboard
│   ├── history.html          # Paginated history + filters
│   ├── login.html            # User authentication
│   ├── signup.html           # Registration
│   └── profile.html          # User profile management
└── README.md                 # Documentation and deployment guide
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AJANYA-K-A/Sentiment-Analysis-on-Amazon-Alexa-Reviews.git
cd Sentiment-Analysis-on-Amazon-Alexa-Reviews

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model files in the project root
#    - sentiment_model.pkl
#    - tfidf_vectorizer.pkl
#    (see ML Model Training section below)

# 5. Run the app
python app.py
```

The app will be available at `http://127.0.0.1:5000`.

---

## Configuration

SentiScope uses environment variables for production configuration:

| Variable       | Default                           | Description                                     |
| -------------- | --------------------------------- | ----------------------------------------------- |
| `SECRET_KEY`   | `dev-secret-change-in-production` | Flask session secret. **Change in production!** |
| `DATABASE_URL` | `sqlite:///sentiscope.db`         | Database URI. Set to PostgreSQL for production. |

### Example for production:

```bash
export SECRET_KEY="your-long-random-secret-key-here"
export DATABASE_URL="postgresql://user:password@localhost:5432/sentiscope"
```

---

## Running the Application

### Development

```bash
python app.py
```

### Production (Gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## Usage Guide

### 1. Dual Text Comparison (`/analyser`)

Enter text in both input boxes and click **Analyse Sentiments**. Voice input is supported on Chrome/Edge via the microphone button. Results show:

- Sentiment label (Positive/Neutral/Negative) with confidence
- VADER compound score (-1 to 1)
- TextBlob polarity and subjectivity
- ML model confidence percentage
- A comparison verdict below the two cards

### 2. Batch Analysis (`/batch`)

Paste multiple texts, one per line (max 50). Click **Analyse All** to process all lines at once. Results are shown in a table and saved to history. Use the **Export CSV** button to download results.

### 3. Dashboard (`/dashboard`)

View overall statistics:

- Total analyses, breakdown by sentiment, average confidence
- Doughnut chart of sentiment distribution
- Line chart of recent 30 analyses (positive=1, neutral=0, negative=-1)
- Table of 10 most recent analyses

### 4. History (`/history`)

Browse all past analyses with pagination (20 per page). Filter by:

- **Label**: Positive / Negative / Neutral
- **Mode**: single / compare / batch
  Export all history as CSV from the filter bar.

### 5. Profile (`/profile`)

- Update your bio
- Change your password
- Clear all analysis history (with confirmation)

---

## ML Model Training

You need to provide `sentiment_model.pkl` and `tfidf_vectorizer.pkl`. Here is a minimal training example using the IMDb or SST-2 dataset:

```python
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    return ' '.join(w for w in text.split() if w not in stop_words)

# Load your dataset (CSV with 'text' and 'label' columns, label: 1=positive, 0=negative)
df = pd.read_csv('dataset.csv')
df['clean'] = df['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_vec, y_train)

print(classification_report(y_test, model.predict(X_test_vec)))

with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")
```

Recommended datasets: **IMDb Movie Reviews**, **Twitter US Airline Sentiment**, **Amazon Product Reviews**.

---

## API Reference

### POST `/api/analyse`

Programmatic JSON endpoint for single-text analysis.

**Request:**

```json
{ "text": "I absolutely love this product!" }
```

**Response (200):**

```json
{
  "ok": true,
  "result": {
    "label": "Positive",
    "emoji": "😊",
    "css": "positive",
    "confidence": 87.3,
    "vader_compound": 0.6369,
    "vader_pos": 24.4,
    "vader_neg": 0.0,
    "vader_neu": 75.6,
    "tb_polarity": 0.375,
    "tb_subjectivity": 0.65,
    "ml_confidence": 92.1,
    "mode": "single",
    "text_preview": "I absolutely love this product!"
  }
}
```

**Error (400):**

```json
{ "ok": false, "error": "Empty text" }
```

Requires login (session cookie).

---

## Database Schema

### `users`

| Column        | Type         | Description           |
| ------------- | ------------ | --------------------- |
| id            | INTEGER PK   | Auto-increment        |
| username      | VARCHAR(64)  | Unique, indexed       |
| email         | VARCHAR(255) | Unique, indexed       |
| password_hash | VARCHAR(256) | Werkzeug hash         |
| bio           | VARCHAR(300) | Optional user bio     |
| created_at    | DATETIME     | Account creation time |

### `analysis_history`

| Column      | Type        | Description                   |
| ----------- | ----------- | ----------------------------- |
| id          | INTEGER PK  | Auto-increment                |
| user_id     | INTEGER FK  | References users.id           |
| text        | TEXT        | Text preview (120 chars)      |
| label       | VARCHAR(20) | Positive / Negative / Neutral |
| confidence  | FLOAT       | Ensemble confidence %         |
| vader_score | FLOAT       | VADER compound score          |
| tb_score    | FLOAT       | TextBlob polarity             |
| mode        | VARCHAR(20) | single / compare / batch      |
| created_at  | DATETIME    | Indexed timestamp             |

---

## Requirements

```
flask>=3.0
flask-sqlalchemy>=3.1
werkzeug>=3.0
vaderSentiment>=3.3
textblob>=0.18
nltk>=3.8
scikit-learn>=1.4
```

Save as `requirements.txt`.

---

## Future Improvements

- [ ] Multi-language sentiment analysis (multilingual BERT)
- [ ] Aspect-based sentiment (ABSA) — analyse sentiment per topic/entity
- [ ] Real-time Twitter/Reddit sentiment streaming
- [ ] REST API with token authentication (Flask-JWT)
- [ ] Admin panel for managing users
- [ ] Email verification on signup
- [ ] Sentiment word highlighting (which words drove the result)
- [ ] Confidence calibration with temperature scaling
- [ ] Docker + CI/CD deployment pipeline

---

## Academic Context

This project was developed as a Final Year Project (FYP) in Computer Science / Artificial Intelligence. It demonstrates:

- **NLP pipeline**: tokenisation, stopword removal, TF-IDF vectorisation
- **Ensemble methods**: combining rule-based and ML models
- **Full-stack web development**: Flask MVC pattern, Jinja2 templating, SQLAlchemy ORM
- **Security**: password hashing, session management, CSRF-safe forms
- **Data visualisation**: Chart.js integration for analytics
- **Software engineering**: modular code structure, environment-based config, RESTful API design

---

_Built with ❤️ using Flask & Python_
#   S e n t i m e n t - A n a l y s i s - o n - A m a z o n - A l e x a - R e v i e w s 
 
 