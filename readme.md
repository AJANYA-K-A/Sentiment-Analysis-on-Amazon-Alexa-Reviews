# SentiScope 🔍 — Amazon Alexa Sentiment Analysis

<div align="center">

  <img src="static/logo.svg" alt="SentiScope Logo" width="80" height="80" />

  # SentiScope
  **An Intelligent AI Platform for Understanding Customer Emotion in Amazon Alexa Reviews**

  [![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-6366f1?style=for-the-badge&logo=github&logoColor=white)](https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/)
  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

  <br />

  <p align="center">
    <b>Ever wondered what thousands of customers really think about their Amazon Alexa?</b><br />
    SentiScope reads customer reviews and instantly tells you whether they're <b>Delighted 😊</b>, <b>Neutral 😐</b>, or <b>Frustrated 😔</b>.
  </p>

</div>

---

> ### 🌐 Try It Right Now (No Installation Needed!)
>
> You can test the application directly in your web browser with zero setup:  
> 👉 **[Click Here to Launch SentiScope Live Demo](https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/)**
>
> *Tip: You can register any username to log in instantly or test the features right away!*

---

## 📖 What is SentiScope? (In Simple Words)

When customers buy an **Amazon Echo**, **Echo Dot**, or **Alexa Smart Home** device, they leave thousands of reviews sharing their honest experiences. Some love the sound quality, some are amazed by the smart features, and others are frustrated by connectivity issues.

Reading through thousands of customer opinions manually would take weeks. **SentiScope** solves this problem by using **Artificial Intelligence and Natural Language Processing (NLP)** to:

1. **Read & understand** human reviews in a fraction of a second.
2. **Detect the underlying emotion**:
   - 🟢 **Positive (😊)**: Customer loves the product, praises features, or expresses happiness.
   - 🟡 **Neutral (😐)**: Factual statement, unbiased observation, or mixed feedback.
   - 🔴 **Negative (😔)**: Disappointment, hardware glitches, setup difficulty, or complaints.
3. **Give you clear, visual insights**: Side-by-side comparison, charts, and downloadable reports.

---

## ✨ What Can You Do With SentiScope?

| Feature | What It Does for You |
| :--- | :--- |
| ⚖️ **Dual Comparison** | Paste two different reviews side-by-side to see which customer is happier and why. |
| 🎙️ **Voice Dictation** | Don't feel like typing? Just click the microphone icon and speak your review aloud! |
| 📦 **Batch Analysis** | Paste up to 50 reviews at once and get an instant summary of positive vs. negative feelings. |
| 📊 **Visual Analytics Dashboard** | Beautiful doughnut charts and trend graphs showing overall sentiment distributions. |
| 📥 **Export to CSV / Excel** | Download your entire history of analysed reviews with a single click. |
| 🌓 **Dark & Light Mode** | Easy on the eyes day or night, with your preference saved automatically. |
| 🔐 **User Accounts & History** | Sign up, log in, manage your profile, and keep track of all your past analyses. |

---

## 🧠 How Does the AI Work? (In Plain English)

Instead of relying on just one algorithm, SentiScope uses an **Ensemble Brain**—combining **three specialized AI models** that work together like a panel of judges:

```
                          ┌───────────────────────────┐
                          │   Customer Review Text    │
                          └─────────────┬─────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
  │   Model 1: VADER  │        │ Model 2: TextBlob │        │ Model 3: ML Model │
  │ (Social & Emojis) │        │(Grammar & Tone)  │        │ (3,150 Alexa Data)│
  └────────┬─────────┘        └────────┬─────────┘        └────────┬─────────┘
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   Weighted Ensemble Vote    │
                         └──────────────┬──────────────┘
                                        ▼
                   ╔═════════════════════════════════════════╗
                   ║  Verdict: Positive / Neutral / Negative ║
                   ║  Confidence Score: 87.5% 😊             ║
                   ╚═════════════════════════════════════════╝
```

1. **🤖 VADER (Valence Aware Dictionary for sEntiment Reasoning)**  
   Specialized in modern online communication. It understands emojis (`❤️`, `🔥`, `😡`), punctuation like exclamation marks (`"Great!!!"` = very positive), and capitalization (`"DO NOT BUY"` = intense negative).
2. **📖 TextBlob**  
   Analyzes formal language grammar, measuring both **Polarity** (how positive or negative the words are) and **Subjectivity** (whether it's a personal opinion or an objective fact).
3. **🎯 Custom Machine Learning Model (TF-IDF + Logistic Regression)**  
   Trained on a real dataset of **3,150 Amazon Alexa customer reviews** to learn the exact vocabulary, phrases, and technical complaints specific to smart speakers and Alexa hardware.

**The Result:** A reliable, balanced verdict with an ensemble confidence percentage that avoids common blind spots!

---

## 💻 Quick Start Guide (Run on Your Computer in 3 Steps)

If you'd like to run the full Python Flask backend locally on your computer:

### Step 1: Clone the Repository
```bash
git clone https://github.com/AJANYA-K-A/Sentiment-Analysis-on-Amazon-Alexa-Reviews.git
cd Sentiment-Analysis-on-Amazon-Alexa-Reviews
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Start the App!
```bash
python app.py
```

Now open your browser and go to:
👉 **`http://127.0.0.1:5000`**

---

## 📁 Project Tour: What's Inside?

Here is a friendly roadmap of the files in this repository:

```
Sentiment-Analysis-on-Amazon-Alexa-Reviews/
│
├── index.html                 # 🌐 The complete web application for GitHub Pages (Runs in browser)
├── app.py                     # 🐍 The Python Flask backend application
│
├── amazon_alexa.tsv           # 📊 Dataset of 3,150 real Amazon Alexa customer reviews
├── sentiment_model.pkl        # 🧠 Trained Machine Learning classification model
├── tfidf_vectorizer.pkl       # 🔤 Vocabulary vectorizer used by the ML model
├── requirements.txt           # 📦 List of Python packages required to run locally
│
├── static/                    # 🎨 Images, icons, and styling assets
│   ├── logo.svg               # SentiScope vector logo
│   └── image_da3f79.png       # Application icon
│
├── templates/                 # 📄 HTML Templates for the Flask backend
│   ├── base.html              # Base layout (header, footer, navigation)
│   ├── index.html             # Dual review comparison tool
│   ├── batch.html             # Batch review analysis tool
│   ├── dashboard.html         # Analytics charts & trend graphs
│   ├── history.html           # Past analysis history table
│   ├── login.html             # User login page
│   ├── signup.html            # User registration page
│   └── profile.html           # Profile settings & data management
│
└── readme.md                  # 📖 This friendly documentation file!
```

---

## 📊 About the Alexa Dataset

The model is trained on the popular **Amazon Alexa Reviews Dataset**:
- **Total Reviews**: 3,150 customer reviews.
- **Product Variations**: Echo Dot, Echo Show, Echo Spot, Echo Plus, and Fire TV Stick with Alexa Voice Remote.
- **Data Points**: Customer rating (1 to 5 stars), review date, variation name, verified feedback, and full review text.

---

## 🛠️ Built With

- **Backend**: Python 3.9+, Flask, SQLAlchemy, Werkzeug
- **AI / NLP**: VADER Sentiment, TextBlob, NLTK, Scikit-learn
- **Frontend**: Modern Vanilla HTML5, CSS3, JavaScript (ES6+), Glassmorphism UI
- **Data Visualizations**: Chart.js 4.4
- **Fonts & Icons**: Google Fonts (Syne & DM Sans)

---

## 💡 How to Publish to GitHub Pages Yourself

If you fork or clone this repository, you can enable your own live link in seconds:

1. Go to your repository on GitHub.
2. Click **Settings** ⚙️ &rarr; **Pages** (in the left menu).
3. Under **Branch**, select `main` and root folder `/`.
4. Click **Save**.
5. Your project will be live at `https://<your-username>.github.io/<repo-name>/`!

---

## 👤 Author & Acknowledgments

- **Author**: [AJANYA K A](https://github.com/AJANYA-K-A)
- **Repository**: [Sentiment-Analysis-on-Amazon-Alexa-Reviews](https://github.com/AJANYA-K-A/Sentiment-Analysis-on-Amazon-Alexa-Reviews)
- **Live Demo**: [ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/](https://ajanya-k-a.github.io/Sentiment-Analysis-on-Amazon-Alexa-Reviews/)

⭐ *If you found this project helpful, please consider giving it a star on GitHub!*