import os
import re
import csv
import pickle
import io
from datetime import datetime
from functools import wraps
from typing import Optional

import nltk
from nltk.corpus import stopwords
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, Response
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
stop_words = set(stopwords.words("english"))

# ── Load saved ML model & vectorizer ─────────────────────────────────────────
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

vader = SentimentIntensityAnalyzer()

# ── App & DB config ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///sentiscope.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ── Models ────────────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(64),  unique=True, nullable=False, index=True)
    email         = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    bio           = db.Column(db.String(300), nullable=True, default="")
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    analyses      = db.relationship("AnalysisHistory", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, plaintext: str) -> None:
        self.password_hash = generate_password_hash(plaintext)

    def check_password(self, plaintext: str) -> bool:
        return check_password_hash(self.password_hash, plaintext)

    def __repr__(self) -> str:
        return f"<User {self.username}>"


class AnalysisHistory(db.Model):
    __tablename__ = "analysis_history"
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    text        = db.Column(db.Text, nullable=False)
    label       = db.Column(db.String(20), nullable=False)
    confidence  = db.Column(db.Float, nullable=False)
    vader_score = db.Column(db.Float, nullable=False)
    tb_score    = db.Column(db.Float, nullable=False)
    mode        = db.Column(db.String(20), default="single")
    created_at  = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id":          self.id,
            "text":        self.text,
            "label":       self.label,
            "confidence":  self.confidence,
            "vader_score": self.vader_score,
            "tb_score":    self.tb_score,
            "mode":        self.mode,
            "created_at":  self.created_at.strftime("%Y-%m-%d %H:%M"),
        }


with app.app_context():
    db.create_all()


# ── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access the analyser.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def current_user() -> Optional[User]:
    if "user_id" in session:
        return db.session.get(User, session["user_id"])
    return None


# ── NLP helpers ───────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = " ".join(w for w in text.split() if w not in stop_words)
    return text


def predict_sentiment(text: str, mode: str = "single") -> Optional[dict]:
    if not text or not text.strip():
        return None

    vader_scores    = vader.polarity_scores(text)
    vader_compound  = vader_scores["compound"]

    blob            = TextBlob(text).sentiment
    tb_polarity     = blob.polarity
    tb_subjectivity = blob.subjectivity

    vader_vote = 1 if vader_compound >= 0.05 else (0 if vader_compound <= -0.05 else None)
    tb_vote    = 1 if tb_polarity > 0.05    else (0 if tb_polarity < -0.05    else None)

    cleaned   = preprocess_text(text)
    tfidf_vec = vectorizer.transform([cleaned])
    ml_pred   = model.predict(tfidf_vec)[0]
    ml_proba  = model.predict_proba(tfidf_vec)[0]
    ml_vote   = int(ml_pred)

    vader_w        = vader_vote if vader_vote is not None else 0.5
    tb_w           = tb_vote    if tb_vote    is not None else 0.5
    positive_score = (vader_w * 2) + tb_w + ml_vote

    if positive_score >= 2.2:
        final_label = "Positive"
    elif positive_score <= 0.8:
        final_label = "Negative"
    else:
        final_label = "Neutral"

    vader_conf = abs(vader_compound)
    ml_conf    = max(ml_proba)
    tb_conf    = min(abs(tb_polarity), 1.0)
    raw_conf   = (vader_conf * 2 + ml_conf + tb_conf) / 4
    confidence = max(round(raw_conf * 100, 1), 50.1)

    LABEL_MAP = {
        "Positive": {"emoji": "😊", "css": "positive"},
        "Negative": {"emoji": "😔", "css": "negative"},
        "Neutral":  {"emoji": "😐", "css": "neutral"},
    }

    return {
        "label":           final_label,
        "emoji":           LABEL_MAP[final_label]["emoji"],
        "css":             LABEL_MAP[final_label]["css"],
        "confidence":      confidence,
        "vader_compound":  round(vader_compound, 4),
        "vader_pos":       round(vader_scores["pos"] * 100, 1),
        "vader_neg":       round(vader_scores["neg"] * 100, 1),
        "vader_neu":       round(vader_scores["neu"] * 100, 1),
        "tb_polarity":     round(tb_polarity, 4),
        "tb_subjectivity": round(tb_subjectivity, 4),
        "ml_confidence":   round(max(ml_proba) * 100, 1),
        "mode":            mode,
        "text_preview":    text[:120] + ("…" if len(text) > 120 else ""),
    }


def save_analysis(result: dict, mode: str = "single") -> None:
    if "user_id" not in session or result is None:
        return
    entry = AnalysisHistory(
        user_id     = session["user_id"],
        text        = result.get("text_preview", ""),
        label       = result["label"],
        confidence  = result["confidence"],
        vader_score = result["vader_compound"],
        tb_score    = result["tb_polarity"],
        mode        = mode,
    )
    db.session.add(entry)
    db.session.commit()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return redirect(url_for("index") if "user_id" in session else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = User.query.filter(User.username.ilike(username)).first()
        if user and user.check_password(password):
            session["user_id"]  = user.id
            session["username"] = user.username
            flash(f"Welcome back, {user.username}!", "success")
            return redirect(url_for("index"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email",    "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm",  "").strip()

        error = None
        if not username or not email or not password:
            error = "All fields are required."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif User.query.filter(User.username.ilike(username)).first():
            error = "Username already taken."
        elif User.query.filter(User.email.ilike(email)).first():
            error = "An account with that email already exists."

        if error:
            flash(error, "error")
        else:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            session["user_id"]  = new_user.id
            session["username"] = new_user.username
            flash(f"Account created! Welcome, {new_user.username}.", "success")
            return redirect(url_for("index"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.", "info")
    return redirect(url_for("login"))


# ── Analyser ──────────────────────────────────────────────────────────────────
@app.route("/analyser", methods=["GET", "POST"])
@login_required
def index():
    result1 = result2 = None
    if request.method == "POST":
        text1 = request.form.get("user_text1", "").strip()
        text2 = request.form.get("user_text2", "").strip()
        result1 = predict_sentiment(text1, mode="compare")
        result2 = predict_sentiment(text2, mode="compare")
        if result1:
            save_analysis(result1, "compare")
        if result2:
            save_analysis(result2, "compare")
    return render_template("index.html", result1=result1, result2=result2)


# ── Batch Analyser ────────────────────────────────────────────────────────────
@app.route("/batch", methods=["GET", "POST"])
@login_required
def batch():
    results   = []
    raw_input = ""
    if request.method == "POST":
        raw_input = request.form.get("batch_text", "").strip()
        lines = [l.strip() for l in raw_input.split("\n") if l.strip()]
        for line in lines[:50]:
            r = predict_sentiment(line, mode="batch")
            if r:
                r["original_text"] = line
                results.append(r)
                save_analysis(r, "batch")
    return render_template("batch.html", results=results, raw_input=raw_input)


# ── Single Analyser API ───────────────────────────────────────────────────────
@app.route("/api/analyse", methods=["POST"])
@login_required
def api_analyse():
    data   = request.get_json(force=True)
    text   = data.get("text", "").strip()
    result = predict_sentiment(text, mode="single")
    if result:
        save_analysis(result, "single")
        return jsonify({"ok": True, "result": result})
    return jsonify({"ok": False, "error": "Empty text"}), 400


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    uid         = session["user_id"]
    all_records = (
        AnalysisHistory.query
        .filter_by(user_id=uid)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(200)
        .all()
    )

    total = len(all_records)
    pos   = sum(1 for r in all_records if r.label == "Positive")
    neg   = sum(1 for r in all_records if r.label == "Negative")
    neu   = sum(1 for r in all_records if r.label == "Neutral")

    chart_data = [r.to_dict() for r in reversed(all_records[:30])]
    recent     = [r.to_dict() for r in all_records[:10]]
    stats      = {"total": total, "positive": pos, "negative": neg, "neutral": neu}
    avg_conf   = round(sum(r.confidence for r in all_records) / total, 1) if total else 0

    return render_template(
        "dashboard.html",
        stats=stats,
        avg_conf=avg_conf,
        chart_data=chart_data,
        recent=recent,
    )


# ── History ───────────────────────────────────────────────────────────────────
@app.route("/history")
@login_required
def history():
    uid   = session["user_id"]
    page  = request.args.get("page", 1, type=int)
    label = request.args.get("label", "")
    mode  = request.args.get("mode", "")

    q = AnalysisHistory.query.filter_by(user_id=uid)
    if label:
        q = q.filter(AnalysisHistory.label == label)
    if mode:
        q = q.filter(AnalysisHistory.mode == mode)

    pagination = q.order_by(AnalysisHistory.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    return render_template("history.html", pagination=pagination, label=label, mode=mode)


# ── Export CSV ────────────────────────────────────────────────────────────────
@app.route("/export/csv")
@login_required
def export_csv():
    uid     = session["user_id"]
    records = (
        AnalysisHistory.query
        .filter_by(user_id=uid)
        .order_by(AnalysisHistory.created_at.desc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Text", "Label", "Confidence (%)",
                     "VADER Score", "TextBlob Score", "Mode", "Created At"])
    for r in records:
        writer.writerow([
            r.id, r.text, r.label, r.confidence,
            r.vader_score, r.tb_score, r.mode,
            r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        ])

    output.seek(0)
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sentiscope_history.csv"},
    )


# ── Profile ───────────────────────────────────────────────────────────────────
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user()
    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_bio":
            user.bio = request.form.get("bio", "").strip()[:300]
            db.session.commit()
            flash("Bio updated!", "success")

        elif action == "change_password":
            current_pw = request.form.get("current_password", "")
            new_pw     = request.form.get("new_password", "")
            confirm_pw = request.form.get("confirm_password", "")
            if not user.check_password(current_pw):
                flash("Current password is incorrect.", "error")
            elif len(new_pw) < 6:
                flash("New password must be at least 6 characters.", "error")
            elif new_pw != confirm_pw:
                flash("New passwords do not match.", "error")
            else:
                user.set_password(new_pw)
                db.session.commit()
                flash("Password changed successfully!", "success")

        elif action == "delete_history":
            AnalysisHistory.query.filter_by(user_id=user.id).delete()
            db.session.commit()
            flash("Analysis history cleared.", "info")

        return redirect(url_for("profile"))

    total = AnalysisHistory.query.filter_by(user_id=user.id).count()
    return render_template("profile.html", user=user, total=total)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)