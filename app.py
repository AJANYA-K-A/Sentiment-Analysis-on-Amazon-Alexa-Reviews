from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment1 = sentiment2 = ""
    if request.method == "POST":
        text1 = request.form["user_text1"]
        text2 = request.form["user_text2"]

        if text1:
            text1_tfidf = vectorizer.transform([text1])
            prediction1 = model.predict(text1_tfidf)[0]
            sentiment1 = "😊 Positive" if prediction1 == 1 else "😔 Negative"

        if text2:
            text2_tfidf = vectorizer.transform([text2])
            prediction2 = model.predict(text2_tfidf)[0]
            sentiment2 = "😊 Positive" if prediction2 == 1 else "😔 Negative"

    return render_template("index.html", sentiment1=sentiment1, sentiment2=sentiment2)

if __name__ == "__main__":
    app.run(debug=True)
