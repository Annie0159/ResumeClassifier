from flask import Flask, request, jsonify
import joblib
from utils import clean_text
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

model = joblib.load("model/resume_classifier.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("resume")
    if not text:
        return jsonify({"error": "No resume text provided"}), 400

    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return jsonify({"category": prediction})


if __name__ == "__main__":
    app.run(debug=True)
