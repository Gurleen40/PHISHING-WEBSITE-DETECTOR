import pickle as pk
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "Random_Forest.pk")
scaler_path = os.path.join(BASE_DIR, "scaler.pk")

with open(model_path, "rb") as file:
    model = pk.load(file)

with open(scaler_path, "rb") as file:
    scaler = pk.load(file)

# Exact column order the model was trained on (from the notebook's final dataset)
FEATURE_ORDER = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL",
    "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
    "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL"
]

# Binary columns were left unscaled in the notebook (nunique == 2); everything
# else was passed through StandardScaler. Indexes below match FEATURE_ORDER.
BINARY_COLS = {"IsDomainIP", "HasObfuscation"}
CONTINUOUS_INDEXES = [i for i, col in enumerate(FEATURE_ORDER) if col not in BINARY_COLS]


def predict(features: dict):
    row = np.array([[float(features[col]) for col in FEATURE_ORDER]])
    row[:, CONTINUOUS_INDEXES] = scaler.transform(row[:, CONTINUOUS_INDEXES])
    prediction = model.predict(row)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(row)[0].tolist()
    return int(prediction), probability


app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict_route():
    input_data = request.get_json()

    missing = [col for col in FEATURE_ORDER if col not in input_data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    label, probability = predict(input_data)

    # Dataset convention (PhiUSIIL Phishing URL Dataset): label 1 = legitimate, 0 = phishing.
    # Flip this if your training data used the opposite convention.
    result = "Legitimate" if label == 1 else "Phishing"

    response = {"Prediction_Label": label, "Prediction": result}
    if probability is not None:
        response["Probability"] = probability

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)