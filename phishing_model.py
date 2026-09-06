import pickle as pk
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from feature_extraction import extract_features, FEATURE_ORDER

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "Random_Forest_medium.pk")
scaler_path = os.path.join(BASE_DIR, "scaler.pk")

with open(model_path, "rb") as f:
    model = pk.load(f)
with open(scaler_path, "rb") as f:
    scaler = pk.load(f)

# Binary columns were left unscaled in the notebook (nunique == 2);
# everything else was passed through StandardScaler.
BINARY_COLS = {"IsDomainIP", "HasObfuscation"}
CONTINUOUS_INDEXES = [i for i, col in enumerate(FEATURE_ORDER) if col not in BINARY_COLS]

app = Flask(__name__)
CORS(app)


def run_prediction(features: dict):
    row = np.array([[float(features[col]) for col in FEATURE_ORDER]])
    row[:, CONTINUOUS_INDEXES] = scaler.transform(row[:, CONTINUOUS_INDEXES])
    prediction = model.predict(row)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(row)[0].tolist()
    return int(prediction), probability


@app.route("/predict", methods=["POST"])
def predict_route():
    input_data = request.get_json(silent=True) or {}

    if "url" in input_data:
        # New path: raw URL in, features computed server-side.
        url = str(input_data["url"]).strip()
        if not url:
            return jsonify({"error": "Empty 'url' field"}), 400
        try:
            features = extract_features(url)
        except Exception as e:
            return jsonify({"error": f"Could not parse URL: {e}"}), 400
    else:
        # Legacy path: caller supplies all 20 raw feature values directly
        # (used by the manual feature-value tester).
        missing = [col for col in FEATURE_ORDER if col not in input_data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        features = input_data

    label, probability = run_prediction(features)

    # Dataset convention (PhiUSIIL Phishing URL Dataset): label 1 = legitimate, 0 = phishing.
    result = "Legitimate" if label == 1 else "Phishing"
    response = {
        "Prediction_Label": label,
        "Prediction": result,
        "features_used": features,
    }
    if probability is not None:
        response["Probability"] = probability
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
