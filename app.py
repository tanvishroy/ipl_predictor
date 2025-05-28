from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Absolute paths for model and encoder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "match_winner_model.pkl")
encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

try:
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    print("Models loaded successfully.")
except Exception as e:
    print("Error loading models:", e)
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        team1 = data.get('team1').upper()
        team2 = data.get('team2').upper()

        valid_teams = ['CSK', 'RCB', 'MI', 'RR', 'KKR', 'PBKS', 'SRH', 'DC', 'LSG', 'GT']
        if team1 not in valid_teams or team2 not in valid_teams:
            return jsonify({'error': 'Invalid team code. Please use valid team codes.'}), 400

        team1_enc = encoder.transform([team1])[0]
        team2_enc = encoder.transform([team2])[0]

        X = pd.DataFrame([[team1_enc, team2_enc]], columns=["team1_enc", "team2_enc"])
        pred_enc = model.predict(X)[0]
        predicted_winner = encoder.inverse_transform([pred_enc])[0]

        return jsonify({'winner': predicted_winner})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)