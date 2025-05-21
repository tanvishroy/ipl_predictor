from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the saved model and label encoder
model = joblib.load("models/match_winner_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Home route - serves the frontend HTML
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
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
        predicted_emoji = encoder.inverse_transform([pred_enc])[0]

        return jsonify({'winner': predicted_emoji})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)