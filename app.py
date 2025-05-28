from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enables Cross-Origin Resource Sharing

@app.route('/')
def home():
    return "Backend is live!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    team1 = data.get('team1')
    team2 = data.get('team2')

    if not team1 or not team2:
        return jsonify({'error': 'Both teams must be provided'}), 400

    if team1 == team2:
        return jsonify({'error': 'Teams must be different'}), 400

    # Dummy prediction logic (replace with your real model)
    winner = team1 if team1 > team2 else team2
    return jsonify({'winner': winner})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)