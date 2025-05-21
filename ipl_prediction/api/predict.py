import joblib
import pandas as pd
import sys

# Load model and label encoder
model = joblib.load("models/match_winner_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Teams seen in training
trained_teams = list(encoder.classes_)

print("\n📋 Valid team codes:")
print(", ".join(trained_teams))

# Input and cleanup
team1 = input("\nEnter Team 1 code: ").strip().upper()
team2 = input("Enter Team 2 code: ").strip().upper()

# Validations
if not team1 or not team2:
    print("❌ Both team codes must be entered.")
    sys.exit(1)

if team1 == team2:
    print("❌ Please enter two different teams.")
    sys.exit(1)

if team1 not in trained_teams or team2 not in trained_teams:
    print("❌ One or both teams are not valid or were not seen during training.")
    print("✅ Valid teams are:", ", ".join(trained_teams))
    sys.exit(1)

# Encode and predict
try:
    team1_enc = encoder.transform([team1])[0]
    team2_enc = encoder.transform([team2])[0]
except ValueError as ve:
    print(f"⚠️ Encoding Error: {ve}")
    sys.exit(1)

X = pd.DataFrame([[team1_enc, team2_enc]], columns=["team1_enc", "team2_enc"])

# Predict
try:
    pred_enc = model.predict(X)[0]
    pred_team = encoder.inverse_transform([pred_enc])[0]
    print(f"\n🏏 Predicted Winner: {pred_team}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    sys.exit(1)