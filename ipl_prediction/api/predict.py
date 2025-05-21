import joblib
import pandas as pd

# Load model and label encoder
model = joblib.load("models/match_winner_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Teams in training
trained_teams = list(encoder.classes_)

print("\n📋 Enter team codes from the following list:")
print(", ".join(trained_teams))

# Get input
team1 = input("\nEnter Team 1 code: ").strip().upper()
team2 = input("Enter Team 2 code: ").strip().upper()

# Validate
if team1 not in trained_teams or team2 not in trained_teams:
    print("\n❌ One or both teams not seen in training.")
    print("✅ Trained teams:", trained_teams)
    exit()

if team1 == team2:
    print("\n❌ Same team entered.")
    exit()
    
# Encode and predict
team1_enc = encoder.transform([team1])[0]
team2_enc = encoder.transform([team2])[0]
X = pd.DataFrame([[team1_enc, team2_enc]], columns=["team1_enc", "team2_enc"])
pred_enc = model.predict(X)[0]
pred_team = encoder.inverse_transform([pred_enc])[0]

print(f"\n🏏 Predicted Winner: {pred_team}")