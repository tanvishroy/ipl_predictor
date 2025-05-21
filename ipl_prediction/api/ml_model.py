import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Paths
matches_file = "data/matches.csv"
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Load match data
print("📂 Loading match data...")
matches_df = pd.read_csv(matches_file)

# Team normalization map (map old names to modern 10-team format)
team_name_map = {
    'Delhi Daredevils': 'DC',
    'Delhi Capitals': 'DC',
    'Rising Pune Supergiants': 'LSG',
    'Rising Pune Super Giant': 'LSG',
    'Pune Warriors': 'LSG',
    'Rising Pune Warriors': 'LSG',
    'Lucknow Super Giants': 'LSG',
    'Lucknow Supergiants': 'LSG',
    'Lucknow SuperGaints': 'LSG',
    'Gujarat Lions': 'GT',
    'Gujarat Titans': 'GT',
    'Kings XI Punjab': 'PBKS',
    'Punjab Kings': 'PBKS',
    'Chennai Super Kings': 'CSK',
    'Royal Challengers Bangalore': 'RCB',
    'Mumbai Indians': 'MI',
    'Rajasthan Royals': 'RR',
    'Kolkata Knight Riders': 'KKR',
    'Sunrisers Hyderabad': 'SRH'
}

# Final valid team list
valid_teams = ['CSK', 'RCB', 'MI', 'RR', 'KKR', 'PBKS', 'SRH', 'DC', 'LSG', 'GT']

# Normalize names
for col in ['team1', 'team2', 'winner']:
    matches_df[col] = matches_df[col].replace(team_name_map)

# Filter to only final 10 teams
matches_df = matches_df[
    matches_df['team1'].isin(valid_teams) &
    matches_df['team2'].isin(valid_teams) &
    matches_df['winner'].isin(valid_teams)
]
# Check which teams are still present after filtering
teams_in_data = set(matches_df['team1']).union(matches_df['team2']).union(matches_df['winner'])
print("🥪 Teams remaining in dataset after filtering:", sorted(teams_in_data))

# Drop rows with missing data
matches_df.dropna(subset=['team1', 'team2', 'winner'], inplace=True)

# Encode
le = LabelEncoder()
df = matches_df.copy()
df['team1_enc'] = le.fit_transform(df['team1'])
df['team2_enc'] = le.transform(df['team2'])
df['winner_enc'] = le.transform(df['winner'])

# Train/test split
X = df[['team1_enc', 'team2_enc']]
y = df['winner_enc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"✅ Model accuracy: {accuracy:.2f}")

# Save model and encoder
joblib.dump(model, os.path.join(models_dir, "match_winner_model.pkl"))
joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
print("📂 Model and encoder saved.")