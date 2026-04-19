from flask import Flask, render_template, request
import pickle
import os
from mlb_data import get_player_stats, get_pitcher_stats

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            player_name = request.form["player_name"]
            pitcher_name = request.form["pitcher_name"]
            matchup = float(request.form["matchup"])
            park_factor = float(request.form["park_factor"])

            player_stats = get_player_stats(player_name)
            pitcher_stats = get_pitcher_stats(pitcher_name)

            if not player_stats:
                error = f"Player '{player_name}' not found."
            elif not pitcher_stats:
                error = f"Pitcher '{pitcher_name}' not found."
            else:
                features = [[
                    player_stats["player_hr_rate"],
                    player_stats["barrel_rate"],
                    pitcher_stats["pitcher_hr9"],
                    pitcher_stats["flyball_rate"],
                    matchup,
                    park_factor
                ]]

                prob = model.predict_proba(features)[0][1]
                prediction = f"{prob:.2%}"

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
