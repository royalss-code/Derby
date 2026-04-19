from flask import Flask, render_template, request
import pickle
import os
import pandas as pd
from mlb_data import get_player_stats, get_pitcher_stats
from today_games import get_today_games

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PLAYER_CSV = os.path.join(BASE_DIR, "player_stats.csv")
PITCHER_CSV = os.path.join(BASE_DIR, "pitcher_stats.csv")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def load_names():
    player_df = pd.read_csv(PLAYER_CSV)
    pitcher_df = pd.read_csv(PITCHER_CSV)

    players = sorted(player_df["name"].dropna().unique().tolist())
    pitchers = sorted(pitcher_df["name"].dropna().unique().tolist())

    return players, pitchers

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    try:
        games = get_today_games()
    except Exception:
        games = []

    players, pitchers = load_names()

    selected_player = ""
    selected_pitcher = ""
    selected_matchup = ""
    selected_park_factor = ""

    if request.method == "POST":
        try:
            selected_player = request.form["player_name"]
            selected_pitcher = request.form["pitcher_name"]
            selected_matchup = request.form["matchup"]
            selected_park_factor = request.form["park_factor"]

            matchup = float(selected_matchup)
            park_factor = float(selected_park_factor)

            player_stats = get_player_stats(selected_player)
            pitcher_stats = get_pitcher_stats(selected_pitcher)

            if not player_stats:
                error = f"Player '{selected_player}' not found."
            elif not pitcher_stats:
                error = f"Pitcher '{selected_pitcher}' not found."
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

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        games=games,
        players=players,
        pitchers=pitchers,
        selected_player=selected_player,
        selected_pitcher=selected_pitcher,
        selected_matchup=selected_matchup,
        selected_park_factor=selected_park_factor
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
