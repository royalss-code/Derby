from flask import Flask, render_template, request
import pickle
import os
import pandas as pd
from mlb_data import get_player_stats, get_pitcher_stats
from today_games import get_today_games
from live_rosters import get_team_roster

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PLAYER_CSV = os.path.join(BASE_DIR, "player_stats.csv")
PITCHER_CSV = os.path.join(BASE_DIR, "pitcher_stats.csv")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

PARK_FACTOR_MAP = {
    "Colorado Rockies": 1.25,
    "New York Yankees": 1.12,
    "Los Angeles Dodgers": 0.98,
    "San Francisco Giants": 0.90,
    "Boston Red Sox": 1.05,
    "Cincinnati Reds": 1.10,
    "Philadelphia Phillies": 1.08,
    "Atlanta Braves": 1.02,
    "Chicago Cubs": 1.03,
    "Texas Rangers": 1.04,
    "Houston Astros": 0.97,
    "Seattle Mariners": 0.92,
    "San Diego Padres": 0.95,
    "Miami Marlins": 0.91,
    "Detroit Tigers": 0.94,
    "Kansas City Royals": 0.96,
    "Oakland Athletics": 0.89,
    "Tampa Bay Rays": 0.93,
    "New York Mets": 0.97,
    "St. Louis Cardinals": 0.99,
    "Milwaukee Brewers": 1.01,
    "Baltimore Orioles": 1.02,
    "Toronto Blue Jays": 1.00,
    "Cleveland Guardians": 0.98,
    "Minnesota Twins": 1.00,
    "Pittsburgh Pirates": 0.95,
    "Washington Nationals": 0.99,
    "Arizona Diamondbacks": 1.01,
    "Los Angeles Angels": 0.96,
    "Chicago White Sox": 1.00,
}

def load_names():
    player_df = pd.read_csv(PLAYER_CSV)
    pitcher_df = pd.read_csv(PITCHER_CSV)

    players = sorted(player_df["name"].dropna().unique().tolist())
    pitchers = sorted(pitcher_df["name"].dropna().unique().tolist())

    return players, pitchers

def american_odds_from_probability(prob):
    if prob <= 0 or prob >= 1:
        return "N/A"

    if prob < 0.5:
        odds = round(((1 - prob) / prob) * 100)
        return f"+{odds}"
    odds = round((prob / (1 - prob)) * 100)
    return f"-{odds}"

def build_features(player_stats, pitcher_stats, park_factor=1.0):
    player_hr_rate = float(player_stats["player_hr_rate"])
    barrel_rate = float(player_stats["barrel_rate"])

    # Current live app approximations
    recent_hr_rate = player_hr_rate
    recent_barrel_rate = barrel_rate
    player_hr_rate_split = player_hr_rate
    barrel_rate_split = barrel_rate
    recent_hr_rate_split = recent_hr_rate
    recent_barrel_rate_split = recent_barrel_rate

    power_index = (
        player_hr_rate * 0.38 +
        barrel_rate * 0.16 +
        recent_hr_rate * 0.12 +
        recent_barrel_rate * 0.14 +
        player_hr_rate_split * 0.10 +
        barrel_rate_split * 0.05 +
        recent_hr_rate_split * 0.03 +
        recent_barrel_rate_split * 0.02
    )

    pitcher_hr9 = float(pitcher_stats["pitcher_hr9"])
    flyball_rate = float(pitcher_stats["flyball_rate"])
    power_vs_pitcher = power_index * pitcher_hr9
    bad_pitcher = int(pitcher_hr9 > 1.2)
    weak_hitter = int(player_hr_rate < 0.12)
    very_weak_hitter = int(player_hr_rate < 0.08)
    strong_hitter = int(player_hr_rate > 0.18)
    split_confidence = 1

    stand = player_stats.get("stand", "R")
    p_throws = pitcher_stats.get("p_throws", "R")

    matchup = int(
        (stand == "L" and p_throws == "R") or
        (stand == "R" and p_throws == "L")
    )

    return [[
        player_hr_rate,
        barrel_rate,
        recent_hr_rate,
        recent_barrel_rate,
        player_hr_rate_split,
        barrel_rate_split,
        recent_hr_rate_split,
        recent_barrel_rate_split,
        power_index,
        pitcher_hr9,
        flyball_rate,
        power_vs_pitcher,
        bad_pitcher,
        weak_hitter,
        very_weak_hitter,
        strong_hitter,
        split_confidence,
        matchup,
        float(park_factor)
    ]]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    rankings = []

    try:
        games = get_today_games()
    except Exception as e:
        games = []
        error = f"Could not load today's games: {str(e)}"

    players, pitchers = load_names()

    selected_player = ""
    selected_pitcher = ""
    selected_matchup = ""
    selected_park_factor = ""

    # Today-only rankings
    try:
        for game in games:
            home_park_factor = PARK_FACTOR_MAP.get(game["home_team"], 1.00)

            # Away hitters vs home probable pitcher
            if game["home_pitcher"] != "TBD":
                away_hitters = get_team_roster(game["away_team"])
                for hitter in away_hitters:
                    try:
                        player_stats = get_player_stats(hitter)
                        pitcher_stats = get_pitcher_stats(game["home_pitcher"])

                        if not player_stats or not pitcher_stats:
                            continue

                        features = build_features(
    			     player_stats=player_stats,
    			     pitcher_stats=pitcher_stats,
    			     park_factor=home_park_factor
			  )

                        prob = model.predict_proba(features)[0][1]

                        rankings.append({
                            "player": hitter,
                            "team": game["away_team"],
                            "game": game["label"],
                            "opposing_pitcher": game["home_pitcher"],
                            "probability": prob,
                            "probability_pct": f"{prob:.2%}",
                            "implied_odds": american_odds_from_probability(prob)
                        })
                    except Exception:
                        continue

            # Home hitters vs away probable pitcher
            if game["away_pitcher"] != "TBD":
                home_hitters = get_team_roster(game["home_team"])
                for hitter in home_hitters:
                    try:
                        player_stats = get_player_stats(hitter)
                        pitcher_stats = get_pitcher_stats(game["away_pitcher"])

                        if not player_stats or not pitcher_stats:
                            continue

                        features = build_features(
                            player_stats=player_stats,
                            pitcher_stats=pitcher_stats,
                            matchup=1.0,
                            park_factor=home_park_factor
                        )

                        prob = model.predict_proba(features)[0][1]

                        rankings.append({
                            "player": hitter,
                            "team": game["home_team"],
                            "game": game["label"],
                            "opposing_pitcher": game["away_pitcher"],
                            "probability": prob,
                            "probability_pct": f"{prob:.2%}",
                            "implied_odds": american_odds_from_probability(prob)
                        })
                    except Exception:
                        continue

        rankings = sorted(rankings, key=lambda x: x["probability"], reverse=True)[:25]

    except Exception as e:
        if not error:
            error = f"Could not build rankings: {str(e)}"

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
                features = build_features(
                    player_stats=player_stats,
                    pitcher_stats=pitcher_stats,
                    matchup=matchup,
                    park_factor=park_factor
                )

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
        rankings=rankings,
        selected_player=selected_player,
        selected_pitcher=selected_pitcher,
        selected_matchup=selected_matchup,
        selected_park_factor=selected_park_factor
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)