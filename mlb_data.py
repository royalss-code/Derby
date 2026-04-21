import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

player_df = pd.read_csv(os.path.join(BASE_DIR, "player_stats.csv"))
pitcher_df = pd.read_csv(os.path.join(BASE_DIR, "pitcher_stats.csv"))

def get_player_stats(player_name):
    player = player_df[player_df["name"].str.lower() == player_name.lower()]
    if player.empty:
        return None

    row = player.iloc[0]
    return {
        "player_hr_rate": float(row["player_hr_rate"]),
        "barrel_rate": float(row["barrel_rate"]),
        "stand": str(row["stand"]).strip().upper() if "stand" in row else "R"
    }

def get_pitcher_stats(pitcher_name):
    pitcher = pitcher_df[pitcher_df["name"].str.lower() == pitcher_name.lower()]
    if pitcher.empty:
        return None

    row = pitcher.iloc[0]
    return {
        "pitcher_hr9": float(row["pitcher_hr9"]),
        "flyball_rate": float(row["flyball_rate"]),
        "p_throws": str(row["p_throws"]).strip().upper() if "p_throws" in row else "R"
    }