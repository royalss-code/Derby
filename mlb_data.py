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
        "recent_hr_rate": float(row.get("recent_hr_rate", row["player_hr_rate"])),
        "recent_barrel_rate": float(row.get("recent_barrel_rate", row["barrel_rate"])),
        "hr_vs_R": float(row.get("hr_vs_R", 0)),
        "hr_vs_L": float(row.get("hr_vs_L", 0)),
        "barrel_vs_R": float(row.get("barrel_vs_R", 0)),
        "barrel_vs_L": float(row.get("barrel_vs_L", 0)),
        "recent_hr_rate_vs_R": float(row.get("recent_hr_rate_vs_R", row.get("hr_vs_R", 0))),
        "recent_hr_rate_vs_L": float(row.get("recent_hr_rate_vs_L", row.get("hr_vs_L", 0))),
        "recent_barrel_rate_vs_R": float(row.get("recent_barrel_rate_vs_R", row.get("barrel_vs_R", 0))),
        "recent_barrel_rate_vs_L": float(row.get("recent_barrel_rate_vs_L", row.get("barrel_vs_L", 0))),
        "stand": str(row["stand"]),
        "team": str(row["team"])
    }


def get_pitcher_stats(pitcher_name):
    pitcher = pitcher_df[pitcher_df["name"].str.lower() == pitcher_name.lower()]
    if pitcher.empty:
        return None

    row = pitcher.iloc[0]
    return {
        "pitcher_hr9": float(row["pitcher_hr9"]),
        "flyball_rate": float(row["flyball_rate"]),
        "hr_allowed_vs_R": float(row.get("hr_allowed_vs_R", 0)),
        "hr_allowed_vs_L": float(row.get("hr_allowed_vs_L", 0)),
        "flyball_vs_R": float(row.get("flyball_vs_R", 0.35)),
        "flyball_vs_L": float(row.get("flyball_vs_L", 0.35)),
        "p_throws": str(row["p_throws"]).strip().upper() if "p_throws" in row else "R"
    }