from pybaseball import batting_stats, pitching_stats
import pandas as pd

def get_player_stats(player_name):
    df = batting_stats(2025, qual=1)
    player = df[df["Name"].str.lower() == player_name.lower()]

    if player.empty:
        return None

    row = player.iloc[0]

    # fallback-friendly estimates
    hr_rate = row["HR"] / row["PA"] if row["PA"] > 0 else 0
    barrel_rate = row["Barrel%"] / 100 if "Barrel%" in row and pd.notna(row["Barrel%"]) else 0.08

    return {
        "player_hr_rate": hr_rate,
        "barrel_rate": barrel_rate
    }

def get_pitcher_stats(pitcher_name):
    df = pitching_stats(2025, qual=1)
    pitcher = df[df["Name"].str.lower() == pitcher_name.lower()]

    if pitcher.empty:
        return None

    row = pitcher.iloc[0]

    hr9 = row["HR/9"] if "HR/9" in row and pd.notna(row["HR/9"]) else 1.1
    flyball_rate = row["FB%"] / 100 if "FB%" in row and pd.notna(row["FB%"]) else 0.35

    return {
        "pitcher_hr9": hr9,
        "flyball_rate": flyball_rate
    }
