from pybaseball import statcast_batter, statcast_pitcher
import pandas as pd
from datetime import date, timedelta

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=30)

RAW_PLAYERS_FILE = "raw_players.csv"
RAW_PITCHERS_FILE = "raw_pitchers.csv"
PLAYER_OUTPUT_FILE = "player_stats.csv"
PITCHER_OUTPUT_FILE = "pitcher_stats.csv"


def calc_barrel_like_rate(df):
    if df.empty:
        return 0.0

    launch_speed = pd.to_numeric(df.get("launch_speed"), errors="coerce")
    launch_angle = pd.to_numeric(df.get("launch_angle"), errors="coerce")

    barrel_like = (
        (launch_speed >= 98) &
        (launch_angle.between(26, 30, inclusive="both"))
    ).fillna(False)

    return float(barrel_like.mean())


def build_player_stats():
    raw_players = pd.read_csv(RAW_PLAYERS_FILE)
    rows = []

    for _, p in raw_players.iterrows():
        name = p["name"]
        mlbam_id = int(p["mlbam_id"])
        stand = str(p["stand"]).strip().upper()

        try:
            df = statcast_batter(START_DATE.isoformat(), END_DATE.isoformat(), mlbam_id)

            if df.empty:
                rows.append({
                    "name": name,
                    "player_hr_rate": 0.0,
                    "barrel_rate": 0.0,
                    "stand": stand
                })
                continue

            hr_rate = float((df["events"] == "home_run").mean())
            barrel_rate = calc_barrel_like_rate(df)

            rows.append({
                "name": name,
                "player_hr_rate": round(hr_rate, 4),
                "barrel_rate": round(barrel_rate, 4),
                "stand": stand
            })

        except Exception:
            rows.append({
                "name": name,
                "player_hr_rate": 0.0,
                "barrel_rate": 0.0,
                "stand": stand
            })

    return pd.DataFrame(rows)


def build_pitcher_stats():
    raw_pitchers = pd.read_csv(RAW_PITCHERS_FILE)
    rows = []

    for _, p in raw_pitchers.iterrows():
        name = p["name"]
        mlbam_id = int(p["mlbam_id"])
        p_throws = str(p["p_throws"]).strip().upper()

        try:
            df = statcast_pitcher(START_DATE.isoformat(), END_DATE.isoformat(), mlbam_id)

            if df.empty:
                rows.append({
                    "name": name,
                    "pitcher_hr9": 1.0,
                    "flyball_rate": 0.35,
                    "p_throws": p_throws
                })
                continue

            hr_allowed_rate = float((df["events"] == "home_run").mean())
            pitcher_hr9 = round(hr_allowed_rate * 9, 4)

            bb_type = df.get("bb_type")
            if bb_type is not None:
                flyball_rate = round((bb_type == "fly_ball").mean(), 4)
            else:
                flyball_rate = 0.35

            rows.append({
                "name": name,
                "pitcher_hr9": pitcher_hr9,
                "flyball_rate": flyball_rate,
                "p_throws": p_throws
            })

        except Exception:
            rows.append({
                "name": name,
                "pitcher_hr9": 1.0,
                "flyball_rate": 0.35,
                "p_throws": p_throws
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    player_df = build_player_stats()
    pitcher_df = build_pitcher_stats()

    player_df.to_csv(PLAYER_OUTPUT_FILE, index=False)
    pitcher_df.to_csv(PITCHER_OUTPUT_FILE, index=False)

    print("Updated player_stats.csv and pitcher_stats.csv")
    print(player_df.head())
    print(pitcher_df.head())