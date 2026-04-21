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

    batted = df[df["launch_speed"].notna() & df["launch_angle"].notna()].copy()

    if batted.empty:
        return 0.0

    ev = pd.to_numeric(batted["launch_speed"], errors="coerce")
    la = pd.to_numeric(batted["launch_angle"], errors="coerce")

    barrel_like = (
        ((ev >= 98) & la.between(26, 30, inclusive="both")) |
        ((ev >= 99) & la.between(25, 31, inclusive="both")) |
        ((ev >= 100) & la.between(24, 33, inclusive="both")) |
        ((ev >= 101) & la.between(23, 34, inclusive="both")) |
        ((ev >= 102) & la.between(22, 35, inclusive="both")) |
        ((ev >= 103) & la.between(21, 36, inclusive="both")) |
        ((ev >= 104) & la.between(20, 37, inclusive="both")) |
        ((ev >= 105) & la.between(19, 38, inclusive="both"))
    )

    return float(barrel_like.mean())


def calc_hr_rate_per_pa(df):
    if df.empty or "events" not in df.columns:
        return 0.0

    # Plate-appearance-ending events
    pa_events = {
        "single", "double", "triple", "home_run",
        "walk", "intent_walk", "strikeout", "strikeout_double_play",
        "hit_by_pitch", "field_out", "grounded_into_double_play",
        "force_out", "field_error", "double_play", "triple_play",
        "fielders_choice", "fielders_choice_out", "sac_fly",
        "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play",
        "catcher_interf"
    }

    pa_df = df[df["events"].isin(pa_events)].copy()

    if pa_df.empty:
        return 0.0

    return float((pa_df["events"] == "home_run").mean())


def build_player_stats():
    raw_players = pd.read_csv(RAW_PLAYERS_FILE)
    rows = []

    for _, p in raw_players.iterrows():
        name = p["name"]
        mlbam_id = int(p["mlbam_id"])
        stand = str(p["stand"]).strip().upper()
        team = str(p.get("team", "")).strip()

        try:
            df = statcast_batter(START_DATE.isoformat(), END_DATE.isoformat(), mlbam_id)

            if df.empty:
                rows.append({
                    "name": name,
                    "player_hr_rate": 0.0,
                    "barrel_rate": 0.0,
                    "stand": stand,
                    "team": team
                })
                continue

            hr_rate = calc_hr_rate_per_pa(df)
            barrel_rate = calc_barrel_like_rate(df)

            rows.append({
                "name": name,
                "player_hr_rate": round(hr_rate, 4),
                "barrel_rate": round(barrel_rate, 4),
                "stand": stand,
                "team": team
            })

        except Exception as e:
            print(f"FAILED hitter: {name} ({mlbam_id}) -> {e}")
            rows.append({
                "name": name,
                "player_hr_rate": 0.0,
                "barrel_rate": 0.0,
                "stand": stand,
                "team": team
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

            # Better denominator for pitchers too: PA-ending events
            pa_events = {
                "single", "double", "triple", "home_run",
                "walk", "intent_walk", "strikeout", "strikeout_double_play",
                "hit_by_pitch", "field_out", "grounded_into_double_play",
                "force_out", "field_error", "double_play", "triple_play",
                "fielders_choice", "fielders_choice_out", "sac_fly",
                "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play",
                "catcher_interf"
            }

            pa_df = df[df["events"].isin(pa_events)].copy()
            if pa_df.empty:
                hr_allowed_rate = 0.0
            else:
                hr_allowed_rate = float((pa_df["events"] == "home_run").mean())

            pitcher_hr9 = round(hr_allowed_rate * 9, 4)

            batted = df[df["bb_type"].notna()].copy() if "bb_type" in df.columns else pd.DataFrame()
            if not batted.empty:
                flyball_rate = round((batted["bb_type"] == "fly_ball").mean(), 4)
            else:
                flyball_rate = 0.35

            rows.append({
                "name": name,
                "pitcher_hr9": pitcher_hr9,
                "flyball_rate": flyball_rate,
                "p_throws": p_throws
            })

        except Exception as e:
            print(f"FAILED pitcher: {name} ({mlbam_id}) -> {e}")
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