from pybaseball import statcast_batter, statcast_pitcher
import pandas as pd
from datetime import date, timedelta
import requests
import csv
import os

END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=30)

RAW_PLAYERS_FILE = "raw_players.csv"
RAW_PITCHERS_FILE = "raw_pitchers.csv"
PLAYER_OUTPUT_FILE = "player_stats.csv"
PITCHER_OUTPUT_FILE = "pitcher_stats.csv"

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people/{person_id}"


def clean_pitcher_duplicates():
    if not os.path.exists(RAW_PITCHERS_FILE):
        print(f"{RAW_PITCHERS_FILE} not found, skipping duplicate cleanup.")
        return

    df = pd.read_csv(RAW_PITCHERS_FILE)

    if "mlbam_id" not in df.columns:
        print(f"'mlbam_id' column not found in {RAW_PITCHERS_FILE}, skipping duplicate cleanup.")
        return

    before = len(df)
    dupes = df[df.duplicated(subset=["mlbam_id"], keep="first")].copy()
    df = df.drop_duplicates(subset=["mlbam_id"], keep="first")
    after = len(df)

    df.to_csv(RAW_PITCHERS_FILE, index=False)

    print(f"Removed {before - after} duplicate pitchers from {RAW_PITCHERS_FILE}.")
    if not dupes.empty:
        print("Removed duplicates:")
        for _, row in dupes.iterrows():
            print(f'{row["name"]},{row["mlbam_id"]},{row["p_throws"]}')


def fetch_schedule_for_date(game_date: str):
    params = {
        "sportId": 1,
        "date": game_date,
        "hydrate": "probablePitcher"
    }

    response = requests.get(SCHEDULE_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def fetch_pitcher_handedness(person_id: int):
    url = PEOPLE_URL.format(person_id=person_id)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()

    people = data.get("people", [])
    if not people:
        return None

    person = people[0]
    pitch_hand = person.get("pitchHand", {})
    code = pitch_hand.get("code")

    if code in ("R", "L"):
        return code

    return ""


def load_existing_pitchers(filepath):
    existing_rows = []
    existing_ids = set()

    if not os.path.exists(filepath):
        return existing_rows, existing_ids

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_rows.append(row)
            existing_ids.add(str(row["mlbam_id"]).strip())

    return existing_rows, existing_ids


def update_pitcher_list():
    _, existing_ids = load_existing_pitchers(RAW_PITCHERS_FILE)
    new_rows = []

    dates_to_check = [
        date.today(),
        date.today() + timedelta(days=1)
    ]

    for d in dates_to_check:
        game_date = d.strftime("%Y-%m-%d")

        try:
            data = fetch_schedule_for_date(game_date)
        except Exception as e:
            print(f"FAILED schedule fetch for {game_date}: {e}")
            continue

        for day in data.get("dates", []):
            for game in day.get("games", []):
                teams = game.get("teams", {})

                for side in ("away", "home"):
                    team_info = teams.get(side, {})
                    probable = team_info.get("probablePitcher")

                    if not probable:
                        continue

                    pitcher_id = probable.get("id")
                    pitcher_name = probable.get("fullName")

                    if not pitcher_id or not pitcher_name:
                        continue

                    pitcher_id_str = str(pitcher_id)

                    if pitcher_id_str in existing_ids:
                        continue

                    try:
                        throws = fetch_pitcher_handedness(pitcher_id)
                    except Exception as e:
                        print(f"FAILED handedness lookup: {pitcher_name} ({pitcher_id}) -> {e}")
                        throws = ""

                    row = {
                        "name": pitcher_name,
                        "mlbam_id": pitcher_id_str,
                        "p_throws": throws
                    }

                    new_rows.append(row)
                    existing_ids.add(pitcher_id_str)

    if new_rows:
        file_exists = os.path.exists(RAW_PITCHERS_FILE)

        with open(RAW_PITCHERS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "mlbam_id", "p_throws"])

            if not file_exists:
                writer.writeheader()

            writer.writerows(new_rows)

    print(f"Added {len(new_rows)} new probable pitchers.")
    if new_rows:
        for row in new_rows:
            print(f'{row["name"]},{row["mlbam_id"]},{row["p_throws"]}')


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

    pa_events = {
        "single", "double", "triple", "home_run", "walk", "intent_walk",
        "strikeout", "strikeout_double_play", "hit_by_pitch", "field_out",
        "grounded_into_double_play", "force_out", "field_error", "double_play",
        "triple_play", "fielders_choice", "fielders_choice_out", "sac_fly",
        "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"
    }

    pa_df = df[df["events"].isin(pa_events)]
    if pa_df.empty:
        return 0.0

    return float((pa_df["events"] == "home_run").mean())


def calc_split_stats(df):
    if df.empty:
        return {
            "hr_vs_R": 0.0,
            "hr_vs_L": 0.0,
            "barrel_vs_R": 0.0,
            "barrel_vs_L": 0.0
        }

    pa_events = {
        "single", "double", "triple", "home_run", "walk", "intent_walk",
        "strikeout", "strikeout_double_play", "hit_by_pitch", "field_out",
        "grounded_into_double_play", "force_out", "field_error", "double_play",
        "triple_play", "fielders_choice", "fielders_choice_out", "sac_fly",
        "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"
    }

    def calc_side(side):
        vs = df[df["p_throws"] == side]

        pa = vs[vs["events"].isin(pa_events)]
        batted_vs = vs[vs["launch_speed"].notna() & vs["launch_angle"].notna()]

        hr_rate = (pa["events"] == "home_run").mean() if not pa.empty else 0.0

        if not batted_vs.empty:
            ev = pd.to_numeric(batted_vs["launch_speed"], errors="coerce")
            la = pd.to_numeric(batted_vs["launch_angle"], errors="coerce")

            barrel = (
                ((ev >= 98) & la.between(26, 30, inclusive="both")) |
                ((ev >= 99) & la.between(25, 31, inclusive="both")) |
                ((ev >= 100) & la.between(24, 33, inclusive="both")) |
                ((ev >= 101) & la.between(23, 34, inclusive="both")) |
                ((ev >= 102) & la.between(22, 35, inclusive="both")) |
                ((ev >= 103) & la.between(21, 36, inclusive="both")) |
                ((ev >= 104) & la.between(20, 37, inclusive="both")) |
                ((ev >= 105) & la.between(19, 38, inclusive="both"))
            )
            barrel_rate = barrel.mean()
        else:
            barrel_rate = 0.0

        return hr_rate, barrel_rate

    hr_R, barrel_R = calc_side("R")
    hr_L, barrel_L = calc_side("L")

    return {
        "hr_vs_R": hr_R,
        "hr_vs_L": hr_L,
        "barrel_vs_R": barrel_R,
        "barrel_vs_L": barrel_L
    }


def calc_pitcher_split_stats(df):
    if df.empty:
        return {
            "hr_allowed_vs_R": 0.0,
            "hr_allowed_vs_L": 0.0,
            "flyball_vs_R": 0.35,
            "flyball_vs_L": 0.35
        }

    pa_events = {
        "single", "double", "triple", "home_run", "walk", "intent_walk",
        "strikeout", "strikeout_double_play", "hit_by_pitch", "field_out",
        "grounded_into_double_play", "force_out", "field_error", "double_play",
        "triple_play", "fielders_choice", "fielders_choice_out", "sac_fly",
        "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"
    }

    def calc_side(stand_side):
        vs = df[df["stand"] == stand_side]

        pa = vs[vs["events"].isin(pa_events)]
        hr_allowed = (pa["events"] == "home_run").mean() if not pa.empty else 0.0

        batted = vs[vs["bb_type"].notna()] if "bb_type" in vs.columns else pd.DataFrame()
        flyball_rate = (batted["bb_type"] == "fly_ball").mean() if not batted.empty else 0.35

        return hr_allowed, flyball_rate

    hr_vs_R, fb_vs_R = calc_side("R")
    hr_vs_L, fb_vs_L = calc_side("L")

    return {
        "hr_allowed_vs_R": hr_vs_R,
        "hr_allowed_vs_L": hr_vs_L,
        "flyball_vs_R": fb_vs_R,
        "flyball_vs_L": fb_vs_L
    }


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
                    "hr_vs_R": 0.0,
                    "hr_vs_L": 0.0,
                    "barrel_vs_R": 0.0,
                    "barrel_vs_L": 0.0,
                    "stand": stand,
                    "team": team
                })
                continue

            hr_rate = calc_hr_rate_per_pa(df)
            barrel_rate = calc_barrel_like_rate(df)
            splits = calc_split_stats(df)

            rows.append({
                "name": name,
                "player_hr_rate": round(hr_rate, 4),
                "barrel_rate": round(barrel_rate, 4),
                "hr_vs_R": round(splits["hr_vs_R"], 4),
                "hr_vs_L": round(splits["hr_vs_L"], 4),
                "barrel_vs_R": round(splits["barrel_vs_R"], 4),
                "barrel_vs_L": round(splits["barrel_vs_L"], 4),
                "stand": stand,
                "team": team
            })

        except Exception as e:
            print(f"FAILED hitter: {name} ({mlbam_id}) -> {e}")
            rows.append({
                "name": name,
                "player_hr_rate": 0.0,
                "barrel_rate": 0.0,
                "hr_vs_R": 0.0,
                "hr_vs_L": 0.0,
                "barrel_vs_R": 0.0,
                "barrel_vs_L": 0.0,
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
                    "hr_allowed_vs_R": 0.0,
                    "hr_allowed_vs_L": 0.0,
                    "flyball_vs_R": 0.35,
                    "flyball_vs_L": 0.35,
                    "p_throws": p_throws
                })
                continue

            pa_events = {
                "single", "double", "triple", "home_run", "walk", "intent_walk",
                "strikeout", "strikeout_double_play", "hit_by_pitch", "field_out",
                "grounded_into_double_play", "force_out", "field_error", "double_play",
                "triple_play", "fielders_choice", "fielders_choice_out", "sac_fly",
                "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"
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

            splits = calc_pitcher_split_stats(df)

            rows.append({
                "name": name,
                "pitcher_hr9": pitcher_hr9,
                "flyball_rate": round(flyball_rate, 4),
                "hr_allowed_vs_R": round(splits["hr_allowed_vs_R"], 4),
                "hr_allowed_vs_L": round(splits["hr_allowed_vs_L"], 4),
                "flyball_vs_R": round(splits["flyball_vs_R"], 4),
                "flyball_vs_L": round(splits["flyball_vs_L"], 4),
                "p_throws": p_throws
            })

        except Exception as e:
            print(f"FAILED pitcher: {name} ({mlbam_id}) -> {e}")
            rows.append({
                "name": name,
                "pitcher_hr9": 1.0,
                "flyball_rate": 0.35,
                "hr_allowed_vs_R": 0.0,
                "hr_allowed_vs_L": 0.0,
                "flyball_vs_R": 0.35,
                "flyball_vs_L": 0.35,
                "p_throws": p_throws
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Cleaning duplicate pitchers...")
    clean_pitcher_duplicates()

    print("Updating probable pitchers for today and tomorrow...")
    update_pitcher_list()

    print("Cleaning duplicate pitchers again after update...")
    clean_pitcher_duplicates()

    player_df = build_player_stats()
    pitcher_df = build_pitcher_stats()

    player_df.to_csv(PLAYER_OUTPUT_FILE, index=False)
    pitcher_df.to_csv(PITCHER_OUTPUT_FILE, index=False)

    print("Updated player_stats.csv and pitcher_stats.csv")
    print(player_df.head())
    print(pitcher_df.head())