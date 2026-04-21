from pybaseball import statcast
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

START_DATE = "2024-03-20"
END_DATE = "2026-04-19"
OUTPUT_FILE = "data.csv"


def pull_statcast_in_chunks(start_date_str, end_date_str, chunk_days=30):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    all_chunks = []
    current_start = start_date

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end_date)

        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")

        print(f"Pulling Statcast chunk: {start_str} to {end_str}")

        try:
            chunk = statcast(start_dt=start_str, end_dt=end_str)
            if chunk is not None and not chunk.empty:
                all_chunks.append(chunk)
        except Exception as e:
            print(f"FAILED chunk {start_str} to {end_str}: {e}")

        current_start = current_end + timedelta(days=1)

    if not all_chunks:
        raise ValueError("No Statcast data was pulled successfully.")

    return pd.concat(all_chunks, ignore_index=True)


print("Pulling Statcast data...")
df = pull_statcast_in_chunks(START_DATE, END_DATE, chunk_days=30)

needed_cols = [
    "game_date",
    "game_pk",
    "batter",
    "pitcher",
    "events",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "stand",
    "p_throws",
    "home_team"
]

df = df[needed_cols].copy()
df["game_date"] = pd.to_datetime(df["game_date"])
df = df.dropna(subset=["game_pk", "batter", "pitcher"])
df["launch_speed"] = pd.to_numeric(df["launch_speed"], errors="coerce")
df["launch_angle"] = pd.to_numeric(df["launch_angle"], errors="coerce")

# Real outcome at event level
df["is_hr"] = (df["events"] == "home_run").astype(int)

# Better barrel-like contact approximation
ev = pd.to_numeric(df["launch_speed"], errors="coerce")
la = pd.to_numeric(df["launch_angle"], errors="coerce")

df["is_barrel_like"] = (
    (
        ((ev >= 98) & la.between(26, 30, inclusive="both")) |
        ((ev >= 99) & la.between(25, 31, inclusive="both")) |
        ((ev >= 100) & la.between(24, 33, inclusive="both")) |
        ((ev >= 101) & la.between(23, 34, inclusive="both")) |
        ((ev >= 102) & la.between(22, 35, inclusive="both")) |
        ((ev >= 103) & la.between(21, 36, inclusive="both")) |
        ((ev >= 104) & la.between(20, 37, inclusive="both")) |
        ((ev >= 105) & la.between(19, 38, inclusive="both"))
    )
).fillna(False).astype(int)

# Simplified fly ball
df["is_flyball"] = (df["bb_type"] == "fly_ball").astype(int)

# Only true batted balls for barrel denominator
df["is_batted_ball"] = (
    df["launch_speed"].notna() & df["launch_angle"].notna()
).astype(int)

# Sort so rolling features only use past games
df = df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

print("Building hitter-game table...")

# Aggregate to one row per hitter per game
hitter_game = df.groupby(["game_date", "game_pk", "batter"]).agg(
    home_run=("is_hr", "max"),
    batter_events=("is_hr", "size"),
    batter_batted_balls=("is_batted_ball", "sum"),
    batter_barrels=("is_barrel_like", "sum"),
    opp_pitcher=("pitcher", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
    stand=("stand", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
    p_throws=("p_throws", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
    home_team=("home_team", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
).reset_index()

# Rolling hitter features BEFORE each game
hitter_game = hitter_game.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)

hitter_game["player_hr_rate"] = np.nan
hitter_game["barrel_rate"] = np.nan
hitter_game["prior_games"] = np.nan
hitter_game["recent_hr_rate"] = np.nan
hitter_game["recent_barrel_rate"] = np.nan
hitter_game["player_hr_rate_split"] = np.nan
hitter_game["barrel_rate_split"] = np.nan
hitter_game["recent_hr_rate_split"] = np.nan
hitter_game["recent_barrel_rate_split"] = np.nan

for batter_id, idx in tqdm(hitter_game.groupby("batter").groups.items(), desc="Hitters"):
    g = hitter_game.loc[list(idx)].copy().sort_values(["game_date", "game_pk"])

    barrel_per_game = (
        g["batter_barrels"] / g["batter_batted_balls"].replace(0, np.nan)
    )

    # Long-term overall stats
    g["player_hr_rate"] = g["home_run"].expanding().mean().shift(1)
    g["barrel_rate"] = barrel_per_game.expanding().mean().shift(1)

    # Recent overall stats
    g["recent_hr_rate"] = g["home_run"].rolling(5, min_periods=2).mean().shift(1)
    g["recent_barrel_rate"] = (
        barrel_per_game
        .rolling(10, min_periods=3)
        .mean()
        .shift(1)
    )

    # Split stats vs pitcher handedness
    split_hr_vals = []
    split_barrel_vals = []
    split_recent_hr_vals = []
    split_recent_barrel_vals = []

    for i in range(len(g)):
        current_throw = g.iloc[i]["p_throws"]
        prior = g.iloc[:i]
        prior_same_hand = prior[prior["p_throws"] == current_throw]

        if len(prior_same_hand) > 0:
            split_hr_vals.append(prior_same_hand["home_run"].mean())
            split_barrel_vals.append(
                (
                    prior_same_hand["batter_barrels"] /
                    prior_same_hand["batter_batted_balls"].replace(0, np.nan)
                ).mean()
            )
        else:
            split_hr_vals.append(np.nan)
            split_barrel_vals.append(np.nan)

        if len(prior_same_hand) >= 3:
            recent_same_hand = prior_same_hand.tail(10)
            split_recent_hr_vals.append(recent_same_hand["home_run"].mean())
            split_recent_barrel_vals.append(
                (
                    recent_same_hand["batter_barrels"] /
                    recent_same_hand["batter_batted_balls"].replace(0, np.nan)
                ).mean()
            )
        else:
            split_recent_hr_vals.append(np.nan)
            split_recent_barrel_vals.append(np.nan)

    g["player_hr_rate_split"] = split_hr_vals
    g["barrel_rate_split"] = split_barrel_vals
    g["recent_hr_rate_split"] = split_recent_hr_vals
    g["recent_barrel_rate_split"] = split_recent_barrel_vals

    g["prior_games"] = np.arange(len(g))

    hitter_game.loc[g.index, "player_hr_rate"] = g["player_hr_rate"]
    hitter_game.loc[g.index, "barrel_rate"] = g["barrel_rate"]
    hitter_game.loc[g.index, "recent_hr_rate"] = g["recent_hr_rate"]
    hitter_game.loc[g.index, "recent_barrel_rate"] = g["recent_barrel_rate"]
    hitter_game.loc[g.index, "player_hr_rate_split"] = g["player_hr_rate_split"]
    hitter_game.loc[g.index, "barrel_rate_split"] = g["barrel_rate_split"]
    hitter_game.loc[g.index, "recent_hr_rate_split"] = g["recent_hr_rate_split"]
    hitter_game.loc[g.index, "recent_barrel_rate_split"] = g["recent_barrel_rate_split"]
    hitter_game.loc[g.index, "prior_games"] = g["prior_games"]

print("Building pitcher-game table...")

# Aggregate pitcher side per game
pitcher_game = df.groupby(["game_date", "game_pk", "pitcher"]).agg(
    hr_allowed=("is_hr", "sum"),
    pitcher_events=("is_hr", "size"),
    flyballs_allowed=("is_flyball", "sum")
).reset_index()

pitcher_game = pitcher_game.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)

pitcher_game["pitcher_hr9"] = np.nan
pitcher_game["flyball_rate"] = np.nan
pitcher_game["prior_pitcher_games"] = np.nan

for pitcher_id, idx in tqdm(pitcher_game.groupby("pitcher").groups.items(), desc="Pitchers"):
    g = pitcher_game.loc[list(idx)].copy().sort_values(["game_date", "game_pk"])

    prior_hr_rate = (g["hr_allowed"] / g["pitcher_events"]).expanding().mean().shift(1)
    prior_fb_rate = (g["flyballs_allowed"] / g["pitcher_events"]).expanding().mean().shift(1)

    g["pitcher_hr9"] = prior_hr_rate * 9
    g["pitcher_hr9"] = g["pitcher_hr9"].clip(0.5, 2.0)
    g["flyball_rate"] = prior_fb_rate
    g["prior_pitcher_games"] = np.arange(len(g))

    pitcher_game.loc[g.index, "pitcher_hr9"] = g["pitcher_hr9"]
    pitcher_game.loc[g.index, "flyball_rate"] = g["flyball_rate"]
    pitcher_game.loc[g.index, "prior_pitcher_games"] = g["prior_pitcher_games"]

print("Merging hitter and pitcher features...")

final_df = hitter_game.merge(
    pitcher_game[[
        "game_date",
        "game_pk",
        "pitcher",
        "pitcher_hr9",
        "flyball_rate",
        "prior_pitcher_games"
    ]],
    left_on=["game_date", "game_pk", "opp_pitcher"],
    right_on=["game_date", "game_pk", "pitcher"],
    how="left"
)

# Real handedness matchup
final_df["matchup"] = (
    ((final_df["stand"] == "L") & (final_df["p_throws"] == "R")) |
    ((final_df["stand"] == "R") & (final_df["p_throws"] == "L"))
).astype(int)

# Park factor map
park_factor_map = {
    "COL": 1.25,
    "NYY": 1.12,
    "LAD": 0.98,
    "SF": 0.90,
    "BOS": 1.05,
    "CIN": 1.10,
    "PHI": 1.08,
    "ATL": 1.02,
    "CHC": 1.03,
    "TEX": 1.04,
    "HOU": 0.97,
    "SEA": 0.92,
    "SD": 0.95,
    "MIA": 0.91,
    "DET": 0.94,
    "KC": 0.96,
    "OAK": 0.89,
    "TB": 0.93,
    "NYM": 0.97,
    "STL": 0.99,
    "MIL": 1.01,
    "BAL": 1.02,
    "TOR": 1.00,
    "CLE": 0.98,
    "MIN": 1.00,
    "PIT": 0.95,
    "WSH": 0.99,
    "ARI": 1.01,
    "LAA": 0.96,
    "CWS": 1.00
}

final_df["park_factor"] = final_df["home_team"].map(park_factor_map).fillna(1.00)
final_df["bad_pitcher"] = (final_df["pitcher_hr9"] > 1.2).astype(int)
final_df["weak_hitter"] = (final_df["player_hr_rate"] < 0.12).astype(int)
final_df["very_weak_hitter"] = (final_df["player_hr_rate"] < 0.08).astype(int)
final_df["strong_hitter"] = (final_df["player_hr_rate"] > 0.18).astype(int)

final_df["power_index"] = (
    final_df["player_hr_rate"] * 0.38 +
    final_df["barrel_rate"] * 0.16 +
    final_df["recent_hr_rate"] * 0.12 +
    final_df["recent_barrel_rate"] * 0.14 +
    final_df["player_hr_rate_split"] * 0.10 +
    final_df["barrel_rate_split"] * 0.05 +
    final_df["recent_hr_rate_split"] * 0.03 +
    final_df["recent_barrel_rate_split"] * 0.02
)

final_df["power_vs_pitcher"] = (
    final_df["power_index"] * final_df["pitcher_hr9"]
)

print("Rows before history filter:", len(final_df))
print("Non-null player_hr_rate:", final_df["player_hr_rate"].notna().sum())
print("Non-null barrel_rate:", final_df["barrel_rate"].notna().sum())
print("Non-null recent_hr_rate:", final_df["recent_hr_rate"].notna().sum())
print("Non-null recent_barrel_rate:", final_df["recent_barrel_rate"].notna().sum())
print("Non-null pitcher_hr9:", final_df["pitcher_hr9"].notna().sum())
print("Non-null flyball_rate:", final_df["flyball_rate"].notna().sum())

# Keep only rows with enough prior history
final_df = final_df[
    (final_df["prior_games"] >= 5) &
    (final_df["prior_pitcher_games"] >= 2)
].copy()

final_df = final_df.dropna(subset=[
    "player_hr_rate",
    "barrel_rate",
    "recent_hr_rate",
    "recent_barrel_rate",
    "player_hr_rate_split",
    "barrel_rate_split",
    "recent_hr_rate_split",
    "recent_barrel_rate_split",
    "pitcher_hr9",
    "flyball_rate",
    "home_run"
])

final_df["split_confidence"] = (
    final_df["player_hr_rate_split"].notna().astype(int)
)

final_df = final_df[[
    "player_hr_rate",
    "barrel_rate",
    "recent_hr_rate",
    "recent_barrel_rate",
    "player_hr_rate_split",
    "barrel_rate_split",
    "recent_hr_rate_split",
    "recent_barrel_rate_split",
    "power_index",
    "pitcher_hr9",
    "flyball_rate",
    "power_vs_pitcher",
    "bad_pitcher",
    "weak_hitter",
    "very_weak_hitter",
    "strong_hitter",
    "split_confidence",
    "matchup",
    "park_factor",
    "home_run"
]]

print("Rows:", len(final_df))
print("HR label rate:", round(final_df["home_run"].mean(), 4))

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {OUTPUT_FILE}")