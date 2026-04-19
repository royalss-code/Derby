import pandas as pd

# Replace these with updated values whenever you want.
# You can expand these lists over time.

players = [
    {"name": "Aaron Judge", "player_hr_rate": 0.080, "barrel_rate": 0.20},
    {"name": "Shohei Ohtani", "player_hr_rate": 0.065, "barrel_rate": 0.17},
    {"name": "Mookie Betts", "player_hr_rate": 0.045, "barrel_rate": 0.11},
    {"name": "Bryce Harper", "player_hr_rate": 0.050, "barrel_rate": 0.13},
    {"name": "Kyle Schwarber", "player_hr_rate": 0.062, "barrel_rate": 0.16},
    {"name": "Pete Alonso", "player_hr_rate": 0.058, "barrel_rate": 0.15},
    {"name": "Matt Olson", "player_hr_rate": 0.055, "barrel_rate": 0.14},
    {"name": "Yordan Alvarez", "player_hr_rate": 0.057, "barrel_rate": 0.15},
]

pitchers = [
    {"name": "Gerrit Cole", "pitcher_hr9": 1.10, "flyball_rate": 0.35},
    {"name": "Zack Wheeler", "pitcher_hr9": 0.90, "flyball_rate": 0.32},
    {"name": "Max Fried", "pitcher_hr9": 0.80, "flyball_rate": 0.30},
    {"name": "Corbin Burnes", "pitcher_hr9": 0.95, "flyball_rate": 0.34},
    {"name": "Chris Sale", "pitcher_hr9": 0.98, "flyball_rate": 0.36},
    {"name": "Sonny Gray", "pitcher_hr9": 0.89, "flyball_rate": 0.31},
    {"name": "Pablo Lopez", "pitcher_hr9": 1.02, "flyball_rate": 0.33},
    {"name": "Logan Webb", "pitcher_hr9": 0.78, "flyball_rate": 0.28},
]

player_df = pd.DataFrame(players)
pitcher_df = pd.DataFrame(pitchers)

player_df.to_csv("player_stats.csv", index=False)
pitcher_df.to_csv("pitcher_stats.csv", index=False)

print("Updated player_stats.csv and pitcher_stats.csv")
