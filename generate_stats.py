import pandas as pd

# Example source data you can keep expanding
# Later this can come from another file or API export

players_data = [
    {"name": "Aaron Judge", "HR": 58, "PA": 722, "Barrel%": 20.0},
    {"name": "Shohei Ohtani", "HR": 44, "PA": 599, "Barrel%": 17.5},
    {"name": "Mookie Betts", "HR": 39, "PA": 693, "Barrel%": 11.2},
    {"name": "Bryce Harper", "HR": 30, "PA": 581, "Barrel%": 13.4},
    {"name": "Kyle Schwarber", "HR": 47, "PA": 720, "Barrel%": 16.1},
    {"name": "Pete Alonso", "HR": 46, "PA": 658, "Barrel%": 15.0},
    {"name": "Matt Olson", "HR": 54, "PA": 718, "Barrel%": 14.8},
    {"name": "Yordan Alvarez", "HR": 31, "PA": 496, "Barrel%": 15.6},
]

pitchers_data = [
    {"name": "Gerrit Cole", "HR_allowed": 24, "IP": 209.0, "FB%": 35.0},
    {"name": "Zack Wheeler", "HR_allowed": 20, "IP": 192.0, "FB%": 32.0},
    {"name": "Max Fried", "HR_allowed": 14, "IP": 174.0, "FB%": 30.0},
    {"name": "Corbin Burnes", "HR_allowed": 22, "IP": 193.2, "FB%": 34.0},
    {"name": "Chris Sale", "HR_allowed": 18, "IP": 177.2, "FB%": 36.0},
    {"name": "Sonny Gray", "HR_allowed": 15, "IP": 184.0, "FB%": 31.0},
    {"name": "Pablo Lopez", "HR_allowed": 23, "IP": 194.0, "FB%": 33.0},
    {"name": "Logan Webb", "HR_allowed": 13, "IP": 216.0, "FB%": 28.0},
]

player_rows = []
for p in players_data:
    hr_rate = p["HR"] / p["PA"] if p["PA"] > 0 else 0
    barrel_rate = p["Barrel%"] / 100
    player_rows.append({
        "name": p["name"],
        "player_hr_rate": round(hr_rate, 4),
        "barrel_rate": round(barrel_rate, 4)
    })

pitcher_rows = []
for p in pitchers_data:
    hr9 = (p["HR_allowed"] * 9) / p["IP"] if p["IP"] > 0 else 0
    flyball_rate = p["FB%"] / 100
    pitcher_rows.append({
        "name": p["name"],
        "pitcher_hr9": round(hr9, 4),
        "flyball_rate": round(flyball_rate, 4)
    })

pd.DataFrame(player_rows).to_csv("player_stats.csv", index=False)
pd.DataFrame(pitcher_rows).to_csv("pitcher_stats.csv", index=False)

print("Generated player_stats.csv and pitcher_stats.csv")
