import requests

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher"

def get_today_games():
    response = requests.get(SCHEDULE_URL, timeout=15)
    response.raise_for_status()
    data = response.json()

    games = []

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            away_team = game["teams"]["away"]["team"]["name"]
            home_team = game["teams"]["home"]["team"]["name"]

            away_probable = game["teams"]["away"].get("probablePitcher", {})
            home_probable = game["teams"]["home"].get("probablePitcher", {})

            away_pitcher = away_probable.get("fullName", "TBD")
            home_pitcher = home_probable.get("fullName", "TBD")

            away_pitcher_id = away_probable.get("id")
            home_pitcher_id = home_probable.get("id")

            games.append({
                "game_pk": game["gamePk"],
                "away_team": away_team,
                "home_team": home_team,
                "away_pitcher": away_pitcher,
                "home_pitcher": home_pitcher,
                "away_pitcher_id": away_pitcher_id,
                "home_pitcher_id": home_pitcher_id,
                "label": f"{away_team} at {home_team}",
            })

    return games