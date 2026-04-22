import requests
from datetime import date

def get_today_games():
    today = date.today().isoformat()

    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1"
        f"&date={today}"
        f"&hydrate=probablePitcher"
    )

    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    games = []

    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            away_team = game["teams"]["away"]["team"]["name"]
            home_team = game["teams"]["home"]["team"]["name"]

            away_probable = game["teams"]["away"].get("probablePitcher", {})
            home_probable = game["teams"]["home"].get("probablePitcher", {})

            games.append({
                "game_pk": game.get("gamePk"),
                "game_date": today,
                "away_team": away_team,
                "home_team": home_team,
                "away_pitcher": away_probable.get("fullName", "TBD"),
                "home_pitcher": home_probable.get("fullName", "TBD"),
                "label": f"{away_team} at {home_team}",
            })

    return games