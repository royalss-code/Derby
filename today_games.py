import requests

def get_today_games():
    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1"
    response = requests.get(url, timeout=10)
    data = response.json()

    games_list = []

    for date in data.get("dates", []):
        for game in date.get("games", []):
            away = game["teams"]["away"]["team"]["name"]
            home = game["teams"]["home"]["team"]["name"]
            games_list.append(f"{away} at {home}")

    return games_list
