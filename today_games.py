import requests
from datetime import date, datetime


def parse_game_hour(game_datetime_str):
    """
    MLB StatsAPI usually returns something like:
    2026-04-22T22:40:00Z

    We convert that to an hour integer. This is UTC in the raw feed,
    but for now we use the local scheduled hour approximation carefully.
    If parsing fails, default to 19 (7 PM).
    """
    if not game_datetime_str:
        return 19

    try:
        dt = datetime.fromisoformat(game_datetime_str.replace("Z", "+00:00"))
        return dt.hour
    except Exception:
        return 19


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

            game_datetime = game.get("gameDate")
            game_hour = parse_game_hour(game_datetime)

            games.append({
                "game_pk": game.get("gamePk"),
                "game_date": today,
                "game_datetime": game_datetime,
                "game_hour": game_hour,
                "away_team": away_team,
                "home_team": home_team,
                "away_pitcher": away_probable.get("fullName", "TBD"),
                "home_pitcher": home_probable.get("fullName", "TBD"),
                "label": f"{away_team} at {home_team}",
            })

    return games