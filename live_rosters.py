import requests

TEAM_LOOKUP_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"

def get_team_id_map():
    response = requests.get(TEAM_LOOKUP_URL, timeout=15)
    response.raise_for_status()
    data = response.json()

    team_map = {}
    for team in data.get("teams", []):
        team_map[team["name"]] = team["id"]
    return team_map

def get_team_roster(team_name):
    team_map = get_team_id_map()
    team_id = team_map.get(team_name)

    if not team_id:
        return []

    response = requests.get(ROSTER_URL.format(team_id=team_id), timeout=15)
    response.raise_for_status()
    data = response.json()

    hitters = []
    for player in data.get("roster", []):
        person = player.get("person", {})
        position = player.get("position", {})
        pos_abbr = position.get("abbreviation", "")

        if pos_abbr != "P":
            hitters.append(person.get("fullName"))

    return sorted(hitters)