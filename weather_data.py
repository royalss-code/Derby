import math
from datetime import datetime
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Approximate stadium coordinates
STADIUM_COORDS = {
    "Arizona Diamondbacks": {"lat": 33.4455, "lon": -112.0667},
    "Atlanta Braves": {"lat": 33.8908, "lon": -84.4677},
    "Baltimore Orioles": {"lat": 39.2840, "lon": -76.6217},
    "Boston Red Sox": {"lat": 42.3467, "lon": -71.0972},
    "Chicago Cubs": {"lat": 41.9484, "lon": -87.6553},
    "Chicago White Sox": {"lat": 41.8299, "lon": -87.6338},
    "Cincinnati Reds": {"lat": 39.0979, "lon": -84.5082},
    "Cleveland Guardians": {"lat": 41.4962, "lon": -81.6852},
    "Colorado Rockies": {"lat": 39.7561, "lon": -104.9942},
    "Detroit Tigers": {"lat": 42.3390, "lon": -83.0485},
    "Houston Astros": {"lat": 29.7573, "lon": -95.3555},
    "Kansas City Royals": {"lat": 39.0517, "lon": -94.4803},
    "Los Angeles Angels": {"lat": 33.8003, "lon": -117.8827},
    "Los Angeles Dodgers": {"lat": 34.0739, "lon": -118.2400},
    "Miami Marlins": {"lat": 25.7781, "lon": -80.2197},
    "Milwaukee Brewers": {"lat": 43.0280, "lon": -87.9712},
    "Minnesota Twins": {"lat": 44.9817, "lon": -93.2776},
    "New York Mets": {"lat": 40.7571, "lon": -73.8458},
    "New York Yankees": {"lat": 40.8296, "lon": -73.9262},
    "Oakland Athletics": {"lat": 37.7516, "lon": -122.2005},
    "Philadelphia Phillies": {"lat": 39.9061, "lon": -75.1665},
    "Pittsburgh Pirates": {"lat": 40.4469, "lon": -80.0057},
    "San Diego Padres": {"lat": 32.7073, "lon": -117.1566},
    "San Francisco Giants": {"lat": 37.7786, "lon": -122.3893},
    "Seattle Mariners": {"lat": 47.5914, "lon": -122.3325},
    "St. Louis Cardinals": {"lat": 38.6226, "lon": -90.1928},
    "Tampa Bay Rays": {"lat": 27.7682, "lon": -82.6534},
    "Texas Rangers": {"lat": 32.7473, "lon": -97.0842},
    "Toronto Blue Jays": {"lat": 43.6414, "lon": -79.3894},
    "Washington Nationals": {"lat": 38.8730, "lon": -77.0074},
}

# Approximate "wind blowing out to CF" direction for each park
OUT_TO_CENTER_BEARING = {
    "Arizona Diamondbacks": 20,
    "Atlanta Braves": 25,
    "Baltimore Orioles": 45,
    "Boston Red Sox": 55,
    "Chicago Cubs": 30,
    "Chicago White Sox": 45,
    "Cincinnati Reds": 20,
    "Cleveland Guardians": 40,
    "Colorado Rockies": 15,
    "Detroit Tigers": 35,
    "Houston Astros": 30,
    "Kansas City Royals": 35,
    "Los Angeles Angels": 40,
    "Los Angeles Dodgers": 35,
    "Miami Marlins": 35,
    "Milwaukee Brewers": 30,
    "Minnesota Twins": 35,
    "New York Mets": 30,
    "New York Yankees": 35,
    "Oakland Athletics": 35,
    "Philadelphia Phillies": 30,
    "Pittsburgh Pirates": 30,
    "San Diego Padres": 30,
    "San Francisco Giants": 45,
    "Seattle Mariners": 35,
    "St. Louis Cardinals": 35,
    "Tampa Bay Rays": 30,
    "Texas Rangers": 35,
    "Toronto Blue Jays": 35,
    "Washington Nationals": 35,
}

ROOF_TEAMS = {
    "Arizona Diamondbacks",
    "Houston Astros",
    "Miami Marlins",
    "Milwaukee Brewers",
    "Seattle Mariners",
    "Tampa Bay Rays",
    "Texas Rangers",
    "Toronto Blue Jays",
}

DEFAULT_WEATHER = {
    "temperature_f": 70.0,
    "wind_speed_mph": 8.0,
    "wind_direction_deg": 0.0,
    "wind_out_mph": 0.0,
    "humidity_pct": 50.0,
    "weather_factor": 1.0,
}


def angle_diff_deg(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def wind_out_component_mph(wind_speed_mph, wind_dir_deg, out_bearing_deg):
    """
    Positive = wind helping balls carry out
    Negative = wind blowing in
    Weather APIs usually give wind direction as where wind comes FROM.
    """
    if wind_speed_mph is None or wind_dir_deg is None or out_bearing_deg is None:
        return 0.0

    wind_to_deg = (float(wind_dir_deg) + 180.0) % 360.0
    radians = math.radians((wind_to_deg - float(out_bearing_deg)) % 360.0)
    return float(wind_speed_mph) * math.cos(radians)


def compute_weather_factor(temp_f, wind_out_mph, humidity_pct=None, roof_open=True):
    factor = 1.0

    # Temperature
    if temp_f is not None:
        factor += max(min((float(temp_f) - 70.0) * 0.003, 0.08), -0.08)

    # Wind out/in
    factor += max(min(float(wind_out_mph) * 0.008, 0.12), -0.12)

    # Humidity (small effect)
    if humidity_pct is not None:
        factor += max(min((float(humidity_pct) - 50.0) * 0.0005, 0.03), -0.03)

    # Reduce weather impact for roof parks
    if not roof_open:
        factor = 1.0 + ((factor - 1.0) * 0.40)

    return max(0.82, min(1.18, factor))


def _pick_closest_hour(hourly, target_hour):
    times = hourly.get("time", [])
    if not times:
        return None

    best_idx = None
    best_diff = None

    for i, ts in enumerate(times):
        dt = datetime.fromisoformat(ts)
        diff = abs(dt.hour - target_hour)

        if best_idx is None or diff < best_diff:
            best_idx = i
            best_diff = diff

    if best_idx is None:
        return None

    return {
        "temperature_f": hourly.get("temperature_2m", [None])[best_idx],
        "wind_speed_mph": hourly.get("wind_speed_10m", [None])[best_idx],
        "wind_direction_deg": hourly.get("wind_direction_10m", [None])[best_idx],
        "humidity_pct": hourly.get("relative_humidity_2m", [None])[best_idx],
    }


def _date_str(value):
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def get_live_game_weather(home_team, target_hour_local=19):
    coords = STADIUM_COORDS.get(home_team)
    if not coords:
        return DEFAULT_WEATHER.copy()

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "forecast_days": 2,
    }

    try:
        r = requests.get(FORECAST_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        pick = _pick_closest_hour(data.get("hourly", {}), target_hour_local)
        if not pick:
            return DEFAULT_WEATHER.copy()

        temperature_f = pick["temperature_f"]
        wind_speed_mph = pick["wind_speed_mph"]
        wind_direction_deg = pick["wind_direction_deg"]
        humidity_pct = pick["humidity_pct"]

        out_bearing = OUT_TO_CENTER_BEARING.get(home_team)
        wind_out_mph = wind_out_component_mph(
            wind_speed_mph=wind_speed_mph,
            wind_dir_deg=wind_direction_deg,
            out_bearing_deg=out_bearing
        )

        roof_open = home_team not in ROOF_TEAMS

        weather_factor = compute_weather_factor(
            temp_f=temperature_f,
            wind_out_mph=wind_out_mph,
            humidity_pct=humidity_pct,
            roof_open=roof_open
        )

        return {
            "temperature_f": float(temperature_f) if temperature_f is not None else 70.0,
            "wind_speed_mph": float(wind_speed_mph) if wind_speed_mph is not None else 8.0,
            "wind_direction_deg": float(wind_direction_deg) if wind_direction_deg is not None else 0.0,
            "wind_out_mph": float(wind_out_mph),
            "humidity_pct": float(humidity_pct) if humidity_pct is not None else 50.0,
            "weather_factor": float(weather_factor),
        }
    except Exception:
        return DEFAULT_WEATHER.copy()


def get_historical_game_weather(home_team, game_date, target_hour_local=19):
    coords = STADIUM_COORDS.get(home_team)
    if not coords:
        return DEFAULT_WEATHER.copy()

    date_str = _date_str(game_date)

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
    }

    try:
        r = requests.get(ARCHIVE_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        pick = _pick_closest_hour(data.get("hourly", {}), target_hour_local)
        if not pick:
            return DEFAULT_WEATHER.copy()

        temperature_f = pick["temperature_f"]
        wind_speed_mph = pick["wind_speed_mph"]
        wind_direction_deg = pick["wind_direction_deg"]
        humidity_pct = pick["humidity_pct"]

        out_bearing = OUT_TO_CENTER_BEARING.get(home_team)
        wind_out_mph = wind_out_component_mph(
            wind_speed_mph=wind_speed_mph,
            wind_dir_deg=wind_direction_deg,
            out_bearing_deg=out_bearing
        )

        roof_open = home_team not in ROOF_TEAMS

        weather_factor = compute_weather_factor(
            temp_f=temperature_f,
            wind_out_mph=wind_out_mph,
            humidity_pct=humidity_pct,
            roof_open=roof_open
        )

        return {
            "temperature_f": float(temperature_f) if temperature_f is not None else 70.0,
            "wind_speed_mph": float(wind_speed_mph) if wind_speed_mph is not None else 8.0,
            "wind_direction_deg": float(wind_direction_deg) if wind_direction_deg is not None else 0.0,
            "wind_out_mph": float(wind_out_mph),
            "humidity_pct": float(humidity_pct) if humidity_pct is not None else 50.0,
            "weather_factor": float(weather_factor),
        }
    except Exception:
        return DEFAULT_WEATHER.copy()