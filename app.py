def build_features(player_stats, pitcher_stats, park_factor=1.0):
    player_hr_rate = float(player_stats["player_hr_rate"])
    barrel_rate = float(player_stats["barrel_rate"])

    # Determine pitcher handedness
    p_throws = pitcher_stats.get("p_throws", "R")

    # Use split stats correctly
    if p_throws == "R":
        player_hr_rate_split = player_stats.get("hr_vs_R", player_hr_rate)
        barrel_rate_split = player_stats.get("barrel_vs_R", barrel_rate)
    else:
        player_hr_rate_split = player_stats.get("hr_vs_L", player_hr_rate)
        barrel_rate_split = player_stats.get("barrel_vs_L", barrel_rate)

    # Recent = same for now
    recent_hr_rate = player_hr_rate
    recent_barrel_rate = barrel_rate
    recent_hr_rate_split = player_hr_rate_split
    recent_barrel_rate_split = barrel_rate_split

    power_index = (
        player_hr_rate * 0.38 +
        barrel_rate * 0.16 +
        recent_hr_rate * 0.12 +
        recent_barrel_rate * 0.14 +
        player_hr_rate_split * 0.10 +
        barrel_rate_split * 0.05 +
        recent_hr_rate_split * 0.03 +
        recent_barrel_rate_split * 0.02
    )

    pitcher_hr9 = float(pitcher_stats["pitcher_hr9"])
    flyball_rate = float(pitcher_stats["flyball_rate"])
    power_vs_pitcher = power_index * pitcher_hr9
    bad_pitcher = int(pitcher_hr9 > 1.2)
    weak_hitter = int(player_hr_rate < 0.12)
    very_weak_hitter = int(player_hr_rate < 0.08)
    strong_hitter = int(player_hr_rate > 0.18)
    split_confidence = 1

    stand = player_stats.get("stand", "R")

    matchup = int(
        (stand == "L" and p_throws == "R") or
        (stand == "R" and p_throws == "L")
    )

    return [[
        player_hr_rate,
        barrel_rate,
        recent_hr_rate,
        recent_barrel_rate,
        player_hr_rate_split,
        barrel_rate_split,
        recent_hr_rate_split,
        recent_barrel_rate_split,
        power_index,
        pitcher_hr9,
        flyball_rate,
        power_vs_pitcher,
        bad_pitcher,
        weak_hitter,
        very_weak_hitter,
        strong_hitter,
        split_confidence,
        matchup,
        float(park_factor)
    ]]