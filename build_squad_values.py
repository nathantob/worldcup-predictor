"""
Step (build) 2: Turn player data into a 'national talent pool value' over time.

For any country on any date, we sum the market values of its most valuable
ACTIVE players (those valued within the last 3 years). This is our genuinely-new
signal. Before modelling, we sanity-check it (2022 World Cup) and check how well
the country names line up with our match data.
"""

import csv
import re
from bisect import bisect_right
from collections import defaultdict
from datetime import date as Date, timedelta

TOP_N = 23          # squad size
ACTIVE_DAYS = 1095  # a player counts only if valued within ~3 years

def to_date(s):
    y, m, d = s.split("-")
    return Date(int(y), int(m), int(d))

# --- citizenship per player ---
citizen = {}
with open("player_profiles.csv", newline="") as f:
    for r in csv.DictReader(f):
        # Dual-nationality players have both countries jammed together
        # (separated by 2+ spaces). Keep only the first (primary) nationality.
        c = re.split(r"\s{2,}", r["citizenship"].strip())[0].strip()
        if c and c != "N/A":
            citizen[r["player_id"]] = c

# --- market value timeline per player ---
timeline = defaultdict(list)   # player_id -> list of (date_obj, value)
with open("player_market_value.csv", newline="") as f:
    for r in csv.DictReader(f):
        try:
            v = float(r["value"])
        except ValueError:
            continue
        try:
            d = to_date(r["date_unix"])
        except Exception:
            continue
        timeline[r["player_id"]].append((d, v))

# Sort each player's timeline and split into parallel lists for fast lookup.
players_by_country = defaultdict(list)
pv = {}
for pid, entries in timeline.items():
    entries.sort()
    pv[pid] = ([e[0] for e in entries], [e[1] for e in entries])
    c = citizen.get(pid)
    if c:
        players_by_country[c].append(pid)

# Our match-dataset names -> the names used in the player data.
ALIAS_TM = {
    "Ivory Coast": "Cote d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "South Korea": "Korea, South",
    "North Korea": "Korea, North",
    "Hong Kong": "Hongkong",
}

_cache = {}
def squad_value(country, on_date):
    """Sum of the TOP_N most valuable active players for a country on a date.
    Cached by (country, year, month) for speed during backtesting."""
    name = ALIAS_TM.get(country, country)
    key = (name, on_date.year, on_date.month)
    if key in _cache:
        return _cache[key]
    active_cutoff = on_date - timedelta(days=ACTIVE_DAYS)
    vals = []
    for pid in players_by_country.get(name, []):
        dates, values = pv[pid]
        i = bisect_right(dates, on_date)
        if i == 0:
            continue
        if dates[i - 1] >= active_cutoff:      # still actively valued
            vals.append(values[i - 1])
    vals.sort(reverse=True)
    total = sum(vals[:TOP_N])
    _cache[key] = total
    return total

if __name__ == "__main__":
    # --- SANITY CHECK: richest squads at the 2022 World Cup ---
    wc2022 = Date(2022, 11, 1)
    ranked = sorted(players_by_country.keys(),
                    key=lambda c: squad_value(c, wc2022), reverse=True)
    print("Richest national talent pools as of Nov 2022 (sanity check):\n")
    for i, c in enumerate(ranked[:15], 1):
        print(f"{i:2}. {c:<18} €{squad_value(c, wc2022)/1e6:,.0f}M")

    # --- country-name overlap with our match data (games since 2010) ---
    tm_countries = set(players_by_country.keys())
    match_teams = set()
    with open("results.csv", newline="") as f:
        for m in csv.DictReader(f):
            if m["date"] >= "2010-01-01":
                match_teams.add(m["home_team"]); match_teams.add(m["away_team"])
    missing = sorted(t for t in match_teams if t not in tm_countries)
    print(f"\nMatch teams since 2010: {len(match_teams)} | "
          f"not found in player data: {len(missing)}")
    print("First 25 unmatched names:")
    print(", ".join(missing[:25]))
