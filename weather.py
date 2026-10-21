"""Kickoff weather for this week's outdoor games, from Open-Meteo (free, no key).

Shown on cards as context only: backtests found no gain from adding weather to projections, because Vegas
totals already price the forecast in. One request covers every stadium. Any failure returns {}.
"""
import json
import urllib.request
from urllib.parse import urlencode

# home stadium coordinates (lat, lon)
STADIUM = {
    'ARI': (33.528, -112.263), 'ATL': (33.755, -84.401), 'BAL': (39.278, -76.623), 'BUF': (42.774, -78.787),
    'CAR': (35.226, -80.853), 'CHI': (41.862, -87.617), 'CIN': (39.095, -84.516), 'CLE': (41.506, -81.700),
    'DAL': (32.748, -97.093), 'DEN': (39.744, -105.020), 'DET': (42.340, -83.046), 'GB': (44.501, -88.062),
    'HOU': (29.685, -95.411), 'IND': (39.760, -86.164), 'JAX': (30.324, -81.637), 'KC': (39.049, -94.484),
    'LA': (33.953, -118.339), 'LAC': (33.953, -118.339), 'LV': (36.091, -115.184), 'MIA': (25.958, -80.239),
    'MIN': (44.974, -93.258), 'NE': (42.091, -71.264), 'NO': (29.951, -90.081), 'NYG': (40.814, -74.074),
    'NYJ': (40.814, -74.074), 'PHI': (39.901, -75.168), 'PIT': (40.447, -80.016), 'SEA': (47.595, -122.332),
    'SF': (37.403, -121.970), 'TB': (27.976, -82.503), 'TEN': (36.166, -86.771), 'WAS': (38.908, -76.864)}


def kickoff_weather(games):
    """games: iterable of (home_team, away_team, gameday 'YYYY-MM-DD', gametime 'HH:MM' ET, roof, location).
    Returns {team: dict(temp=F, wind=mph, pop=% chance of precipitation, snow=bool)} for outdoor home games."""
    todo = [g for g in games if g[4] in ('outdoors', 'open', None, '') and g[5] != 'Neutral' and g[0] in STADIUM]
    if not todo:
        return {}
    lat = ','.join(str(STADIUM[g[0]][0]) for g in todo)
    lon = ','.join(str(STADIUM[g[0]][1]) for g in todo)
    q = urlencode(dict(latitude=lat, longitude=lon, hourly='temperature_2m,precipitation_probability,wind_speed_10m,snowfall',
                       wind_speed_unit='mph', temperature_unit='fahrenheit', timezone='America/New_York', forecast_days=16))
    try:
        req = urllib.request.Request('https://api.open-meteo.com/v1/forecast?' + q, headers={'User-Agent': 'startsit'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        print('WARN weather unavailable', e)
        return {}
    data = data if isinstance(data, list) else [data]
    out = {}
    for g, d in zip(todo, data):
        h = d.get('hourly') or {}
        times = h.get('time') or []
        hh = (g[3] or '13:00')[:2]
        key = f'{g[2]}T{hh}:00'
        if key not in times:
            continue
        i = times.index(key)
        w = dict(temp=round(h['temperature_2m'][i]), wind=round(h['wind_speed_10m'][i]),
                 pop=int(h['precipitation_probability'][i] or 0), snow=bool((h.get('snowfall') or [0] * len(times))[i] or 0))
        out[g[0]] = out[g[1]] = w
    return out
