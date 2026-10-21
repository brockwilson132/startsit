"""Player news and status from free public feeds.

ESPN injuries feed: status (Active / Questionable / Doubtful / Out / Injured Reserve / Suspension),
  a one-line news blurb, and an expected return date, for about 800 players. Fetched every run.
Sleeper players file: current team (catches free agents and signings), depth chart order and injury
  notes. Large (~15 MB) and Sleeper asks callers to fetch it at most once a day, so it is cached.
Sleeper trending: how many fantasy managers added each player in the last 24 hours. Fetched every run.

Every fetch is optional: if a feed is down the pipeline carries on without it.
"""
import json
import os
import re
import time
import urllib.request

UA = {'User-Agent': 'Mozilla/5.0 (startsit fantasy app)'}
ESPN_STATUS = {'Out': 'Out', 'Doubtful': 'Doubtful', 'Questionable': 'Questionable',
               'Injured Reserve': 'IR', 'Suspension': 'Suspended'}


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _cached(path, url, max_age_h):
    stamp = path + '.fetched'
    fetched = float(open(stamp).read()) if os.path.exists(stamp) else 0
    if os.path.exists(path) and time.time() - fetched < max_age_h * 3600:
        return json.load(open(path))
    try:
        data = _get(url)
        json.dump(data, open(path, 'w'))
        open(stamp, 'w').write(str(time.time()))
        return data
    except Exception as e:
        print('WARN news fetch failed', url, e)
        return json.load(open(path)) if os.path.exists(path) else None


def load(cache_dir, offline=False):
    """Returns dict with espn {espn_id: {...}}, sleeper {gsis_id: {...}}, trending {gsis_id: adds}."""
    age = 1e9 if offline else None
    espn_raw = _cached(os.path.join(cache_dir, 'espn_injuries.json'),
                       'https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries', age or 0.01)
    sleeper_raw = _cached(os.path.join(cache_dir, 'sleeper_players.json'),
                          'https://api.sleeper.app/v1/players/nfl', age or 20)
    trend_raw = _cached(os.path.join(cache_dir, 'sleeper_trending.json'),
                        'https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=100', age or 0.01)

    espn = {}
    for t in (espn_raw or {}).get('injuries', []):
        team_name = (t.get('displayName') or '').lower()
        for e in t.get('injuries', []):
            a = e.get('athlete', {})
            links = a.get('links') or [{}]
            m = re.search(r'/id/(\d+)/', links[0].get('href', ''))
            if not m:
                continue
            det = e.get('details') or {}
            espn[m.group(1)] = dict(
                status=ESPN_STATUS.get(e.get('status'), ''), raw=e.get('status') or '', news=(e.get('shortComment') or '').strip(),
                date=(e.get('date') or '')[:10], ts=(e.get('date') or '')[:16], ret=det.get('returnDate'),
                injury=det.get('type') or '', team_name=team_name,
                name=a.get('displayName'))

    sleeper, sl_by_id = {}, {}
    if isinstance(sleeper_raw, dict):
        for sid, v in sleeper_raw.items():
            if v.get('position') not in ('QB', 'RB', 'WR', 'TE', 'K'):
                continue
            g = (v.get('gsis_id') or '').strip()
            rec = dict(name=v.get('full_name'), pos=v.get('position'), team=v.get('team'), active=v.get('active'),
                       espn_id=str(v.get('espn_id')) if v.get('espn_id') else None, gsis_id=g or None,
                       status=v.get('status'), body=v.get('injury_body_part'), notes=v.get('injury_notes'), inj=v.get('injury_status'),
                       depth=v.get('depth_chart_order'), rank=v.get('search_rank') or 9999,
                       ts=time.strftime('%Y-%m-%dT%H:%M', time.gmtime(v['news_updated'] / 1000)) if v.get('news_updated') else None)
            sl_by_id[sid] = rec
            if g:
                sleeper[g] = rec
    trending = {}
    for t in trend_raw or []:
        rec = sl_by_id.get(str(t.get('player_id')))
        if rec:
            c = int(t.get('count', 0))
            # 2 significant figures, so the hourly check doesn't redeploy for tiny count changes
            trending[rec['gsis_id'] or 'sleeper:' + str(t.get('player_id'))] = int(float(f'{c:.2g}'))
    print(f'news: espn {len(espn)} players, sleeper {len(sleeper)}, trending {len(trending)}')
    return dict(espn=espn, sleeper=sleeper, sleeper_all=sl_by_id, trending=trending)
