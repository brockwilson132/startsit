"""One place that decides, for every player, whether he can play and which team he is on.

Every source makes claims with a timestamp, and the freshest claim wins:

  source          fetched     claims                                   timestamp used
  weekly roster   daily       team; active / IR / exempt / cut         his team's game day that week
  injury report   daily       this week: Out / Doubtful / Questionable   two days before his game
  ESPN            every run   status (Active / IR / Suspension / Out...),  date of ESPN's latest note
                              headline events (activated, placed on IR,
                              released, traded, signed, suspended)
  Sleeper         daily       team; IR / PUP / not-active / suspended    Sleeper's per-player news time

Two separate questions are resolved:
  long  - multi-week availability: active, IR, Exempt, Suspended, FA (released / unsigned)
  week  - this week's game designation: '', Questionable, Doubtful, Out (only claims made this week count)
"""
import re
from dataclasses import dataclass

TEAMS = {
    'ARI': 'Cardinals', 'ATL': 'Falcons', 'BAL': 'Ravens', 'BUF': 'Bills', 'CAR': 'Panthers', 'CHI': 'Bears',
    'CIN': 'Bengals', 'CLE': 'Browns', 'DAL': 'Cowboys', 'DEN': 'Broncos', 'DET': 'Lions', 'GB': 'Packers',
    'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars', 'KC': 'Chiefs', 'LA': 'Rams', 'LAC': 'Chargers',
    'LV': 'Raiders', 'MIA': 'Dolphins', 'MIN': 'Vikings', 'NE': 'Patriots', 'NO': 'Saints', 'NYG': 'Giants',
    'NYJ': 'Jets', 'PHI': 'Eagles', 'PIT': 'Steelers', 'SEA': 'Seahawks', 'SF': '49ers', 'TB': 'Buccaneers',
    'TEN': 'Titans', 'WAS': 'Commanders'}
NICK = {v.lower(): k for k, v in TEAMS.items()}
CITY = {'arizona': 'ARI', 'atlanta': 'ATL', 'baltimore': 'BAL', 'buffalo': 'BUF', 'carolina': 'CAR', 'chicago': 'CHI',
        'cincinnati': 'CIN', 'cleveland': 'CLE', 'dallas': 'DAL', 'denver': 'DEN', 'detroit': 'DET', 'green bay': 'GB',
        'houston': 'HOU', 'indianapolis': 'IND', 'jacksonville': 'JAX', 'kansas city': 'KC', 'las vegas': 'LV',
        'miami': 'MIA', 'minnesota': 'MIN', 'new england': 'NE', 'new orleans': 'NO', 'philadelphia': 'PHI',
        'pittsburgh': 'PIT', 'seattle': 'SEA', 'san francisco': 'SF', 'tampa bay': 'TB', 'tennessee': 'TEN',
        'washington': 'WAS'}   # no New York / Los Angeles: ambiguous without the nickname
ESPN_ABBR = {'WSH': 'WAS', 'LAR': 'LA', 'JAC': 'JAX'}

ROSTER_LONG = {'ACT': 'active', 'INA': 'active', 'RES': 'IR', 'EXE': 'Exempt', 'SUS': 'Suspended',
               'CUT': 'FA', 'RET': 'FA', 'DEV': 'FA'}
ESPN_LONG = {'Active': 'active', 'Injured Reserve': 'IR', 'Suspension': 'Suspended'}
ESPN_WEEK = {'Active': '', 'Questionable': 'Questionable', 'Doubtful': 'Doubtful', 'Out': 'Out'}
SLEEPER_INJ_LONG = {'IR': 'IR', 'PUP': 'IR', 'NFI': 'IR', 'NA': 'Exempt', 'Sus': 'Suspended', 'SUS': 'Suspended'}
SLEEPER_WEEK = {'Questionable': 'Questionable', 'Doubtful': 'Doubtful', 'Out': 'Out'}

# ---------------------------------------------------------------- headlines
_NAME = r"[A-Z][\w'.-]+(?: [A-Z][\w'.-]+){0,2}(?:'s)?(?: \([^)]*\))?"
_FUTURE = re.compile(r"\b(expected|could|will|would|may|might|hopes?|likely|eligible|set|poised|candidate|window|"
                     r"plans?|planning|considering|on track|in line|figures)\b[^.;]{0,35}?\b(remov|reinstat|activat|"
                     r"releas|waiv|cut|trad|sign|plac|suspend)", re.I)
_RULES = [
    # (kind, long, regex)  - order matters: first match wins
    # recognized, but not a change: still on IR while practicing; practice-squad moves
    ('no change', None, re.compile(r"\bdesignated\b[^.;]{0,50}\breturn\b|\b(return|practice) window\b|"
                                   r"\bpractice squad\b(?![^.;]{0,30}\b(53-man|active) roster)", re.I)),
    ('cleared', 'active', re.compile(r"\b(has|have|had|was|were|been|is|got|gets)\b[^.;]{0,30}?\b(removed from|reinstated|"
                                     r"activated|lifted from|returned from)\b|\b(activated|reinstated)\b " + _NAME +
                                     r" (from|off)\b|\breinstated\b|\bremoved from the (commissioner's )?exempt", re.I)),
    ('ir', 'IR', re.compile(r"\b(placed|put|moved|reverted|landed|lands)\b[^.;]{0,40}?\b(injured reserve|IR|reserve/PUP|"
                            r"physically unable to perform|PUP list|reserve/NFI|non-football injury list)\b", re.I)),
    ('suspended', 'Suspended', re.compile(r"\b(has been|was|is|been)\b[^.;]{0,15}?\bsuspended\b|\bsuspension\b[^.;]{0,70}?"
                                          r"\b(upheld|denied|announced|handed|issued|reduced)\b|\bappeal\b[^.;]{0,40}\bdenied\b", re.I)),
    ('exempt', 'Exempt', re.compile(r"\b(placed|put)\b[^.;]{0,30}?\b(commissioner's )?exempt list\b", re.I)),
    ('released', 'FA', re.compile(r"\b(released|waived|cut|terminated)\b(?![^.;]{0,20}\b(from|off) (injured reserve|IR|the "
                                  r"practice squad)\b)|\bparted ways\b", re.I)),
    ('traded', None, re.compile(r"\b(traded|dealt|shipped|acquired)\b", re.I)),
    ('signed', 'active', re.compile(r"\b(signed|signs|re-signed|claimed off waivers|promoted|agreed to (a )?(one-year|two-year|"
                                    r"multi-year|\d+-year)? ?(deal|contract))\b", re.I)),
]
# words that mean the headline is talking about status at all; used to catch phrasings the rules miss
STATUS_WORDS = re.compile(r"\b(exempt|injured reserve|reserve/|activated|reinstated|released|waived|traded|"
                          r"suspend|signed|PUP|non-football)\b", re.I)


def team_in(txt):
    """Team a traded / signed player is joining, e.g. 'traded to the Texans', 'signed with Houston'."""
    names = '|'.join(re.escape(v) for v in TEAMS.values())
    cities = '|'.join(c.title() for c in CITY)
    m = re.search(r"\b(?:to|with|by|joins?|joining)\s+(?:the\s+)?(?:[A-Z][\w'.]+\s+){0,2}?(" + names + r")\b", txt)
    if m:
        return NICK[m.group(1).lower()]
    m = re.search(r"\b(?:to|with|by|joins?|joining)\s+(?:the\s+)?(" + cities + r")\b", txt)
    return CITY[m.group(1).lower()] if m else None


def parse_headline(txt):
    """Returns (kind, long_status, new_team) for a status-changing headline, or None.
    kind is None (with long None) when the headline talks about status but no rule understood it."""
    if not txt:
        return None
    first = re.split(r"(?<=[a-z0-9)])\.\s", txt.strip())[0]       # the lead sentence carries the news
    if _FUTURE.search(first):
        return None
    for kind, long, rx in _RULES:
        if rx.search(first):
            if kind == 'no change':
                return None
            team = team_in(first) if kind in ('traded', 'signed') else None
            if kind == 'signed' and not team:
                m = re.match(r"(?:The\s+)?(" + '|'.join(re.escape(v) for v in TEAMS.values()) + '|' +
                             '|'.join(c.title() for c in CITY) + r")\s+(?:signed|claimed|promoted)\b", first)
                if m:
                    team = NICK.get(m.group(1).lower()) or CITY.get(m.group(1).lower())
            if kind == 'traded' and not team:
                return None                                       # "traded for X" about someone else: ignore
            return kind, long, team
    if STATUS_WORDS.search(first):
        return None, None, None
    return None


# ---------------------------------------------------------------- resolution
@dataclass
class Claim:
    src: str
    ts: str                  # 'YYYY-MM-DDTHH:MM' (UTC-ish; only order matters)
    long: str = None         # active / IR / Exempt / Suspended / FA
    week: str = None         # '' / Questionable / Doubtful / Out
    team: str = None         # team abbreviation or 'FA'
    note: str = None


def _ts(x):
    return (x or '')[:16].replace(' ', 'T')


def resolve(claims, week_start):
    """Freshest claim wins, separately for team, multi-week status, and this week's designation.
    Returns dict(long, week, team, long_src, team_src) with the winning claims."""
    out = dict(long='active', week='', team=None, long_src=None, team_src=None, week_src=None)
    for field in ('team', 'long', 'week'):
        best = None
        for c in claims:
            v = getattr(c, field)
            if v is None or (field == 'week' and c.ts < week_start):
                continue
            # ties go to the later-listed (more authoritative for that field) source
            if best is None or c.ts >= best.ts:
                best = c
        if best is not None:
            out[field] = getattr(best, field)
            out[field + '_src'] = best
    return out


def final_status(r):
    """One status string for the app and model: long-term status if he's out for weeks, else this week's tag."""
    if r.get('team') == 'FA':
        return 'FA'
    if r['long'] in ('IR', 'Exempt', 'Suspended', 'FA'):
        return r['long']
    return r['week'] or ''
