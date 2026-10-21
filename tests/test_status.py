"""Run: python3 tests/test_status.py   (also runs in GitHub Actions before every update)

Real-world headline phrasings and source conflicts. When a new phrasing slips through, add it here."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from status import Claim, parse_headline, resolve, final_status  # noqa: E402

HEADLINES = [
    # (headline, expected kind, expected long, expected team)
    ("Jacobs (personal) has officially been removed from the Commissioner's Exempt List, Matt Schneidman of The Athletic reports.", 'cleared', 'active', None),
    ("The Packers activated Smith from injured reserve Monday.", 'cleared', 'active', None),
    ("Smith was activated from injured reserve Saturday.", 'cleared', 'active', None),
    ("Smith has been reinstated by the NFL and is eligible to play Sunday.", 'cleared', 'active', None),
    ("Smith was activated off the PUP list Wednesday.", 'cleared', 'active', None),
    ("The Eagles placed Hurts (calf) on injured reserve Tuesday.", 'ir', 'IR', None),
    ("Hurts was placed on IR on Tuesday with a calf strain, Adam Schefter of ESPN reports.", 'ir', 'IR', None),
    ("Charbonnet (knee) reverted to the reserve/PUP list Wednesday.", 'ir', 'IR', None),
    ("Smith was moved to injured reserve with a torn ACL.", 'ir', 'IR', None),
    ("The NFL announced Tuesday that Sweat has been suspended two games without pay for violating the league's PED policy.", 'suspended', 'Suspended', None),
    ("Greenlaw's appeal of a one-game suspension was denied by the NFL on Tuesday, Adam Schefter of ESPN reports.", 'suspended', 'Suspended', None),
    ("Jackson's suspension for an act of unsportsmanlike conduct has been reduced to one game, Tom Pelissero reports.", 'suspended', 'Suspended', None),
    ("Smith was placed on the commissioner's exempt list Friday.", 'exempt', 'Exempt', None),
    ("The Jets released Smith on Tuesday.", 'released', 'FA', None),
    ("Smith was waived by the Bears on Monday.", 'released', 'FA', None),
    ("The Cardinals traded Harrison to the Texans on Tuesday in exchange for a 2027 third-round pick.", 'traded', None, 'HOU'),
    ("Harrison was dealt to Houston's Texans Tuesday.", 'traded', None, 'HOU'),
    ("The Bears activated Turner (knee) from their reserve/PUP list Tuesday.", 'cleared', 'active', None),
    ("The 49ers activated Williams (knee) from the reserve/PUP list Saturday, and he no longer carries an injury designation.", 'cleared', 'active', None),
    ("The Browns signed Estime to the 53-man roster Tuesday.", 'signed', 'active', 'CLE'),
    ("Dallas signed Demercado to its 53-man roster Tuesday, Tommy Yarrish reports.", 'signed', 'active', 'DAL'),
    ("The Broncos signed Humphrey off the practice squad to the active roster Friday.", 'signed', 'active', 'DEN'),
    ("Smith signed a one-year contract with the Steelers on Wednesday.", 'signed', 'active', 'PIT'),
    ("Smith was claimed off waivers by the Dolphins on Wednesday.", 'signed', 'active', 'MIA'),
]
NOT_EVENTS = [
    # future tense, speculation, or game recaps: must not change status
    "Smith is expected to be activated from IR this week.",
    "Smith has been designated to return from IR and could be activated Saturday.",
    "Smith could be released if he doesn't return to form.",
    "Smith caught eight of nine targets for 53 yards in Sunday's 30-23 win over the Falcons.",
    "Coach Mike LaFleur has ruled out Williams (concussion) for Sunday's game against the Rams.",
    "Smith rushed 25 times for 151 yards and two touchdowns while bringing in five of six targets.",
    "Smith is likely to be placed on injured reserve.",
    "The Panthers designated Wharton (neck) to return from reserve/PUP list Tuesday.",
    "The Texans opened Ingram's (elbow) 21-day return window from injured reserve Wednesday.",
    "Lane (wrist) was a full participant in practice Wednesday after being designated for return from injured reserve.",
    "The Chargers signed Browning to the practice squad Tuesday, Noah Furtado of The Athletic reports.",
]


def test_headlines():
    bad = []
    for txt, kind, long, team in HEADLINES:
        r = parse_headline(txt)
        if r is None or r[0] != kind or r[1] != long or r[2] != team:
            bad.append(f'{txt[:70]!r}: got {r}, want {(kind, long, team)}')
    for txt in NOT_EVENTS:
        r = parse_headline(txt)
        if r is not None and r[0] is not None:
            bad.append(f'{txt[:70]!r}: should not be an event, got {r}')
    assert not bad, '\n'.join(bad)


WEEK_START = '2026-10-13T00:00'


def test_jacobs_reinstated_after_roster_snapshot():
    r = resolve([Claim('roster', '2026-10-18T13:00', long='Exempt', team='GB'),
                 Claim('sleeper', '2026-10-14T15:10', long='Exempt', team='GB'),
                 Claim('espn', '2026-10-19T20:00', long='active')], WEEK_START)
    assert final_status(r) == '' and r['team'] == 'GB', r


def test_stale_active_cannot_undo_ir():
    # Hurts: ESPN and Sleeper still said Active from before the IR move that the game-day roster shows
    r = resolve([Claim('roster', '2026-10-18T13:00', long='IR', team='PHI'),
                 Claim('sleeper', '2026-10-14T18:55', long='active', team='PHI'),
                 Claim('espn', '2026-10-11T19:21', long='active', week='')], WEEK_START)
    assert final_status(r) == 'IR', r


def test_trade_from_fresher_source():
    r = resolve([Claim('roster', '2026-10-12T13:00', long='active', team='ARI'),
                 Claim('sleeper', '2026-10-14T09:00', long='active', team='HOU')], WEEK_START)
    assert r['team'] == 'HOU' and final_status(r) == '', r


def test_release_from_headline():
    r = resolve([Claim('roster', '2026-10-12T13:00', long='active', team='NYJ'),
                 Claim('espn', '2026-10-14T16:00', long='FA', team='FA')], WEEK_START)
    assert final_status(r) == 'FA' and r['team'] == 'FA', r


def test_week_tag_only_from_this_week():
    r = resolve([Claim('roster', '2026-10-12T13:00', long='active', team='DAL'),
                 Claim('espn', '2026-10-05T10:00', week='Out'),                    # last week's news
                 Claim('injury report', '2026-10-16T16:00', week='Questionable')], WEEK_START)
    assert final_status(r) == 'Questionable', r


def test_game_day_active_clears_questionable():
    r = resolve([Claim('injury report', '2026-10-16T16:00', week='Questionable'),
                 Claim('espn', '2026-10-18T11:30', long='active', week='')], WEEK_START)
    assert final_status(r) == '', r


if __name__ == '__main__':
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('ok  ', name)
            except AssertionError as e:
                fails += 1
                print('FAIL', name, '\n', e)
    sys.exit(1 if fails else 0)
