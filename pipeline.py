"""Start/Sit pipeline.
Downloads nflverse + FantasyPros consensus, fits the model on the prior season,
tests it on the season before that, projects the next unplayed week, scores this
season so far, and writes data.js for the app.

Run:  python pipeline.py            (auto-detect season/week)
      python pipeline.py --offline  (use files already in ./cache)
"""
import os, sys, re, json, datetime as dt, urllib.request
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import kickers
import news as newsfeed
import defense as dstm

POS = ['QB', 'RB', 'WR', 'TE']
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist')
NV = 'https://github.com/nflverse/nflverse-data/releases/download'
OFFLINE = '--offline' in sys.argv
for p in (CACHE, HIST, OUT): os.makedirs(p, exist_ok=True)


def fetch(url, name, refresh=True):
    path = os.path.join(CACHE, name)
    if OFFLINE or (os.path.exists(path) and not refresh):
        return path
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'startsit-pipeline'})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, 'wb') as f:
            f.write(r.read())
    except Exception as e:
        if not os.path.exists(path):
            raise
        print('WARN using cached', name, e)
    return path


# ---------------------------------------------------------------- load
games = pd.read_csv(fetch('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv', 'games.csv'))
games = games[games.game_type == 'REG']
today = dt.date.today()
SEASON = int(games[pd.to_datetime(games.gameday).dt.date <= today + dt.timedelta(days=10)].season.max())
gs = games[games.season == SEASON]
unplayed = gs[gs.home_score.isna()]
if unplayed.empty:
    print('Season over; nothing to project'); sys.exit(0)
WEEK = int(unplayed.week.min())
SEASONS = [SEASON - 4, SEASON - 3, SEASON - 2, SEASON - 1, SEASON]
print('Projecting', SEASON, 'week', WEEK)

frames = []
for y in SEASONS:
    frames.append(pd.read_csv(fetch(f'{NV}/stats_player/stats_player_week_{y}.csv', f'w{y}.csv', refresh=y == SEASON), low_memory=False))
kd = kickers.load(frames)
d = pd.concat(frames)
d = d[(d.season_type == 'REG') & d.position.isin(POS)].copy()
for c in ['targets', 'carries', 'attempts', 'receiving_air_yards']:
    d[c] = d[c].fillna(0)
d['ppr'] = d.fantasy_points_ppr.fillna(0)
d['std'] = d.fantasy_points.fillna(0)
d['half'] = (d.ppr + d['std']) / 2
d = d[(d.targets + d.carries + d.attempts) > 0]

ids = pd.read_csv(fetch('https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv', 'ids.csv'))

# red zone usage (inside the 20) from play-by-play
rz = []
for y in SEASONS:
    p = pd.read_parquet(fetch(f'{NV}/pbp/play_by_play_{y}.parquet', f'pbp{y}.parquet', refresh=y == SEASON),
                        columns=['season', 'week', 'season_type', 'yardline_100', 'receiver_player_id', 'rusher_player_id', 'pass_attempt', 'rush_attempt'])
    p = p[(p.season_type == 'REG') & (p.yardline_100 <= 20)]
    a = p[p.pass_attempt == 1].groupby(['season', 'week', 'receiver_player_id']).size().rename('rz_tgt')
    b = p[p.rush_attempt == 1].groupby(['season', 'week', 'rusher_player_id']).size().rename('rz_car')
    a.index.names = b.index.names = ['season', 'week', 'player_id']
    rz.append(pd.concat([a, b], axis=1))
rz = pd.concat(rz).fillna(0).reset_index()
d = d.merge(rz, on=['season', 'week', 'player_id'], how='left')
d[['rz_tgt', 'rz_car']] = d[['rz_tgt', 'rz_car']].fillna(0)

# team defense scoring per game, from the same play-by-play files
DST_COLS = ['season', 'week', 'season_type', 'defteam', 'posteam', 'sack', 'interception', 'fumble_lost', 'touchdown',
            'td_team', 'safety', 'punt_blocked', 'field_goal_result', 'play_type']
pbp_d = pd.concat([pd.read_parquet(os.path.join(CACHE, f'pbp{y}.parquet'), columns=DST_COLS) for y in SEASONS])
dw, dgive = dstm.weekly(pbp_d, games[games.season.isin(SEASONS)])
del pbp_d
TEAM_NAME = {'ARI': 'Cardinals', 'ATL': 'Falcons', 'BAL': 'Ravens', 'BUF': 'Bills', 'CAR': 'Panthers', 'CHI': 'Bears',
             'CIN': 'Bengals', 'CLE': 'Browns', 'DAL': 'Cowboys', 'DEN': 'Broncos', 'DET': 'Lions', 'GB': 'Packers',
             'HOU': 'Texans', 'IND': 'Colts', 'JAX': 'Jaguars', 'KC': 'Chiefs', 'LA': 'Rams', 'LAC': 'Chargers',
             'LV': 'Raiders', 'MIA': 'Dolphins', 'MIN': 'Vikings', 'NE': 'Patriots', 'NO': 'Saints', 'NYG': 'Giants',
             'NYJ': 'Jets', 'PHI': 'Eagles', 'PIT': 'Steelers', 'SEA': 'Seahawks', 'SF': '49ers', 'TB': 'Buccaneers',
             'TEN': 'Titans', 'WAS': 'Commanders'}

# snap share
sn = []
for y in SEASONS:
    s = pd.read_csv(fetch(f'{NV}/snap_counts/snap_counts_{y}.csv', f'snaps{y}.csv', refresh=y == SEASON))
    sn.append(s[s.game_type == 'REG'][['season', 'week', 'pfr_player_id', 'offense_pct']])
sn = pd.concat(sn).merge(ids[['pfr_id', 'gsis_id']].dropna(), left_on='pfr_player_id', right_on='pfr_id')
sn = sn.groupby(['season', 'week', 'gsis_id']).offense_pct.max().rename('snap').reset_index().rename(columns={'gsis_id': 'player_id'})
d = d.merge(sn, on=['season', 'week', 'player_id'], how='left')
d['snap'] = d.snap.fillna(d.groupby('position').snap.transform('median'))

g = games[games.season.isin(SEASONS)].copy()
g['home_imp'] = g.total_line / 2 + g.spread_line / 2
g['away_imp'] = g.total_line / 2 - g.spread_line / 2
tg = pd.concat([
    g[['season', 'week', 'home_team', 'away_team', 'home_imp']].rename(columns={'home_team': 'team', 'away_team': 'opp_team', 'home_imp': 'imp'}),
    g[['season', 'week', 'away_team', 'home_team', 'away_imp']].rename(columns={'away_team': 'team', 'home_team': 'opp_team', 'away_imp': 'imp'})])
LEAGUE_IMP = float(tg.imp.mean())
roof = {}
for r in g.itertuples():
    v = 1 if r.roof in ('dome', 'closed') else 0
    roof[(r.season, r.week, r.home_team)] = roof[(r.season, r.week, r.away_team)] = v
posavg = d[d.season >= SEASON - 2].groupby(['season', 'week', 'team', 'position']).ppr.sum().groupby('position').mean()

# ---------------------------------------------------------------- as-of features
UCOLS = ['targets', 'carries', 'attempts', 'receiving_air_yards', 'rz_tgt', 'rz_car', 'snap']
K_PRIOR, K_NEW, K_USE = 4.0, 3.0, 2.5
HALF_LIFE = 5.0   # games; recent games count more (best of 3/5/8/none on the held-out season)


def asof(season, week):
    cur = d[(d.season == season) & (d.week < week)].copy()
    pri = d[d.season == season - 1]
    cur['w'] = 0.5 ** ((week - 1 - cur.week) / HALF_LIFE)
    n = cur.groupby('player_id').size()
    pos = pd.concat([pri, cur]).groupby('player_id').position.last()
    idx = pos.index
    out = pd.DataFrame(index=idx)
    out['position'] = pos
    out['n_cur'] = n.reindex(idx).fillna(0)
    nw = cur.groupby('player_id').w.sum().reindex(idx).fillna(0)
    np_ = pri.groupby('player_id').size().reindex(idx).fillna(0)
    for col in ['ppr', 'std', 'half']:
        repl = pri.groupby('player_id').agg(m=(col, 'mean'), p=('position', 'last')).groupby('p').m.quantile(.35)
        cs = (cur[col] * cur.w).groupby(cur.player_id).sum().reindex(idx).fillna(0)
        pm = pri.groupby('player_id')[col].mean().reindex(idx)
        kp = np.where(pm.notna(), K_PRIOR * np.minimum(np_, 17) / 17, K_NEW)
        prior = pm.fillna(out.position.map(repl))
        out['base_' + col] = (cs + kp * prior) / (nw + kp)
    for col in UCOLS:
        cs = (cur[col] * cur.w).groupby(cur.player_id).sum().reindex(idx).fillna(0)
        pm = pri.groupby('player_id')[col].mean().reindex(idx)
        cm = cs / nw.replace(0, np.nan)
        out['u_' + col] = (cs + K_USE * pm.fillna(cm).fillna(0)) / (nw + K_USE)
    # role trend: last 2 games snap share vs this player's blended snap share
    last2 = cur.sort_values('week').groupby('player_id').tail(2).groupby('player_id').snap.mean().reindex(idx)
    out['snap_trend'] = (last2 - out.u_snap).fillna(0)
    # volatility: coefficient of variation over last ~24 games, shrunk to position
    hist = d[((d.season == season) & (d.week < week)) | (d.season == season - 1)]
    hist = hist.sort_values(['season', 'week']).groupby('player_id').tail(24)
    st = hist.groupby('player_id').ppr.agg(['std', 'mean', 'count']).reindex(idx)
    cv = (st['std'] / st['mean'].clip(lower=3))
    poscv = cv.groupby(out.position).median()
    out['cv'] = ((cv.fillna(0) * st['count'].fillna(0) + 8 * out.position.map(poscv)) / (st['count'].fillna(0) + 8))
    return out


keys = [(s, w) for s in SEASONS[1:] for w in range(1, 19) if not (s == SEASON and w > WEEK)]
A = {k: asof(*k) for k in keys}


def defense_adj(season, week):
    rows = []
    for s, wmax, wt in [(season, week, 1.0), (season - 1, 19, 0.35)]:
        sub = d[(d.season == s) & (d.week < wmax)]
        for w, grp in sub.groupby('week'):
            if (s, w) not in A:
                continue
            m = grp.join(A[(s, w)]['base_ppr'], on='player_id')
            m['poe'] = m.ppr - m.base_ppr
            a = m.groupby(['opponent_team', 'position']).poe.sum().reset_index()
            a['wt'] = wt
            rows.append(a)
    a = pd.concat(rows)
    a['wpoe'] = a.poe * a.wt
    agg = a.groupby(['opponent_team', 'position']).agg(wpoe=('wpoe', 'sum'), n=('wt', 'sum'))
    return agg.wpoe / (agg.n + 2.0)


DA = {}
def dadj(s, w):
    if (s, w) not in DA:
        DA[(s, w)] = defense_adj(s, w)
    return DA[(s, w)]


def frame(season, week, rows):
    """attach as-of features to rows (players in games of season/week)."""
    f = rows.join(A[(season, week)].drop(columns=['position']), on='player_id')
    f = f.merge(tg[['season', 'week', 'team', 'imp']], on=['season', 'week', 'team'], how='left')
    da = dadj(season, week)
    f['dadj'] = [da.get((o, p), 0.0) for o, p in zip(f.opponent_team, f.position)]
    f['x_def'] = f.dadj / f.position.map(posavg)
    f['x_imp'] = f.imp / LEAGUE_IMP - 1
    cur = d[(d.season == season) & (d.week < week)].groupby('player_id').ppr.mean()
    pri = d[d.season == season - 1].groupby('player_id').ppr.mean()
    f['naive'] = f.player_id.map(cur).fillna(f.player_id.map(pri))
    return f


VCOLS = ['targets', 'carries', 'receiving_air_yards', 'rz_tgt', 'rz_car']
ABSORB = 0.75   # share of a missing regular's targets/carries his teammates pick up (best of 0.5/0.75/1 on the held-out season)


def vacated(season, week, rows, absent_by_team):
    """Usage left behind by regular teammates who won't play, split among the players in `rows` (same team)
    in proportion to their own share. absent_by_team: {team: [player_id, ...]}."""
    a = A[(season, week)]
    out = pd.DataFrame(0.0, index=rows.index, columns=['vac_' + c for c in VCOLS])
    src = pd.Series('', index=rows.index)
    for team, g in rows.groupby('team'):
        ab = [p for p in absent_by_team.get(team, []) if p in a.index]
        if not ab:
            continue
        ab = a.loc[ab]
        reg = ab[(ab.u_snap >= .35) | (ab.u_targets >= 3) | (ab.u_carries >= 5)]
        if reg.empty:
            continue
        act = a.reindex(g.player_id)
        here = ~g.player_id.isin(absent_by_team.get(team, [])).values      # absent players don't absorb anything
        for c in VCOLS:
            v = float(reg['u_' + c].sum())
            if v <= 0:
                continue
            elig = (act.position.isin(['RB']).values if c in ('carries', 'rz_car') else act.position.isin(['WR', 'TE', 'RB']).values) & here
            sh = act['u_' + c].fillna(0).values * elig
            if sh.sum() > 0:
                out.loc[g.index, 'vac_' + c] = v * sh / sh.sum()
        src.loc[g.index[here]] = ', '.join(reg.index)
    out['vac_src'] = src
    return out


def absorb(f):
    f = f.copy()
    for c in VCOLS:
        f['u_' + c] = f['u_' + c] + ABSORB * f['vac_' + c]
    return f


def hist_absent(season, week, rows):
    """Backtests: regulars who played in one of the team's last two games but not this one."""
    prev = d[(d.season == season) & (d.week < week)]
    now = set(rows.player_id)
    res = {}
    for team in rows.team.unique():
        tw = prev[prev.team == team]
        if tw.empty:
            continue
        last2 = sorted(tw.week.unique())[-2:]
        res[team] = [p for p in set(tw[tw.week.isin(last2)].player_id) if p not in now]
    return res


def build(seasons, wmin=3):
    out = []
    for s in seasons:
        for w in range(wmin, 19):
            if (s, w) not in A or (s == SEASON and w >= WEEK):
                continue
            rows = d[(d.season == s) & (d.week == w)][['season', 'week', 'player_id', 'player_display_name', 'position', 'team', 'opponent_team', 'ppr', 'std', 'half']]
            f_ = frame(s, w, rows)
            out.append(absorb(pd.concat([f_, vacated(s, w, f_, hist_absent(s, w, f_))], axis=1)))
    f = pd.concat(out)
    return f[f.base_ppr >= 4].dropna(subset=['imp', 'naive'])


USE = {
    'QB': ['u_attempts', 'u_carries', 'u_rz_car'],
    'RB': ['u_carries', 'u_targets', 'u_rz_car', 'u_rz_tgt'],
    'WR': ['u_targets', 'u_receiving_air_yards', 'u_rz_tgt'],
    'TE': ['u_targets', 'u_receiving_air_yards', 'u_rz_tgt'],
}


class Model:
    def fit(self, f):
        self.W, self.C, self.Q = {}, {}, {}
        f = f.copy()
        for p in POS:
            a = f.position == p
            X = f.loc[a, USE[p]].values
            Xs = np.c_[X, f.loc[a, 'u_targets'] * f.loc[a, 'snap_trend']] if p != 'QB' else X
            self.W[p] = lstsq(Xs, f.loc[a, 'ppr'], rcond=None)[0]
            f.loc[a, 'xfp'] = Xs @ self.W[p]
            X3 = np.c_[f.loc[a, 'base_ppr'], f.loc[a, 'xfp'], f.loc[a, 'base_ppr'] * f.loc[a, 'x_imp']]
            self.C[p] = lstsq(X3, f.loc[a, 'ppr'], rcond=None)[0]
            f.loc[a, 'res'] = f.loc[a, 'ppr'] - X3 @ self.C[p]
        mx = (f.base_ppr * f.x_def).values
        self.cm = max(0.0, float(np.dot(mx, f.res) / np.dot(mx, mx)))
        f = self.apply(f)
        # range width: scale per-player cv so ~60% of outcomes land in [lo, hi]
        best = (1e9, 1.0)
        for m in np.arange(.3, 1.6, .02):
            lo, hi = self.range(f.proj, f.cv, m)
            cov = ((f.ppr >= lo) & (f.ppr <= hi)).mean()
            if abs(cov - .6) < best[0]:
                best = (abs(cov - .6), m)
        self.rm = best[1]
        return self

    def xfp(self, f, p):
        X = f[USE[p]].values
        Xs = np.c_[X, f['u_targets'] * f['snap_trend']] if p != 'QB' else X
        return Xs @ self.W[p]

    def apply(self, f):
        f = f.copy()
        for p in POS:
            a = f.position == p
            if not a.any():
                continue
            c = self.C[p]
            f.loc[a, 'xfp'] = self.xfp(f[a], p)
            f.loc[a, 'c_form'] = c[0] * f.loc[a, 'base_ppr']
            f.loc[a, 'c_usage'] = c[1] * f.loc[a, 'xfp']
            f.loc[a, 'c_vegas'] = c[2] * f.loc[a, 'base_ppr'] * f.loc[a, 'x_imp']
            f.loc[a, 'c_match'] = self.cm * f.loc[a, 'base_ppr'] * f.loc[a, 'x_def']
        f['proj'] = (f.c_form + f.c_usage + f.c_vegas + f.c_match).clip(lower=0)
        return f

    @staticmethod
    def range(proj, cv, m):
        return proj * (1 - m * cv).clip(lower=.1), proj * (1 + m * cv)


def pairs(f, col, close=None):
    hits = tot = 0
    for _, grp in f.groupby(['season', 'week', 'position']):
        v, a, nv = grp[col].values, grp.ppr.values, grp.naive.values
        i, j = np.triu_indices(len(v), 1)
        mk = (v[i] != v[j]) & (a[i] != a[j])
        if close:
            mk &= np.abs(nv[i] - nv[j]) < close
        hits += np.sum(((v[i] > v[j]) == (a[i] > a[j]))[mk])
        tot += mk.sum()
    return hits / max(tot, 1), int(tot)


# ---------------------------------------------------------------- backtest: fit on S-2, test on S-1
train, test = build([SEASON - 3, SEASON - 2]), build([SEASON - 1])
mt = Model().fit(train)
te = mt.apply(test)
bt = dict(train=f'{SEASON - 3}–{SEASON - 2}', test=SEASON - 1,
          mae_naive=float(np.mean(abs(te.ppr - te.naive))), mae_model=float(np.mean(abs(te.ppr - te.proj))),
          pair_naive=pairs(te, 'naive')[0] * 100, pair_model=pairs(te, 'proj')[0] * 100,
          close_naive=pairs(te, 'naive', 3)[0] * 100, close_model=pairs(te, 'proj', 3)[0] * 100,
          bypos={p: [pairs(te[te.position == p], 'naive', 3)[0] * 100, pairs(te[te.position == p], 'proj', 3)[0] * 100] for p in POS})
lo, hi = Model.range(te.proj, te.cv, mt.rm)
bt['range_cover'] = float(((te.ppr >= lo) & (te.ppr <= hi)).mean() * 100)
print('backtest', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in bt.items() if k != 'bypos'})

# ---------------------------------------------------------------- final model: fit on S-2 and S-1
model = Model().fit(pd.concat([train, test]))

# ---------------------------------------------------------------- kickers: fit on S-2, test on S-1, then refit on both
def kbuild(s):
    out_ = []
    for w in range(3, 19):
        rows_ = kd[(kd.season == s) & (kd.week == w)][['season', 'week', 'player_id', 'team', 'kpts']]
        if len(rows_):
            out_.append(kickers.frame(kd, tg, roof, s, w, rows_))
    f_ = pd.concat(out_).dropna(subset=['imp', 'naive', 'base'])
    return f_.assign(position='K', ppr=f_.kpts)


ktr, kte = kbuild(SEASON - 2), kbuild(SEASON - 1)
kte = kte.assign(proj=kickers.KModel().fit(ktr, LEAGUE_IMP).predict(kte))
bt['bypos']['K'] = [pairs(kte, 'naive', 3)[0] * 100, pairs(kte, 'proj', 3)[0] * 100]
print('kicker backtest', [round(x, 1) for x in bt['bypos']['K']])
kmodel = kickers.KModel().fit(pd.concat([ktr, kte]), LEAGUE_IMP)
kicker_asof = kickers.asof(kd, SEASON, WEEK)

# ---------------------------------------------------------------- team defenses: fit on S-2, test on S-1, then refit on both
LEAGUE_GIVE = float(dgive.give.mean())


def dbuild(s):
    out_ = []
    for w in range(3, 19):
        r_ = dw[(dw.season == s) & (dw.week == w)][['season', 'week', 'team', 'opp', 'dpts']]
        if len(r_):
            out_.append(dstm.frame(dw, dgive, tg, s, w, r_))
    f_ = pd.concat(out_).dropna(subset=['opp_imp', 'naive'])
    return f_.assign(position='DST', ppr=f_.dpts)


dtr, dte = dbuild(SEASON - 2), dbuild(SEASON - 1)
dte = dte.assign(proj=dstm.DModel().fit(dtr, LEAGUE_IMP, LEAGUE_GIVE).predict(dte))
bt['bypos']['DST'] = [pairs(dte, 'naive', 3)[0] * 100, pairs(dte, 'proj', 3)[0] * 100]
print('defense backtest', [round(x, 1) for x in bt['bypos']['DST']])
dmodel = dstm.DModel().fit(pd.concat([dtr, dte]), LEAGUE_IMP, LEAGUE_GIVE)
dst_asof = dstm.asof(dw, dgive, SEASON, WEEK)

# ---------------------------------------------------------------- live season scoring (as-of projections, no leakage)
live = None
if WEEK > 2:
    cur = build([SEASON], wmin=2)
    if len(cur):
        cur = model.apply(cur)
        live = dict(weeks=[int(cur.week.min()), int(cur.week.max())],
                    close_naive=pairs(cur, 'naive', 3)[0] * 100, close_model=pairs(cur, 'proj', 3)[0] * 100,
                    n_close=pairs(cur, 'proj', 3)[1],
                    mae_naive=float(np.mean(abs(cur.ppr - cur.naive))), mae_model=float(np.mean(abs(cur.ppr - cur.proj))))

# expert comparison from saved weekly snapshots (history/ecr_<season>_<week>.csv)
expert = None
ex_rows = []
for fn in sorted(os.listdir(HIST)):
    if not fn.startswith(f'ecr_{SEASON}_'):
        continue
    w = int(fn.split('_')[2].split('.')[0])
    if w >= WEEK:
        continue
    e = pd.read_csv(os.path.join(HIST, fn))
    act = d[(d.season == SEASON) & (d.week == w)][['player_id', 'ppr']]
    e = e.merge(act, left_on='id', right_on='player_id').dropna(subset=['fp_pts'])
    e['season'], e['week'], e['position'] = SEASON, w, e.pos
    ex_rows.append(e)
if ex_rows:
    e = pd.concat(ex_rows)
    e['naive'] = e.model  # only used for "close" filter: use model-vs-expert disagreement set
    expert = dict(weeks=sorted(set(int(x) for x in e.week)),
                  mae_model=float(np.mean(abs(e.ppr - e.model))), mae_expert=float(np.mean(abs(e.ppr - e.fp_pts))),
                  pair_model=pairs(e, 'model')[0] * 100, pair_expert=pairs(e, 'fp_pts')[0] * 100, n=int(len(e)))

# ---------------------------------------------------------------- project the current week
# The fantasy week runs Thursday to Monday. Every game in WEEK is shown, including ones
# already played (marked final, with actual points), so no team drops out mid-week.
cur = d[d.season == SEASON]
wk_games = games[(games.season == SEASON) & (games.week == WEEK)]
kick, final = {}, {}
for _, r in wk_games.iterrows():
    done = not pd.isna(r.home_score)
    kick[r.home_team] = (r.away_team, 1, r.gameday, r.gametime, r.weekday)
    kick[r.away_team] = (r.home_team, 0, r.gameday, r.gametime, r.weekday)
    final[r.home_team] = final[r.away_team] = done
all_teams = set(tg[tg.season == SEASON].team)
bye_teams = all_teams - set(kick)

# player universe = current NFL rosters (right team after trades; injured reserve known),
# not just "who has recorded a stat", so returning and traded players aren't missed
ro = pd.read_csv(fetch(f'{NV}/weekly_rosters/roster_weekly_{SEASON}.csv', f'roster{SEASON}.csv'), low_memory=False)
ro = ro[ro.position.isin(POS + ['K'])].sort_values('week').groupby('gsis_id').last().reset_index()
ro = ro[ro.status.isin(['ACT', 'INA', 'RES', 'EXE'])]
stat_names = pd.concat([d, kd]).sort_values(['season', 'week']).groupby('player_id').player_display_name.last()
ro['pname'] = ro.gsis_id.map(stat_names).fillna(ro.full_name)
ON_IR = set(ro[ro.status == 'RES'].gsis_id)
EXEMPT = set(ro[ro.status == 'EXE'].gsis_id)
universe = ro.rename(columns={'gsis_id': 'player_id'})[['player_id', 'pname', 'position', 'team']]
universe = universe[universe.team.isin(all_teams)]

# injuries: latest report this week
inj = pd.read_csv(fetch(f'{NV}/injuries/injuries_{SEASON}.csv', f'inj{SEASON}.csv'))
iw = inj[(inj.season == SEASON) & (inj.week == WEEK)]
status = dict(zip(iw.gsis_id, iw.report_status.fillna('')))
for pid in ON_IR:
    if status.get(pid, '') in ('', 'Questionable'):
        status[pid] = 'IR'
for pid in EXEMPT:
    status[pid] = 'Exempt'   # league exempt/suspension list: not eligible to play
practice = dict(zip(iw.gsis_id, iw.practice_status.fillna('')))
injury = dict(zip(iw.gsis_id, iw.report_primary_injury.fillna('')))
OUT_SET = ('Out', 'IR', 'Exempt', 'Suspended', 'FA')

# live news and status (ESPN every run, Sleeper daily): fresher than the weekly official report
NEWS = newsfeed.load(CACHE, offline=OFFLINE)
espn2gsis = {}
for e_, g_ in zip(ids.espn_id, ids.gsis_id):
    if pd.notna(e_) and isinstance(g_, str):
        espn2gsis[str(int(e_))] = g_
for g_, v in NEWS['sleeper'].items():
    if v.get('espn_id'):
        espn2gsis.setdefault(v['espn_id'], g_)
SEVERITY = {'': 0, 'Questionable': 1, 'Doubtful': 2, 'Out': 3, 'Suspended': 4, 'IR': 5, 'Exempt': 6}
news_by = {}
for eid, v in NEWS['espn'].items():
    g_ = espn2gsis.get(eid)
    if not g_:
        continue
    news_by[g_] = v
    if SEVERITY.get(v['status'], 0) > SEVERITY.get(status.get(g_, ''), 0):
        status[g_] = v['status']
    if v['injury'] and not injury.get(g_):
        injury[g_] = v['injury']
print('statuses after news:', pd.Series(list(status.values())).value_counts().to_dict())
prev_wk = pd.concat([cur, kd[kd.season == SEASON]])
prev_wk = prev_wk[prev_wk.week == WEEK - 1]
played_prev, teams_prev = set(prev_wk.player_id), set(prev_wk.team)
played_season = set(cur.player_id) | set(kd[kd.season == SEASON].player_id)

# actual points for games already played this week
act_rows = cur[cur.week == WEEK].set_index('player_id')
act = {pid: (float(r.ppr), float(r.half), float(r['std'])) for pid, r in act_rows.iterrows()}
for pid, v in kd[(kd.season == SEASON) & (kd.week == WEEK)].set_index('player_id').kpts.items():
    act[pid] = (float(v),) * 3
for t_, v in dw[(dw.season == SEASON) & (dw.week == WEEK)].set_index('team').dpts.items():
    act['DST-' + t_] = (float(v),) * 3

# expert consensus
fp = pd.read_csv(fetch('https://raw.githubusercontent.com/dynastyprocess/data/master/files/fp_latest_weekly.csv', 'fp.csv'))
fp = fp[fp.page.isin(['qb', 'ppr-rb', 'ppr-wr', 'ppr-te', 'k', 'dst'])]
fmap = ids[['fantasypros_id', 'gsis_id']].dropna()
fmap['fantasypros_id'] = fmap.fantasypros_id.astype(int)
fp = fp.merge(fmap, on='fantasypros_id', how='left')
# the ID table misses some players (often rookies and kickers): fall back to name + position
def _norm(n):
    n = re.sub(r"[^a-z ]", "", str(n).lower().replace('-', ' '))
    return re.sub(r" (jr|sr|ii|iii|iv|v)$", "", n).strip()
_by_name = {(_norm(r.pname), r.position): r.player_id for r in universe.itertuples()}
_miss = fp.gsis_id.isna()
fp.loc[_miss, 'gsis_id'] = [_by_name.get((_norm(n), p_)) for n, p_ in zip(fp.loc[_miss, 'player_name'], fp.loc[_miss, 'pos'])]
# second fallback for nicknames (Matt / Matthew): first initial + last name + position + team
_TEAM = {'LAR': 'LA', 'JAC': 'JAX', 'WSH': 'WAS', 'LVR': 'LV'}
def _k2(n, p_, t):
    w = _norm(n).split()
    return (w[0][:1], w[-1], p_, _TEAM.get(t, t)) if w else None
_by_k2 = {}
for r in universe.itertuples():
    _by_k2.setdefault(_k2(r.pname, r.position, r.team), []).append(r.player_id)
_miss = fp.gsis_id.isna()
fp.loc[_miss, 'gsis_id'] = [(lambda c: c[0] if c and len(c) == 1 else None)(_by_k2.get(_k2(n, p_, t)))
                            for n, p_, t in zip(fp.loc[_miss, 'player_name'], fp.loc[_miss, 'pos'], fp.loc[_miss, 'team'])]
# FantasyPros posts the new week's projections a day or two after Monday night. Until then the file
# still holds last week's numbers, so keep only rows whose kickoff falls inside the week being projected.
fp_all = fp.copy()
_gd = pd.to_datetime(wk_games.gameday)
_lo, _hi = _gd.min() - pd.Timedelta(days=1), _gd.max() + pd.Timedelta(days=2)
_ko = pd.to_datetime(fp.player_game_kickoff_ts, unit='s', errors='coerce')
fp = fp[(_ko >= _lo) & (_ko <= _hi)]
print('expert rows for this week:', len(fp), 'of', len(fp_all))
_dst = fp.pos == 'DST'
fp.loc[_dst, 'gsis_id'] = ['DST-' + _TEAM.get(t, t) for t in fp.loc[_dst, 'team']]
fp_rank = dict(zip(fp.gsis_id, fp.pos_rank))
fp_pts = dict(zip(fp.gsis_id, fp.r2p_pts))
fp_fresh = str(fp.scrape_date.max()) if len(fp) else None


def ecr_num(er):
    m = re.search(r'(\d+)$', er) if isinstance(er, str) else None
    return int(m.group(1)) if m else None


# skill players
sk = universe[universe.position.isin(POS) & universe.team.isin(kick)]
rows = [dict(season=SEASON, week=WEEK, player_id=r.player_id, player_display_name=r.pname, position=r.position,
             team=r.team, opponent_team=kick[r.team][0]) for r in sk.itertuples()]
# teammates who won't play: ruled out / on IR / suspended / exempt, or no longer on that team's roster
on_team = dict(zip(universe.player_id, universe.team))
recent_regulars = {}
for t_, tw in cur.groupby('team'):
    last2 = sorted(tw.week.unique())[-2:]
    recent_regulars[t_] = set(tw[tw.week.isin(last2)].player_id)
LONG_OUT = ('IR', 'Exempt', 'Suspended')
absent_now = {t_: [p_ for p_ in ps if status.get(p_, '') in OUT_SET or on_team.get(p_) != t_]
              for t_, ps in recent_regulars.items()}
absent_long = {t_: [p_ for p_ in ps if status.get(p_, '') in LONG_OUT or on_team.get(p_) != t_]
               for t_, ps in recent_regulars.items()}
fwk = frame(SEASON, WEEK, pd.DataFrame(rows)).dropna(subset=['base_ppr', 'imp'])
fwk = absorb(pd.concat([fwk, vacated(SEASON, WEEK, fwk, absent_now)], axis=1))
up = model.apply(fwk)
name_of = dict(zip(universe.player_id, universe.pname))
name_of.update({k: v for k, v in stat_names.items() if k not in name_of})
up['kind'] = 'skill'
no_hist = sk[~sk.player_id.isin(up.player_id)]

# kickers
ka = kicker_asof
kk = universe[(universe.position == 'K') & universe.team.isin(kick)]
krows = pd.DataFrame([dict(season=SEASON, week=WEEK, player_id=r.player_id, team=r.team) for r in kk.itertuples()])
kf = kickers.frame(kd, tg, roof, SEASON, WEEK, krows).dropna(subset=['base', 'imp']) if len(krows) else pd.DataFrame()
if len(kf):
    kf['proj'] = kmodel.predict(kf)
    kf['c_form'], kf['c_vegas'], kf['c_match'] = kmodel.parts(kf)
    kf['position'] = 'K'

out = []


def sleeper_note(pid):
    v = NEWS['sleeper'].get(pid) or {}
    note = ', '.join(x for x in [v.get('body'), v.get('notes')] if x)
    return f'Injury: {note}.' if note and v.get('body') not in ('Personal',) else None


def base_entry(pid, name, pos, team):
    o, h, day, tm, wd = kick.get(team, (None, 0, None, None, None))
    stt = status.get(pid, '')
    a = act.get(pid) if final.get(team) else None
    er = fp_rank.get(pid)
    return dict(id=pid, name=name, pos=pos, team=team, opp=o, home=h, day=wd, date=day, time=tm,
                final=bool(final.get(team, False)), bye=team in bye_teams,
                act=None if a is None else round(a[0], 1), act_h=None if a is None else round(a[1], 1), act_s=None if a is None else round(a[2], 1),
                ecr=ecr_num(er), fp=None if pd.isna(fp_pts.get(pid, np.nan)) else float(fp_pts[pid]),
                status=stt, practice=practice.get(pid, ''), injury=injury.get(pid, ''),
                miss=bool(pos != 'DST' and pid not in played_prev and team in teams_prev and team not in bye_teams),
                new=bool(pos != 'DST' and pid not in played_season and WEEK >= 3),
                news=(news_by.get(pid) or {}).get('news') or sleeper_note(pid), news_date=(news_by.get(pid) or {}).get('date') or None,
                ret=(news_by.get(pid) or {}).get('ret'), adds=NEWS['trending'].get(pid))


for r in up.itertuples():
    e = base_entry(r.player_id, r.player_display_name, r.position, r.team)
    proj = 0.0 if e['status'] in OUT_SET else float(r.proj)
    if r.proj < 1.5 and e['ecr'] is None:
        continue
    lo_, hi_ = Model.range(pd.Series([proj]), pd.Series([r.cv]), model.rm)
    sc_std = r.base_std / r.base_ppr if r.base_ppr > 0 else 1
    sc_half = r.base_half / r.base_ppr if r.base_ppr > 0 else 1
    e.update(proj=round(proj, 2), std=round(proj * sc_std, 2), half=round(proj * sc_half, 2),
             lo=round(float(lo_.iloc[0]), 1), hi=round(float(hi_.iloc[0]), 1),
             form=round(r.c_form, 2), usage=round(r.c_usage, 2), matchup=round(r.c_match, 2), vegas=round(r.c_vegas, 2),
             base=round(r.base_ppr, 1), xfp=round(r.xfp, 1), imp=round(r.imp, 1), xdef=round(r.x_def * 100, 1),
             tgt=round(r.u_targets, 1), car=round(r.u_carries, 1), att=round(r.u_attempts, 1),
             rz=round(r.u_rz_tgt + r.u_rz_car, 1), snap=round(r.u_snap * 100), trend=round(r.snap_trend * 100),
             cv=round(r.cv, 2), gp=int(r.n_cur),
             vac_tgt=round(ABSORB * r.vac_targets, 1), vac_car=round(ABSORB * r.vac_carries, 1),
             vac_from=[name_of.get(x, x) for x in r.vac_src.split(', ')] if r.vac_src else [])
    out.append(e)

# no NFL stats yet (rookies, longtime backups): use the expert projection if experts rank them
for r in no_hist.itertuples():
    x = fp_pts.get(r.player_id)
    if x is None or pd.isna(x):
        continue
    e = base_entry(r.player_id, r.pname, r.position, r.team)
    proj = 0.0 if e['status'] in OUT_SET else float(x)
    e.update(proj=round(proj, 2), std=round(proj * .8, 2), half=round(proj * .9, 2), lo=round(proj * .4, 1), hi=round(proj * 1.6, 1),
             form=round(proj, 2), usage=0.0, matchup=0.0, vegas=0.0, base=0.0, xfp=0.0, imp=0.0, xdef=0.0,
             tgt=0, car=0, att=0, rz=0, snap=0, trend=0, cv=0.6, gp=0, src='experts')
    out.append(e)

for r in (kf.itertuples() if len(kf) else []):
    name = universe.set_index('player_id').pname.get(r.player_id, r.player_id)
    e = base_entry(r.player_id, name, 'K', r.team)
    proj = 0.0 if e['status'] in OUT_SET else float(r.proj)
    lo_, hi_ = kickers.KModel.range(pd.Series([proj]), pd.Series([r.cv]), kmodel.rm)
    e.update(proj=round(proj, 2), std=round(proj, 2), half=round(proj, 2),
             lo=round(float(lo_.iloc[0]), 1), hi=round(float(hi_.iloc[0]), 1),
             form=round(r.c_form, 2), usage=0.0, matchup=round(r.c_match, 2), vegas=round(r.c_vegas, 2),
             base=round(r.base, 1), xfp=round(r.base, 1), imp=round(r.imp, 1), xdef=0.0,
             tgt=0, car=0, att=0, rz=0, snap=0, trend=0, cv=round(r.cv, 2), gp=int(r.n_cur),
             fga=round(r.fga, 1), pata=round(r.pata, 1), fg50=round(r.fg50, 2), dome=int(r.dome))
    out.append(e)

# team defenses
drows = pd.DataFrame([dict(season=SEASON, week=WEEK, team=t_, opp=kick[t_][0]) for t_ in sorted(kick)])
dfw = dstm.frame(dw, dgive, tg, SEASON, WEEK, drows).dropna(subset=['opp_imp'])
dfw['proj'] = dmodel.predict(dfw)
dfw['c_base'], dfw['c_vegas'], dfw['c_match'] = dmodel.parts(dfw)
for r in dfw.itertuples():
    e = base_entry('DST-' + r.team, TEAM_NAME.get(r.team, r.team) + ' D/ST', 'DST', r.team)
    proj = float(r.proj)
    e.update(proj=round(proj, 2), std=round(proj, 2), half=round(proj, 2),
             lo=round(proj - .84 * dmodel.sd, 1), hi=round(proj + .84 * dmodel.sd, 1),
             form=round(r.c_base, 2), usage=0.0, matchup=round(r.c_match, 2), vegas=round(r.c_vegas, 2),
             base=round(r.base, 1), xfp=round(r.base, 1), imp=round(r.opp_imp, 1), xdef=0.0,
             tgt=0, car=0, att=0, rz=0, snap=0, trend=0, cv=round(r.cv, 2), gp=int(r.n_cur),
             sacks=round(r.sacks, 1), takeaways=round(r.ints + r.fum, 1), pa=round(r.pa, 1),
             opp_give=round(r.opp_give, 1))
    out.append(e)
for t_ in sorted(bye_teams):
    e = base_entry('DST-' + t_, TEAM_NAME.get(t_, t_) + ' D/ST', 'DST', t_)
    e.update(proj=0.0, std=0.0, half=0.0, lo=0.0, hi=0.0, form=0.0, usage=0.0, matchup=0.0, vegas=0.0,
             base=round(float(dst_asof.base.get(t_, 0)), 1), xfp=0.0, imp=0.0, xdef=0.0, tgt=0, car=0, att=0,
             rz=0, snap=0, trend=0, cv=0.0, gp=0)
    out.append(e)

# notable free agents (not on any NFL roster), from Sleeper's player file
in_app = {e['id'] for e in out} | set(universe.player_id)
abase_fa = A[(SEASON, WEEK)]
for v in NEWS['sleeper'].values():
    g_ = v.get('gsis_id')
    if not g_ or v.get('team') or not v.get('active') or v['pos'] not in POS + ['K'] or v['rank'] > 300 or g_ in in_app:
        continue
    b_ = abase_fa.base_ppr.get(g_)
    if b_ is None or pd.isna(b_) or b_ < 6:
        continue
    note = ', '.join(x for x in [v.get('body'), v.get('notes')] if x)
    e = base_entry(g_, name_of.get(g_, v['name']), v['pos'], 'FA')
    e.update(status='FA', proj=0.0, std=0.0, half=0.0, lo=0.0, hi=0.0, form=0.0, usage=0.0, matchup=0.0, vegas=0.0,
             base=round(float(b_), 1), xfp=0.0, imp=0.0, xdef=0.0, tgt=0, car=0, att=0, rz=0, snap=0, trend=0, cv=0.0, gp=0,
             miss=False, new=False, injury=v.get('body') or '',
             news=e['news'] or ('Not on an NFL roster.' + (f' Last injury: {note}.' if note else '')))
    out.append(e)

# players on bye: listed (so they can be found) but not projected this week
abase = A[(SEASON, WEEK)]
kbase = kicker_asof
for r in universe[universe.team.isin(bye_teams)].itertuples():
    b = (kbase.base.get(r.player_id) if r.position == 'K' else abase.base_ppr.get(r.player_id))
    if b is None or pd.isna(b) or b < (5 if r.position == 'K' else 6):
        continue
    e = base_entry(r.player_id, r.pname, r.position, r.team)
    e.update(proj=0.0, std=0.0, half=0.0, lo=0.0, hi=0.0, form=0.0, usage=0.0, matchup=0.0, vegas=0.0,
             base=round(float(b), 1), xfp=0.0, imp=0.0, xdef=0.0, tgt=0, car=0, att=0, rz=0, snap=0, trend=0,
             cv=0.0, gp=0)
    out.append(e)

out.sort(key=lambda x: -x['proj'])
rk = {}
for e in out:
    if e['bye'] or e['status'] in OUT_SET:
        e['rank'] = None
        continue
    rk[e['pos']] = rk.get(e['pos'], 0) + 1
    e['rank'] = rk[e['pos']]
print('this week:', len(out), 'players |', sum(e['final'] for e in out), 'final |', sum(e['bye'] for e in out), 'on bye |',
      sum(e['pos'] == 'K' for e in out), 'kickers')


da = dadj(SEASON, WEEK)
raw = d[(d.season == SEASON) & (d.week < WEEK)].groupby(['opponent_team', 'position', 'week']).ppr.sum().groupby(['opponent_team', 'position']).mean()
teams = sorted(set(tg[tg.season == SEASON].team))
defense = {}
for p in POS:
    v = np.array([da.get((t, p), 0.0) for t in teams])
    m, s = v.mean(), v.std() or 1
    defense[p] = sorted([dict(team=t, adj=round(float(x), 2), z=round(float((x - m) / s), 2), raw=round(float(raw.get((t, p), 0)), 1))
                         for t, x in zip(teams, v)], key=lambda r: -r['z'])


# ---------------------------------------------------------------- rest of season (waivers)
FUTURE_MATCH = 0.5   # future matchups count half: defenses drift over a season


def team_strength(season, week):
    """expected implied points per team: this season's lines so far, blended with last season"""
    cur = tg[(tg.season == season) & (tg.week < week)].groupby('team').imp.agg(['sum', 'count'])
    pri = tg[tg.season == season - 1].groupby('team').imp.mean()
    out = {}
    for t in set(tg[tg.season == season].team):
        s_, n_ = (cur.loc[t, 'sum'], cur.loc[t, 'count']) if t in cur.index else (0.0, 0)
        out[t] = (s_ + 4 * pri.get(t, LEAGUE_IMP)) / (n_ + 4)
    return out


def ros(season, week, mdl, players, absent=None):
    """players: frame with player_id, position, team. Returns dict id -> list of weekly projections week..18 (None on bye)."""
    a = A[(season, week)]
    da = dadj(season, week)
    ts = team_strength(season, week)
    sched = {}
    for _, r in games[(games.season == season) & (games.week >= week)].iterrows():
        sched[(r.home_team, r.week)] = r.away_team
        sched[(r.away_team, r.week)] = r.home_team
    f = players.join(a.drop(columns=['position']), on='player_id').dropna(subset=['base_ppr'])
    out = {}
    for p in POS:
        sub = f[f.position == p]
        if sub.empty:
            continue
        c = mdl.C[p]
        if absent is not None:
            sub = absorb(pd.concat([sub, vacated(season, week, sub, absent)], axis=1))
        core = c[0] * sub.base_ppr + c[1] * mdl.xfp(sub, p)
        for (_, r), cv in zip(sub.iterrows(), core):
            wk = []
            for w in range(week, 19):
                o = sched.get((r.team, w))
                if o is None:
                    wk.append(None); continue
                x_imp = ts.get(r.team, LEAGUE_IMP) / LEAGUE_IMP - 1
                x_def = da.get((o, p), 0.0) / posavg[p] * FUTURE_MATCH
                wk.append(max(0.0, cv + c[2] * r.base_ppr * x_imp + mdl.cm * r.base_ppr * x_def))
            out[r.player_id] = (wk, float(cv))
    return out


# backtest: at several points in the test season, does the rest-of-season projection
# beat "season average so far" at ranking two similar players over the rest of the year?
rb = []
for w in [4, 6, 8, 10, 12]:
    pl = d[(d.season == SEASON - 1) & (d.week < w)].groupby('player_id').agg(position=('position', 'last'), team=('team', 'last'))
    pl = pl.reset_index()
    pr = ros(SEASON - 1, w, mt, pl)
    fut = d[(d.season == SEASON - 1) & (d.week >= w)].groupby('player_id').ppr.agg(['mean', 'count'])
    so_far = d[(d.season == SEASON - 1) & (d.week < w)].groupby('player_id').ppr.mean()
    for pid, (wk, _) in pr.items():
        if pid not in fut.index or fut.loc[pid, 'count'] < 3:
            continue
        g_ = [x for x in wk if x is not None]
        rb.append(dict(season=SEASON - 1, week=w, position=pl.set_index('player_id').position[pid],
                       proj=float(np.mean(g_)), naive=float(so_far.get(pid, np.nan)), ppr=float(fut.loc[pid, 'mean'])))
rb = pd.DataFrame(rb).dropna()
rb = rb[rb.naive >= 4]
ros_bt = dict(close_naive=pairs(rb, 'naive', 3)[0] * 100, close_model=pairs(rb, 'proj', 3)[0] * 100,
              mae_naive=float(np.mean(abs(rb.ppr - rb.naive))), mae_model=float(np.mean(abs(rb.ppr - rb.proj))), n=int(len(rb)))
ros_bt['sd'] = float(np.std(rb.ppr - rb.proj))
print('ROS backtest', {k: round(v, 2) for k, v in ros_bt.items()})

# project every rostered-caliber player, including teams on bye this week
# last-known rostered %: players on bye or injured drop out of the weekly rankings, so remember them
OWN_PATH = os.path.join(HIST, 'ownership.csv')
oh = pd.read_csv(OWN_PATH) if os.path.exists(OWN_PATH) else pd.DataFrame(columns=['fantasypros_id', 'player_owned_avg', 'date'])
now = fp_all[['fantasypros_id', 'player_owned_avg']].dropna().assign(date=str(today))
oh = pd.concat([oh, now]).sort_values('date').groupby('fantasypros_id').last().reset_index()
oh.to_csv(OWN_PATH, index=False)
oh = oh.merge(fmap, on='fantasypros_id', how='inner')
own = dict(zip(oh.gsis_id, oh.player_owned_avg))
_fa = fp_all.copy()
_fa.loc[_fa.pos == 'DST', 'gsis_id'] = ['DST-' + _TEAM.get(t, t) for t in _fa.loc[_fa.pos == 'DST', 'team']]
own.update({g_: v_ for g_, v_ in zip(_fa.gsis_id, _fa.player_owned_avg) if isinstance(g_, str) and not pd.isna(v_)})
pool = universe[universe.position.isin(POS)]
pr = ros(SEASON, WEEK, model, pool[['player_id', 'position', 'team']], absent=absent_long)
cvmap = A[(SEASON, WEEK)].cv
inj_out = {k for k, v in status.items() if v == 'Out'}
IR_WEEKS = 4   # minimum stay on injured reserve; flagged so the user can check the real timeline
waiver = []


# game date for each team-week, so an expected return date can zero out exactly the right weeks
team_week_date = {}
for r in games[(games.season == SEASON) & (games.week >= WEEK)].itertuples():
    team_week_date[(r.home_team, r.week)] = team_week_date[(r.away_team, r.week)] = r.gameday
pid_team = dict(zip(universe.player_id, universe.team))


def zero_out(wk, pid):
    """Weeks a player is expected to miss count as zero. Uses ESPN's expected return date when there is one."""
    ret = (news_by.get(pid) or {}).get('ret')
    st = status.get(pid, '')
    if ret and st in OUT_SET + ('Doubtful',):
        t_ = pid_team.get(pid)
        return [x if x is None or (team_week_date.get((t_, WEEK + i)) or '9999') >= ret else 0.0 for i, x in enumerate(wk)]
    if pid in EXEMPT:
        return [None if x is None else 0.0 for x in wk]
    if pid in ON_IR or st == 'IR':
        n = 0
        for i, x in enumerate(wk):
            if x is not None and n < IR_WEEKS:
                wk[i] = 0.0
                n += 1
    elif st in ('Out', 'Suspended') and wk and wk[0] is not None:
        wk[0] = 0.0
    return wk


for r in pool.itertuples():
    if r.player_id not in pr:
        continue
    wk, core = pr[r.player_id]
    wk = zero_out(list(wk), r.player_id)
    tot = sum(x for x in wk if x is not None)
    if core < 3.5:
        continue
    a_ = A[(SEASON, WEEK)].loc[r.player_id]
    o_ = own.get(r.player_id)
    waiver.append(dict(id=r.player_id, name=r.pname, pos=r.position, team=r.team,
                       wk=[None if x is None else round(x, 1) for x in wk], ros=round(tot, 1),
                       ppg=round(core, 1), cv=round(float(cvmap.get(r.player_id, .6)), 2),
                       trend=round(float(a_.snap_trend) * 100), snap=round(float(a_.u_snap) * 100),
                       own=None if o_ is None or pd.isna(o_) else round(float(o_)),
                       rs=round(float(a_.base_std / a_.base_ppr), 3) if a_.base_ppr > 0 else 1, rh=round(float(a_.base_half / a_.base_ppr), 3) if a_.base_ppr > 0 else 1,
                       status=status.get(r.player_id, ''), adds=NEWS['trending'].get(r.player_id),
                       news=(news_by.get(r.player_id) or {}).get('news')))

# kickers, rest of season: baseline scaled by each team's expected scoring, plus dome games
ts_ = team_strength(SEASON, WEEK)
fut = games[(games.season == SEASON) & (games.week >= WEEK)]
sched_k = {}
for r in fut.itertuples():
    sched_k[(r.home_team, r.week)] = sched_k[(r.away_team, r.week)] = 1 if r.roof in ('dome', 'closed') else 0
for r in universe[universe.position == 'K'].itertuples():
    if r.player_id not in kicker_asof.index:
        continue
    b_ = float(kicker_asof.base[r.player_id])
    xi = ts_.get(r.team, LEAGUE_IMP) / LEAGUE_IMP - 1
    wk = []
    for w in range(WEEK, 19):
        if (r.team, w) not in sched_k:
            wk.append(None)
        else:
            wk.append(max(0.0, kmodel.c[0] * b_ + kmodel.c[1] * b_ * xi + kmodel.c[2] * sched_k[(r.team, w)]))
    wk = zero_out(wk, r.player_id)
    core = kmodel.c[0] * b_ + kmodel.c[1] * b_ * xi
    if core < 4:
        continue
    o_ = own.get(r.player_id)
    waiver.append(dict(id=r.player_id, name=r.pname, pos='K', team=r.team,
                       wk=[None if x is None else round(x, 1) for x in wk], ros=round(sum(x for x in wk if x is not None), 1),
                       ppg=round(core, 1), cv=round(float(kicker_asof.cv[r.player_id]), 2), trend=0, snap=0,
                       own=None if o_ is None or pd.isna(o_) else round(float(o_)), rs=1, rh=1,
                       status=status.get(r.player_id, ''), adds=NEWS['trending'].get(r.player_id),
                       news=(news_by.get(r.player_id) or {}).get('news')))

# team defenses, rest of season: baseline plus each future opponent's expected scoring and mistakes
sched_o = {}
for r in fut.itertuples():
    sched_o[(r.home_team, r.week)], sched_o[(r.away_team, r.week)] = r.away_team, r.home_team
dc = dmodel.c
for t_ in sorted(all_teams):
    if t_ not in dst_asof.index:
        continue
    b_ = float(dst_asof.base[t_])
    wk = []
    for w in range(WEEK, 19):
        o = sched_o.get((t_, w))
        wk.append(None if o is None else
                  dc[0] * b_ + dc[3] + dc[1] * (LEAGUE_IMP - ts_.get(o, LEAGUE_IMP)) + dc[2] * (float(dst_asof.give.get(o, LEAGUE_GIVE)) - LEAGUE_GIVE))
    g_ = [x for x in wk if x is not None]
    o_ = own.get('DST-' + t_)
    waiver.append(dict(id='DST-' + t_, name=TEAM_NAME.get(t_, t_) + ' D/ST', pos='DST', team=t_,
                       wk=[None if x is None else round(x, 1) for x in wk], ros=round(sum(g_), 1),
                       ppg=round(float(np.mean(g_)) if g_ else 0, 1), cv=round(float(dst_asof.cv[t_]), 2), trend=0, snap=0,
                       own=None if o_ is None or pd.isna(o_) else round(float(o_)), rs=1, rh=1, status=''))

waiver.sort(key=lambda x: -x['ros'])
# replacement level in a 12-team league (QB12, RB30, WR36, TE12, K12) so "All" can mix positions fairly
REPL_N = {'QB': 12, 'RB': 30, 'WR': 36, 'TE': 12, 'K': 12, 'DST': 12}
repl = {}
for p in POS + ['K', 'DST']:
    v = sorted([x['ppg'] for x in waiver if x['pos'] == p], reverse=True)
    repl[p] = v[min(REPL_N[p], len(v)) - 1] if v else 0
for x in waiver:
    x['vor'] = round(x['ppg'] - repl[x['pos']], 1)
ros_bt['repl'] = {p: round(v, 1) for p, v in repl.items()}
print('waiver pool', len(waiver))


# ---------------------------------------------------------------- live scoreboard: model vs experts vs Vegas
# Three projections are saved before kickoff each week, then scored once the games are played.
#   model   = this app's projection
#   experts = FantasyPros consensus projection
#   vegas   = the player's average, adjusted only by the Vegas line for that game (sportsbooks don't
#             publish free player-level fantasy projections, so this is what the game lines alone imply)
def vegas_bench(season, week, rows):
    """rows: iterable of (pid, pos, team, opp, imp, opp_imp). Returns dict pid -> projection."""
    ts_w = team_strength(season, week)
    a_w = A.get((season, week))
    k_w = kickers.asof(kd, season, week)
    d_w = dstm.asof(dw, dgive, season, week)
    res = {}
    for pid, pos, team, opp, imp, opp_imp in rows:
        if pd.isna(imp) or pd.isna(opp_imp):
            continue
        if pos == 'DST':
            b_ = d_w.base.get(team)
            v = None if b_ is None else b_ * ts_w.get(opp, LEAGUE_IMP) / max(opp_imp, 10)
        else:
            b_ = (k_w.base.get(pid) if pos == 'K' else (a_w.base_ppr.get(pid) if a_w is not None else None))
            v = None if b_ is None or pd.isna(b_) else b_ * imp / ts_w.get(team, LEAGUE_IMP)
        if v is not None:
            res[pid] = float(v)
    return res


impk = tg.set_index(['season', 'week', 'team']).imp
vb_now = vegas_bench(SEASON, WEEK, [(e['id'], e['pos'], e['team'], e['opp'], impk.get((SEASON, WEEK, e['team']), np.nan),
                                     impk.get((SEASON, WEEK, e['opp']), np.nan)) for e in out if not e['bye']])
for e in out:
    v = vb_now.get(e['id'])
    e['vegas_bench'] = None if v is None else round(0.0 if e['status'] in OUT_SET else v, 2)

# snapshot: only games not yet played are written; started games keep the numbers saved before kickoff
snap = pd.DataFrame([dict(id=p['id'], pos=p['pos'], model=p['proj'], fp_pts=p['fp'], vegas=p['vegas_bench']) for p in out
                     if not p['final'] and not p['bye']])
snap_path = os.path.join(HIST, f'ecr_{SEASON}_{WEEK}.csv')
if os.path.exists(snap_path):
    old = pd.read_csv(snap_path)
    snap = pd.concat([snap, old[~old.id.isin(snap.id)]])
snap.to_csv(snap_path, index=False)


def week_actuals(season, week):
    rows = {}
    for r in d[(d.season == season) & (d.week == week)].itertuples():
        rows[r.player_id] = (float(r.ppr), r.team, r.opponent_team)
    for r in kd[(kd.season == season) & (kd.week == week)].itertuples():
        rows[r.player_id] = (float(r.kpts), r.team, None)
    for r in dw[(dw.season == season) & (dw.week == week)].itertuples():
        rows['DST-' + r.team] = (float(r.dpts), r.team, r.opp)
    return rows


SRC = ['model', 'fp_pts', 'vegas']


def score(f):
    out_ = {}
    for c in SRC:
        h = t = 0
        for _, g_ in f.groupby('pos'):
            v, a = g_[c].values, g_.act.values
            i, j = np.triu_indices(len(v), 1)
            mk = (v[i] != v[j]) & (a[i] != a[j])
            h += np.sum(((v[i] > v[j]) == (a[i] > a[j]))[mk]); t += mk.sum()
        out_[c] = dict(pick=round(h / max(t, 1) * 100, 1), mae=round(float(np.mean(abs(f.act - f[c]))), 2))
    em = np.abs(f.model - f.act)
    h2h = {}
    for c, k_ in [('fp_pts', 'experts'), ('vegas', 'vegas')]:
        eo = np.abs(f[c] - f.act)
        h2h[k_] = round(float(((em < eo) + 0.5 * (em == eo)).mean() * 100), 1)   # % of players the model was closer
    return {'model': out_['model'], 'experts': out_['fp_pts'], 'vegas': out_['vegas'], 'h2h': h2h, 'n': int(len(f))}


sb_weeks, sb_rows = [], []
for fn in sorted(os.listdir(HIST), key=lambda x: (len(x), x)):
    if not fn.startswith(f'ecr_{SEASON}_'):
        continue
    w = int(fn.split('_')[2].split('.')[0])
    if w > WEEK:
        continue
    e_ = pd.read_csv(os.path.join(HIST, fn))
    acts = week_actuals(SEASON, w)
    if 'vegas' not in e_:
        e_['vegas'] = np.nan
    miss_v = e_.vegas.isna()
    if miss_v.any():   # weeks saved before the Vegas benchmark existed: rebuild it from pre-game data
        rows_ = []
        for r in e_[miss_v].itertuples():
            if r.id in acts and acts[r.id][2] is not None:
                t_, o_ = acts[r.id][1], acts[r.id][2]
                rows_.append((r.id, r.pos, t_, o_, impk.get((SEASON, w, t_), np.nan), impk.get((SEASON, w, o_), np.nan)))
            elif r.id in acts:   # kickers: opponent from the schedule
                t_ = acts[r.id][1]
                gm = games[(games.season == SEASON) & (games.week == w) & ((games.home_team == t_) | (games.away_team == t_))]
                if len(gm):
                    o_ = gm.away_team.iloc[0] if gm.home_team.iloc[0] == t_ else gm.home_team.iloc[0]
                    rows_.append((r.id, r.pos, t_, o_, impk.get((SEASON, w, t_), np.nan), impk.get((SEASON, w, o_), np.nan)))
        vb = vegas_bench(SEASON, w, rows_)
        e_.loc[miss_v, 'vegas'] = e_.loc[miss_v, 'id'].map(vb)
    e_['act'] = e_.id.map(lambda x: acts[x][0] if x in acts else np.nan)
    e_ = e_.dropna(subset=['act', 'model', 'fp_pts', 'vegas'])
    e_ = e_[(e_.fp_pts >= 3) | e_.pos.isin(['K', 'DST'])]
    if len(e_) < 20:
        continue
    e_['week'] = w
    sb_rows.append(e_)
    wk_ = score(e_)
    wk_['week'], wk_['partial'] = w, bool(w == WEEK)
    wk_['pos'] = {p_: score(g_) for p_, g_ in e_.groupby('pos') if len(g_) >= 8}
    sb_weeks.append(wk_)
scoreboard = None
if sb_rows:
    allr = pd.concat(sb_rows)
    scoreboard = dict(weeks=sb_weeks, total=score(allr),
                      pos={p_: score(g_) for p_, g_ in allr.groupby('pos') if len(g_) >= 8})
    print('scoreboard', scoreboard['total'])


# ---------------------------------------------------------------- season so far (Rankings: season view)
# One row per player who has played this season: weekly points (PPR and standard), usage, team byes.
cur_all = d[d.season == SEASON]
last_wk = int(max(cur_all.week.max() if len(cur_all) else 0, 0))
team_weeks = {}
for r in games[(games.season == SEASON)].itertuples():
    team_weeks.setdefault(r.home_team, set()).add(r.week)
    team_weeks.setdefault(r.away_team, set()).add(r.week)
team_bye = {t_: [w for w in range(1, 19) if w not in ws_] for t_, ws_ in team_weeks.items()}
cur_team = dict(zip(universe.player_id, universe.team))
cur_name = dict(zip(universe.player_id, universe.pname))
status_now = {e['id']: e['status'] for e in out}


def weekly_list(rows, col):
    m = dict(zip(rows.week, rows[col]))
    return [None if w not in m else round(float(m[w]), 1) for w in range(1, last_wk + 1)]


season_tbl = []
for pid, g_ in cur_all.groupby('player_id'):
    g_ = g_.sort_values('week')
    pos = g_.position.iloc[-1]
    team = cur_team.get(pid, g_.team.iloc[-1])
    gp = len(g_)
    season_tbl.append(dict(
        id=pid, name=cur_name.get(pid, g_.player_display_name.iloc[-1]), pos=pos, team=team, gp=gp,
        wp=weekly_list(g_, 'ppr'), ws=weekly_list(g_, 'std'), bye=team_bye.get(team, []), status=status_now.get(pid, ''),
        u=dict(tgt=round(g_.targets.mean(), 1), car=round(g_.carries.mean(), 1), att=round(g_.attempts.mean(), 1),
               rec=round(g_.receptions.fillna(0).mean(), 1),
               yds=round((g_.passing_yards.fillna(0) + g_.rushing_yards.fillna(0) + g_.receiving_yards.fillna(0)).mean(), 1),
               td=int((g_.passing_tds.fillna(0) + g_.rushing_tds.fillna(0) + g_.receiving_tds.fillna(0)).sum()),
               snap=round(g_.snap.mean() * 100), ts=None if g_.target_share.isna().all() else round(g_.target_share.mean() * 100, 1),
               rz=round((g_.rz_tgt + g_.rz_car).mean(), 1))))
for pid, g_ in kd[kd.season == SEASON].groupby('player_id'):
    g_ = g_.sort_values('week')
    team = cur_team.get(pid, g_.team.iloc[-1])
    season_tbl.append(dict(
        id=pid, name=cur_name.get(pid, g_.player_display_name.iloc[-1]), pos='K', team=team, gp=len(g_),
        wp=weekly_list(g_, 'kpts'), ws=weekly_list(g_, 'kpts'), bye=team_bye.get(team, []), status=status_now.get(pid, ''),
        u=dict(fgm=int(g_.fg_made.fillna(0).sum()), fga=int(g_.fg_att.fillna(0).sum()), fg50=int(g_.fg50.sum()),
               pat=int(g_.pat_made.fillna(0).sum()), lng=int(g_.fg_long.fillna(0).max()))))
for t_, g_ in dw[dw.season == SEASON].groupby('team'):
    g_ = g_.sort_values('week')
    season_tbl.append(dict(
        id='DST-' + t_, name=TEAM_NAME.get(t_, t_) + ' D/ST', pos='DST', team=t_, gp=len(g_),
        wp=weekly_list(g_, 'dpts'), ws=weekly_list(g_, 'dpts'), bye=team_bye.get(t_, []), status='',
        u=dict(sacks=round(g_.sacks.mean(), 1), to=round((g_.ints + g_.fum).mean(), 1), pa=round(g_.pa.mean(), 1),
               td=int((g_.td_d + g_.td_k).sum()))))
print('season table:', len(season_tbl), 'players through week', last_wk)

meta = dict(season=SEASON, week=WEEK, generated=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
            updated=dt.datetime.now(ZoneInfo('America/New_York')).strftime('%b %-d, %-I:%M %p ET'), league_imp=round(LEAGUE_IMP, 1), expert_date=fp_fresh,
            coef={p: [round(float(x), 3) for x in model.C[p]] for p in POS}, matchup_coef=round(model.cm, 3),
            backtest=bt, live=live, expert=expert, ros_backtest=ros_bt, scoreboard=scoreboard, n_injury_reports=int(len(iw)))
payload = json.dumps(dict(players=out, defense=defense, waiver=waiver, season=season_tbl, meta=meta), separators=(',', ':'), default=float)
open(os.path.join(OUT, 'data.json'), 'w').write(payload)
open(os.path.join(OUT, 'data.js'), 'w').write('window.SS_DATA=' + payload + ';\n')
print(f'wrote {len(out)} players, {len(payload)//1024} KB')
