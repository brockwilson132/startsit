"""Start/Sit pipeline.
Downloads nflverse + FantasyPros consensus, fits the model on the prior season,
tests it on the season before that, projects the next unplayed week, scores this
season so far, and writes data.js for the app.

Run:  python pipeline.py            (auto-detect season/week)
      python pipeline.py --offline  (use files already in ./cache)
"""
import os, sys, json, datetime as dt, urllib.request
import numpy as np, pandas as pd
from numpy.linalg import lstsq

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
SEASONS = [SEASON - 3, SEASON - 2, SEASON - 1, SEASON]
print('Projecting', SEASON, 'week', WEEK)

frames = []
for y in SEASONS:
    frames.append(pd.read_csv(fetch(f'{NV}/stats_player/stats_player_week_{y}.csv', f'w{y}.csv', refresh=y == SEASON), low_memory=False))
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
posavg = d[d.season >= SEASON - 2].groupby(['season', 'week', 'team', 'position']).ppr.sum().groupby('position').mean()

# ---------------------------------------------------------------- as-of features
UCOLS = ['targets', 'carries', 'attempts', 'receiving_air_yards', 'rz_tgt', 'rz_car', 'snap']
K_PRIOR, K_NEW, K_USE = 4.0, 3.0, 2.5


def asof(season, week):
    cur = d[(d.season == season) & (d.week < week)]
    pri = d[d.season == season - 1]
    n = cur.groupby('player_id').size()
    pos = pd.concat([pri, cur]).groupby('player_id').position.last()
    idx = pos.index
    out = pd.DataFrame(index=idx)
    out['position'] = pos
    out['n_cur'] = n.reindex(idx).fillna(0)
    np_ = pri.groupby('player_id').size().reindex(idx).fillna(0)
    for col in ['ppr', 'std', 'half']:
        repl = pri.groupby('player_id').agg(m=(col, 'mean'), p=('position', 'last')).groupby('p').m.quantile(.35)
        cs = cur.groupby('player_id')[col].sum().reindex(idx).fillna(0)
        pm = pri.groupby('player_id')[col].mean().reindex(idx)
        kp = np.where(pm.notna(), K_PRIOR * np.minimum(np_, 17) / 17, K_NEW)
        prior = pm.fillna(out.position.map(repl))
        out['base_' + col] = (cs + kp * prior) / (out.n_cur + kp)
    for col in UCOLS:
        cs = cur.groupby('player_id')[col].sum().reindex(idx).fillna(0)
        pm = pri.groupby('player_id')[col].mean().reindex(idx)
        cm = (cs / out.n_cur.clip(lower=1))
        out['u_' + col] = (cs + K_USE * pm.fillna(cm)) / (out.n_cur + K_USE)
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


def build(seasons, wmin=3):
    out = []
    for s in seasons:
        for w in range(wmin, 19):
            if (s, w) not in A or (s == SEASON and w >= WEEK):
                continue
            rows = d[(d.season == s) & (d.week == w)][['season', 'week', 'player_id', 'player_display_name', 'position', 'team', 'opponent_team', 'ppr', 'std', 'half']]
            out.append(frame(s, w, rows))
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
train, test = build([SEASON - 2]), build([SEASON - 1])
mt = Model().fit(train)
te = mt.apply(test)
bt = dict(train=SEASON - 2, test=SEASON - 1,
          mae_naive=float(np.mean(abs(te.ppr - te.naive))), mae_model=float(np.mean(abs(te.ppr - te.proj))),
          pair_naive=pairs(te, 'naive')[0] * 100, pair_model=pairs(te, 'proj')[0] * 100,
          close_naive=pairs(te, 'naive', 3)[0] * 100, close_model=pairs(te, 'proj', 3)[0] * 100,
          bypos={p: [pairs(te[te.position == p], 'naive', 3)[0] * 100, pairs(te[te.position == p], 'proj', 3)[0] * 100] for p in POS})
lo, hi = Model.range(te.proj, te.cv, mt.rm)
bt['range_cover'] = float(((te.ppr >= lo) & (te.ppr <= hi)).mean() * 100)
print('backtest', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in bt.items() if k != 'bypos'})

# ---------------------------------------------------------------- final model: fit on S-2 and S-1
model = Model().fit(pd.concat([train, test]))

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

# ---------------------------------------------------------------- project the upcoming week
cur = d[d.season == SEASON]
last = cur.sort_values('week').groupby('player_id').last()
kick = {}
for _, r in unplayed[unplayed.week == WEEK].iterrows():
    kick[r.home_team] = (r.away_team, 1, r.gameday, r.gametime, r.weekday)
    kick[r.away_team] = (r.home_team, 0, r.gameday, r.gametime, r.weekday)
rows = []
for pid, r in last.iterrows():
    if r.team in kick:
        rows.append(dict(season=SEASON, week=WEEK, player_id=pid, player_display_name=r.player_display_name, position=r.position,
                         team=r.team, opponent_team=kick[r.team][0]))
up = model.apply(frame(SEASON, WEEK, pd.DataFrame(rows)))
up = up[up.n_cur >= 1]

# injuries: latest report this week
inj = pd.read_csv(fetch(f'{NV}/injuries/injuries_{SEASON}.csv', f'inj{SEASON}.csv'))
iw = inj[(inj.season == SEASON) & (inj.week == WEEK)]
status = dict(zip(iw.gsis_id, iw.report_status.fillna('')))
practice = dict(zip(iw.gsis_id, iw.practice_status.fillna('')))
injury = dict(zip(iw.gsis_id, iw.report_primary_injury.fillna('')))
prev_wk = cur[cur.week == WEEK - 1]
played_prev, teams_prev = set(prev_wk.player_id), set(prev_wk.team)

# expert consensus
fp = pd.read_csv(fetch('https://raw.githubusercontent.com/dynastyprocess/data/master/files/fp_latest_weekly.csv', 'fp.csv'))
fp = fp[fp.page.isin(['qb', 'ppr-rb', 'ppr-wr', 'ppr-te'])]
fmap = ids[['fantasypros_id', 'gsis_id']].dropna()
fmap['fantasypros_id'] = fmap.fantasypros_id.astype(int)
fp = fp.merge(fmap, on='fantasypros_id', how='left')
fp_rank = dict(zip(fp.gsis_id, fp.pos_rank))
fp_pts = dict(zip(fp.gsis_id, fp.r2p_pts))
fp_fresh = str(fp.scrape_date.max()) if len(fp) else None

out = []
up = up.sort_values('proj', ascending=False)
up['pos_rank'] = up.groupby('position').proj.rank(ascending=False, method='first').astype(int)
pos_count = up[up.proj >= 4].groupby('position').size()
for _, r in up.iterrows():
    stt = status.get(r.player_id, '')
    proj = 0.0 if stt == 'Out' else float(r.proj)
    if r.proj < 4:
        continue
    lo_, hi_ = Model.range(pd.Series([proj]), pd.Series([r.cv]), model.rm)
    sc_std = r.base_std / r.base_ppr if r.base_ppr > 0 else 1
    sc_half = r.base_half / r.base_ppr if r.base_ppr > 0 else 1
    o, h, day, tm, wd = kick[r.team]
    er = fp_rank.get(r.player_id)
    out.append(dict(
        id=r.player_id, name=r.player_display_name, pos=r.position, team=r.team, opp=o, home=h, day=wd, date=day, time=tm,
        proj=round(proj, 2), std=round(proj * sc_std, 2), half=round(proj * sc_half, 2),
        lo=round(float(lo_.iloc[0]), 1), hi=round(float(hi_.iloc[0]), 1),
        form=round(r.c_form, 2), usage=round(r.c_usage, 2), matchup=round(r.c_match, 2), vegas=round(r.c_vegas, 2),
        base=round(r.base_ppr, 1), xfp=round(r.xfp, 1), imp=round(r.imp, 1), xdef=round(r.x_def * 100, 1),
        tgt=round(r.u_targets, 1), car=round(r.u_carries, 1), att=round(r.u_attempts, 1),
        rz=round(r.u_rz_tgt + r.u_rz_car, 1), snap=round(r.u_snap * 100), trend=round(r.snap_trend * 100),
        cv=round(r.cv, 2), gp=int(r.n_cur), rank=int(r.pos_rank),
        ecr=int(er.lstrip('QBRWTE')) if isinstance(er, str) and er[2:].isdigit() else None,
        fp=None if pd.isna(fp_pts.get(r.player_id, np.nan)) else float(fp_pts[r.player_id]),
        status=stt, practice=practice.get(r.player_id, ''), injury=injury.get(r.player_id, ''),
        miss=bool(r.player_id not in played_prev and r.team in teams_prev)))

# save a snapshot of this week's model + expert numbers so we can score both after the games
snap = pd.DataFrame([dict(id=p['id'], pos=p['pos'], model=p['proj'], fp_pts=p['fp']) for p in out])
snap_path = os.path.join(HIST, f'ecr_{SEASON}_{WEEK}.csv')
if os.path.exists(snap_path):  # keep earlier rows for players whose game already kicked off (e.g. Thursday)
    old = pd.read_csv(snap_path)
    snap = pd.concat([snap, old[~old.id.isin(snap.id)]])
snap.to_csv(snap_path, index=False)

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


def ros(season, week, mdl, players):
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
now = fp[['fantasypros_id', 'player_owned_avg']].dropna().assign(date=str(today))
oh = pd.concat([oh, now]).sort_values('date').groupby('fantasypros_id').last().reset_index()
oh.to_csv(OWN_PATH, index=False)
oh = oh.merge(fmap, on='fantasypros_id', how='inner')
own = dict(zip(oh.gsis_id, oh.player_owned_avg))
pool = cur.sort_values('week').groupby('player_id').agg(position=('position', 'last'), team=('team', 'last'),
                                                       pname=('player_display_name', 'last')).reset_index()
pr = ros(SEASON, WEEK, model, pool[['player_id', 'position', 'team']])
cvmap = A[(SEASON, WEEK)].cv
inj_out = {k for k, v in status.items() if v == 'Out'}
waiver = []
for _, r in pool.iterrows():
    if r.player_id not in pr:
        continue
    wk, core = pr[r.player_id]
    if r.player_id in inj_out and wk and wk[0] is not None:
        wk = [0.0] + wk[1:]
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
                       status=status.get(r.player_id, '')))
waiver.sort(key=lambda x: -x['ros'])
# replacement level in a 12-team league (QB12, RB30, WR36, TE12) so "All" can mix positions fairly
REPL_N = {'QB': 12, 'RB': 30, 'WR': 36, 'TE': 12}
repl = {}
for p in POS:
    v = sorted([x['ppg'] for x in waiver if x['pos'] == p], reverse=True)
    repl[p] = v[min(REPL_N[p], len(v)) - 1] if v else 0
for x in waiver:
    x['vor'] = round(x['ppg'] - repl[x['pos']], 1)
ros_bt['repl'] = {p: round(v, 1) for p, v in repl.items()}
print('waiver pool', len(waiver))

meta = dict(season=SEASON, week=WEEK, generated=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
            updated=today.strftime('%b %-d'), league_imp=round(LEAGUE_IMP, 1), expert_date=fp_fresh,
            coef={p: [round(float(x), 3) for x in model.C[p]] for p in POS}, matchup_coef=round(model.cm, 3),
            backtest=bt, live=live, expert=expert, ros_backtest=ros_bt, n_injury_reports=int(len(iw)))
payload = json.dumps(dict(players=out, defense=defense, waiver=waiver, meta=meta), separators=(',', ':'), default=float)
open(os.path.join(OUT, 'data.json'), 'w').write(payload)
open(os.path.join(OUT, 'data.js'), 'w').write('window.SS_DATA=' + payload + ';\n')
print(f'wrote {len(out)} players, {len(payload)//1024} KB')
