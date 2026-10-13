"""Team defense / special teams (DST) projections.

Scoring (Sleeper/ESPN-style defaults, the same in PPR, half and standard):
  sack 1, interception 2, fumble recovery 2, safety 2, blocked kick 2,
  defensive or return touchdown 6,
  points allowed: 0 -> 10, 1-6 -> 7, 7-13 -> 4, 14-20 -> 1, 21-27 -> 0, 28-34 -> -1, 35+ -> -4.

Model: a defense scores when the offense it faces struggles. Vegas's implied total
for the opponent is the best single signal (it drives points allowed, and sacks and
turnovers come with it). So the projection is the defense's blended points per game,
adjusted by the opponent's implied total and by how often that opponent gives up
sacks and turnovers.
"""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

K_PRIOR, K_GIVE = 5.0, 4.0


def pa_points(pa):
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34], [10, 7, 4, 1, 0, -1], -4)


def weekly(pbp, games):
    """One row per team-game: the team's DST fantasy points, plus what its offense gave up."""
    p = pbp[pbp.season_type == 'REG']
    by = lambda mask, team_col: p[mask].groupby(['season', 'week', team_col]).size()
    parts = {
        'sacks': by(p.sack == 1, 'defteam'),
        'ints': by(p.interception == 1, 'defteam'),
        'fum': by(p.fumble_lost == 1, 'defteam'),
        'safety': by(p.safety == 1, 'defteam'),
        'blk': by((p.punt_blocked == 1) | (p.field_goal_result == 'blocked'), 'defteam'),
        # touchdowns by the team without the ball (turnover returns, punt returns, blocked kicks)
        'td_d': by((p.touchdown == 1) & (p.td_team == p.defteam) & (p.play_type != 'kickoff'), 'defteam'),
        # kickoff return touchdowns (the receiving team is listed as having the ball)
        'td_k': by((p.touchdown == 1) & (p.play_type == 'kickoff') & (p.td_team == p.posteam), 'posteam'),
    }
    df = pd.concat(parts, axis=1).fillna(0)
    df.index.names = ['season', 'week', 'team']
    df = df.reset_index()
    g = games[games.home_score.notna()]
    sc = pd.concat([
        g[['season', 'week', 'home_team', 'away_team', 'away_score']].rename(columns={'home_team': 'team', 'away_team': 'opp', 'away_score': 'pa'}),
        g[['season', 'week', 'away_team', 'home_team', 'home_score']].rename(columns={'away_team': 'team', 'home_team': 'opp', 'home_score': 'pa'})])
    df = sc.merge(df, on=['season', 'week', 'team'], how='left').fillna(0)
    df['dpts'] = (df.sacks + 2 * (df.ints + df.fum + df.safety + df.blk) + 6 * (df.td_d + df.td_k) + pa_points(df.pa))
    # what each offense gave up: sacks + turnovers allowed, keyed by the offense
    give = df[['season', 'week', 'opp', 'sacks', 'ints', 'fum']].rename(columns={'opp': 'team'})
    give['give'] = give.sacks + give.ints + give.fum
    return df, give[['season', 'week', 'team', 'give']]


def asof(dw, give, season, week):
    cur = dw[(dw.season == season) & (dw.week < week)]
    pri = dw[dw.season == season - 1]
    teams = sorted(set(dw[dw.season.isin([season, season - 1])].team))
    out = pd.DataFrame(index=teams)
    cs = cur.groupby('team').dpts.sum().reindex(teams).fillna(0)
    n = cur.groupby('team').size().reindex(teams).fillna(0)
    pm = pri.groupby('team').dpts.mean().reindex(teams).fillna(dw.dpts.mean())
    out['base'] = (cs + K_PRIOR * pm) / (n + K_PRIOR)
    out['n_cur'] = n
    gc = give[(give.season == season) & (give.week < week)]
    gp = give[give.season == season - 1]
    gs = gc.groupby('team').give.sum().reindex(teams).fillna(0)
    gn = gc.groupby('team').size().reindex(teams).fillna(0)
    gm = gp.groupby('team').give.mean().reindex(teams).fillna(give.give.mean())
    out['give'] = (gs + K_GIVE * gm) / (gn + K_GIVE)
    for col in ['sacks', 'ints', 'fum']:
        c = cur.groupby('team')[col].sum().reindex(teams).fillna(0)
        pm_ = pri.groupby('team')[col].mean().reindex(teams).fillna(0)
        out[col] = (c + K_PRIOR * pm_) / (n + K_PRIOR)
    out['pa'] = ((cur.groupby('team').pa.sum().reindex(teams).fillna(0) + K_PRIOR * pri.groupby('team').pa.mean().reindex(teams).fillna(22))
                 / (n + K_PRIOR))
    hist = pd.concat([pri, cur]).sort_values(['season', 'week']).groupby('team').tail(24)
    st = hist.groupby('team').dpts.agg(['std', 'mean']).reindex(teams)
    out['cv'] = (st['std'] / st['mean'].clip(lower=4)).fillna(.8).clip(.4, 1.3)
    return out


def frame(dw, give, tg, season, week, rows):
    """rows: season, week, team, opp. Adds as-of features, the opponent's implied total and giveaway rate."""
    a = asof(dw, give, season, week)
    f = rows.join(a[['base', 'n_cur', 'cv', 'sacks', 'ints', 'fum', 'pa']], on='team')
    f['opp_give'] = f.opp.map(a.give)
    imp = tg.set_index(['season', 'week', 'team']).imp
    f['opp_imp'] = [imp.get((s, w, o), np.nan) for s, w, o in zip(f.season, f.week, f.opp)]
    cur = dw[(dw.season == season) & (dw.week < week)].groupby('team').dpts.mean()
    pri = dw[dw.season == season - 1].groupby('team').dpts.mean()
    f['naive'] = f.team.map(cur).fillna(f.team.map(pri))
    return f


class DModel:
    def fit(self, f, league_imp, league_give):
        self.li, self.lg = league_imp, league_give
        X = self.X(f)
        self.c = lstsq(X, f.dpts.values, rcond=None)[0]
        proj = X @ self.c
        sd = float(np.std(f.dpts.values - proj))
        self.sd = sd
        return self

    def X(self, f):
        return np.c_[f.base, (self.li - f.opp_imp), (f.opp_give - self.lg), np.ones(len(f))]

    def parts(self, f):
        c = self.c
        return c[0] * f.base + c[3], c[1] * (self.li - f.opp_imp), c[2] * (f.opp_give - self.lg)

    def predict(self, f):
        a, b, c = self.parts(f)
        return a + b + c
