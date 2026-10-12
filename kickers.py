"""Kicker projections.

Scoring (ESPN/Sleeper default): PAT 1, FG 0-39 yds 3, 40-49 yds 4, 50+ yds 5,
missed FG -1, missed PAT -1. Same in PPR, half and standard.

Model: a kicker's points mostly follow how many scoring drives his offense gets,
which the Vegas implied team total measures. So the projection is his blended
points per game (this season + last season, shrunk toward a typical starter),
scaled by this week's implied total, plus a small dome/indoor bump.
"""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

K_PRIOR, K_NEW = 5.0, 4.0


def kicker_points(df):
    f = lambda c: df[c].fillna(0) if c in df else 0
    return (f('pat_made') - f('pat_missed')
            + 3 * (f('fg_made_0_19') + f('fg_made_20_29') + f('fg_made_30_39'))
            + 4 * f('fg_made_40_49') + 5 * (f('fg_made_50_59') + f('fg_made_60_'))
            - f('fg_missed'))


def load(frames):
    k = pd.concat(frames)
    k = k[(k.season_type == 'REG') & (k.position == 'K')].copy()
    k['kpts'] = kicker_points(k)
    k['fga'] = k.fg_att.fillna(0)
    k['pata'] = k.pat_att.fillna(0)
    k['fg50'] = (k.fg_made_50_59.fillna(0) + k.fg_made_60_.fillna(0))
    k = k[(k.fga + k.pata) > 0]
    return k


def asof(k, season, week):
    cur = k[(k.season == season) & (k.week < week)]
    pri = k[k.season == season - 1]
    idx = pd.concat([pri, cur]).player_id.unique()
    out = pd.DataFrame(index=idx)
    n = cur.groupby('player_id').size().reindex(idx).fillna(0)
    np_ = pri.groupby('player_id').size().reindex(idx).fillna(0)
    repl = pri.groupby('player_id').kpts.mean().quantile(.4) if len(pri) else 7.0
    cs = cur.groupby('player_id').kpts.sum().reindex(idx).fillna(0)
    pm = pri.groupby('player_id').kpts.mean().reindex(idx)
    kp = np.where(pm.notna(), K_PRIOR * np.minimum(np_, 17) / 17, K_NEW)
    out['base'] = (cs + kp * pm.fillna(repl)) / (n + kp)
    out['n_cur'] = n
    for col in ['fga', 'pata', 'fg50']:
        c = cur.groupby('player_id')[col].sum().reindex(idx).fillna(0)
        p = pri.groupby('player_id')[col].mean().reindex(idx)
        out[col] = (c + 2.5 * p.fillna(c / n.clip(lower=1))) / (n + 2.5)
    hist = pd.concat([pri, cur]).sort_values(['season', 'week']).groupby('player_id').tail(24)
    st = hist.groupby('player_id').kpts.agg(['std', 'mean', 'count']).reindex(idx)
    cv = st['std'] / st['mean'].clip(lower=3)
    med = cv.median()
    out['cv'] = (cv.fillna(med) * st['count'].fillna(0) + 8 * med) / (st['count'].fillna(0) + 8)
    out['team'] = pd.concat([pri, cur]).sort_values(['season', 'week']).groupby('player_id').team.last().reindex(idx)
    out['name'] = pd.concat([pri, cur]).sort_values(['season', 'week']).groupby('player_id').player_display_name.last().reindex(idx)
    return out


def frame(k, games_tg, roof, season, week, rows=None):
    """rows: kicker weekly rows to attach features to (actuals) or None for the asof table only."""
    a = asof(k, season, week)
    f = rows.join(a[['base', 'n_cur', 'fga', 'pata', 'fg50', 'cv']], on='player_id')
    f = f.merge(games_tg[['season', 'week', 'team', 'imp']], on=['season', 'week', 'team'], how='left')
    f['dome'] = [roof.get((s, w, t), 0) for s, w, t in zip(f.season, f.week, f.team)]
    cur = k[(k.season == season) & (k.week < week)].groupby('player_id').kpts.mean()
    pri = k[k.season == season - 1].groupby('player_id').kpts.mean()
    f['naive'] = f.player_id.map(cur).fillna(f.player_id.map(pri))
    return f


class KModel:
    def fit(self, f, league_imp):
        self.li = league_imp
        X = self.X(f)
        self.c = lstsq(X, f.kpts.values, rcond=None)[0]
        f = f.assign(proj=X @ self.c)
        best = (1e9, 1.0)
        for m in np.arange(.3, 1.6, .02):
            lo, hi = self.range(f.proj, f.cv, m)
            cov = ((f.kpts >= lo) & (f.kpts <= hi)).mean()
            if abs(cov - .6) < best[0]:
                best = (abs(cov - .6), m)
        self.rm = best[1]
        return self

    def X(self, f):
        x_imp = f.imp / self.li - 1
        return np.c_[f.base, f.base * x_imp, f.dome]

    def parts(self, f):
        x_imp = f.imp / self.li - 1
        return self.c[0] * f.base, self.c[1] * f.base * x_imp, self.c[2] * f.dome

    def predict(self, f):
        a, b, c = self.parts(f)
        return (a + b + c).clip(lower=0)

    @staticmethod
    def range(proj, cv, m):
        return proj * (1 - m * cv).clip(lower=.1), proj * (1 + m * cv)
