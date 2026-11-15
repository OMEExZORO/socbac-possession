"""Possession methods under test. Each takes a degraded match dict and returns int8 codes per
frame (0 H, 1 A, 2 D), the codes evaluation/possession_metrics.py uses.

  baseline_smooth50   the organisers' published floor, re-implemented on bench arrays
  nb_viterbi          ours v0: naive-Bayes evidence fitted on training matches + a 3-state Viterbi
"""

import numpy as np

H, A, D = 0, 1, 2
PITCH = (105.0, 68.0)


# -- shared features ---------------------------------------------------------------------------

def nearest(match):
    """Distance (m) from the ball to the nearest home and nearest away player; NaN if no ball."""
    ball, players = match["ball"], match["players"]
    d = np.linalg.norm(players - ball[:, None, None, :2], axis=3)  # (n, 2, slots)
    with np.errstate(all="ignore"):
        dh = np.nanmin(np.where(np.isnan(d[:, 0]), np.inf, d[:, 0]), axis=1)
        da = np.nanmin(np.where(np.isnan(d[:, 1]), np.inf, d[:, 1]), axis=1)
    missing = np.isnan(ball[:, 0])
    dh[missing] = np.nan
    da[missing] = np.nan
    return dh, da


def team_speed(match):
    """Median player speed (m/s) per frame over all tracked slots, from position differences."""
    p = match["players"].reshape(len(match["ball"]), -1, 2)
    v = np.linalg.norm(np.diff(p, axis=0), axis=2) * 25.0
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(v > 12, np.nan, v), axis=1)
    med = np.concatenate(([med[0] if len(med) else 0.0], med))
    return np.nan_to_num(med, nan=0.0)


def features(match):
    """Discrete per-frame features for the naive-Bayes evidence. Each column is a small int."""
    dh, da = nearest(match)
    missing = np.isnan(dh)
    margin = np.clip(np.nan_to_num(da - dh), -9.99, 9.99)            # + means home closer
    near = np.nan_to_num(np.minimum(dh, da), nan=99.0)
    bx, by = match["ball"][:, 0], match["ball"][:, 1]
    out_of_play = (~missing) & ((bx < -0.5) | (bx > PITCH[0] + 0.5) | (by < -0.5) | (by > PITCH[1] + 0.5))
    speed = team_speed(match)
    # smoothed speed over 1 s, the "calm" signal of a stoppage
    k = np.ones(25) / 25.0
    speed = np.convolve(speed, k, mode="same")

    # how long the ball has been missing, and whether it was last seen near a line: a short gap in
    # open play is an occlusion, a long gap that starts at the touchline is a stoppage
    n = len(missing)
    idx = np.where(~missing, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    gap = np.where(last < 0, 9999, np.arange(n) - last)
    lx, ly = bx[np.maximum(last, 0)], by[np.maximum(last, 0)]
    near_line = (lx < 2) | (lx > PITCH[0] - 2) | (ly < 2) | (ly > PITCH[1] - 2)
    f_gap = np.where(missing, 1 + np.digitize(gap, [12, 25, 50, 100, 200]), 0).astype(np.int8)
    f_edge = np.where(missing, 1 + near_line.astype(np.int8), 0).astype(np.int8)

    f_missing = missing.astype(np.int8)
    f_margin = np.where(missing, 0, 1 + np.digitize(margin, [-6, -3, -1.5, -0.5, 0, 0.5, 1.5, 3, 6])).astype(np.int8)
    f_near = np.where(missing, 0, 1 + np.digitize(near, [1, 2, 4, 8, 15])).astype(np.int8)
    f_out = out_of_play.astype(np.int8)
    f_speed = np.digitize(speed, [0.6, 1.0, 1.5, 2.2, 3.0]).astype(np.int8)
    return np.stack([f_missing, f_margin, f_near, f_out, f_speed, f_gap, f_edge], axis=1)


# -- baseline ----------------------------------------------------------------------------------

def baseline_smooth50(match, window=50):
    """Nearest tracked player to the ball, D where the ball is missing, majority vote +/- window.
    Same algorithm as starter/baseline.py, applied per period."""
    dh, da = nearest(match)
    raw = np.where(np.isnan(dh), -1, np.where(dh <= da, H, A))
    out = np.full(len(raw), D, np.int8)
    for p in np.unique(match["period"]):
        idx = np.flatnonzero(match["period"] == p)
        codes = raw[idx]
        hc = np.concatenate(([0], np.cumsum(codes == H)))
        ac = np.concatenate(([0], np.cumsum(codes == A)))
        i = np.arange(len(codes))
        lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
        h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
        lab = np.full(len(codes), D, np.int8)
        seen = (h + a) > 0
        lab[seen] = np.where(h[seen] >= a[seen], H, A)
        out[idx] = lab
    return out


# -- ours v0: naive-Bayes evidence + Viterbi ---------------------------------------------------

class NBViterbi:
    """Per-frame log P(features | state) from counts on training matches, then a 3-state Viterbi
    with one cost for changing team and one for changing between alive and dead."""

    def __init__(self, switch_team=8.0, switch_dead=8.0, alpha=1.0):
        self.switch_team, self.switch_dead, self.alpha = switch_team, switch_dead, alpha
        self.tables = None

    def fit(self, matches, labels):
        feats = np.concatenate([features(m) for m in matches])
        y = np.concatenate(labels)
        self.tables = []
        for j in range(feats.shape[1]):
            k = int(feats[:, j].max()) + 1
            table = np.zeros((3, k))
            for s in range(3):
                counts = np.bincount(feats[y == s, j], minlength=k) + self.alpha
                table[s] = np.log(counts / counts.sum())
            self.tables.append(table)
        return self

    def emissions(self, match):
        f = features(match)
        e = np.zeros((len(f), 3))
        for j, table in enumerate(self.tables):
            col = np.minimum(f[:, j], table.shape[1] - 1)
            e += table[:, col].T
        return e

    def decode(self, e):
        n = len(e)
        cost = np.array([[0, self.switch_team, self.switch_dead],
                         [self.switch_team, 0, self.switch_dead],
                         [self.switch_dead, self.switch_dead, 0]])
        score = e[0].copy()
        back = np.zeros((n, 3), np.int8)
        for t in range(1, n):
            cand = score[:, None] - cost          # from-state x to-state
            back[t] = np.argmax(cand, axis=0)
            score = cand[back[t], np.arange(3)] + e[t]
        path = np.zeros(n, np.int8)
        path[-1] = int(np.argmax(score))
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def predict(self, match):
        e = self.emissions(match)
        out = np.full(len(e), D, np.int8)
        for p in np.unique(match["period"]):
            idx = np.flatnonzero(match["period"] == p)
            out[idx] = self.decode(e[idx])
        return out


# -- ours v1: windowed features + LightGBM evidence + Viterbi ----------------------------------

def _centred_mean(x, w):
    """Mean of x over frames t-w..t+w (edges shrink). x must be finite."""
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def _since(event, cap=500):
    """Frames since the last True (cap if none yet)."""
    n = len(event)
    last = np.maximum.accumulate(np.where(event, np.arange(n), -1))
    return np.where(last < 0, cap, np.minimum(cap, np.arange(n) - last)).astype(np.float32)


def _until(event, cap=500):
    return _since(event[::-1], cap)[::-1]


def dense_features(match):
    """Continuous per-frame features, offline (they may look ahead as well as back)."""
    dh, da = nearest(match)
    ball = match["ball"]
    missing = np.isnan(dh)
    n = len(missing)
    dhf, daf = np.nan_to_num(dh, nan=30.0).clip(0, 30), np.nan_to_num(da, nan=30.0).clip(0, 30)
    margin = np.where(missing, 0.0, (daf - dhf).clip(-15, 15))
    near = np.minimum(dhf, daf)

    # ball kinematics from positions (NaN through gaps)
    vel = np.diff(ball[:, :2], axis=0, prepend=ball[:1, :2]) * 25.0
    speed = np.linalg.norm(vel, axis=1)
    acc = np.linalg.norm(np.diff(vel, axis=0, prepend=vel[:1]), axis=1) * 25.0
    speed_f = np.nan_to_num(speed, nan=-1.0).clip(-1, 40)
    acc_f = np.nan_to_num(acc, nan=-1.0).clip(-1, 400)

    # a touch: a sharp change of ball velocity with a player close
    touch = (np.nan_to_num(acc) > 60) & (near < 2.0)
    touch_h = touch & (dhf <= daf)
    touch_a = touch & (daf < dhf)

    # gap context, both directions
    gap_back, gap_fwd = _since(~missing), _until(~missing)
    seen_idx = np.where(~missing, np.arange(n), -1)
    last = np.maximum.accumulate(seen_idx)
    nxt = np.minimum.accumulate(np.where(~missing, np.arange(n), n)[::-1])[::-1]
    bx, by = ball[:, 0], ball[:, 1]

    def edge_dist(ix):
        ok = (ix >= 0) & (ix < n)
        j = np.clip(ix, 0, n - 1)
        d = np.minimum.reduce([bx[j], PITCH[0] - bx[j], by[j], PITCH[1] - by[j]])
        return np.where(ok, np.nan_to_num(d, nan=30.0), 30.0).clip(-5, 30)

    calm = np.convolve(team_speed(match), np.ones(25) / 25.0, mode="same")

    cols = {
        "missing": missing.astype(np.float32), "dh": dhf, "da": daf, "margin": margin, "near": near,
        "speed": speed_f, "acc": acc_f, "calm": calm,
        "gap_back": gap_back, "gap_fwd": gap_fwd, "gap_total": np.minimum(1000, gap_back + gap_fwd),
        "edge_last": edge_dist(last), "edge_next": edge_dist(nxt),
        "since_touch_h": _since(touch_h), "since_touch_a": _since(touch_a),
        "until_touch_h": _until(touch_h), "until_touch_a": _until(touch_a),
    }
    for w in (12, 25, 50, 125):
        cols["close_h_%d" % w] = _centred_mean((~missing & (dhf < 2.0)).astype(float), w)
        cols["close_a_%d" % w] = _centred_mean((~missing & (daf < 2.0)).astype(float), w)
        cols["lead_h_%d" % w] = _centred_mean((~missing & (margin > 0.5)).astype(float), w)
        cols["lead_a_%d" % w] = _centred_mean((~missing & (margin < -0.5)).astype(float), w)
        cols["missing_%d" % w] = _centred_mean(missing.astype(float), w)
        cols["calm_%d" % w] = _centred_mean(calm, w)
        cols["touch_h_%d" % w] = _centred_mean(touch_h.astype(float), w)
        cols["touch_a_%d" % w] = _centred_mean(touch_a.astype(float), w)
    names = list(cols)
    return np.column_stack([np.asarray(cols[k], np.float32) for k in names]), names


SWAP = np.array([A, H, D])   # label code -> code with the teams exchanged


def swap_teams(match):
    """The same match with home and away exchanged (players only; the ball is shared)."""
    out = dict(match)
    out["players"] = match["players"][:, ::-1]
    return out


def viterbi(e, switch_team, switch_dead):
    cost = np.array([[0, switch_team, switch_dead],
                     [switch_team, 0, switch_dead],
                     [switch_dead, switch_dead, 0]], dtype=float)
    n = len(e)
    score = e[0].copy()
    back = np.zeros((n, 3), np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + e[t]
    path = np.zeros(n, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_periods(e, period, switch_team, switch_dead):
    out = np.full(len(e), D, np.int8)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = viterbi(e[idx], switch_team, switch_dead)
    return out


class LGBMViterbi:
    """LightGBM P(H, A, D | windowed features) as scaled likelihoods (posterior / prior), then the
    same 3-state Viterbi. Deterministic: fixed seed, single thread, deterministic=True."""

    PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, seed=0, deterministic=True, num_threads=1, verbose=-1)

    def __init__(self, switch_team=10.0, switch_dead=40.0, rounds=300, stride=5, symmetric=True):
        self.switch_team, self.switch_dead = switch_team, switch_dead
        self.rounds, self.stride = rounds, stride
        # Possession has no preferred side: train on every match twice (teams swapped, H<->A) and
        # average the two views at prediction, so nothing learned about one league's home sides
        # can tilt a new match towards H or A.
        self.symmetric = symmetric

    def fit(self, matches, labels):
        import lightgbm as lgb
        xs, ys = [], []
        for m, y in zip(matches, labels):
            x, self.names = dense_features(m)
            xs.append(x[::self.stride]); ys.append(y[::self.stride])
            if self.symmetric:
                x, _ = dense_features(swap_teams(m))
                xs.append(x[::self.stride]); ys.append(SWAP[y][::self.stride])
        x, y = np.concatenate(xs), np.concatenate(ys)
        self.prior = np.bincount(y, minlength=3) / len(y)
        self.model = lgb.train(self.PARAMS, lgb.Dataset(x, y), num_boost_round=self.rounds)
        return self

    def emissions(self, match):
        x, _ = dense_features(match)
        lp = np.log(self.model.predict(x) + 1e-6)
        if self.symmetric:
            x, _ = dense_features(swap_teams(match))
            lp = 0.5 * (lp + np.log(self.model.predict(x) + 1e-6)[:, SWAP])
        return lp - np.log(self.prior)

    def predict(self, match):
        return decode_periods(self.emissions(match), match["period"], self.switch_team, self.switch_dead)
