"""Decoupled degradation (v4). Does not modify bench/degrade.py.

v2 draws two ball-hiding chains and switches between them on the TRUE ball-alive flag, so gap
edges mark dead-ball boundaries (54% of alive<->dead transitions have a gap edge within +-2
frames, chance ~1%). v4 never reads `alive` or `owner`: the hiding hazard depends only on
physical conditions a camera could miss:

  - ball in the air (z > AIR_Z)
  - crowd: >= 3 players within CROWD_RADIUS m of the ball
  - ball near or over a touchline / goal line (within LINE_MARGIN m, or outside the pitch);
    gaps that start there are LINE_DUR_MULT times longer (the ball is off the pitch / behind the
    boards)

Gap length is lognormal (median / mean calibrated to the real tracking aggregates: visible 59.3%
of graded frames, gap median 268, mean 431, p90 845 frames). Everything else (ball smoothing,
PLAYER_SMOOTH = 12, bursty player drops, duplicates) is v2's, with v2's constants.
"""

import numpy as np

from bench import degrade as v2

PITCH = (105.0, 68.0)
PLAYER_SMOOTH = 12
AIR_Z = 2.0
LINE_MARGIN = 1.0
AIR_MULT, CROWD_MULT, LINE_MULT = 2.0, 2.0, 3.0   # hazard multipliers (relative, not calibrated)
LINE_DUR_MULT = 2.0
# calibrated by bench/calibrate_v4.py (label-free aggregates only)
BASE_RATE = 1.0e-3       # per-frame gap start probability in open conditions (see CALIB below)
LOG_SIGMA = 0.95
DUR_MEDIAN = 190.0       # frames, median of a gap that starts away from a line
REV = "v4a"


def physical_conditions(ball, players):
    n = len(ball)
    flat = players.reshape(n, -1, 2)
    dist = np.linalg.norm(flat - ball[:, None, :2], axis=2)
    crowd = (np.nan_to_num(dist, nan=99.0) < v2.CROWD_RADIUS).sum(axis=1) >= 3
    air = np.nan_to_num(ball[:, 2]) > AIR_Z
    bx, by = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        line = ((bx < LINE_MARGIN) | (bx > PITCH[0] - LINE_MARGIN) |
                (by < LINE_MARGIN) | (by > PITCH[1] - LINE_MARGIN))
    line = np.nan_to_num(line, nan=0).astype(bool)
    return crowd, air, line


def ball_hidden_mask(rng, ball, players, base_rate=None, dur_median=None, sigma=None):
    base_rate = BASE_RATE if base_rate is None else base_rate
    dur_median = DUR_MEDIAN if dur_median is None else dur_median
    sigma = LOG_SIGMA if sigma is None else sigma
    n = len(ball)
    crowd, air, line = physical_conditions(ball, players)
    mult = np.ones(n)
    mult = np.where(air, mult * AIR_MULT, mult)
    mult = np.where(crowd, mult * CROWD_MULT, mult)
    mult = np.where(line, mult * LINE_MULT, mult)
    p_start = np.minimum(0.5, base_rate * mult)
    starts = rng.random(n) < p_start
    z = rng.standard_normal(n)       # drawn for every frame so the stream length is fixed
    hidden = np.zeros(n, bool)
    t = 0
    idx = np.flatnonzero(starts)
    for a in idx:
        if a < t:
            continue
        d = dur_median * np.exp(sigma * z[a]) * (LINE_DUR_MULT if line[a] else 1.0)
        b = min(n, a + max(1, int(round(d))))
        hidden[a:b] = True
        t = b
    return hidden


def degrade(match, seed, player_smooth=PLAYER_SMOOTH, **kw):
    """Degraded copy of a bench.dfl.load() dict. Reads only ball and players (never alive/owner)."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in match.items()}
    ball, players = out["ball"], out["players"]
    n = len(ball)

    hidden = ball_hidden_mask(rng, ball, players, **kw)
    ball[:, :2] = v2._smooth(ball[:, :2], v2.BALL_SMOOTH)
    ball[hidden] = np.nan

    players[:] = v2._smooth(players, player_smooth)
    for side in range(2):
        for slot in range(players.shape[2]):
            present = ~np.isnan(players[:, side, slot, 0])
            if not present.any():
                continue
            drop = v2._bursts(rng, n, v2.PLAYER_DROP, v2.PLAYER_DROP_GAP)
            players[drop & present, side, slot] = np.nan

    dups = []
    for side in range(2):
        on = v2._bursts(rng, n, v2.DUP_SHARE, v2.DUP_GAP)
        copy = np.full((n, 2), np.nan, np.float32)
        edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            live = np.flatnonzero(~np.isnan(players[a, side, :, 0]))
            if len(live) == 0:
                continue
            k = live[rng.integers(len(live))]
            copy[a:b] = players[a:b, side, k] + rng.normal(0, v2.DUP_OFFSET, 2).astype(np.float32)
        dups.append(copy)
    extra = np.stack(dups, axis=1)[:, :, None, :]
    out["players"] = np.concatenate([players, extra], axis=2)
    out["ball_hidden_share"] = float(np.isnan(ball[:, 0]).mean())
    return out


def degrade_v2s(match, seed):
    """The old v2 degrader with PLAYER_SMOOTH = 12 (module constant patched for the call)."""
    old = v2.PLAYER_SMOOTH
    v2.PLAYER_SMOOTH = 12
    try:
        return v2.degrade(match, seed)
    finally:
        v2.PLAYER_SMOOTH = old


DEGRADERS = {"v4": degrade, "v2s": degrade_v2s}


def gap_lengths(ball_x, period=None):
    """Lengths (frames) of runs of NaN ball, per period."""
    miss = np.isnan(ball_x)
    if period is None:
        period = np.zeros(len(miss), np.int8)
    out = []
    for p in np.unique(period):
        m = miss[period == p].astype(np.int8)
        e = np.diff(np.concatenate(([0], m, [0])))
        out.extend((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist())
    return np.asarray(out)


def edge_probe(alive, ball_x, tol=2):
    """Share of alive<->dead transitions with a ball visible<->hidden change within +-tol frames,
    and the same share at random positions (chance)."""
    vis_edge = np.flatnonzero(np.diff(np.isnan(ball_x).astype(np.int8)) != 0)
    tr = np.flatnonzero(np.diff(alive.astype(np.int8)) != 0)

    def near(pos):
        if len(vis_edge) == 0 or len(pos) == 0:
            return 0.0
        j = np.searchsorted(vis_edge, pos)
        lo = np.abs(pos - vis_edge[np.clip(j - 1, 0, len(vis_edge) - 1)])
        hi = np.abs(pos - vis_edge[np.clip(j, 0, len(vis_edge) - 1)])
        return float((np.minimum(lo, hi) <= tol).mean())
    rng = np.random.default_rng(0)
    return near(tr), near(rng.integers(0, len(ball_x) - 1, 5000)), len(tr)
