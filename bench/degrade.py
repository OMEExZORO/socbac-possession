"""Make clean open-data tracking look like the challenge's automatic tracking (ATD).

v2 is calibrated against match-level aggregates of the real ATD (python -m bench.fingerprint
prints them side by side; shares, means and rates only, no labels). See docs/team/BENCH.md.

Ball gaps are drawn as bursts from a two-state Markov chain, with a higher hide rate while the
ball is dead than while it is alive, and gaps start more often when the ball is in the air or in a
crowd. Positions are smoothed rather than jittered, players drop out in bursts, and a duplicate
of one player sometimes appears. Every draw is seeded, so a (match, seed) pair always gives the
same degraded copy.
"""

import numpy as np

# v2 calibration (2026-11-07) against match-level aggregates of the real ATD (bench/fingerprint.py):
# the ATD loses the ball rarely but for long stretches (about 1.4 gaps a minute, mean ~430
# frames, 59% present), its ball track is smooth, its tracklets last ~100 s, and a team shows more
# than 11 tracked objects on ~13% of frames (duplicates). v1 of this file hid the ball 10x as
# often for 1/8 the time, which taught the model that "missing for long" means "dead".
ALIVE_HIDE, ALIVE_GAP = 0.29, 500
DEAD_HIDE, DEAD_GAP = 0.55, 800
CROWD_RADIUS = 3.0      # m: three or more players this close to the ball counts as a crowd
CROWD_BOOST = 1.6       # odds multiplier for starting a gap in a crowd or with the ball in the air
PLAYER_DROP = 0.07      # share of player-frames lost, in bursts
PLAYER_DROP_GAP = 200   # frames, mean length of a lost-player burst
DUP_SHARE = 0.15        # per team: share of frames with one duplicated player
DUP_GAP = 100           # frames, mean length of a duplicate burst
DUP_OFFSET = 0.8        # m, how far a duplicate sits from the player it copies
# The ATD is temporally smooth: its median player speed matches clean DFL (white jitter inflated
# it 3x and made every real frame look like a stoppage) and its ball acceleration matches clean
# DFL averaged over ~1 s. So positions are smoothed, not jittered.
BALL_SMOOTH = 25        # frames, centred moving average on the ball track
PLAYER_SMOOTH = 5       # frames, centred moving average on player tracks


def _bursts(rng, n, hide_rate, mean_gap, boost=None):
    """Boolean mask of hidden frames from a two-state chain with the given stationary rate."""
    p_end = 1.0 / mean_gap
    # stationary hide share h = p_start / (p_start + p_end)  ->  p_start = h * p_end / (1 - h)
    p_start = hide_rate * p_end / (1.0 - hide_rate)
    p_start = np.full(n, p_start) if boost is None else np.minimum(0.5, p_start * boost)
    draws = rng.random(n)
    hidden = np.zeros(n, bool)
    state = False
    for t in range(n):
        state = (draws[t] >= p_end) if state else (draws[t] < p_start[t])
        hidden[t] = state
    return hidden


def _smooth(a, w):
    """Centred moving average along time that ignores NaN (NaN stays NaN)."""
    if w <= 1:
        return a
    k = np.ones(w) / w
    flat = a.reshape(len(a), -1).copy()
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = ~np.isnan(col)
        if not ok.any():
            continue
        num = np.convolve(np.where(ok, col, 0.0), k, "same")
        den = np.convolve(ok.astype(float), k, "same")
        flat[:, j] = np.where(ok, num / np.maximum(den, 1e-9), np.nan)
    return flat.reshape(a.shape).astype(a.dtype)


def degrade(match, seed):
    """Return a degraded copy of a bench.dfl.load() dict. Labels are left untouched."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in match.items()}
    ball, players, alive = out["ball"], out["players"], out["alive"]
    n = len(alive)

    # crowd / air boost for starting a gap
    flat = players.reshape(n, -1, 2)
    dist = np.linalg.norm(flat - ball[:, None, :2], axis=2)
    crowd = (np.nan_to_num(dist, nan=99.0) < CROWD_RADIUS).sum(axis=1) >= 3
    air = np.nan_to_num(ball[:, 2]) > 2.0
    boost = np.where(crowd | air, CROWD_BOOST, 1.0)

    hide_alive = _bursts(rng, n, ALIVE_HIDE, ALIVE_GAP, boost)
    hide_dead = _bursts(rng, n, DEAD_HIDE, DEAD_GAP)
    hidden = np.where(alive, hide_alive, hide_dead)
    ball[:, :2] = _smooth(ball[:, :2], BALL_SMOOTH)
    ball[hidden] = np.nan

    # players: smoothing + dropped bursts per slot
    players[:] = _smooth(players, PLAYER_SMOOTH)
    for side in range(2):
        for slot in range(players.shape[2]):
            present = ~np.isnan(players[:, side, slot, 0])
            if not present.any():
                continue
            drop = _bursts(rng, n, PLAYER_DROP, PLAYER_DROP_GAP)
            players[drop & present, side, slot] = np.nan

    # duplicates: the tracker sometimes follows one player twice, in a spare slot of that team
    dups = []
    for side in range(2):
        on = _bursts(rng, n, DUP_SHARE, DUP_GAP)
        copy = np.full((n, 2), np.nan, np.float32)
        edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            live = np.flatnonzero(~np.isnan(players[a, side, :, 0]))
            if len(live) == 0:
                continue
            k = live[rng.integers(len(live))]       # one player copied for the whole burst
            copy[a:b] = players[a:b, side, k] + rng.normal(0, DUP_OFFSET, 2).astype(np.float32)
        dups.append(copy)
    extra = np.stack(dups, axis=1)[:, :, None, :]            # (n, 2, 1, 2)
    out["players"] = players = np.concatenate([players, extra], axis=2)

    out["ball_hidden_share"] = float(np.isnan(ball[:, 0]).mean())
    return out
