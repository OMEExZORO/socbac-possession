"""Realistic dead-ball degradation (v3). Does not modify bench/degrade.py.

v2 hides the ball with one Markov chain on dead frames and another on live
frames, and it lets a player dropout last a single frame. Rival measurements
of the organiser feed disagree with both:

- the ball is visible on about a quarter of dead frames
- a stoppage is almost all-or-nothing, and a lost stoppage loses the ball at
  its first frame
- player dropouts last at least 2.04 s

Live-ball gaps, smoothing and the duplicate-player draws keep v2's constants.
The live hidden mask matches v2 for the same seed: the same generator runs
first, and the draw v2 uses for dead frames is consumed and discarded so the
player-dropout stream stays aligned. Dead frames then get the stoppage rule
below instead of that discarded draw.

Keep probabilities are set so the frame-weighted visibility on these DFL
matches lands near 25%. About 78% of dead frames sit in runs of 20 s or more,
so the long bucket has to carry that target. Each run is then fully kept or
hidden from the first frame, with at most a 6% tail left visible.
"""

import numpy as np

from bench.degrade import (
    ALIVE_GAP,
    ALIVE_HIDE,
    BALL_SMOOTH,
    CROWD_BOOST,
    CROWD_RADIUS,
    DEAD_GAP,
    DEAD_HIDE,
    DUP_GAP,
    DUP_OFFSET,
    DUP_SHARE,
    PLAYER_DROP,
    PLAYER_DROP_GAP,
    PLAYER_SMOOTH,
    _bursts,
    _smooth,
)

FPS = 25
MIN_DROPOUT = 51  # 2.04 s at 25 Hz
P_KEEP_SHORT = 0.80  # dead run < 6 s
P_KEEP_MID = 0.35    # 6 s to 20 s
P_KEEP_LONG = 0.18   # >= 20 s
TAIL_FRAC = 0.06
REV = "v3a"


def _extend_min(mask, min_len):
    """Lengthen every True-run to at least min_len, growing forward.

    A run that would pass the end of the array stops at the end.
    """
    if min_len <= 1:
        return np.asarray(mask, bool)
    src = np.asarray(mask, bool)
    if not src.any():
        return src.copy()
    edges = np.diff(np.concatenate(([0], src.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    out = src.copy()
    n = len(src)
    for a, b in zip(starts, ends):
        if b - a < min_len:
            out[a:min(n, a + min_len)] = True
    return out


def _dead_hidden(alive, rng):
    """Hide dead frames per stoppage: keep the run, or hide it from the start."""
    hidden = np.zeros(len(alive), bool)
    dead = ~np.asarray(alive, bool)
    if not np.any(dead):
        return hidden
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for a, b in zip(starts, ends):
        length = int(b - a)
        sec = length / float(FPS)
        if sec < 6.0:
            p_keep = P_KEEP_SHORT
        elif sec < 20.0:
            p_keep = P_KEEP_MID
        else:
            p_keep = P_KEEP_LONG
        if float(rng.random()) < p_keep:
            continue
        tail_cap = int(TAIL_FRAC * length)
        tail = int(rng.integers(0, tail_cap + 1)) if tail_cap > 0 else 0
        stop = b - tail
        if stop > a:
            hidden[a:stop] = True
    return hidden


def degrade(match, seed):
    """Degraded copy of a bench.dfl.load() dict. Labels are not modified."""
    rng = np.random.default_rng(int(seed))
    out = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in match.items()}
    ball = out["ball"]
    players = out["players"]
    alive = np.asarray(out["alive"], bool)
    n = len(alive)

    flat = players.reshape(n, -1, 2)
    dist = np.linalg.norm(flat - ball[:, None, :2], axis=2)
    crowd = (np.nan_to_num(dist, nan=99.0) < CROWD_RADIUS).sum(axis=1) >= 3
    air = np.nan_to_num(ball[:, 2]) > 2.0
    boost = np.where(crowd | air, CROWD_BOOST, 1.0)

    hide_alive = _bursts(rng, n, ALIVE_HIDE, ALIVE_GAP, boost)
    _bursts(rng, n, DEAD_HIDE, DEAD_GAP)  # discarded; keeps v2's rng alignment
    dead_rng = np.random.default_rng(int(seed) + 17_000_011)
    hide_dead = _dead_hidden(alive, dead_rng)
    hidden = np.where(alive, hide_alive, hide_dead)

    ball[:, :2] = _smooth(ball[:, :2], BALL_SMOOTH)
    ball[hidden] = np.nan
    players[:] = _smooth(players, PLAYER_SMOOTH)
    for side in range(2):
        for slot in range(players.shape[2]):
            present = ~np.isnan(players[:, side, slot, 0])
            if not present.any():
                continue
            drop = _extend_min(_bursts(rng, n, PLAYER_DROP, PLAYER_DROP_GAP), MIN_DROPOUT)
            players[drop & present, side, slot] = np.nan

    dups = []
    for side in range(2):
        on = _bursts(rng, n, DUP_SHARE, DUP_GAP)
        copy = np.full((n, 2), np.nan, np.float32)
        edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            live = np.flatnonzero(~np.isnan(players[a, side, :, 0]))
            if len(live) == 0:
                continue
            k = int(live[rng.integers(len(live))])
            copy[a:b] = players[a:b, side, k] + rng.normal(0, DUP_OFFSET, 2).astype(np.float32)
        dups.append(copy)
    extra = np.stack(dups, axis=1)[:, :, None, :]
    out["players"] = np.concatenate([players, extra], axis=2)
    visible = ~np.isnan(ball[:, 0])
    dead = ~alive
    out["ball_hidden_share"] = float((~visible).mean())
    out["ball_dead_visible"] = float(visible[dead].mean()) if np.any(dead) else float("nan")
    out["ball_live_visible"] = float(visible[alive].mean()) if np.any(alive) else float("nan")
    out["degrade"] = REV
    return out


def _run_bounds(mask):
    m = np.asarray(mask, np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return starts, ends


def summarise(clean, degraded):
    """Match-level shares only. No coordinates and no per-frame records."""
    alive = np.asarray(clean["alive"], bool)
    visible = ~np.isnan(degraded["ball"][:, 0])
    dead = ~alive
    starts, ends = _run_bounds(dead)
    vis_share = []
    lost = 0
    lost_at_start = 0
    for a, b in zip(starts, ends):
        share = float(visible[a:b].mean()) if b > a else 1.0
        vis_share.append(share)
        if share < 0.5:
            lost += 1
            hid = np.flatnonzero(~visible[a:b])
            if len(hid) and int(hid[0]) == 0:
                lost_at_start += 1
    shares = np.asarray(vis_share, np.float64) if vis_share else np.zeros(0)
    if len(shares):
        all_or_nothing = float(np.mean((shares >= 0.9) | (shares <= 0.1)))
    else:
        all_or_nothing = float("nan")

    clean_p = clean["players"]
    deg_p = degraded["players"][:, :, :clean_p.shape[2], :]
    present = ~np.isnan(clean_p[..., 0])
    missing = present & np.isnan(deg_p[..., 0])
    lengths = []
    violations = 0
    checked = 0
    for side in range(missing.shape[1]):
        for slot in range(missing.shape[2]):
            ms, me = _run_bounds(missing[:, side, slot])
            pres = present[:, side, slot]
            for a, b in zip(ms, me):
                length = int(b - a)
                lengths.append(length)
                horizon = min(len(pres), a + MIN_DROPOUT)
                if horizon - a >= MIN_DROPOUT and bool(pres[a:horizon].all()):
                    checked += 1
                    if length < MIN_DROPOUT:
                        violations += 1
    lengths = np.asarray(lengths, np.int32) if lengths else np.zeros(0, np.int32)
    if len(lengths):
        short_share = float(np.mean(lengths < MIN_DROPOUT))
        med = float(np.median(lengths))
        min_len = int(lengths.min())
    else:
        short_share = 0.0
        med = 0.0
        min_len = 0
    return {
        "dead_visible": float(visible[dead].mean()) if np.any(dead) else float("nan"),
        "live_visible": float(visible[alive].mean()) if np.any(alive) else float("nan"),
        "hidden_share": float((~visible).mean()),
        "n_dead_runs": int(len(shares)),
        "all_or_nothing": all_or_nothing,
        "lost_at_start": float(lost_at_start / lost) if lost else float("nan"),
        "dropout_runs": int(len(lengths)),
        "dropout_min": min_len,
        "dropout_median": med,
        "dropout_short_share": short_share,
        "dropout_violations": int(violations),
        "dropout_checked": int(checked),
    }


def aggregate(matches, seed=0):
    """Mean of per-match shares. Prints nothing."""
    from bench.dfl import load
    from bench.degrade import degrade as degrade_v2

    rows = []
    live_gap = []
    for i, match in enumerate(matches):
        clean = load(match)
        deg = degrade(clean, seed=1000 * seed + i)
        base = degrade_v2(clean, seed=1000 * seed + i)
        rec = summarise(clean, deg)
        alive = np.asarray(clean["alive"], bool)
        v2_live = ~np.isnan(base["ball"][:, 0])
        v3_live = ~np.isnan(deg["ball"][:, 0])
        rec["live_mask_match"] = bool(np.array_equal(v2_live[alive], v3_live[alive]))
        live_gap.append(rec["live_mask_match"])
        rows.append(rec)
        del clean, deg, base
    keys = [k for k in rows[0] if k != "live_mask_match"]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["live_mask_match"] = bool(all(live_gap))
    out["matches"] = len(rows)
    return out
