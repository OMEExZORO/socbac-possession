"""Alternative, label-independent degradation of open tracking (robustness test and mixture training).

bench/degrade.py hides the ball with two Markov chains switched on the TRUE alive flag, so ball-gap
edges line up with alive/dead changes (54% of transitions have a gap edge within +-2 frames against
a 1% base rate) and dead frames are hidden far more often than alive ones. A model trained on it can
use "the ball just vanished" as a dead-ball shortcut that the real tracking may not offer.

Here the ball-hiding hazard depends only on observable geometry of the clean track, never on the
alive flag or the owner:
  * a gap STARTS on a visible frame with probability base * boost(t), where boost multiplies
    AIR_X if the ball is above AIR_Z metres, CROWD_X if CROWD_N or more players are within
    CROWD_RADIUS of it, and LINE_X if it is within LINE_DIST of a touchline or goal line or beyond one;
  * a gap LASTS a lognormal number of frames with median GAP_MEDIAN and mean GAP_MEAN (the real ATD
    has gaps of median ~270 and mean ~430 frames: match-level aggregates of the tracking, no labels);
  * base is solved per match from the same random draws so the hidden share lands on HIDE_SHARE
    (the real ATD shows the ball on ~59% of graded frames). Calibration uses no labels.
The multipliers are assumptions (the real hazard is unknown); they only have to make hiding depend on
play geometry instead of on the label.

Smoothing, player dropout bursts and duplicate players are exactly bench.degrade's (same constants,
including the current value of bench.degrade.PLAYER_SMOOTH, which the v2 pipeline sets to 12).
Every draw is seeded: a (match, seed) pair always gives the same copy.
"""

import numpy as np

from bench import degrade as dg

HIDE_SHARE = 0.41          # ball missing on ~41% of frames (present ~59%)
GAP_MEDIAN = 270.0         # frames
GAP_MEAN = 430.0           # frames
AIR_Z, AIR_X = 2.0, 3.0    # m, hazard multiplier
CROWD_RADIUS, CROWD_N, CROWD_X = 3.0, 3, 2.0
LINE_DIST, LINE_X = 3.0, 3.0
PITCH = (105.0, 68.0)
CALIB_PASSES = 4
REV = "alt1"

_SIGMA = float(np.sqrt(2.0 * np.log(GAP_MEAN / GAP_MEDIAN)))  # lognormal: mean = median * exp(s^2/2)


def boost(match):
    """Per-frame hazard multiplier from ball height, crowding and lines (clean track, no labels)."""
    ball, players = match["ball"], match["players"]
    n = len(ball)
    flat = players.reshape(n, -1, 2)
    dist = np.linalg.norm(flat - ball[:, None, :2], axis=2)
    crowd = (np.nan_to_num(dist, nan=99.0) < CROWD_RADIUS).sum(axis=1) >= CROWD_N
    air = np.nan_to_num(ball[:, 2]) > AIR_Z
    bx, by = ball[:, 0], ball[:, 1]
    edge = np.minimum.reduce([bx, PITCH[0] - bx, by, PITCH[1] - by])
    line = np.nan_to_num(edge, nan=99.0) < LINE_DIST          # includes beyond the line (edge < 0)
    b = np.ones(n)
    b = np.where(air, b * AIR_X, b)
    b = np.where(crowd, b * CROWD_X, b)
    b = np.where(line, b * LINE_X, b)
    return b


def _hide(u, z, hazard_boost, base):
    """Hidden mask from pre-drawn uniforms u (gap starts) and normals z (gap lengths, in order)."""
    n = len(u)
    hidden = np.zeros(n, bool)
    p = np.minimum(0.5, base * hazard_boost)
    t, k = 0, 0
    while t < n:
        if u[t] < p[t]:
            length = max(1, int(round(GAP_MEDIAN * np.exp(_SIGMA * z[k % len(z)]))))
            k += 1
            hidden[t:t + length] = True
            t += length
        else:
            t += 1
    return hidden


def hidden_mask(match, rng):
    """Calibrated label-free hidden mask for one match."""
    n = len(match["ball"])
    b = boost(match)
    u = rng.random(n)
    z = rng.standard_normal(max(16, n // 50))
    h = HIDE_SHARE
    # start rate per visible frame that gives share h with mean gap length GAP_MEAN
    rate = h / ((1.0 - h) * GAP_MEAN)
    base = rate / float(b.mean())
    hidden = _hide(u, z, b, base)
    for _ in range(CALIB_PASSES):
        got = float(np.clip(hidden.mean(), 1e-3, 0.999))
        base *= (h / (1.0 - h)) / (got / (1.0 - got))
        hidden = _hide(u, z, b, base)
    return hidden


def degrade(match, seed):
    """Return a degraded copy of a bench.dfl.load() dict. Labels are left untouched and never read."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in match.items()}
    ball, players = out["ball"], out["players"]
    n = len(ball)

    hidden = hidden_mask(match, rng)
    ball[:, :2] = dg._smooth(ball[:, :2], dg.BALL_SMOOTH)
    ball[hidden] = np.nan

    # players: exactly bench.degrade (smoothing, dropout bursts, duplicates), own seeded stream
    players[:] = dg._smooth(players, dg.PLAYER_SMOOTH)
    for side in range(2):
        for slot in range(players.shape[2]):
            present = ~np.isnan(players[:, side, slot, 0])
            if not present.any():
                continue
            drop = dg._bursts(rng, n, dg.PLAYER_DROP, dg.PLAYER_DROP_GAP)
            players[drop & present, side, slot] = np.nan
    dups = []
    for side in range(2):
        on = dg._bursts(rng, n, dg.DUP_SHARE, dg.DUP_GAP)
        copy = np.full((n, 2), np.nan, np.float32)
        edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            live = np.flatnonzero(~np.isnan(players[a, side, :, 0]))
            if len(live) == 0:
                continue
            k = live[rng.integers(len(live))]
            copy[a:b] = players[a:b, side, k] + rng.normal(0, dg.DUP_OFFSET, 2).astype(np.float32)
        dups.append(copy)
    extra = np.stack(dups, axis=1)[:, :, None, :]
    out["players"] = np.concatenate([players, extra], axis=2)
    out["ball_hidden_share"] = float(np.isnan(ball[:, 0]).mean())
    return out


def gap_stats(ball):
    """(visible share, n gaps, median, mean, p90 gap length in frames)."""
    vis = ~np.isnan(ball[:, 0])
    edges = np.diff(np.concatenate(([0], (~vis).astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    if len(lengths) == 0:
        return float(vis.mean()), 0, 0.0, 0.0, 0.0
    return (float(vis.mean()), int(len(lengths)), float(np.median(lengths)), float(lengths.mean()),
            float(np.percentile(lengths, 90)))


def coupling(match, deg):
    """Label-side diagnostics (open data only): share of alive/dead transitions with a gap edge
    within +-2 frames, and the hidden share on alive vs dead frames."""
    alive = np.asarray(match["alive"], bool)
    vis = ~np.isnan(deg["ball"][:, 0])
    tr = np.flatnonzero(alive[1:] != alive[:-1]) + 1
    ge = np.flatnonzero(vis[1:] != vis[:-1]) + 1
    if len(tr) == 0:
        return 0.0, float((~vis[alive]).mean()), float("nan")
    near = np.zeros(len(alive) + 5, bool)
    for g in ge:
        near[max(0, g - 2):g + 3] = True
    return (float(near[tr].mean()), float((~vis[alive]).mean()),
            float((~vis[~alive]).mean()) if (~alive).any() else float("nan"))


def main():
    from bench import dfl
    dg.PLAYER_SMOOTH = 12
    print("alt degradation %s: target visible %.2f, gap median %.0f mean %.0f" % (
        REV, 1 - HIDE_SHARE, GAP_MEDIAN, GAP_MEAN))
    for kind, fn in (("coupled", dg.degrade), ("alt", degrade)):
        for i, m in enumerate(["J03WPY", "J03WQQ"]):
            clean = dfl.load(m)
            d = fn(clean, seed=4 + i)
            vis, ng, med, mean, p90 = gap_stats(d["ball"])
            ce, ha, hd = coupling(clean, d)
            print("%-8s %s visible %.3f gaps %d (%.2f/min) median %.0f mean %.0f p90 %.0f | "
                  "transitions with gap edge +-2f %.2f, hidden alive %.2f dead %.2f" % (
                      kind, m, vis, ng, ng / (len(d["ball"]) / 1500.0), med, mean, p90, ce, ha, hd))


if __name__ == "__main__":
    main()
