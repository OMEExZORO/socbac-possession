"""Keeper-proof attack direction (from Aditya, findings-aditya bc74612). Label-free, positions only."""
import numpy as np


def outfield_gap(players, period):
    """{period: median over frames of (trimmed mean x of side 1 - trimmed mean x of side 0)}.
    Trimmed = each team's smallest and largest x are dropped per frame, so a goalkeeper (own or
    mis-filed) cannot move the team's position. Positive -> side 0 defends the x=0 goal."""
    out = {}
    x = players[..., 0]
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        means = []
        for side in (0, 1):
            v = np.sort(x[idx, side, :], axis=1)                 # NaN sorts last
            cnt = np.isfinite(v).sum(axis=1)
            tot = np.nansum(v, axis=1) - v[:, 0]                  # drop the smallest
            last = v[np.arange(len(idx)), np.maximum(cnt - 1, 0)]
            tot = tot - np.where(cnt >= 3, last, 0.0)             # drop the largest
            m = np.where(cnt >= 3, tot / np.maximum(cnt - 2, 1), np.nan)
            means.append(m)
        d = means[1] - means[0]
        d = d[np.isfinite(d)]
        if len(d):
            out[int(p)] = (float(np.median(d)), float(np.mean(d > 0)))
    return out


def attack_sign_robust(xy, period):
    """Drop-in for bench.feats3._attack_sign: +1 for the team defending x=0, from outfield_gap."""
    n = xy.shape[0]
    sign = np.ones((n, 2), np.float32)
    sign[:, 1] = -1.0
    for p, (gap, share) in outfield_gap(xy, np.asarray(period)).items():
        idx = np.flatnonzero(np.asarray(period) == p)
        if abs(gap) < 1.0:
            if p == 0:
                continue                    # outside the graded halves: nothing to orient, keep default
            raise RuntimeError("attack direction ambiguous in period %s (outfield gap %.2f m)" % (p, gap))
        left = 0 if gap > 0 else 1
        sign[idx, :] = -1.0
        sign[idx, left] = 1.0
    return sign
