"""Noise fingerprint: a few match-level aggregates, side by side for the real ATD and the bench.

    python -m bench.fingerprint

It answers one question: does bench/degrade.py make open DFL tracking as noisy as BAR-ATH's ATD,
in the ways that matter for possession? Every number printed is a match-wide aggregate (a share,
a mean, a rate per minute) over the graded halves. No rows, values or per-frame records are
printed (AGENTS.md section 1). Labels are never read: the public GT is not touched.
"""

import numpy as np

from bench import dfl, methods
from bench.degrade import degrade
from bench.run import TEST, TRAIN

PITCH = methods.PITCH


def _runs(mask):
    """Lengths of the True runs in a boolean vector."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def fingerprint(match):
    period = match["period"]
    rows = []
    for p in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == p)
        sub = {k: match[k][idx] for k in ("ball", "players", "period")}
        rows.append(_one(sub))
    tot = {k: sum(r[k] for r in rows) for k in rows[0]}
    mins = tot["frames"] / 1500.0
    vis = max(tot["visible"], 1)
    return {
        "ball present": tot["visible"] / tot["frames"],
        "ball gap mean (frames)": tot["gap_frames"] / max(tot["gaps"], 1),
        "ball gaps / min": tot["gaps"] / mins,
        "ball step > 1.2 m (of visible pairs)": tot["jumps"] / max(tot["pairs"], 1),
        "ball off pitch > 1 m (of visible)": tot["off"] / vis,
        "tracked per frame": tot["tracked"] / tot["frames"],
        "frames with a team > 11": tot["over11"] / tot["frames"],
        "tracklets started / min": tot["tracklets"] / mins,
        "tracklet mean length (s)": tot["tracklet_frames"] / max(tot["tracklets"], 1) / 25.0,
        "nearest-team flips / min": tot["flips"] / mins,
        "close-team flips / min (both < 3 m)": tot["close_flips"] / mins,
        "baseline possessions / 10 min": 10 * tot["base_poss"] / mins,
    }


def _one(m):
    ball, players = m["ball"], m["players"]
    n = len(ball)
    vis = ~np.isnan(ball[:, 0])
    gaps = _runs(~vis)

    both = vis[1:] & vis[:-1]
    step = np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1)
    jumps = int(np.sum(both & (np.nan_to_num(step) > 1.2)))
    bx, by = ball[:, 0], ball[:, 1]
    off = vis & ((bx < -1) | (bx > PITCH[0] + 1) | (by < -1) | (by > PITCH[1] + 1))

    present = ~np.isnan(players[..., 0])                      # (n, 2, slots)
    per_team = present.sum(axis=2)
    tl = [_runs(present[:, s, k]) for s in range(2) for k in range(present.shape[2])]
    tl = np.concatenate(tl) if tl else np.zeros(0, int)

    dh, da = methods.nearest(m)
    team = np.where(dh <= da, 0, 1)
    ok = vis[1:] & vis[:-1]
    flip = ok & (team[1:] != team[:-1])
    close = flip & (np.nan_to_num(dh[1:], nan=99) < 3) & (np.nan_to_num(da[1:], nan=99) < 3)

    base = methods.baseline_smooth50({"ball": ball, "players": players, "period": m["period"]})
    teamrun = np.where(base == methods.D, -1, base)
    starts = np.sum((teamrun[1:] != teamrun[:-1]) & (teamrun[1:] >= 0)) + int(teamrun[0] >= 0)

    return {"frames": n, "visible": int(vis.sum()), "gaps": len(gaps), "gap_frames": int(gaps.sum()),
            "pairs": int(both.sum()), "jumps": jumps, "off": int(off.sum()),
            "tracked": int(per_team.sum()), "over11": int(np.sum((per_team > 11).any(axis=1))),
            "tracklets": len(tl), "tracklet_frames": int(tl.sum()),
            "flips": int(flip.sum()), "close_flips": int(close.sum()), "base_poss": int(starts)}


def main():
    import sys
    cols = {}
    if "--no-atd" not in sys.argv:
        from socbac import atd
        cols["ATD (BAR-ATH)"] = fingerprint(atd.load())
    clean = {m: dfl.load(m) for m in TRAIN + TEST}
    fps_clean = [fingerprint(clean[m]) for m in clean]
    fps_deg = [fingerprint(degrade(clean[m], seed=i)) for i, m in enumerate(clean)]
    cols["DFL degraded"] = {k: np.mean([f[k] for f in fps_deg]) for k in fps_deg[0]}
    cols["DFL clean"] = {k: np.mean([f[k] for f in fps_clean]) for k in fps_clean[0]}

    names = list(cols)
    print("{:<40}".format("aggregate over graded halves") + "".join("{:>16}".format(c) for c in names))
    for k in cols[names[0]]:
        print("{:<40}".format(k) + "".join("{:>16.3f}".format(cols[c][k]) for c in names))


if __name__ == "__main__":
    main()


def feature_shift():
    """Per-feature standardised mean difference, ATD vs degraded DFL (aggregates only)."""
    from socbac import atd
    m = atd.load()
    xa, names = methods.dense_features(m)
    xa = xa[m["period"] > 0]
    xs = []
    for i, k in enumerate(TRAIN + TEST):
        d = degrade(dfl.load(k), seed=i)
        x, _ = methods.dense_features(d)
        xs.append(x[d["period"] > 0])
    xb = np.concatenate(xs)
    ma, mb = xa.mean(0), xb.mean(0)
    sd = np.sqrt(0.5 * (xa.var(0) + xb.var(0))) + 1e-9
    smd = (ma - mb) / sd
    print("{:<18}{:>10}{:>10}{:>8}".format("feature", "ATD mean", "DFL mean", "SMD"))
    for j in np.argsort(-np.abs(smd))[:15]:
        print("{:<18}{:>10.3f}{:>10.3f}{:>8.2f}".format(names[j], ma[j], mb[j], smd[j]))
