"""Provenance-clean degrader (prov_clean, 2026-11-13). New module; reads only an IDSSE bench.dfl.load() dict.

Numbers taken from the organisers' docs (SOCBAC\\DATA.md), nothing else:
  VIS_TARGET 0.593   ball visible on 59.3% of graded frames (DATA.md "94,214 / 158,857")
  MEAN_OBJ   20.5    mean tracked player objects over graded frames (DATA.md "the mean over graded frames is 20.5")
  N_SLOTS    46      track slots (DATA.md); used only as a sanity cap on objects per frame
Everything else is a BROAD RANGE drawn per copy (two copies per match are drawn stratified, so they sit in
different halves of every range):
  gap law       lognormal; median = 190 x {0.5, 1, 2} (the two copies get two different scales),
                log-sigma U[0.6, 1.3]; per-frame start rate solved by bisection so visible share = 59.3%
  hazard        x2 ball in the air (z > 2 m), x2 crowd (>= 3 players within 3 m) as v4 (free physical
                choices, class c); line hazard ON (x3 start, x2 duration within 1 m of a line) in one copy,
                OFF in the other
  ball smooth   centred box, integer U[5, 40] frames
  player smooth centred box, integer U[3, 25] frames
  duplicates    per team share U[0.02, 0.30]; burst mean log-U[25, 250] frames; offset U[0.3, 1.5] m
  player drops  share solved so that mean player objects (real + duplicates) = 20.5
                (drop = 1 - (20.5 - 2 * dup) / mean clean players on the pitch); burst mean log-U[50, 500] frames
Not used: v5ball / any rival-reported constant, gap median/mean/p90 of the provided tracking, PLAYER_SMOOTH 12,
BALL_SMOOTH 25 or DUP_SHARE .15 as anchors (the x1 gap scale reuses v4's 190-frame median as the centre of the
x0.5..x2 range, as the brief allows). alive/owner are never read.
"""
import numpy as np

from bench import degrade as v2

PITCH = (105.0, 68.0)
VIS_TARGET = 0.593
MEAN_OBJ = 20.5
N_SLOTS = 46
AIR_Z, CROWD_R, LINE_MARGIN = 2.0, 3.0, 1.0
AIR_MULT, CROWD_MULT = 2.0, 2.0
GAP_MEDIAN_X1 = 190.0
SCALES = (0.5, 1.0, 2.0)
REV = "pc1"


def draw_regimes(seed, mi, n_copies=2):
    """Regimes of the copies of match index mi for a seed: stratified so the copies differ on every knob."""
    rng = np.random.default_rng(880000 + 1000 * seed + mi)
    scales = rng.choice(len(SCALES), size=n_copies, replace=False)
    out = []
    u = (np.array([rng.permutation(n_copies) for _ in range(6)]) + rng.random((6, n_copies))) / n_copies
    lo = lambda a, b, x: float(a + (b - a) * x)
    loglo = lambda a, b, x: float(np.exp(np.log(a) + (np.log(b) - np.log(a)) * x))
    for k in range(n_copies):
        out.append(dict(
            gap_scale=SCALES[int(scales[k])],
            sigma=round(lo(0.6, 1.3, u[0, k]), 3),
            ball_smooth=int(round(lo(5, 40, u[1, k]))),
            player_smooth=int(round(lo(3, 25, u[2, k]))),
            dup_share=round(lo(0.02, 0.30, u[3, k]), 3),
            dup_gap=round(loglo(25, 250, u[4, k]), 1),
            drop_gap=round(loglo(50, 500, u[5, k]), 1),
            dup_offset=round(lo(0.3, 1.5, rng.random()), 3),
            mask_seed=int(rng.integers(2 ** 31))))
    first = bool(rng.integers(2))
    for k, r in enumerate(out):
        r["line"] = first if k % 2 == 0 else not first
    return out


def conditions(ball, players):
    n = len(ball)
    flat = players.reshape(n, -1, 2)
    dist = np.linalg.norm(flat - ball[:, None, :2], axis=2)
    crowd = (np.nan_to_num(dist, nan=99.0) < CROWD_R).sum(axis=1) >= 3
    air = np.nan_to_num(ball[:, 2]) > AIR_Z
    bx, by = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        line = ((bx < LINE_MARGIN) | (bx > PITCH[0] - LINE_MARGIN) | (by < LINE_MARGIN) | (by > PITCH[1] - LINE_MARGIN))
    return crowd, air, np.nan_to_num(line, nan=0).astype(bool)


def hidden_mask(cond, n, rate, median, sigma, line_on, seed):
    crowd, air, line = cond
    rng = np.random.default_rng(seed)
    mult = np.ones(n)
    mult = np.where(air, mult * AIR_MULT, mult)
    mult = np.where(crowd, mult * CROWD_MULT, mult)
    if line_on:
        mult = np.where(line, mult * 3.0, mult)
    starts = rng.random(n) < np.minimum(0.5, rate * mult)
    z = rng.standard_normal(n)
    hidden = np.zeros(n, bool)
    t = 0
    for a in np.flatnonzero(starts):
        if a < t:
            continue
        d = median * np.exp(sigma * z[a]) * (2.0 if (line_on and line[a]) else 1.0)
        b = min(n, a + max(1, int(round(d))))
        hidden[a:b] = True
        t = b
    return hidden


def solve_mask(ball, players, reg):
    """Bisection on the start rate (log scale) so that the visible share of the copy is VIS_TARGET."""
    n = len(ball)
    cond = conditions(ball, players)
    med = GAP_MEDIAN_X1 * reg["gap_scale"]
    lo, hi = np.log(1e-6), np.log(0.2)
    best = None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        h = hidden_mask(cond, n, float(np.exp(mid)), med, reg["sigma"], reg["line"], reg["mask_seed"])
        vis = 1.0 - h.mean()
        if best is None or abs(vis - VIS_TARGET) < abs(best[1] - VIS_TARGET):
            best = (h, vis, float(np.exp(mid)))
        if vis > VIS_TARGET:
            lo = mid
        else:
            hi = mid
        if abs(vis - VIS_TARGET) < 0.002:
            break
    return best


def degrade(match, seed, reg):
    """Degraded copy of a bench.dfl.load() dict under regime reg (see draw_regimes). Reads only ball and players."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in match.items()}
    ball, players = out["ball"], out["players"]
    n = len(ball)
    hidden, vis, rate = solve_mask(ball, players, reg)
    ball[:, :2] = v2._smooth(ball[:, :2], reg["ball_smooth"])
    ball[hidden] = np.nan

    clean_count = float((~np.isnan(players[..., 0])).sum(axis=(1, 2)).mean())
    drop = float(np.clip(1.0 - (MEAN_OBJ - 2.0 * reg["dup_share"]) / clean_count, 0.0, 0.5))
    players[:] = v2._smooth(players, reg["player_smooth"])
    for side in range(2):
        for slot in range(players.shape[2]):
            present = ~np.isnan(players[:, side, slot, 0])
            if not present.any():
                continue
            d = v2._bursts(rng, n, drop, reg["drop_gap"])
            players[d & present, side, slot] = np.nan
    dups = []
    for side in range(2):
        on = v2._bursts(rng, n, reg["dup_share"], reg["dup_gap"])
        copy = np.full((n, 2), np.nan, np.float32)
        edges = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            live = np.flatnonzero(~np.isnan(players[a, side, :, 0]))
            if len(live) == 0:
                continue
            k = live[rng.integers(len(live))]
            copy[a:b] = players[a:b, side, k] + rng.normal(0, reg["dup_offset"], 2).astype(np.float32)
        dups.append(copy)
    out["players"] = np.concatenate([players, np.stack(dups, axis=1)[:, :, None, :]], axis=2)
    cnt = (~np.isnan(out["players"][..., 0])).sum(axis=(1, 2))
    assert cnt.max() <= N_SLOTS
    out["ball_hidden_share"] = float(np.isnan(ball[:, 0]).mean())
    info = dict(reg, rate=rate, vis=float(vis), drop=drop, clean_players=clean_count,
                mean_objects=float(cnt.mean()))
    return out, info
