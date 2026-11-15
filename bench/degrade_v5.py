"""Degrader v5: v4 as the base setting plus switchable stress options. Label independent.

Never reads `alive` or `owner`. Every option is OFF by default, so degrade(match, seed) is
bit-identical to bench.degrade_v4.degrade(match, seed). Every option draws from its own random
stream (np.random.default_rng([seed, 5, k])), so switching one option on never changes what another
produces, and every call is deterministic given (match, seed, options).

Options
  ball_stress=True   Replaces v4's ball hiding by a regime-matched stress setting from rival
                     measurements of the real feed (not our data): about 10% of live frames hidden
                     (short gaps), about 75% of dead-like frames hidden, all-or-nothing per dead run
                     and lost within the first 10 frames of it. "Dead-like" is a label-free physical
                     proxy: ball outside the pitch, or ball nearly still for >= 3 s.
  swaps=r            cross-team ID swaps: r events per 10 000 frames; each exchanges the trajectories
                     of one home and one nearby away tracklet for 0.5-4 s.
  touchline=p        a run of a player within 2 m of a touchline is dropped with probability p
                     (dropout starts at the run's start and ends 25 frames after it).
  extra_object=True  one referee-like slot (follows the ball loosely), assigned to a random team side,
                     present about 85% of frames. Appends one slot to both sides.
  gk_missing=s       the goalkeeper slot of one side is missing for stretches (10-60 s) covering about
                     share s of the frames of the match (summed over both sides).

Use `regime_report(clean, degraded)` (reads alive, for MEASUREMENT only, never as an input) to print
the visible share by live/dead.

    python -m bench.degrade_v5      # unit tests on synthetic matches of a few thousand frames
"""

import numpy as np
from scipy.signal import lfilter

from bench import degrade as v2
from bench import degrade_v4 as v4

REV = "v5a"
PITCH = v4.PITCH
FPS = 25

# ball regime (rival measurements of the real feed, via research/13 section 1.2)
LIVE_HIDE = 0.10
DEAD_HIDE_PROB = 0.77        # per dead-like run; with the <= 10 frame lag this gives ~75% of its frames
DEAD_LAG_MAX = 10
LIVE_DUR_MEDIAN = 50.0       # frames
LIVE_SIGMA = 0.8
STILL_SPEED = 0.5            # m/s
STILL_MIN_FRAMES = 75        # 3 s

TOUCH_MARGIN = 2.0
TOUCH_PAD = 25
SWAP_MIN, SWAP_MAX, SWAP_DIST = 12, 100, 4.0
SWAP_SCAN = 250
EXTRA_PRESENT = 0.85
GK_LEN = (250, 1500)


def _rng(seed, k):
    return np.random.default_rng([int(seed), 5, int(k)])


def _runs(mask):
    """(starts, stops) of maximal True runs."""
    m = np.concatenate(([0], np.asarray(mask, np.int8), [0]))
    e = np.diff(m)
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


# ---- ball regime ---------------------------------------------------------------------------

def deadlike_mask(ball):
    """Label-free proxy for dead-ball time, from the ball track only."""
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        outside = (x < 0) | (x > PITCH[0]) | (y < 0) | (y > PITCH[1])
    outside = np.nan_to_num(outside, nan=0).astype(bool)
    n = len(ball)
    step = np.zeros(n)
    step[1:] = np.linalg.norm(np.diff(ball[:, :2], axis=0), axis=1) * FPS
    step = np.nan_to_num(step, nan=99.0)
    k = 25
    sm = np.convolve(step, np.ones(k) / k, mode="same")
    still = sm < STILL_SPEED
    dead = outside.copy()
    for a, b in zip(*_runs(still)):
        if b - a >= STILL_MIN_FRAMES:
            dead[max(0, a - k // 2):b + k // 2] = True   # undo the smoothing window's shrinkage
    return dead


def ball_hidden_stress(rng, ball, players, live_share=LIVE_HIDE, dead_prob=DEAD_HIDE_PROB,
                       live_dur_median=LIVE_DUR_MEDIAN, sigma=LIVE_SIGMA):
    n = len(ball)
    dead = deadlike_mask(ball)
    hidden = np.zeros(n, bool)
    starts, stops = _runs(dead)
    u = rng.random(len(starts))
    lag = rng.integers(0, DEAD_LAG_MAX + 1, len(starts))
    for a, b, ui, li in zip(starts, stops, u, lag):
        if ui < dead_prob:
            hidden[min(a + li, b):b] = True
    crowd, air, line = v4.physical_conditions(ball, players)
    mult = np.ones(n)
    mult = np.where(air, mult * v4.AIR_MULT, mult)
    mult = np.where(crowd, mult * v4.CROWD_MULT, mult)
    mult = np.where(line, mult * v4.LINE_MULT, mult)
    live = ~dead
    mean_mult = float(mult[live].mean()) if live.any() else 1.0
    d_mean = live_dur_median * np.exp(sigma ** 2 / 2.0)
    odds = live_share / (1.0 - live_share)
    rate = odds / d_mean / mean_mult
    p_start = np.minimum(0.5, rate * mult)
    s = rng.random(n) < p_start
    z = rng.standard_normal(n)
    t = 0
    for a in np.flatnonzero(s & live):
        if a < t:
            continue
        d = max(1, int(round(live_dur_median * np.exp(sigma * z[a]))))
        b = min(n, a + d)
        hidden[a:b] = True
        t = b
    return hidden


# ---- player-side stress --------------------------------------------------------------------

def apply_swaps(rng, pl, s0, rate):
    """Exchange one home and one nearby away tracklet for a short stretch. Returns the event count."""
    n = len(pl)
    k = int(rng.poisson(rate * n / 10000.0))
    t0 = rng.integers(0, max(1, n), k)
    ln = rng.integers(SWAP_MIN, SWAP_MAX + 1, k)
    done = 0
    for a0, L in zip(t0, ln):
        for t in range(int(a0), min(n, int(a0) + SWAP_SCAN), 5):
            h = pl[t, 0, :s0]
            a = pl[t, 1, :s0]
            fh = np.flatnonzero(np.isfinite(h[:, 0]))
            fa = np.flatnonzero(np.isfinite(a[:, 0]))
            if len(fh) == 0 or len(fa) == 0:
                continue
            d = np.linalg.norm(h[fh][:, None, :] - a[fa][None, :, :], axis=2)
            i, j = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[i, j] > SWAP_DIST:
                continue
            hi, aj = fh[i], fa[j]
            b = min(n, t + int(L))
            tmp = pl[t:b, 0, hi].copy()
            pl[t:b, 0, hi] = pl[t:b, 1, aj]
            pl[t:b, 1, aj] = tmp
            done += 1
            break
    return done


def apply_touchline(rng, pl, s0, p):
    n = len(pl)
    for side in range(2):
        for k in range(s0):
            y = pl[:, side, k, 1]
            with np.errstate(invalid="ignore"):
                near = (y < TOUCH_MARGIN) | (y > PITCH[1] - TOUCH_MARGIN)
            near = np.nan_to_num(near, nan=0).astype(bool)
            st, sp = _runs(near)
            if len(st) == 0:
                continue
            u = rng.random(len(st))
            for a, b, ui in zip(st, sp, u):
                if ui < p:
                    pl[a:min(n, b + TOUCH_PAD), side, k] = np.nan


def make_extra_object(rng, match, n):
    """(n, 2, 2) referee-like track: loose follower of the ball, present on one random side."""
    ball = np.asarray(match["ball"][:, :2], np.float64)
    ball = np.where(np.isfinite(ball), ball, np.nanmean(ball, axis=0))
    k = 125
    pad = np.pad(ball, ((k // 2, k // 2), (0, 0)), mode="edge")
    c = np.cumsum(np.vstack([np.zeros((1, 2)), pad]), axis=0)
    base = (c[k:] - c[:-k])[:n] / k
    phi = 0.995
    e = rng.normal(0, 0.1 * 8.0 * np.sqrt(1 - phi ** 2) * 3.0, (n, 2))
    off = lfilter([1.0], [1.0, -phi], e, axis=0)
    pos = np.clip(base + off, [0.0, 0.0], [PITCH[0], PITCH[1]])
    # presence: alternating on / off runs, ~85% on
    present = np.zeros(n, bool)
    t = 0
    on = True
    while t < n:
        L = int(rng.integers(500, 3000)) if on else int(rng.integers(100, 600))
        present[t:t + L] = on
        t += L
        on = not on
    side = int(rng.integers(2))
    out = np.full((n, 2, 2), np.nan, np.float32)
    out[present, side] = pos[present]
    return out


def goalkeeper_slots(pl, s0, period):
    """Per period and side: slot closest (median) to the goal that side defends. Label free."""
    out = {}
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        for side in range(2):
            x = pl[idx, side, :s0, 0]
            fin = np.isfinite(x)
            if not fin.any():
                continue
            goal = 0.0 if np.nanmean(x) < PITCH[0] / 2 else PITCH[0]
            best, bk = None, None
            for k in range(s0):
                if fin[:, k].sum() < 0.2 * len(idx):
                    continue
                d = float(np.nanmedian(np.abs(x[:, k] - goal)))
                if best is None or d < best:
                    best, bk = d, k
            if bk is not None:
                out[(int(p), side)] = bk
    return out


def apply_gk_missing(rng, pl, s0, period, share):
    n = len(pl)
    mean_len = 0.5 * (GK_LEN[0] + GK_LEN[1])
    k = int(round(share * n / mean_len))
    a = rng.integers(0, max(1, n), k)
    L = rng.integers(GK_LEN[0], GK_LEN[1] + 1, k)
    side = rng.integers(0, 2, k)
    gks = goalkeeper_slots(pl, s0, period)
    for ai, li, si in zip(a, L, side):
        slot = gks.get((int(period[ai]), int(si)))
        if slot is None:
            continue
        b = min(n, int(ai) + int(li))
        same = period[int(ai):b] == period[int(ai)]
        pl[int(ai):b][same, int(si), slot] = np.nan


# ---- main ----------------------------------------------------------------------------------

def degrade(match, seed, ball_stress=False, swaps=0.0, touchline=0.0, extra_object=False,
            gk_missing=0.0, **v4kw):
    """Degraded copy of a bench.dfl.load() dict. v4 base setting; options listed in the module doc."""
    out = v4.degrade(match, seed, **v4kw)
    s0 = match["players"].shape[2]
    n = len(out["ball"])
    if ball_stress:
        ball = np.array(match["ball"], np.float32, copy=True)
        ball[:, :2] = v2._smooth(ball[:, :2], v2.BALL_SMOOTH)
        hidden = ball_hidden_stress(_rng(seed, 1), ball, match["players"])
        ball[hidden] = np.nan
        out["ball"] = ball
        out["ball_hidden_share"] = float(np.isnan(ball[:, 0]).mean())
    pl = out["players"]
    if swaps and swaps > 0:
        out["n_swaps"] = apply_swaps(_rng(seed, 2), pl, s0, float(swaps))
    if touchline and touchline > 0:
        apply_touchline(_rng(seed, 3), pl, s0, float(touchline))
    if gk_missing and gk_missing > 0:
        apply_gk_missing(_rng(seed, 5), pl, s0, np.asarray(match["period"]), float(gk_missing))
    if extra_object:
        obj = make_extra_object(_rng(seed, 4), match, n)
        out["players"] = np.concatenate([pl, obj[:, :, None, :]], axis=2)
    return out


STRESS = dict(ball_stress=True, swaps=4.0, touchline=0.5, extra_object=True, gk_missing=0.10)


def _mk(**kw):
    def f(match, seed):
        return degrade(match, seed, **kw)
    return f


DEGRADERS5 = {
    "v5": _mk(),                                    # == v4
    "v5ball": _mk(ball_stress=True),
    "v5player": _mk(swaps=4.0, touchline=0.5, extra_object=True, gk_missing=0.10),
    "v5stress": _mk(**STRESS),
}


def register():
    """Make the v5 names visible to bench.run_v3cand (which reads degrade_v4.DEGRADERS)."""
    v4.DEGRADERS.update(DEGRADERS5)


def regime_report(clean, deg):
    """MEASUREMENT only (reads alive): visible share of the ball on live and dead frames."""
    alive = np.asarray(clean["alive"], bool)
    vis = ~np.isnan(deg["ball"][:, 0])
    return {"live_hidden": float(1 - vis[alive].mean()) if alive.any() else 0.0,
            "dead_hidden": float(1 - vis[~alive].mean()) if (~alive).any() else 0.0,
            "all_hidden": float(1 - vis.mean())}


# ---- tests ---------------------------------------------------------------------------------

def _synth(n=8000, s0=6, seed=0):
    """Ball: dead (still) 150 frames every 400, otherwise moves. Players wander; slot 0 is a keeper."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cyc = t % 400
    still = cyc >= 250
    ball = np.zeros((n, 3), np.float32)
    pos = np.array([52.0, 34.0])
    vel = np.array([0.3, 0.1])
    for i in range(n):
        if not still[i]:
            if cyc[i] == 0:
                vel = rng.normal(0, 0.35, 2)
            pos = pos + vel
            if pos[0] < 3 or pos[0] > 102:
                vel[0] = -vel[0]
            if pos[1] < 1 or pos[1] > 67:
                vel[1] = -vel[1]
        ball[i, :2] = pos
    ball[:, 2] = 0.0
    players = np.full((n, 2, s0, 2), np.nan, np.float32)
    for side in range(2):
        for k in range(s0):
            home_x = 5.0 if k == 0 else rng.uniform(20, 85)
            home_x = home_x if side == 0 else 105 - home_x
            y0 = rng.uniform(3, 65)
            walk = np.cumsum(rng.normal(0, 0.03, (n, 2)), axis=0)
            players[:, side, k, 0] = np.clip(home_x + walk[:, 0] + 0.3 * (ball[:, 0] - 52) * (k > 0), 0, 105)
            y = y0 + walk[:, 1]
            y = np.where(k == 1, 1.0 + 0.5 * np.sin(t / 400.0), y)   # slot 1 hugs the touchline
            players[:, side, k, 1] = np.clip(y, 0, 68)
    # one home/away pair that crosses closely so swaps can happen
    players[:, 1, 2, :] = players[:, 0, 2, :] + np.array([1.5, 0.5], np.float32)
    return {"ball": ball, "players": players, "period": np.ones(n, np.int8),
            "alive": ~still, "owner": np.zeros(n, np.int8)}


def _same(a, b):
    return np.array_equal(a, b, equal_nan=True)


def _expect(name, cond, detail=""):
    if not cond:
        raise AssertionError("{} ({})".format(name, detail))
    print("PASS", name)


def _selftest():
    m = _synth()
    snap = {k: v.copy() for k, v in m.items()}

    base = v4.degrade(m, 3)
    d0 = degrade(m, 3)
    _expect("defaults are bit-identical to v4",
            _same(d0["ball"], base["ball"]) and _same(d0["players"], base["players"]))
    _expect("input not modified", all(_same(m[k], snap[k]) for k in m))

    kw = dict(STRESS)
    a, b = degrade(m, 3, **kw), degrade(m, 3, **kw)
    _expect("deterministic", _same(a["ball"], b["ball"]) and _same(a["players"], b["players"]))
    c = degrade(m, 4, **kw)
    _expect("seed changes output", not _same(a["players"], c["players"]))

    # label independence: scramble alive/owner, nothing may move
    m2 = {k: v.copy() for k, v in m.items()}
    m2["alive"] = ~m["alive"]
    m2["owner"] = np.ones_like(m["owner"])
    e = degrade(m2, 3, **kw)
    _expect("ignores alive and owner", _same(a["ball"], e["ball"]) and _same(a["players"], e["players"]))

    # ball regime over several seeds: players must equal the v4 base, ball must hit the targets
    live, dead = [], []
    for s in range(8):
        d = degrade(m, s, ball_stress=True)
        rep = regime_report(m, d)
        live.append(rep["live_hidden"])
        dead.append(rep["dead_hidden"])
        if s == 0:
            _expect("ball stress leaves players as in v4", _same(d["players"], v4.degrade(m, 0)["players"]))
    _expect("ball stress live ~10%", 0.04 <= np.mean(live) <= 0.18, "live %.3f" % np.mean(live))
    _expect("ball stress dead ~75%", 0.55 <= np.mean(dead) <= 0.90, "dead %.3f" % np.mean(dead))
    v4live = np.mean([regime_report(m, v4.degrade(m, s))["live_hidden"] for s in range(4)])
    print("INFO v4 live hidden %.3f, stress live %.3f dead %.3f" % (v4live, np.mean(live), np.mean(dead)))

    # swaps: the per-frame sum of all positions is conserved, something changed, v4 ball untouched
    d = degrade(m, 3, swaps=40.0)
    _expect("swaps happened", d.get("n_swaps", 0) > 0 and not _same(d["players"], base["players"]),
            str(d.get("n_swaps")))
    _expect("swaps conserve positions per frame",
            np.allclose(np.nansum(d["players"], axis=(1, 2)), np.nansum(base["players"], axis=(1, 2)), atol=1e-3))
    _expect("swaps leave the ball alone", _same(d["ball"], base["ball"]))
    sw = (d["players"][:, 0, :6] != base["players"][:, 0, :6]) & ~np.isnan(base["players"][:, 0, :6])
    _expect("swaps cross teams (home trajectories changed)", bool(sw.any()))

    # touchline
    d = degrade(m, 3, touchline=1.0)
    new_nan = np.isnan(d["players"][:, :, :6, 0]) & ~np.isnan(base["players"][:, :, :6, 0])
    _expect("touchline drops frames", int(new_nan.sum()) > 0)
    near = np.zeros(new_nan.shape, bool)
    py = base["players"][:, :, :6, 1]
    with np.errstate(invalid="ignore"):
        near = (py < TOUCH_MARGIN) | (py > PITCH[1] - TOUCH_MARGIN)
    near = np.nan_to_num(near, nan=0).astype(bool)
    ok = True
    for sd in range(2):
        for k in range(6):
            nn = np.flatnonzero(new_nan[:, sd, k])
            nr = np.flatnonzero(near[:, sd, k])
            if len(nn) and (len(nr) == 0 or np.abs(nn[:, None] - nr[None, :]).min(axis=1).max() > TOUCH_PAD + 1):
                ok = False
    _expect("touchline drops only near the line", ok)
    d_off = degrade(m, 3, touchline=0.0)
    _expect("touchline p=0 is off", _same(d_off["players"], base["players"]))

    # extra object
    d = degrade(m, 3, extra_object=True)
    _expect("extra object adds one slot", d["players"].shape[2] == base["players"].shape[2] + 1)
    _expect("other slots untouched", _same(d["players"][:, :, :-1], base["players"]))
    ex = d["players"][:, :, -1, 0]
    one_side = (np.isfinite(ex).sum(axis=1) <= 1).all()
    share = float(np.isfinite(ex).any(axis=1).mean())
    _expect("extra object on one side, mostly present", one_side and 0.5 <= share <= 0.98, "%.2f" % share)

    # goalkeeper
    d = degrade(m, 3, gk_missing=0.2)
    new_nan = np.isnan(d["players"][:, :, :, 0]) & ~np.isnan(base["players"][:, :, :, 0])
    cols = np.flatnonzero(new_nan.any(axis=(0, 1)))
    share = float(new_nan.any(axis=(1, 2)).mean())
    _expect("goalkeeper slot only", list(cols) == [0], str(list(cols)))
    _expect("goalkeeper missing share ~ target", 0.04 <= share <= 0.4, "%.3f" % share)

    # options are independent streams: ball+swaps equals ball alone with swaps alone
    both = degrade(m, 3, ball_stress=True, swaps=40.0)
    only_b = degrade(m, 3, ball_stress=True)
    only_s = degrade(m, 3, swaps=40.0)
    _expect("option streams independent",
            _same(both["ball"], only_b["ball"]) and _same(both["players"], only_s["players"]))
    print("degrade_v5 selftest ok")


if __name__ == "__main__":
    _selftest()
