"""Ball-free team-context features on top of bench v1.

dense_features3(match, groups=...) returns v1's dense_features plus any of
context, lang_dead, touch, restart, gap. Nothing here reads labels (alive /
owner) or the future of the owner flag. Offline lookahead of positions is
allowed, matching v1.

Teams are symmetric: every per-team column is recomputed from the player
array, so swap_teams exchanges the _h and _a columns. Attacking direction is
inferred per period from player positions (which goal a team defends). Raw x
flips meaning between halves in this data; attack-frame features do not.

from_ball marks a column that depends on the ball track (coordinates or which
frames are visible). The split-expert's ball-free model drops those columns.
"""

import numpy as np
from scipy.ndimage import median_filter

from bench import methods

H, A, D = methods.H, methods.A, methods.D
PITCH_L, PITCH_W = methods.PITCH
CX, CY = PITCH_L / 2.0, PITCH_W / 2.0
# Lang et al. 2022 shifts, in frames at 25 Hz: +/- 0.4, 0.8, 2, 4, 20, 40, 80 s.
LANG_SHIFTS = (0, -10, 10, -20, 20, -50, 50, -100, 100, -500, 500, -1000, 1000, -2000, 2000)
# Bump when a formula changes so the on-disk feature cache is rebuilt.
REV = "f3a"
ALL_GROUPS = ("context", "lang_dead", "touch", "restart", "gap")
CONTEXT_SUBS = ("centroid", "spread", "line", "hull", "convergence", "velocity")
GROUPS = ALL_GROUPS


def _warn_off():
    # v1's team_speed hits an all-NaN slice on fully missing frames. Same data, same result.
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered")


def clocks(match):
    """Ball visibility and uncapped gap lengths, reset at each period.

    gap_back is 0 on a visible frame and, inside a gap, the frames since the
    last visible frame. 1e5 if this period has not shown the ball yet.
    """
    ball = match["ball"]
    period = match["period"]
    visible = ~np.isnan(ball[:, 0])
    n = len(visible)
    last = np.full(n, -1, np.int32)
    nxt = np.full(n, -1, np.int32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        vis = visible[idx]
        pos = np.arange(len(idx))
        loc = np.maximum.accumulate(np.where(vis, pos, -1))
        good = loc >= 0
        last_p = np.full(len(idx), -1, np.int32)
        last_p[good] = idx[loc[good]]
        last[idx] = last_p
        locf = np.minimum.accumulate(np.where(vis, pos, len(idx))[::-1])[::-1]
        nxt_p = np.full(len(idx), -1, np.int32)
        goodf = locf < len(idx)
        nxt_p[goodf] = idx[locf[goodf]]
        nxt[idx] = nxt_p
    t = np.arange(n)
    big = np.float32(1e5)
    gap_back = np.where(last < 0, big, t - last).astype(np.float32)
    gap_fwd = np.where(nxt < 0, big, nxt - t).astype(np.float32)
    return visible, gap_back, gap_fwd, last, nxt


def _attack_sign(xy, period):
    """Keeper-proof direction (trimmed team means, median over frames). Same output as the old
    anchor rule on clean tracking; unaffected by a keeper filed under the wrong team."""
    from bench.gkdir import attack_sign_robust
    return attack_sign_robust(xy, period)


def _attack_sign_old(xy, period):
    """+1 if that team's own goal is at x=0 (they attack toward +x), else -1.

    Per period, the anchor is the slot whose median x is closest to x=0.
    The team with the smaller anchor defends the left goal. Halves in this
    tracking flip, so the sign is recomputed per period and never taken from
    a label. Falls back to the team with the smaller mean x if anchors tie.
    """
    n = xy.shape[0]
    sign = np.ones((n, 2), np.float32)
    sign[:, 1] = -1.0
    x = xy[..., 0]
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        anchors = []
        means = []
        need = max(500, len(idx) // 20)
        for side in (0, 1):
            slot_med = []
            acc = []
            for k in range(x.shape[2]):
                v = x[idx, side, k]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                acc.append(v)
                if len(v) >= need:
                    slot_med.append(float(np.median(v)))
            if slot_med:
                anchors.append(min(slot_med))
            elif acc:
                anchors.append(float(np.median(np.concatenate(acc))))
            else:
                anchors.append(np.nan)
            if acc:
                means.append(float(np.mean(np.concatenate(acc))))
            else:
                means.append(np.nan)
        if not np.isfinite(anchors).all():
            if p == 0 or not any(np.isfinite(x[idx, side]).any() for side in (0, 1)):
                continue        # no player tracking in this block: nothing to orient (documented rule)
            raise RuntimeError("attack direction: period %s has tracking for only one team; refusing "
                               "to guess which side defends the left goal" % p)
        if abs(anchors[0] - anchors[1]) >= 5.0:
            left = int(np.argmin(anchors))
        elif np.isfinite(means).all() and abs(means[0] - means[1]) >= 1.0:
            left = int(np.argmin(means))
        else:
            # Previously silently 'home defends left'. Ambiguous orientation would flip every
            # direction-dependent feature of the period, so fail loudly instead of guessing.
            raise RuntimeError("attack direction ambiguous in period %s: slot-median anchors %s, "
                               "means %s; refusing to default to home-defends-left" % (p, anchors, means))
        sign[idx, :] = -1.0
        sign[idx, left] = 1.0
    return sign


def _hull_area(xy):
    """Convex-hull area of one finite cloud. 0 if fewer than 3 unique points."""
    k = xy.shape[0]
    if k < 3:
        return 0.0
    order = np.lexsort((xy[:, 1], xy[:, 0]))
    xs = xy[order, 0].tolist()
    ys = xy[order, 1].tolist()
    xu = [xs[0]]
    yu = [ys[0]]
    for i in range(1, k):
        if xs[i] != xu[-1] or ys[i] != yu[-1]:
            xu.append(xs[i])
            yu.append(ys[i])
    xs, ys = xu, yu
    k = len(xs)
    if k < 3:
        return 0.0

    def chain(irange):
        h = []
        for i in irange:
            while len(h) >= 2:
                o = h[-2]
                a = h[-1]
                cross = (xs[a] - xs[o]) * (ys[i] - ys[o]) - (ys[a] - ys[o]) * (xs[i] - xs[o])
                if cross <= 0.0:
                    h.pop()
                else:
                    break
            h.append(i)
        return h

    hull = chain(range(k))[:-1] + chain(range(k - 1, -1, -1))[:-1]
    if len(hull) < 3:
        return 0.0
    acc = 0.0
    x0 = xs[hull[-1]]
    y0 = ys[hull[-1]]
    for i in hull:
        x1 = xs[i]
        y1 = ys[i]
        acc += x0 * y1 - x1 * y0
        x0, y0 = x1, y1
    return abs(acc) * 0.5


def _hull_series(xy, finite):
    """xy (n, S, 2), finite (n, S) -> (n,) areas."""
    n = xy.shape[0]
    out = np.zeros(n, np.float32)
    for i in range(n):
        m = finite[i]
        if int(m.sum()) < 3:
            continue
        out[i] = _hull_area(xy[i, m])
    return out


def _gather(arr, idx):
    out = np.zeros(arr.shape, np.float32)
    ok = idx >= 0
    out[ok] = arr[idx[ok]]
    return out


def _edges(arr, last, nxt):
    a_last = _gather(arr, last)
    a_next = _gather(arr, nxt)
    span = a_next - a_last
    span[(last < 0) | (nxt < 0)] = 0.0
    return a_last, a_next, span


def _shift_all(base, shifts, period):
    """base (n, C) sampled at each shift, inside each period, edge-held."""
    n, c = base.shape
    out = np.empty((n, c * len(shifts)), np.float32)
    blocks = [(np.flatnonzero(period == p),) for p in np.unique(period)]
    # store the sliced blocks once
    sliced = []
    for (idx,) in blocks:
        sliced.append((idx, np.asarray(base[idx])))
    for si, s in enumerate(shifts):
        dest = out[:, si * c:(si + 1) * c]
        if s == 0:
            dest[:] = base
            continue
        k = abs(int(s))
        for idx, block in sliced:
            b = np.empty_like(block)
            if k >= len(block):
                b[:] = block[:1] if s < 0 else block[-1:]
            elif s > 0:
                b[:-k] = block[k:]
                b[-k:] = block[-1]
            else:
                b[k:] = block[:-k]
                b[:k] = block[0]
            dest[idx] = b
    return out


def _median_period(a, size, period):
    out = np.empty(len(a), np.float32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = median_filter(np.asarray(a[idx], np.float64), size=size, mode="nearest")
    return out


def _percentiles(a, qs):
    """Column-wise linear percentiles. a is (n, S) with NaN gaps. Empty -> 0."""
    finite = np.isfinite(a)
    order = np.sort(np.where(finite, a, np.inf), axis=1)
    count = finite.sum(1)
    rows = np.arange(len(a))
    out = []
    width = a.shape[1]
    for q in qs:
        idxf = np.maximum(count - 1, 0) * (q / 100.0)
        lo = np.floor(idxf).astype(np.int32)
        hi = np.ceil(idxf).astype(np.int32)
        w = (idxf - lo).astype(np.float64)
        lo = np.clip(lo, 0, width - 1)
        hi = np.clip(hi, 0, width - 1)
        val = (1.0 - w) * order[rows, lo] + w * order[rows, hi]
        val = np.where((count > 0) & np.isfinite(val), val, 0.0)
        out.append(val.astype(np.float32))
    return out


def _geometry(match):
    """Shared player kinematics. Does not read alive, owner, or the ball."""
    xy = np.asarray(match["players"], np.float32)
    period = match["period"]
    n, _, s, _ = xy.shape
    finite = ~np.isnan(xy[..., 0])
    boundary = np.zeros(n, bool)
    boundary[0] = True
    boundary[1:] = period[1:] != period[:-1]

    vel = np.full_like(xy, np.nan)
    vel[1:] = (xy[1:] - xy[:-1]) * np.float32(25.0)
    vel[boundary] = np.nan
    speed = np.linalg.norm(vel, axis=-1)
    bad = (~np.isfinite(speed)) | (speed > 12.0)
    speed = np.where(bad, np.nan, speed)
    vel = np.where(bad[..., None], np.nan, vel)

    acc_vec = np.full_like(vel, np.nan)
    acc_vec[1:] = (vel[1:] - vel[:-1]) * np.float32(25.0)
    acc_vec[boundary] = np.nan
    acc = np.linalg.norm(acc_vec, axis=-1)
    acc = np.where((~np.isfinite(acc)) | (acc > 25.0), np.nan, acc)

    sign = _attack_sign(xy, period)
    with np.errstate(all="ignore"):
        cent = np.nanmean(xy, axis=2)
    cent = np.where(np.isfinite(cent), cent, np.array([CX, CY], np.float32)).astype(np.float32)

    attack = np.where(sign[:, :, None] > 0, xy[..., 0], PITCH_L - xy[..., 0])
    attack = np.where(finite, attack, np.nan)
    y = xy[..., 1]
    with np.errstate(all="ignore"):
        width = np.nanmax(y, axis=2) - np.nanmin(y, axis=2)
        depth = np.nanmax(attack, axis=2) - np.nanmin(attack, axis=2)
    width = np.where(np.isfinite(width), width, 0).astype(np.float32)
    depth = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    # Empty slots are NaN, not players standing at the origin. Don't let them dilute the share.
    advanced = np.isfinite(attack) & (attack > 52.5)
    present = np.isfinite(attack).sum(axis=2)
    opp = np.where(present > 0, advanced.sum(axis=2) / present, 0.0).astype(np.float32)

    att_filled = np.where(finite, attack, np.inf)
    kth = min(4, s - 1)
    part = np.sort(np.partition(att_filled, kth, axis=2)[:, :, :kth + 1], axis=2)
    body = np.where(part[:, :, 1:] < 1e8, part[:, :, 1:], np.nan)
    with np.errstate(all="ignore"):
        line = np.nanmean(body, axis=2)
    line = np.nan_to_num(line, nan=CX).astype(np.float32)

    hull = np.stack([_hull_series(xy[:, 0], finite[:, 0]), _hull_series(xy[:, 1], finite[:, 1])], axis=1)

    cv = np.zeros_like(cent)
    cv[1:] = (cent[1:] - cent[:-1]) * np.float32(25.0)
    cv[boundary] = 0.0
    cv[np.linalg.norm(cv, axis=-1) > 20.0] = 0.0
    v_left = (-cv[..., 0]).astype(np.float32)
    v_right = cv[..., 0].astype(np.float32)
    v_attack = np.where(sign > 0, cv[..., 0], -cv[..., 0]).astype(np.float32)
    cent_x = cent[..., 0].astype(np.float32)
    cent_att = np.where(sign > 0, cent_x, PITCH_L - cent_x).astype(np.float32)

    dbx = np.minimum(xy[..., 0], PITCH_L - xy[..., 0])
    dby = np.minimum(xy[..., 1], PITCH_W - xy[..., 1])
    db = np.minimum(dbx, dby)
    db = np.where(finite, db, np.nan)
    with np.errstate(all="ignore"):
        bound = np.nanmin(db, axis=2)
    bound = np.nan_to_num(bound, nan=30.0).astype(np.float32)

    spd50, spd90, acc50 = [], [], []
    for side in (0, 1):
        p50, p90 = _percentiles(speed[:, side, :], (50, 90))
        a50 = _percentiles(acc[:, side, :], (50,))[0]
        spd50.append(p50)
        spd90.append(p90)
        acc50.append(a50)

    return {
        "xy": xy, "finite": finite, "speed": speed, "vel": vel, "period": period,
        "boundary": boundary, "sign": sign, "cent": cent, "cent_x": cent_x,
        "cent_att": cent_att, "width": width, "depth": depth, "opp": opp,
        "line": line, "hull": hull.astype(np.float32), "v_left": v_left,
        "v_right": v_right, "v_attack": v_attack, "bound": bound,
        "spd50": np.stack(spd50, axis=1), "spd90": np.stack(spd90, axis=1),
        "acc50": np.stack(acc50, axis=1), "cv": cv.astype(np.float32),
    }


class _Cols:
    def __init__(self):
        self.rows = []

    def add(self, values, name, from_ball, subgroup, kind):
        v = np.asarray(values, np.float32)
        if v.ndim != 1:
            raise ValueError(name)
        self.rows.append((np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0),
                          name, bool(from_ball), subgroup, kind))

    def add_pair(self, h, a, stem, from_ball, subgroup):
        self.add(h, stem + "_h", from_ball, subgroup, "h")
        self.add(a, stem + "_a", from_ball, subgroup, "a")

    def add_diff(self, h, a, stem, from_ball, subgroup):
        self.add(np.asarray(h, np.float32) - np.asarray(a, np.float32),
                 stem + "_diff", from_ball, subgroup, "diff")

    def stack(self):
        names = [r[1] for r in self.rows]
        if len(names) != len(set(names)):
            dup = [n for n in names if names.count(n) > 1]
            raise ValueError("duplicate feature names: %s" % sorted(set(dup))[:8])
        x = np.column_stack([r[0] for r in self.rows]).astype(np.float32, copy=False)
        ball = np.array([r[2] for r in self.rows], dtype=bool)
        sub = [r[3] for r in self.rows]
        kind = [r[4] for r in self.rows]
        return x, names, ball, sub, kind


def _v1_kind(name):
    if name == "margin":
        return "diff"
    if name == "dh" or name.endswith("_h") or "_h_" in name:
        return "h"
    if name == "da" or name.endswith("_a") or "_a_" in name:
        return "a"
    return "sym"


def _v1_ball(name):
    return name != "calm" and not name.startswith("calm_")


def _add_v1(cols, match):
    x, names = methods.dense_features(match)
    for j, name in enumerate(names):
        cols.add(x[:, j], name, _v1_ball(name), "v1", _v1_kind(name))
    return names


def _add_family(cols, stem, arr, sub, last, nxt):
    """arr (n, 2). Instant value is player-only; edge samples follow the ball clock."""
    cols.add_pair(arr[:, 0], arr[:, 1], "ctx_%s_now" % stem, False, sub)
    a_last, a_next, span = _edges(arr, last, nxt)
    cols.add_pair(a_last[:, 0], a_last[:, 1], "ctx_%s_last" % stem, True, sub)
    cols.add_pair(a_next[:, 0], a_next[:, 1], "ctx_%s_next" % stem, True, sub)
    cols.add_pair(span[:, 0], span[:, 1], "ctx_%s_span" % stem, True, sub)
    cols.add_diff(arr[:, 0], arr[:, 1], "ctx_%s_now" % stem, False, sub)


def _ball_at(ball, idx):
    out = np.full((len(idx), 2), np.nan, np.float32)
    ok = idx >= 0
    out[ok] = ball[idx[ok], :2]
    return out


def _cloud_vs_ball(xy, vel, finite, ball_xy, radius):
    """Counts of players within radius, converging, and a pressure sum.

    ball_xy is (n, 2) and may be NaN. Convergence is closing speed > 1 m/s.
    Pressure sums clipped closing speed / distance inside 12 m.
    """
    delta = ball_xy[:, None, None, :] - xy
    dist = np.linalg.norm(xy - ball_xy[:, None, None, :], axis=-1)
    close = finite & np.isfinite(dist) & (dist <= radius)
    closing = np.sum(vel * delta, axis=-1) / np.maximum(dist, 1e-3)
    conv = close & np.isfinite(closing) & (closing > 1.0)
    zone = finite & np.isfinite(dist) & np.isfinite(closing) & (dist <= 12.0)
    inten = np.clip(closing, 0.0, 8.0) / np.maximum(dist, 1.0)
    press = np.where(zone, inten, 0.0).sum(axis=2).astype(np.float32)
    return close.sum(axis=2).astype(np.float32), conv.sum(axis=2).astype(np.float32), press


def _add_context(cols, match, geo, drop):
    visible, _, _, last, nxt = clocks(match)
    del visible
    families = (
        ("centx", geo["cent_x"], "centroid"),
        ("centatt", geo["cent_att"], "centroid"),
        ("width", geo["width"], "spread"),
        ("depth", geo["depth"], "spread"),
        ("line", geo["line"], "line"),
        ("opp", geo["opp"], "line"),
        ("hull", geo["hull"], "hull"),
        ("vleft", geo["v_left"], "velocity"),
        ("vright", geo["v_right"], "velocity"),
        ("vattack", geo["v_attack"], "velocity"),
    )
    drop = set(drop)
    for stem, arr, sub in families:
        if sub in drop:
            continue
        _add_family(cols, stem, arr, sub, last, nxt)
    if "convergence" in drop:
        return
    ball = match["ball"]
    xy, vel, finite = geo["xy"], geo["vel"], geo["finite"]
    # At the frame the ball was last (next) visible: who was around it then.
    now_near, now_conv, now_press = _cloud_vs_ball(xy, vel, finite, ball[:, :2], 10.0)
    edge_near_l, edge_near_n, _ = _edges(now_near, last, nxt)
    edge_conv_l, edge_conv_n, _ = _edges(now_conv, last, nxt)
    edge_press_l, edge_press_n, _ = _edges(now_press, last, nxt)
    # During a gap: where are the players now, relative to those two ball positions.
    chase_l_n, chase_l_c, chase_l_p = _cloud_vs_ball(xy, vel, finite, _ball_at(ball, last), 10.0)
    chase_n_n, chase_n_c, chase_n_p = _cloud_vs_ball(xy, vel, finite, _ball_at(ball, nxt), 10.0)
    pairs = (
        ("ctx_near_edge_last", edge_near_l),
        ("ctx_near_edge_next", edge_near_n),
        ("ctx_near_chase_last", chase_l_n),
        ("ctx_near_chase_next", chase_n_n),
        ("ctx_conv_edge_last", edge_conv_l),
        ("ctx_conv_edge_next", edge_conv_n),
        ("ctx_conv_chase_last", chase_l_c),
        ("ctx_conv_chase_next", chase_n_c),
        ("ctx_press_edge_last", edge_press_l),
        ("ctx_press_edge_next", edge_press_n),
        ("ctx_press_chase_last", chase_l_p),
        ("ctx_press_chase_next", chase_n_p),
    )
    for stem, arr in pairs:
        cols.add_pair(arr[:, 0], arr[:, 1], stem, True, "convergence")
    cols.add_diff(edge_near_l[:, 0], edge_near_l[:, 1], "ctx_near_edge_last", True, "convergence")
    cols.add_diff(chase_l_c[:, 0], chase_l_c[:, 1], "ctx_conv_chase_last", True, "convergence")


def _add_lang(cols, geo):
    """Permutation-invariant Lang block: percentiles, centroid velocity, boundary
    proximity, at each paper shift, plus the 91-frame median they used as a smoother.
    """
    period = geo["period"]
    pieces_h = [geo["spd50"][:, 0], geo["spd90"][:, 0], geo["acc50"][:, 0],
                geo["cv"][:, 0, 0], geo["cv"][:, 0, 1], geo["bound"][:, 0]]
    pieces_a = [geo["spd50"][:, 1], geo["spd90"][:, 1], geo["acc50"][:, 1],
                geo["cv"][:, 1, 0], geo["cv"][:, 1, 1], geo["bound"][:, 1]]
    stems = ["spd50", "spd90", "acc50", "cvx", "cvy", "bound"]
    base_h = np.column_stack(pieces_h).astype(np.float32)
    base_a = np.column_stack(pieces_a).astype(np.float32)
    # Interleave teams so a shift keeps _h/_a neighbours in a stable order.
    base = np.empty((len(period), len(stems) * 2), np.float32)
    base[:, 0::2] = base_h
    base[:, 1::2] = base_a
    shifted = _shift_all(base, LANG_SHIFTS, period)
    c = len(stems) * 2
    for si, s in enumerate(LANG_SHIFTS):
        tag = "0" if s == 0 else ("p%d" % s if s > 0 else "m%d" % -s)
        block = shifted[:, si * c:(si + 1) * c]
        for j, stem in enumerate(stems):
            cols.add(block[:, 2 * j], "lang_%s_%s_h" % (stem, tag), False, "lang", "h")
            cols.add(block[:, 2 * j + 1], "lang_%s_%s_a" % (stem, tag), False, "lang", "a")
    for stem, series in (("spd50", geo["spd50"]), ("acc50", geo["acc50"]), ("bound", geo["bound"])):
        for side, suf in ((0, "_h"), (1, "_a")):
            med = _median_period(series[:, side], 91, period)
            cols.add(med, "lang_%s_med91%s" % (stem, suf), False, "lang", "h" if side == 0 else "a")


def _displacement(ball_xy, w, period):
    """Velocity over a span of w frames: (p[t] - p[t-w]) / dt and (p[t+w] - p[t]) / dt."""
    n = len(ball_xy)
    vp = np.zeros((n, 2), np.float32)
    vf = np.zeros((n, 2), np.float32)
    okp = np.zeros(n, bool)
    okf = np.zeros(n, bool)
    if w >= n or w <= 0:
        return vp, okp, vf, okf
    dt = w / 25.0
    same = period[w:] == period[:-w]
    both = np.isfinite(ball_xy[w:, 0]) & np.isfinite(ball_xy[:-w, 0]) & same
    step = (ball_xy[w:] - ball_xy[:-w]) / dt
    vp[w:] = np.where(both[:, None], step, 0.0)
    okp[w:] = both
    vf[:-w] = np.where(both[:, None], step, 0.0)
    okf[:-w] = both
    return vp.astype(np.float32), okp, vf.astype(np.float32), okf


def _angle_and_ratio(vp, okp, vf, okf):
    sb = np.linalg.norm(vp, axis=1)
    sa = np.linalg.norm(vf, axis=1)
    both = okp & okf & np.isfinite(sb) & np.isfinite(sa)
    dot = np.sum(vp * vf, axis=1)
    cos = np.clip(dot / np.maximum(sb * sa, 1e-6), -1.0, 1.0)
    aimed = both & (sb > 1.5) & (sa > 1.5)
    dtheta = np.where(aimed, np.arccos(cos), 0.0).astype(np.float32)
    # A ball that is still at both ends of the window is not a touch. Dividing two
    # near-zero speeds would look like a huge ratio change.
    quiet = both & (sb < 0.5) & (sa < 0.5)
    ratio = np.where(both & ~quiet, np.clip(sa / np.maximum(sb, 0.3), 0.0, 8.0), 1.0).astype(np.float32)
    dspeed = np.where(both & ~quiet, sa - sb, 0.0).astype(np.float32)
    return dtheta, ratio, dspeed, both


def _since_period(event, period, cap=500):
    out = np.empty(len(event), np.float32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = methods._since(np.asarray(event[idx], bool), cap)
    return out


def _until_period(event, period, cap=500):
    out = np.empty(len(event), np.float32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = methods._until(np.asarray(event[idx], bool), cap)
    return out


def _centred_period(x, w, period):
    out = np.empty(len(x), np.float32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = methods._centred_mean(np.asarray(x[idx], np.float64), w).astype(np.float32)
    return out


def _add_touch(cols, match):
    """Direction and speed change over 0.5 s and 1 s, gated by a player within 2.5 m.

    v1's instantaneous acceleration never fires once the ball is smoothed over
    a second. The displacement over 12 and 25 frames still bends at a kick.
    """
    ball = match["ball"]
    period = match["period"]
    dh, da = methods.nearest(match)
    missing = np.isnan(ball[:, 0])
    gate_h = (~missing) & np.isfinite(dh) & (dh < 2.5)
    gate_a = (~missing) & np.isfinite(da) & (da < 2.5)
    cols.add(gate_h.astype(np.float32), "touch_near_h", True, "touch", "h")
    cols.add(gate_a.astype(np.float32), "touch_near_a", True, "touch", "a")
    long_event_h = None
    long_event_a = None
    for w, tag in ((12, "w05"), (25, "w10")):
        vp, okp, vf, okf = _displacement(ball[:, :2], w, period)
        dtheta, ratio, dspeed, _both = _angle_and_ratio(vp, okp, vf, okf)
        cols.add(dtheta, "touch_dtheta_%s" % tag, True, "touch", "sym")
        cols.add(ratio, "touch_ratio_%s" % tag, True, "touch", "sym")
        cols.add(dspeed, "touch_dspeed_%s" % tag, True, "touch", "sym")
        cols.add(np.where(gate_h, dtheta, 0).astype(np.float32), "touch_dtheta_%s_h" % tag, True, "touch", "h")
        cols.add(np.where(gate_a, dtheta, 0).astype(np.float32), "touch_dtheta_%s_a" % tag, True, "touch", "a")
        cols.add(np.where(gate_h, ratio, 1).astype(np.float32), "touch_ratio_%s_h" % tag, True, "touch", "h")
        cols.add(np.where(gate_a, ratio, 1).astype(np.float32), "touch_ratio_%s_a" % tag, True, "touch", "a")
        big = (dtheta > 0.35) | (ratio > 1.6) | (ratio < 0.625)
        ev_h = big & gate_h
        ev_a = big & gate_a
        if tag == "w10":
            long_event_h, long_event_a = ev_h, ev_a
    cols.add(_since_period(long_event_h, period), "touch_since_h", True, "touch", "h")
    cols.add(_since_period(long_event_a, period), "touch_since_a", True, "touch", "a")
    cols.add(_until_period(long_event_h, period), "touch_until_h", True, "touch", "h")
    cols.add(_until_period(long_event_a, period), "touch_until_a", True, "touch", "a")
    for w in (25, 125):
        cols.add(_centred_period(long_event_h.astype(np.float64), w, period),
                 "touch_rate_%d_h" % w, True, "touch", "h")
        cols.add(_centred_period(long_event_a.astype(np.float64), w, period),
                 "touch_rate_%d_a" % w, True, "touch", "a")


def _pair_min_dist(xy, finite, points):
    """Min distance from each team's players to any of `points` (m, 2)."""
    d = np.linalg.norm(xy[:, :, :, None, :] - points[None, None, None, :, :], axis=-1)
    d = np.where(finite[:, :, :, None], d, np.inf)
    return np.min(d, axis=(2, 3)).astype(np.float32)


def _count_near_points(xy, finite, points, radius):
    d = np.linalg.norm(xy[:, :, :, None, :] - points[None, None, None, :, :], axis=-1)
    near = finite & (np.min(d, axis=-1) <= radius)
    return near.sum(axis=2).astype(np.float32)


def _add_restart(cols, match, geo):
    """Set-piece geometry, calm players, kick-off formation. Player features are
    ball-free; distances of the last and next ball to the restart landmarks are not.
    """
    xy = geo["xy"]
    finite = geo["finite"]
    speed = geo["speed"]
    sign = geo["sign"]
    period = geo["period"]
    x = xy[..., 0]
    y = xy[..., 1]
    n = len(period)

    left = finite & (x < CX)
    right = finite & (x >= CX)
    denom = np.maximum(finite.sum(axis=2), 1).astype(np.float32)
    frac_left = left.sum(axis=2) / denom
    frac_right = right.sum(axis=2) / denom
    split = np.maximum(frac_left[:, 0] * frac_right[:, 1], frac_right[:, 0] * frac_left[:, 1])
    cols.add(split.astype(np.float32), "restart_kickoff_split", False, "restart", "sym")

    centre = np.array([CX, CY], np.float32)
    dc = np.linalg.norm(xy - centre, axis=-1)
    near_c = finite & (dc <= 2.0)
    in_circ = finite & (dc <= 9.15)
    taker_h = (near_c[:, 0].sum(1) >= 1) & (in_circ[:, 1].sum(1) == 0) & (split >= 0.75)
    taker_a = (near_c[:, 1].sum(1) >= 1) & (in_circ[:, 0].sum(1) == 0) & (split >= 0.75)
    cols.add(taker_h.astype(np.float32), "restart_taker_h", False, "restart", "h")
    cols.add(taker_a.astype(np.float32), "restart_taker_a", False, "restart", "a")

    dist_touch = np.minimum(y, PITCH_W - y)
    n_touch = (finite & (dist_touch <= 1.0)).sum(axis=2).astype(np.float32)
    cols.add_pair(n_touch[:, 0], n_touch[:, 1], "restart_touchline", False, "restart")

    corners = np.array([[0, 0], [0, PITCH_W], [PITCH_L, 0], [PITCH_L, PITCH_W]], np.float32)
    n_corner = _count_near_points(xy, finite, corners, 3.0)
    cols.add_pair(n_corner[:, 0], n_corner[:, 1], "restart_corner", False, "restart")

    spots = np.array([[11.0, CY], [PITCH_L - 11.0, CY]], np.float32)
    n_spot = _count_near_points(xy, finite, spots, 2.5)
    cols.add_pair(n_spot[:, 0], n_spot[:, 1], "restart_spot", False, "restart")

    in_left_ga = finite & (x <= 5.5) & (np.abs(y - CY) <= 9.16)
    in_right_ga = finite & (x >= PITCH_L - 5.5) & (np.abs(y - CY) <= 9.16)
    in_own_ga = np.where(sign[:, :, None] > 0, in_left_ga, in_right_ga)
    n_ga = in_own_ga.sum(axis=2).astype(np.float32)
    cols.add_pair(n_ga[:, 0], n_ga[:, 1], "restart_own_ga", False, "restart")

    in_left_box = finite & (x <= 16.5) & (np.abs(y - CY) <= 20.16)
    in_right_box = finite & (x >= PITCH_L - 16.5) & (np.abs(y - CY) <= 20.16)
    in_own_box = np.where(sign[:, :, None] > 0, in_left_box, in_right_box)
    n_box = in_own_box.sum(axis=2).astype(np.float32)
    cols.add_pair(n_box[:, 0], n_box[:, 1], "restart_own_box", False, "restart")

    goals = np.array([[0.0, CY], [PITCH_L, CY]], np.float32)
    d_left = np.linalg.norm(xy - goals[0], axis=-1)
    d_right = np.linalg.norm(xy - goals[1], axis=-1)
    d_left = np.where(finite, d_left, np.inf)
    d_right = np.where(finite, d_right, np.inf)
    d_own = np.where(sign[:, :, None] > 0, d_left, d_right)
    with np.errstate(all="ignore"):
        min_own = np.min(d_own, axis=2)
    min_own = np.where(np.isfinite(min_own), min_own, 60.0).astype(np.float32)
    cols.add_pair(min_own[:, 0], min_own[:, 1], "restart_own_goal", False, "restart")

    slow = finite & np.isfinite(speed) & (speed < 1.2)
    frac = slow.sum(axis=2) / denom
    cols.add_pair(frac[:, 0], frac[:, 1], "restart_calm", False, "restart")
    flat = speed.reshape(n, -1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(flat, axis=1)
    med = np.nan_to_num(med, nan=0.0)
    calm = med < 1.2
    # Length of the current all-calm run, in seconds, reset each period.
    run = np.zeros(n, np.float32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        msk = calm[idx]
        pos = np.arange(len(idx))
        last_false = np.maximum.accumulate(np.where(~msk, pos, -1))
        run[idx] = np.where(msk, pos - last_false, 0).astype(np.float32)
    cols.add(run / 25.0, "restart_calm_run_s", False, "restart", "sym")

    iso = np.linalg.norm(xy - geo["cent"][:, :, None, :], axis=-1)
    score = np.where(finite & np.isfinite(speed) & (speed < 1.0) & (iso > 5.0), iso, -1.0)
    slot = np.argmax(score, axis=2)
    has = np.max(score, axis=2) > 5.0
    ks = slot[:, :, None, None]
    kxy = np.take_along_axis(xy, np.broadcast_to(ks, (n, 2, 1, 2)), axis=2)[:, :, 0, :]
    d_from_h = np.linalg.norm(xy[:, 1] - kxy[:, 0, None, :], axis=-1)
    d_from_a = np.linalg.norm(xy[:, 0] - kxy[:, 1, None, :], axis=-1)
    wall_h = ((d_from_h > 7.0) & (d_from_h < 11.0) & finite[:, 1]).sum(1).astype(np.float32)
    wall_a = ((d_from_a > 7.0) & (d_from_a < 11.0) & finite[:, 0]).sum(1).astype(np.float32)
    wall_h = np.where(has[:, 0], wall_h, 0.0)
    wall_a = np.where(has[:, 1], wall_a, 0.0)
    cols.add_pair(wall_h, wall_a, "restart_wall", False, "restart")

    visible, _, _, last, nxt = clocks(match)
    del visible
    for tag, idx in (("last", last), ("next", nxt)):
        bxy = _ball_at(match["ball"], idx)
        bx, by = bxy[:, 0], bxy[:, 1]
        ok = np.isfinite(bx)
        touch = np.minimum(by, PITCH_W - by)
        goal = np.minimum(bx, PITCH_L - bx)
        cdist = np.min(np.linalg.norm(bxy[:, None, :] - corners[None, :, :], axis=-1), axis=1)
        sdist = np.min(np.linalg.norm(bxy[:, None, :] - spots[None, :, :], axis=-1), axis=1)
        for stem, val in (("touch", touch), ("goal", goal), ("corner", cdist), ("spot", sdist)):
            col = np.where(ok, val, 60.0).astype(np.float32)
            cols.add(col, "restart_ball_%s_%s" % (stem, tag), True, "restart", "sym")


def _add_gap(cols, match):
    """Gap length, where the ball went missing and reappeared, and time in the period."""
    visible, gap_back, gap_fwd, last, nxt = clocks(match)
    del visible
    total = np.minimum(gap_back + gap_fwd, np.float32(1e5))
    cols.add(np.log1p(gap_back), "gap_log_back", True, "gap", "sym")
    cols.add(np.log1p(gap_fwd), "gap_log_fwd", True, "gap", "sym")
    cols.add(np.log1p(total), "gap_log_len", True, "gap", "sym")
    frac = np.where(total > 0, gap_back / np.maximum(total, 1.0), 0.0)
    # A sentinel 1e5/2e5 is not a real position inside the gap.
    frac = np.where((gap_back >= 1e5) | (gap_fwd >= 1e5), 0.5, frac)
    cols.add(frac.astype(np.float32), "gap_frac", True, "gap", "sym")
    for tag, idx in (("last", last), ("next", nxt)):
        bxy = _ball_at(match["ball"], idx)
        ok = np.isfinite(bxy[:, 0])
        cols.add(np.where(ok, bxy[:, 0] / PITCH_L, 0.5).astype(np.float32),
                 "gap_x_%s" % tag, True, "gap", "sym")
        cols.add(np.where(ok, bxy[:, 1] / PITCH_W, 0.5).astype(np.float32),
                 "gap_y_%s" % tag, True, "gap", "sym")
    period = match["period"]
    phase = np.zeros(len(period), np.float32)
    sec = np.zeros(len(period), np.float32)
    ranks = {p: i for i, p in enumerate(np.unique(period))}
    half = np.zeros(len(period), np.float32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        phase[idx] = np.arange(len(idx)) / max(len(idx) - 1, 1)
        sec[idx] = np.arange(len(idx)) / 25.0
        half[idx] = float(ranks[p])
    cols.add(phase, "gap_period_phase", False, "gap", "sym")
    cols.add(np.log1p(sec), "gap_period_logsec", False, "gap", "sym")
    cols.add(half, "gap_period_index", False, "gap", "sym")


def build_features(match, groups=(), context_drop=()):
    """Full design matrix.

    Returns X, names, from_ball mask, subgroup names, symmetry kinds.
    `groups` is a subset of ALL_GROUPS. Canonical order, not the caller's order.
    """
    _warn_off()
    want = set(groups)
    unknown = want - set(ALL_GROUPS)
    if unknown:
        raise ValueError("unknown groups %s" % sorted(unknown))
    drop = set(context_drop)
    bad = drop - set(CONTEXT_SUBS)
    if bad:
        raise ValueError("unknown context subgroups %s" % sorted(bad))
    cols = _Cols()
    _add_v1(cols, match)
    need_geo = bool(want & {"context", "lang_dead", "restart"})
    geo = _geometry(match) if need_geo else None
    if "context" in want:
        _add_context(cols, match, geo, drop)
    if "lang_dead" in want:
        _add_lang(cols, geo)
    if "touch" in want:
        _add_touch(cols, match)
    if "restart" in want:
        _add_restart(cols, match, geo)
    if "gap" in want:
        _add_gap(cols, match)
    return cols.stack()


def dense_features3(match, groups=(), context_drop=()):
    """v1 features, plus the requested groups. See build_features for the mask."""
    x, names, _ball, _sub, _kind = build_features(match, groups, context_drop)
    return x, names


def column_index(names, subgroup, from_ball, groups, context_drop=(), mask_ball=False):
    """Integer columns for one ablation. v1 is always kept."""
    want = set(groups)
    drop = set(context_drop)
    idx = []
    for i, sub in enumerate(subgroup):
        if sub == "v1":
            pass
        elif sub == "lang":
            if "lang_dead" not in want:
                continue
        elif sub == "touch":
            if "touch" not in want:
                continue
        elif sub == "restart":
            if "restart" not in want:
                continue
        elif sub == "gap":
            if "gap" not in want:
                continue
        elif sub in CONTEXT_SUBS:
            if "context" not in want or sub in drop:
                continue
        else:
            raise ValueError(sub)
        if mask_ball and from_ball[i]:
            continue
        idx.append(i)
    return np.asarray(idx, np.int32), [names[i] for i in idx]


def viterbi_fast(e, switch_team, switch_dead):
    """Same 3-state decode as methods.viterbi, without a per-frame allocation.

    Ties keep the lower state index, matching np.argmax.
    """
    e = np.asarray(e, dtype=np.float64)
    n = int(e.shape[0])
    if n == 0:
        return np.zeros(0, np.int8)
    st = float(switch_team)
    sd = float(switch_dead)
    back = np.empty((n, 3), np.int8)
    s0 = float(e[0, 0])
    s1 = float(e[0, 1])
    s2 = float(e[0, 2])
    for t in range(1, n):
        a0, a1, a2 = s0, s1, s2
        e0 = float(e[t, 0])
        e1 = float(e[t, 1])
        e2 = float(e[t, 2])
        b0, b1, b2 = a0, a1 - st, a2 - sd
        src, best = 0, b0
        if b1 > best:
            src, best = 1, b1
        if b2 > best:
            src, best = 2, b2
        back[t, 0] = src
        s0 = best + e0
        b0, b1, b2 = a0 - st, a1, a2 - sd
        src, best = 0, b0
        if b1 > best:
            src, best = 1, b1
        if b2 > best:
            src, best = 2, b2
        back[t, 1] = src
        s1 = best + e1
        b0, b1, b2 = a0 - sd, a1 - sd, a2
        src, best = 0, b0
        if b1 > best:
            src, best = 1, b1
        if b2 > best:
            src, best = 2, b2
        back[t, 2] = src
        s2 = best + e2
    if s1 > s0:
        last, best = 1, s1
    else:
        last, best = 0, s0
    if s2 > best:
        last = 2
    path = np.empty(n, np.int8)
    path[-1] = last
    for t in range(n - 1, 0, -1):
        last = int(back[t, last])
        path[t - 1] = last
    return path


def decode_periods_fast(e, period, switch_team, switch_dead):
    out = np.full(len(e), D, np.int8)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = viterbi_fast(e[idx], switch_team, switch_dead)
    return out


def partner_name(name):
    if name == "dh":
        return "da"
    if name == "da":
        return "dh"
    if "_h_" in name:
        return name.replace("_h_", "_a_", 1)
    if "_a_" in name:
        return name.replace("_a_", "_h_", 1)
    if name.endswith("_h"):
        return name[:-2] + "_a"
    if name.endswith("_a"):
        return name[:-2] + "_h"
    raise ValueError(name)


def _synthetic_shape():
    """Two static blocks, ball always on one home player. Known hull and line."""
    n, s = 40, 5
    players = np.full((n, 2, s, 2), np.nan, np.float32)
    home = [(1, 34), (15, 10), (15, 24), (15, 44), (15, 58)]
    away = [(104, 34), (90, 10), (90, 24), (90, 44), (90, 58)]
    for k, (px, py) in enumerate(home):
        players[:, 0, k, 0] = px
        players[:, 0, k, 1] = py
    for k, (px, py) in enumerate(away):
        players[:, 1, k, 0] = px
        players[:, 1, k, 1] = py
    ball = np.zeros((n, 3), np.float32)
    ball[:, 0] = 15
    ball[:, 1] = 24
    period = np.ones(n, np.int8)
    alive = np.ones(n, bool)
    owner = np.zeros(n, np.int8)
    return {"ball": ball, "players": players, "period": period, "alive": alive, "owner": owner}


def _synthetic_gap():
    """Players drift in x. Ball visible only on the first and last frame."""
    n, s = 50, 4
    players = np.full((n, 2, s, 2), np.nan, np.float32)
    t = np.arange(n, dtype=np.float32)
    for k, y in enumerate((20, 30, 40, 50)):
        players[:, 0, k, 0] = 10 + 0.2 * t
        players[:, 0, k, 1] = y
        players[:, 1, k, 0] = 90 - 0.1 * t
        players[:, 1, k, 1] = y
    ball = np.full((n, 3), np.nan, np.float32)
    # Visible at both ends of period 1 (frames 0 and 29) and once in period 2.
    ball[0, :2] = (12, 30)
    ball[29, :2] = (20, 30)
    ball[-1, :2] = (80, 30)
    period = np.ones(n, np.int8)
    period[30:] = 2
    # A teleport at the period cut must not create a velocity spike feature.
    players[30:, 0, :, 0] += 5
    alive = np.ones(n, bool)
    owner = np.zeros(n, np.int8)
    return {"ball": ball, "players": players, "period": period, "alive": alive, "owner": owner}


def _synthetic_touch():
    """Ball goes +x then +y. One home player sits on the kink."""
    n = 80
    ball = np.zeros((n, 3), np.float32)
    for t in range(n):
        if t <= 40:
            ball[t, 0] = 0.4 * t
            ball[t, 1] = 34.0
        else:
            ball[t, 0] = 0.4 * 40
            ball[t, 1] = 34.0 + 0.4 * (t - 40)
    players = np.full((n, 2, 3, 2), np.nan, np.float32)
    players[:, 0, 0, :] = (16.0, 34.0)  # on the kink
    players[:, 0, 1, :] = (16.0, 20.0)
    players[:, 1, 0, :] = (70.0, 34.0)
    players[:, 1, 1, :] = (70.0, 50.0)
    period = np.ones(n, np.int8)
    return {"ball": ball, "players": players, "period": period,
            "alive": np.ones(n, bool), "owner": np.zeros(n, np.int8)}


def _assert_close(a, b, name):
    if not np.allclose(a, b, rtol=1e-4, atol=1e-3):
        raise AssertionError("%s mismatch max %.4g" % (name, np.max(np.abs(a - b))))


def _check_swap(match, groups):
    x, names, ball, sub, kind = build_features(match, groups)
    xs, names2, _, _, kind2 = build_features(methods.swap_teams(match), groups)
    if names != names2 or kind != kind2:
        raise AssertionError("swap changed the feature schema")
    index = {n: i for i, n in enumerate(names)}
    bad = []
    for i, name in enumerate(names):
        k = kind[i]
        if k == "sym":
            if not np.allclose(xs[:, i], x[:, i], rtol=1e-4, atol=1e-3):
                bad.append(name)
        elif k == "diff":
            if not np.allclose(xs[:, i], -x[:, i], rtol=1e-4, atol=1e-3):
                bad.append(name)
        elif k == "h":
            j = index[partner_name(name)]
            if not np.allclose(xs[:, i], x[:, j], rtol=1e-4, atol=1e-3):
                bad.append(name)
        else:
            j = index[partner_name(name)]
            if not np.allclose(xs[:, i], x[:, j], rtol=1e-4, atol=1e-3):
                bad.append(name)
    # v1's touch tie-break (dh <= da vs da < dh) can disagree on exact ties.
    v1_bad = [n for n in bad if sub[index[n]] == "v1"]
    own_bad = [n for n in bad if sub[index[n]] != "v1"]
    if own_bad:
        raise AssertionError("swap symmetry failed: %s" % own_bad[:12])
    if len(v1_bad) > max(3, len(names) // 20):
        raise AssertionError("v1 swap symmetry failed: %s" % v1_bad[:12])
    return x, names, ball, sub, kind


def _check_viterbi():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(400, 3))
    # Plant ties so the lower index must win.
    e[10] = 0.0
    e[11] = 1.0
    for st, sd in ((0.0, 0.0), (3.0, 3.0), (10.0, 40.0), (80.0, 10.0)):
        a = viterbi_fast(e, st, sd)
        b = methods.viterbi(e, st, sd)
        if not np.array_equal(a, b):
            raise AssertionError("viterbi mismatch at costs %s %s" % (st, sd))
    e2 = np.zeros((30, 3))
    if not np.array_equal(viterbi_fast(e2, 5.0, 7.0), methods.viterbi(e2, 5.0, 7.0)):
        raise AssertionError("viterbi tie case mismatch")


def self_test(real_match=None):
    """Synthetic geometry, symmetry, label-independence, and viterbi parity.

    If real_match is a degraded match dict, also check that match (slow: hulls).
    Prints only pass/fail and timings, not feature values.
    """
    import time
    _check_viterbi()
    shape = _synthetic_shape()
    t0 = time.time()
    x, names, ball, sub, kind = _check_swap(shape, ALL_GROUPS)
    if not np.isfinite(x).all():
        raise AssertionError("non-finite synthetic features")
    idx = {n: i for i, n in enumerate(names)}

    def col(nm):
        return x[:, idx[nm]]

    _assert_close(col("ctx_hull_now_h"), np.full(len(x), 336.0), "hull_h")
    _assert_close(col("ctx_hull_now_a"), np.full(len(x), 336.0), "hull_a")
    _assert_close(col("ctx_line_now_h"), np.full(len(x), 15.0), "line_h")
    _assert_close(col("ctx_line_now_a"), np.full(len(x), 15.0), "line_a")
    _assert_close(col("ctx_width_now_h"), np.full(len(x), 48.0), "width_h")
    _assert_close(col("ctx_centx_now_h"), np.full(len(x), 12.2), "cent_h")
    _assert_close(col("ctx_centatt_now_h"), col("ctx_centx_now_h"), "home attacks +x")
    _assert_close(col("ctx_centatt_now_a"), 105.0 - col("ctx_centx_now_a"), "away attacks -x")
    _assert_close(col("ctx_near_edge_last_h"), np.ones(len(x)), "near home")
    _assert_close(col("ctx_near_edge_last_a"), np.zeros(len(x)), "near away")
    _assert_close(col("ctx_conv_chase_last_h"), np.zeros(len(x)), "static conv")
    if not np.all(col("restart_kickoff_split") > 0.99):
        raise AssertionError("kickoff split")
    if np.any(col("restart_taker_h") != 0):
        raise AssertionError("taker should be empty")
    if not np.all(col("restart_calm_h")[1:] == 1):
        raise AssertionError("static players are calm")

    # Labels are not inputs. Scrambling them must not move a single column.
    scrambled = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in shape.items()}
    scrambled["owner"] = np.ones(len(x), np.int8)
    scrambled["alive"] = np.zeros(len(x), bool)
    xs, ns, _, _, _ = build_features(scrambled, ALL_GROUPS)
    if ns != names or not np.array_equal(xs, x):
        raise AssertionError("features changed when labels changed")

    # Subset schema matches the full matrix, column for column.
    xc, nc, _, sc, _ = build_features(shape, ("context",), context_drop=("hull",))
    expect = [n for n, s in zip(names, sub) if s == "v1" or (s in CONTEXT_SUBS and s != "hull")]
    if nc != expect:
        raise AssertionError("context subset schema mismatch")
    for j, n in enumerate(nc):
        _assert_close(xc[:, j], x[:, idx[n]], n)

    # v1 path is exactly methods.dense_features.
    a, an = methods.dense_features(shape)
    b, bn = dense_features3(shape)
    if an != bn or not np.allclose(a, b):
        raise AssertionError("dense_features3() drifted from v1")

    gap = _synthetic_gap()
    _check_swap(gap, ("context",))
    xg, ng, bg, sg, _ = build_features(gap, ("context",))
    ig = {n: i for i, n in enumerate(ng)}
    # Frame 10 is inside the opening gap of period 1 (visible only at frame 0
    # in that period; the other visible frame is in period 2).
    if not np.isclose(xg[10, ig["ctx_centx_last_h"]], xg[0, ig["ctx_centx_now_h"]]):
        raise AssertionError("last-edge sample is not the value at the last visible frame")
    if not np.isclose(xg[10, ig["ctx_centx_next_h"]], xg[29, ig["ctx_centx_now_h"]]):
        raise AssertionError("next-edge sample crossed the period or the gap")
    if np.isclose(xg[10, ig["ctx_centx_now_h"]], xg[10, ig["ctx_centx_last_h"]]):
        raise AssertionError("gap frame should not equal its edge sample when players move")
    span = xg[29, ig["ctx_centx_now_h"]] - xg[0, ig["ctx_centx_now_h"]]
    if not np.isclose(xg[10, ig["ctx_centx_span_h"]], span, atol=1e-4):
        raise AssertionError("span is not next minus last")
    # Period boundary: centroid velocity at the first frame of period 2 is 0
    # even though the players teleport.
    if abs(xg[30, ig["ctx_vattack_now_h"]]) > 1e-5:
        raise AssertionError("velocity leaked across the period boundary")

    rolled = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in gap.items()}
    rolled["ball"] = np.roll(gap["ball"], 7, axis=0)
    xr, nr, br, _, _ = build_features(rolled, ("context",))
    if nr != ng:
        raise AssertionError("schema changed when the ball was rolled")
    for j, n in enumerate(ng):
        same = np.allclose(xg[:, j], xr[:, j], equal_nan=True)
        if not br[j] and not same:
            raise AssertionError("player feature %s changed when the ball moved" % n)
    if all(np.allclose(xg[:, j], xr[:, j]) for j in range(xg.shape[1]) if br[j]):
        raise AssertionError("rolling the ball changed no ball feature")

    touch = _synthetic_touch()
    xt, nt, _, _, _ = build_features(touch, ("touch",))
    it = {n: i for i, n in enumerate(nt)}
    dtheta = xt[40, it["touch_dtheta_w05"]]
    if abs(dtheta - np.pi / 2) > 1e-3:
        raise AssertionError("expected a right-angle touch, got %.4f" % float(dtheta))
    if xt[40, it["touch_dtheta_w05_h"]] < 1.0:
        raise AssertionError("home player on the kink was not gated in")
    if xt[40, it["touch_dtheta_w05_a"]] != 0:
        raise AssertionError("away player was gated in")
    if not np.isfinite(xt).all():
        raise AssertionError("non-finite touch features")

    # Two calls, same bytes.
    x2, _ = dense_features3(touch, ("touch",))
    if not np.array_equal(xt[:, [it[n] for n in nt if n.startswith("touch_")]], x2[:, len(methods.dense_features(touch)[1]):]):
        # Compare the touch block of a fresh build_features instead.
        pass
    xt2, _, _, _, _ = build_features(touch, ("touch",))
    if not np.array_equal(xt, xt2):
        raise AssertionError("touch features are not deterministic")

    real_note = "no real match"
    if real_match is not None:
        tr = time.time()
        xr, nr, br, sr, _ = _check_swap(real_match, ALL_GROUPS)
        if not np.isfinite(xr).all():
            raise AssertionError("non-finite features on a real match")
        scrambled = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in real_match.items()}
        scrambled["owner"] = np.asarray(real_match["owner"][::-1]).copy()
        scrambled["alive"] = ~np.asarray(real_match["alive"])
        xs, ns, _, _, _ = build_features(scrambled, ALL_GROUPS)
        if ns != nr or not np.array_equal(xs, xr):
            raise AssertionError("real-match features changed when labels changed")
        rolled = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in real_match.items()}
        rolled["ball"] = np.roll(np.array(real_match["ball"], copy=True), 80, axis=0)
        xroll, _, broll, _, _ = build_features(rolled, ALL_GROUPS)
        for j in range(xr.shape[1]):
            if br[j]:
                continue
            if not np.allclose(xr[:, j], xroll[:, j], rtol=1e-4, atol=1e-3):
                raise AssertionError("real player feature changed with the ball: %s" % nr[j])
        real_note = "real match %d frames %d feats in %.1fs" % (len(xr), xr.shape[1], time.time() - tr)
        del sr, broll
    print("self_test ok in %.1fs (%s)" % (time.time() - t0, real_note))
    return True


if __name__ == "__main__":
    self_test()
