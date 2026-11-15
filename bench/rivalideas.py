"""Rival-team ideas measured on top of v1+context.

R1  minimum stoppage and short-gap cleanup (post-processing)
R2  heavier LightGBM weights on frames next to a true label change
R4  team-shape samples at t+25, t+50, t+125 and past-vs-future contrasts
R5  label-free sanity checks on any H/A/D array

R3 lives in bench/degrade_v3.py. Nothing here reads competition data or the
public five-minute window.
"""

import numpy as np

from bench.methods import A, D, H

FPS = 25
# Attack-frame shape already computed as ctx_*_now_* . Fixed shifts of these
# channels are the look-ahead: the shape after a team wins the ball, not a
# centred window. vleft/vright are omitted; vattack is the same velocity in
# the attack frame.
LA_STEMS = ("centx", "centatt", "width", "depth", "line", "opp", "hull", "vattack")
LA_SHIFTS = (25, 50, 125)  # 1 s, 2 s, 5 s at 25 Hz
LA_REV = "la1"


def maximal_runs(labels):
    """[[start, end, code], ...) maximal constant runs. end is exclusive."""
    y = np.asarray(labels)
    n = int(len(y))
    if n == 0:
        return []
    change = np.flatnonzero(y[1:] != y[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    return [[int(s), int(e), int(y[s])] for s, e in zip(starts, ends)]


def _paint(dest, idx, runs):
    for s, e, k in runs:
        dest[idx[s:e]] = k


def _drop_short_d(runs, td):
    """Absorb D runs shorter than td into the surrounding state.

    Same neighbours on both sides become that label. Different neighbours:
    the longer one takes the run, and a tie goes to the earlier run.
    A short D that fills the whole period stays D.
    """
    if td <= 0:
        return runs
    guard = 0
    limit = max(4, len(runs) + 2)
    changed = True
    while changed and guard < limit:
        guard += 1
        changed = False
        out = []
        i = 0
        n = len(runs)
        while i < n:
            s, e, k = runs[i]
            if k == D and (e - s) < td:
                left = out[-1] if out else None
                right = runs[i + 1] if i + 1 < n else None
                if left is None and right is None:
                    out.append([s, e, k])
                elif left is None:
                    right[0] = s
                    changed = True
                elif right is None:
                    left[1] = e
                    changed = True
                elif left[2] == right[2]:
                    left[1] = right[1]
                    i += 1
                    changed = True
                else:
                    if (left[1] - left[0]) >= (right[1] - right[0]):
                        left[1] = e
                    else:
                        right[0] = s
                    changed = True
            else:
                if out and out[-1][2] == k and out[-1][1] == s:
                    out[-1][1] = e
                else:
                    out.append([s, e, int(k)])
            i += 1
        runs = out
    return runs


def _bridge(runs, tm):
    """Merge same-team possessions separated by one shorter D or opposing run."""
    if tm <= 0:
        return runs
    guard = 0
    limit = max(4, len(runs) + 2)
    changed = True
    while changed and guard < limit:
        guard += 1
        changed = False
        out = []
        i = 0
        n = len(runs)
        while i < n:
            if (
                i + 2 < n
                and runs[i][2] == runs[i + 2][2]
                and runs[i][2] != D
                and runs[i + 1][2] != runs[i][2]
                and (runs[i + 1][1] - runs[i + 1][0]) < tm
            ):
                s = runs[i][0]
                e = runs[i + 2][1]
                k = runs[i][2]
                if out and out[-1][2] == k and out[-1][1] == s:
                    out[-1][1] = e
                else:
                    out.append([s, e, int(k)])
                i += 3
                changed = True
                continue
            s, e, k = runs[i]
            if out and out[-1][2] == k and out[-1][1] == s:
                out[-1][1] = e
            else:
                out.append([s, e, int(k)])
            i += 1
        runs = out
    return runs


def cleanup_labels(pred, period, t_d, t_m, fps=FPS):
    """Post-process one match. Periods are cleaned separately.

    t_d, t_m are seconds. A threshold of 0 turns that rule off. A run is
    removed only when it is strictly shorter than the threshold.
    """
    out = np.asarray(pred, np.int8).copy()
    period = np.asarray(period)
    td = int(round(float(t_d) * fps))
    tm = int(round(float(t_m) * fps))
    if td <= 0 and tm <= 0:
        return out
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        runs = maximal_runs(out[idx])
        if td > 0:
            runs = _drop_short_d(runs, td)
        if tm > 0:
            runs = _bridge(runs, tm)
        _paint(out, idx, runs)
    return out


def boundary_weights(y, period, radius, mult):
    """1 everywhere, `mult` within `radius` frames of a same-period label change.

    The change frame is the first frame of the new label. Period boundaries
    are not label changes. radius is in frames (25 = 1 s).
    """
    y = np.asarray(y)
    period = np.asarray(period)
    w = np.ones(len(y), np.float32)
    radius = int(radius)
    if radius <= 0 or float(mult) == 1.0:
        return w
    mult = np.float32(mult)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        yy = y[idx]
        if len(yy) < 2:
            continue
        change = np.flatnonzero(yy[1:] != yy[:-1]) + 1
        if len(change) == 0:
            continue
        mark = np.zeros(len(yy), bool)
        for t in change:
            lo = max(0, int(t) - radius)
            hi = min(len(yy), int(t) + radius + 1)
            mark[lo:hi] = True
        ww = w[idx]
        ww[mark] = mult
        w[idx] = ww
    return w


def label_sanity(labels, period, fps=FPS):
    """Aggregates on one label array. Safe to print.

    play_share is the fraction of frames that are H or A.
    play_min is that time in minutes.
    poss_per_90 counts H/A spells per 90 minutes of tracking clock
    (the clock includes dead frames).
    D quantiles are run lengths in seconds.
    poss_under_4s is the share of possessions shorter than 4 s.
    home_share is P(H | not D).
    """
    labels = np.asarray(labels)
    period = np.asarray(period)
    n = int(len(labels))
    play = labels != D
    play_frames = int(play.sum())
    clock_min = (n / float(fps)) / 60.0 if n else 0.0
    poss_dur = []
    d_len = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        for s, e, k in maximal_runs(labels[idx]):
            dur = (e - s) / float(fps)
            if k == D:
                d_len.append(dur)
            elif k == H or k == A:
                poss_dur.append(dur)
    n_poss = len(poss_dur)
    if d_len:
        qs = np.quantile(np.asarray(d_len, np.float64), [0.25, 0.5, 0.9])
        d_p25, d_p50, d_p90 = (float(qs[0]), float(qs[1]), float(qs[2]))
    else:
        d_p25 = d_p50 = d_p90 = float("nan")
    if poss_dur:
        under = float(np.mean(np.asarray(poss_dur) < 4.0))
    else:
        under = float("nan")
    if play_frames:
        home = float(np.mean(labels[play] == H))
    else:
        home = float("nan")
    return {
        "frames": n,
        "play_share": float(play_frames / n) if n else float("nan"),
        "play_min": float(play_frames / float(fps) / 60.0) if n else float("nan"),
        "poss": int(n_poss),
        "poss_per_90": float(n_poss * 90.0 / clock_min) if clock_min else float("nan"),
        "d_p25": d_p25,
        "d_p50": d_p50,
        "d_p90": d_p90,
        "n_d_runs": int(len(d_len)),
        "poss_under_4s": under,
        "home_share": home,
    }


def pool_sanity(labels_list, periods):
    """Sanity on several matches at once. Period ids are shifted so halves stay apart."""
    parts_y = []
    parts_p = []
    off = 0
    for labels, period in zip(labels_list, periods):
        period = np.asarray(period)
        parts_y.append(np.asarray(labels))
        parts_p.append(np.asarray(period, np.int32) + off)
        off += int(period.max()) + 1 if len(period) else 1
    if not parts_y:
        return label_sanity(np.zeros(0, np.int8), np.zeros(0, np.int8))
    return label_sanity(np.concatenate(parts_y), np.concatenate(parts_p))


def shift_cols(base, shifts, period):
    """base (n, C) sampled at each shift, inside each period, edge-held.

    Positive shift looks forward. Matches bench.feats3._shift_all.
    """
    base = np.asarray(base, np.float32)
    n, c = base.shape
    out = np.empty((n, c * len(shifts)), np.float32)
    sliced = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        sliced.append((idx, np.asarray(base[idx], np.float32)))
    for si, s in enumerate(shifts):
        dest = out[:, si * c:(si + 1) * c]
        s = int(s)
        if s == 0:
            dest[:] = base
            continue
        k = abs(s)
        for idx, block in sliced:
            b = np.empty_like(block)
            if k >= len(block):
                b[:] = block[-1] if s > 0 else block[0]
            elif s > 0:
                b[:-k] = block[k:]
                b[-k:] = block[-1]
            else:
                b[k:] = block[:-k]
                b[:k] = block[0]
            dest[idx] = b
    return out


def lookahead_names(stems=LA_STEMS, shifts=LA_SHIFTS):
    names = []
    for s in shifts:
        for stem in stems:
            for side in ("h", "a"):
                names.append("la_%s_p%d_%s" % (stem, int(s), side))
        for stem in stems:
            for side in ("h", "a"):
                names.append("la_%s_c%d_%s" % (stem, int(s), side))
    return names


def lookahead_from_context(x, names, period, stems=LA_STEMS, shifts=LA_SHIFTS):
    """Future samples and (t+h) - (t-h) contrasts of context 'now' columns.

    x is one team-view of the v1+context (or fuller) matrix. The swapped view
    is produced by calling this on the swapped matrix, not by swapping columns
    after the fact.
    """
    names = list(names)
    pos = {n: i for i, n in enumerate(names)}
    cols = []
    for stem in stems:
        for side in ("h", "a"):
            key = "ctx_%s_now_%s" % (stem, side)
            if key not in pos:
                raise KeyError(key)
            cols.append(pos[key])
    cols = np.asarray(cols, np.int32)
    base = np.asarray(x[:, cols], np.float32)
    order = []
    for s in shifts:
        order.append(-int(s))
        order.append(int(s))
    shifted = shift_cols(base, order, period)
    c = base.shape[1]
    pieces = []
    for si, _s in enumerate(shifts):
        past = shifted[:, (2 * si) * c:(2 * si + 1) * c]
        fut = shifted[:, (2 * si + 1) * c:(2 * si + 2) * c]
        pieces.append(fut)
        pieces.append(fut - past)
    out = np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
    return out, lookahead_names(stems, shifts)


def choose_costs_grid(score_fn, grid_team, grid_dead):
    """score_fn(st, sd) -> mean spellF1 over the matches being tuned."""
    best = None
    for st in grid_team:
        for sd in grid_dead:
            mean = float(score_fn(float(st), float(sd)))
            if best is None or mean > best[0]:
                best = (mean, float(st), float(sd))
    return best


def self_test():
    """Invariants that do not need a match file."""
    # Short D between the same team is absorbed. 3 frames < 1 s.
    y = np.array([H] * 10 + [D] * 3 + [H] * 10, np.int8)
    p = np.ones(len(y), np.int8)
    out = cleanup_labels(y, p, 1, 0)
    assert np.all(out == H), "short D between H should become H"

    # A D run of exactly 1 s is not shorter than 1 s.
    y = np.array([H] * 10 + [D] * 25 + [H] * 10, np.int8)
    p = np.ones(len(y), np.int8)
    out = cleanup_labels(y, p, 1, 0)
    assert np.array_equal(out, y)

    # Different neighbours: the longer side takes the short D.
    y = np.array([H] * 30 + [D] * 2 + [A] * 10, np.int8)
    p = np.ones(len(y), np.int8)
    out = cleanup_labels(y, p, 1, 0)
    assert np.all(out[:32] == H) and np.all(out[32:] == A)

    # Equal neighbours: the earlier run wins the tie.
    y = np.array([H] * 10 + [D] * 2 + [A] * 10, np.int8)
    p = np.ones(len(y), np.int8)
    out = cleanup_labels(y, p, 1, 0)
    assert np.all(out[:12] == H) and np.all(out[12:] == A)

    # Bridge a short opponent, and do not cross a period boundary.
    y = np.array([H] * 20 + [A] * 5 + [H] * 20, np.int8)
    p = np.ones(len(y), np.int8)
    out = cleanup_labels(y, p, 0, 1)
    assert np.all(out == H)
    y2 = np.array([H] * 10 + [A] * 10, np.int8)
    p2 = np.array([1] * 10 + [2] * 10, np.int8)
    assert np.array_equal(cleanup_labels(y2, p2, 0, 3), y2)

    # Threshold 0 is a no-op. Order is min-D first, then the bridge, so a D
    # that survives T_d can still be bridged by a larger T_m.
    y = np.array([H] * 10 + [D] * 30 + [H] * 10, np.int8)
    p = np.ones(len(y), np.int8)
    kept = cleanup_labels(y, p, 1, 0)
    assert np.array_equal(kept, y)
    bridged = cleanup_labels(y, p, 1, 2)
    assert np.all(bridged == H)

    # A short live spell between two dead runs is not removed. The dead runs
    # themselves are longer than T_d, so this is not the min-stoppage rule.
    y = np.array([D] * 40 + [H] * 3 + [D] * 40, np.int8)
    p = np.ones(len(y), np.int8)
    assert np.array_equal(cleanup_labels(y, p, 1, 1), y)

    rng = np.random.default_rng(0)
    for n in (1, 2, 17, 80, 240):
        y = rng.integers(0, 3, size=n).astype(np.int8)
        p = np.ones(n, np.int8)
        if n > 20:
            p[n // 2:] = 2
        td_frames, tm_frames = 10, 20
        out = cleanup_labels(y, p, td_frames / FPS, tm_frames / FPS)
        assert out.shape == y.shape
        assert set(int(v) for v in np.unique(out)).issubset({H, A, D})
        for pp in np.unique(p):
            runs = maximal_runs(out[p == pp])
            covered = sum(e - s for s, e, _k in runs)
            assert covered == int(np.sum(p == pp))
            for i, (s, e, k) in enumerate(runs):
                if k == D and (e - s) < td_frames:
                    assert len(runs) == 1
                if 0 < i < len(runs) - 1 and runs[i - 1][2] == runs[i + 1][2] and runs[i - 1][2] != D:
                    assert k != runs[i - 1][2]
                    assert (e - s) >= tm_frames

    y = np.zeros(100, np.int8)
    y[50:] = A
    p = np.ones(100, np.int8)
    w = boundary_weights(y, p, 25, 3)
    assert w[0] == 1 and w[24] == 1 and w[25] == 3 and w[50] == 3 and w[75] == 3 and w[76] == 1
    y = np.array([H] * 10 + [A] * 10, np.int8)
    p = np.array([1] * 10 + [2] * 10, np.int8)
    assert np.all(boundary_weights(y, p, 5, 3) == 1)

    # fps=1 so a frame is one second.
    y = np.array([H, H, H, D, D, D, D, A, A], np.int8)
    p = np.ones(len(y), np.int8)
    s = label_sanity(y, p, fps=1)
    assert abs(s["play_share"] - 5 / 9) < 1e-9
    assert s["poss"] == 2
    assert abs(s["home_share"] - 0.6) < 1e-9
    assert abs(s["poss_under_4s"] - 1.0) < 1e-9
    assert abs(s["d_p50"] - 4.0) < 1e-9
    assert abs(s["poss_per_90"] - 1200.0) < 1e-6

    n = 80
    ramp = np.arange(n, dtype=np.float32)
    base = np.column_stack([ramp, -ramp])
    period = np.ones(n, np.int8)
    shifted = shift_cols(base, (-4, 4), period)
    # Interior: future - past = 8 for a unit ramp.
    contrast = shifted[:, 2:] - shifted[:, :2]
    assert np.allclose(contrast[10:-10, 0], 8.0)
    assert np.allclose(contrast[10:-10, 1], -8.0)
    # Edge hold: the last row's forward shift equals the last sample.
    assert np.allclose(shifted[-1, 2:], base[-1])

    from bench.feats3 import _shift_all
    base = rng.normal(size=(100, 3)).astype(np.float32)
    period = np.array([1] * 60 + [2] * 40, np.int8)
    shifts = (-25, 25, -50, 50, -125, 125)
    assert np.allclose(shift_cols(base, shifts, period), _shift_all(base, shifts, period))

    got = lookahead_names()
    assert len(got) == len(LA_STEMS) * 2 * len(LA_SHIFTS) * 2
    assert len(got) == len(set(got))
    assert got[0] == "la_centx_p25_h" and got[1] == "la_centx_p25_a"
    cols = {}
    for stem in LA_STEMS:
        cols["ctx_%s_now_h" % stem] = np.arange(n, dtype=np.float32)
        cols["ctx_%s_now_a" % stem] = np.zeros(n, np.float32)
    x = np.column_stack([cols[k] for k in cols])
    la, la_names = lookahead_from_context(x, list(cols), np.ones(n, np.int8))
    assert la_names[0] == "la_centx_p25_h"
    # Interior frame 40: future t+25 = 65, past t-25 = 15, contrast = 50.
    assert abs(float(la[40, 0]) - 65.0) < 1e-5
    assert abs(float(la[40, 16]) - 50.0) < 1e-5
    assert abs(float(la[40, 1])) < 1e-5

    from bench.degrade_v3 import _extend_min
    mask = np.zeros(200, bool)
    mask[10:12] = True
    mask[80:90] = True
    mask[190:193] = True
    ext = _extend_min(mask, 51)
    runs = maximal_runs(ext.astype(np.int8))
    true_runs = [(s, e) for s, e, k in runs if k == 1]
    assert true_runs[0] == (10, 61)
    assert true_runs[1] == (80, 131)
    # The last burst is clipped by the end of the array.
    assert true_runs[2][0] == 190 and true_runs[2][1] == 200
    for s, e in true_runs:
        if e < len(ext):
            assert e - s >= 51

    return "ok"
