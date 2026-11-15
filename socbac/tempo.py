"""Per-match tempo normalisation as a wrapper around prediction (label-free, no retraining).

pace(match)        one number per match: the 75th percentile, over the frames of the graded halves,
                   of the per-frame median tracked-player speed (m/s). An upper quantile reflects
                   live-play pace and is little affected by how much dead time the match has.
warp(match, k)     the match resampled in time inside each period: warped frame j shows the source at
                   time j*k (linear interpolation of positions, NaN kept; other arrays by nearest
                   frame). Speeds are multiplied by k.
to_source(...)     maps per-frame outputs of the warped match back to the source frames.

Use: k = PACE_REF / pace(match); if |k - 1| > DEADBAND: predict on warp(match, k) and map the
emissions back with to_source, then decode on the source timeline. PACE_REF is the mean pace of the
training matches (a constant stored with the model).
"""
import numpy as np

PACE_REF = 2.23      # mean pace of the 4 TRAIN matches (degrade_v4 seed-0 copies, open data); as real_gk_check.py
DEADBAND = 0.03
K_RANGE = (0.75, 1.33)


def pace(match, q=75):
    p = match["players"]
    n = len(p)
    flat = p.reshape(n, -1, 2)
    v = np.linalg.norm(np.diff(flat, axis=0), axis=2) * 25.0
    v = np.where(v > 12, np.nan, v)
    with np.errstate(all="ignore"):
        med = np.nanmedian(v, axis=1)
    per = np.asarray(match["period"])[1:]
    same = np.asarray(match["period"])[1:] == np.asarray(match["period"])[:-1]
    ok = (per > 0) & same & np.isfinite(med)
    return float(np.percentile(med[ok], q))


def _plan(period, k):
    period = np.asarray(period)
    cut = np.flatnonzero(period[1:] != period[:-1]) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(period)]))
    i0s, i1s, fs, rs, blocks = [], [], [], [], []
    w0 = 0
    for a, b in zip(starts, ends):
        n_p = b - a
        t = np.arange(max(1, int(round(n_p / k)))) * k
        t = t[t <= n_p - 1] if n_p > 1 else np.zeros(1)
        i0 = np.floor(t).astype(np.int64)
        i0s.append(a + i0)
        i1s.append(a + np.minimum(i0 + 1, n_p - 1))
        fs.append(t - i0)
        rs.append(a + np.minimum(np.rint(t).astype(np.int64), n_p - 1))
        blocks.append((a, b, w0, w0 + len(t)))
        w0 += len(t)
    return tuple(map(np.concatenate, (i0s, i1s, fs, rs))) + (blocks,)


def warp(match, k, extra=None):
    """(warped match, warped copies of the arrays in `extra` by nearest frame, blocks)."""
    i0, i1, f, r, blocks = _plan(match["period"], k)

    def interp(arr):
        w = f.reshape((-1,) + (1,) * (arr.ndim - 1)).astype(np.float32)
        out = (1 - w) * arr[i0] + w * arr[i1]
        exact = f == 0
        out[exact] = arr[i0][exact]
        return out.astype(arr.dtype)

    out = dict(match)
    out["ball"] = interp(match["ball"])
    out["players"] = interp(match["players"])
    for key in ("period", "alive", "owner"):
        if key in match:
            out[key] = np.asarray(match[key])[r]
    ex = {name: np.asarray(v)[r] for name, v in (extra or {}).items()}
    return out, ex, blocks


def to_source(values_warped, blocks, k, n_source):
    """Per-frame array of the warped match -> per-frame array on the source timeline (nearest)."""
    out = np.empty((n_source,) + values_warped.shape[1:], values_warped.dtype)
    for a, b, wa, wb in blocks:
        j = np.arange(b - a)
        jw = np.minimum(np.rint(j / k).astype(np.int64), wb - wa - 1)
        out[a:b] = values_warped[wa + jw]
    return out


def factor(match_pace, pace_ref, one_sided=True):
    """Warp factor for a match. one_sided (default): only matches SLOWER than the training pace are
    resampled (k > 1). The model is already robust to faster matches (bench: 15% faster costs
    0.002) and slowing them down cost 0.015 on L1, so they are left alone."""
    k = pace_ref / match_pace
    k = float(np.clip(k, *K_RANGE))
    if abs(k - 1.0) <= DEADBAND or (one_sided and k < 1.0):
        return 1.0
    return k
