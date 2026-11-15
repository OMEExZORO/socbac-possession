"""One command from the open data to the BAR-ATH submission, v2 (insurance pipeline).

    python -m bench.dfl fetch && python -m bench.dfl convert          # open training data, once
    python -m socbac.pipeline_v2 --out evaluation/results/socbac_v2.csv
    python -m socbac.pipeline_v2 --out evaluation/results/socbac_v2.csv --fresh   # ignore all caches

Method (see research/runs/pipeline/FINAL.md for the bench evidence):
  1. Training data: the 7 open IDSSE/DFL matches (bench.dfl, CC-BY 4.0), each degraded once with
     bench.degrade (seed = match index; PLAYER_SMOOTH = 12, calibrated on match-level aggregates of
     the real tracking) and, with --mixture, once more with the label-free bench.degrade_alt.
     No label, frame or statistic of BAR-ATH is used for training or tuning.
  2. Labels L1 (bench.labels_l1): the provider dead-ball flag is unchanged; a change of possession
     counts only if the new team keeps the ball for >= 1 s (written rule: controlled possession).
  3. Features v2 = v1 (bench.methods.dense_features) + ball-free team context + the Lang dead-ball
     block (bench.feats3 groups context + lang_dead, 351 columns), both team views.
  4. Team-symmetric LightGBM exactly as bench.methods.LGBMViterbi (its PARAMS, stride 5, 300 rounds,
     one thread, deterministic): emissions = mean of the two views' log posteriors minus log prior,
     then the 3-state Viterbi per half with switch costs team / dead (chosen on the bench TRAIN
     matches, see FINAL.md).
  5. BAR-ATH: real tracking via socbac.atd.load; graded halves from games.half_video_blocks. The ball
     is not in play before the kick-off: graded frames up to the organisers' published kick-off
     anchor (games.py, FAQ.md) are D, and the model labels each half from the next frame on. X
     outside the graded halves.

Prints aggregates only (shapes, timings, label shares and possession statistics of OUR output).
Per-frame intermediates are cached under data/derived/pipeline_v2/ (gitignored) only.
"""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from bench import degrade as dg  # noqa: E402
from bench import degrade_alt, dfl, feats3, methods  # noqa: E402
from bench.labels_l1 import labels_l1  # noqa: E402
from bench.run import labels as labels_provider  # noqa: E402

CACHE = os.path.join(ROOT, "data", "derived", "pipeline_v2")
GROUPS = ("context", "lang_dead")
N_FEATURES = 351
N_V1 = 49                      # the first 49 columns are exactly methods.dense_features (v1)
FEAT_REV = feats3.REV + "v2"
PLAYER_SMOOTH = 12
STRIDE = 5
ROUNDS = 300
SWITCH_TEAM, SWITCH_DEAD = 5.0, 20.0
MATCHES = list(dfl.MATCHES)    # J03WMX J03WN1 J03WOH J03WOY (bench TRAIN) J03WPY J03WQQ J03WR9 (TEST)
FPS = 25


def coupled_seed(i, bench_seed=0):
    """Degradation seed of match index i: bench.run's scheme, so the pipeline's copies are bench seed 0's."""
    return 1000 * bench_seed + i


def alt_seed(i, bench_seed=0):
    return 1000 * bench_seed + 500 + i


def log(msg):
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)


# -- degradation, labels, features ------------------------------------------------------------------

@contextlib.contextmanager
def player_smooth(w):
    old = dg.PLAYER_SMOOTH
    dg.PLAYER_SMOOTH = w
    try:
        yield
    finally:
        dg.PLAYER_SMOOTH = old


def degrade_copy(clean, kind, seed):
    with player_smooth(PLAYER_SMOOTH):
        if kind == "coupled":
            return dg.degrade(clean, seed=seed)
        if kind == "alt":
            return degrade_alt.degrade(clean, seed=seed)
    raise ValueError(kind)


def features_both(match):
    """v2 design matrix of both team views (float32), and the column names."""
    x, names, _, _, _ = feats3.build_features(match, GROUPS)
    xs, names2, _, _, _ = feats3.build_features(methods.swap_teams(match), GROUPS)
    if names != names2 or len(names) != N_FEATURES:
        raise RuntimeError("unexpected feature schema (%d columns)" % len(names))
    if not (np.isfinite(x).all() and np.isfinite(xs).all()):
        raise RuntimeError("non-finite features")
    return x, xs, names


def _save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(arr))
    os.replace(tmp, path)


def ext_stem(kind, seed, match, cache=CACHE):
    return os.path.join(cache, "ext", "%s_ps%d_s%d_%s_%s" % (kind, PLAYER_SMOOTH, seed, match, FEAT_REV))


def ext_labels(match, cache=CACHE):
    """(provider, L1, period) full-length codes of one open match (cached; labels need no degradation)."""
    path = os.path.join(cache, "ext", "labels_%s.npz" % match)
    if os.path.isfile(path):
        with np.load(path) as z:
            return z["prov"], z["l1"], z["period"]
    clean = dfl.load(match)
    out = (labels_provider(clean), labels_l1(clean), clean["period"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", prov=out[0], l1=out[1], period=out[2])
    os.replace(path + ".tmp.npz", path)
    return out


def ext_rows(kind, seed, match, cache=CACHE, full=False, cache_rows=True, mmap=False):
    """Stride-5 rows of both team views of one degraded open match, cached on disk.

    With full=True the full-length matrices are returned too (computed, never cached: they are
    only needed for bench scoring). With mmap=True cached rows come back memory-mapped (read-only).
    Returns dict with x5, xs5 (and x, xs if full).
    """
    stem = ext_stem(kind, seed, match, cache)
    mode = "r" if mmap else None
    have = os.path.isfile(stem + ".x5.npy") and os.path.isfile(stem + ".xs5.npy")
    if have and not full:
        return {"x5": np.load(stem + ".x5.npy", mmap_mode=mode), "xs5": np.load(stem + ".xs5.npy", mmap_mode=mode)}
    deg = degrade_copy(dfl.load(match), kind, seed)
    x, xs, _ = features_both(deg)
    del deg
    out = {"x5": np.ascontiguousarray(x[::STRIDE]), "xs5": np.ascontiguousarray(xs[::STRIDE])}
    if not have and cache_rows:
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        _save(stem + ".x5.npy", out["x5"])
        _save(stem + ".xs5.npy", out["xs5"])
        if mmap:
            out = {"x5": np.load(stem + ".x5.npy", mmap_mode=mode), "xs5": np.load(stem + ".xs5.npy", mmap_mode=mode)}
    if full:
        out["x"], out["xs"] = x, xs
    return out


# -- model ------------------------------------------------------------------------------------------

def fit(blocks, n_cols=None):
    """Team-symmetric LightGBM on stride-5 rows, rows in LGBMViterbi.fit order.

    blocks: list of (x5, xs5, y_full). Returns (booster, prior).
    """
    import lightgbm as lgb
    xs_, ys_ = [], []
    for x5, xs5, y in blocks:
        y5 = np.asarray(y)[::STRIDE]
        if len(y5) != len(x5):
            raise RuntimeError("rows and labels disagree")
        for part, lab in ((x5, y5), (xs5, methods.SWAP[y5])):
            part = part if n_cols is None else part[:, :n_cols]
            xs_.append(part if part.flags["C_CONTIGUOUS"] else np.ascontiguousarray(part))
            ys_.append(lab)
    y = np.concatenate(ys_).astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)
    # A list of row blocks is the same Dataset as their concatenation (LightGBM samples bin bounds
    # over the global row index and pushes rows in order; checked bit-identical, model text and
    # predictions) without a second copy of the matrix in memory.
    data = lgb.Dataset(xs_, y)
    del xs_
    booster = lgb.train(dict(methods.LGBMViterbi.PARAMS), data, num_boost_round=ROUNDS)
    return booster, prior


def emissions(booster, prior, x, xs, n_cols=None, threads=None):
    """Exactly LGBMViterbi.emissions: mean log posterior of both views (teams swapped back) / prior.

    threads only parallelises prediction over rows (each row's sum over trees is unchanged); the
    pipeline itself predicts with the model's own single-thread setting.
    """
    if n_cols is not None:
        x, xs = np.ascontiguousarray(x[:, :n_cols]), np.ascontiguousarray(xs[:, :n_cols])
    kw = {} if threads is None else {"num_threads": int(threads)}
    lp = np.log(booster.predict(x, **kw) + 1e-6)
    lp = 0.5 * (lp + np.log(booster.predict(xs, **kw) + 1e-6)[:, methods.SWAP])
    return lp - np.log(prior)


def decode(e, period, switch_team, switch_dead):
    """methods.decode_periods via feats3's loop-level port (asserted identical, ties included)."""
    return feats3.decode_periods_fast(e, period, switch_team, switch_dead)


# -- statistics of a label sequence (aggregates only) ----------------------------------------------

def _runs_in_blocks(codes, period):
    """(start, stop, code) maximal runs inside each contiguous period block."""
    out = []
    n = len(codes)
    br = np.flatnonzero(period[1:] != period[:-1]) + 1
    bounds = np.concatenate(([0], br, [n]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = codes[a:b]
        cb = np.flatnonzero(c[1:] != c[:-1]) + 1
        s = np.concatenate(([0], cb))
        e = np.concatenate((cb, [len(c)]))
        out.extend((a + int(i), a + int(j), int(c[i])) for i, j in zip(s, e))
    return out


def sanity_stats(codes, period):
    """Label-free summary of one label sequence (0 H, 1 A, 2 D) over its frames."""
    codes = np.asarray(codes)
    n = len(codes)
    runs = _runs_in_blocks(codes, np.asarray(period))
    poss = np.array([(b - a) for a, b, k in runs if k != methods.D], float) / FPS
    dead = np.array([(b - a) for a, b, k in runs if k == methods.D], float) / FPS
    alive = codes != methods.D
    minutes = n / FPS / 60.0
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else 0.0  # noqa: E731
    return {
        "frames": int(n),
        "share_H": float((codes == methods.H).mean()), "share_A": float((codes == methods.A).mean()),
        "share_D": float((codes == methods.D).mean()),
        "possessions": int(len(poss)),
        "poss_per_90": float(len(poss) / minutes * 90.0) if minutes else 0.0,
        "poss_mean_s": float(poss.mean()) if len(poss) else 0.0,
        "poss_median_s": q(poss, 50),
        "poss_share_under_4s": float((poss < 4).mean()) if len(poss) else 0.0,
        "home_share_alive": float((codes[alive] == methods.H).mean()) if alive.any() else 0.0,
        "dead_runs": int(len(dead)),
        "dead_run_s_q10_25_50_75_90": [round(q(dead, p), 1) for p in (10, 25, 50, 75, 90)],
        "play_minutes": float(alive.sum() / FPS / 60.0),
        "play_share": float(alive.mean()),
    }


def format_stats(name, s):
    return ("%-34s H %.3f A %.3f D %.3f | poss %4d (%.0f/90) mean %.1f s median %.1f s <4s %.2f | "
            "home %.3f | D runs %d q10-90 %s s | play %.1f min (%.3f)" % (
                name, s["share_H"], s["share_A"], s["share_D"], s["possessions"], s["poss_per_90"],
                s["poss_mean_s"], s["poss_median_s"], s["poss_share_under_4s"], s["home_share_alive"],
                s["dead_runs"], s["dead_run_s_q10_25_50_75_90"], s["play_minutes"], s["play_share"]))


# -- the real match ---------------------------------------------------------------------------------

def real_layout(real, use_anchor=True):
    """Indices of the frames the model labels (graded halves after the kick-off anchor), and the
    graded blocks. Nothing here looks at tracking values."""
    from games import GAME, GAMES, anchor_video_frame, half_video_blocks
    anchors = {1: anchor_video_frame(GAMES[GAME]["kickoff"]), 2: anchor_video_frame(GAMES[GAME]["kickoff_2nd"])}
    blocks = half_video_blocks(GAME)
    idx = []
    for half, first, last in blocks:
        start = first
        k = anchors.get(half)
        if use_anchor and k is not None and first <= k < last:
            start = k + 1          # FAQ: half 2 graded from 110808, kick-off 113742, frames before are D
        idx.append(np.arange(start, last + 1))
    idx = np.concatenate(idx)
    if not (real["period"][idx] > 0).all():
        raise RuntimeError("model frames outside the graded halves")
    return idx, blocks


def real_features(real, idx, cache, key):
    path_x = os.path.join(cache, "real_%s.x.npy" % key)
    path_xs = os.path.join(cache, "real_%s.xs.npy" % key)
    if os.path.isfile(path_x) and os.path.isfile(path_xs):
        return np.load(path_x), np.load(path_xs)
    m = {"ball": real["ball"][idx], "players": real["players"][idx], "period": real["period"][idx]}
    x, xs, _ = features_both(m)
    os.makedirs(cache, exist_ok=True)
    _save(path_x, x)
    _save(path_xs, xs)
    return x, xs


def train_final(mixture, cache=CACHE, label="l1"):
    """The final model: all 7 open matches, coupled copy (+ alt copy if mixture)."""
    blocks = []
    for i, m in enumerate(MATCHES):
        prov, l1, _period = ext_labels(m, cache)
        y = l1 if label == "l1" else prov
        r = ext_rows("coupled", coupled_seed(i), m, cache)
        blocks.append((r["x5"], r["xs5"], y))
        if mixture:
            r = ext_rows("alt", alt_seed(i), m, cache)
            blocks.append((r["x5"], r["xs5"], y))
        log("training rows ready: %s (%d copies)" % (m, 2 if mixture else 1))
    rows = sum(2 * len(b[0]) for b in blocks)
    t0 = time.time()
    booster, prior = fit(blocks)
    log("fitted LightGBM on %d rows x %d features in %.0fs (prior H %.3f A %.3f D %.3f)" % (
        rows, blocks[0][0].shape[1], time.time() - t0, prior[0], prior[1], prior[2]))
    return booster, prior


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fresh", action="store_true",
                    help="delete this pipeline's own cache and re-parse the tracking (no cache read)")
    ap.add_argument("--mixture", action="store_true", help="also train on the label-free alt degradation")
    ap.add_argument("--switch-team", type=float, default=SWITCH_TEAM)
    ap.add_argument("--switch-dead", type=float, default=SWITCH_DEAD)
    ap.add_argument("--no-kickoff-anchor", action="store_true",
                    help="let the model label the pre-kick-off frames of each half too")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--summary-json", default=None,
                    help="optional: also write the run summary (aggregates of our output only) here")
    args = ap.parse_args(argv)
    t_all = time.time()
    feats3._warn_off()

    if args.fresh and os.path.isdir(args.cache):
        shutil.rmtree(args.cache)
        log("fresh run: removed this pipeline's cache")
    os.makedirs(args.cache, exist_ok=True)
    log("pipeline_v2: features v2 (%s), labels L1, PLAYER_SMOOTH %d, mixture %s, costs team %.0f dead %.0f" % (
        "+".join(GROUPS), PLAYER_SMOOTH, args.mixture, args.switch_team, args.switch_dead))

    booster, prior = train_final(args.mixture, args.cache)

    from socbac import atd
    from games import GAME, GAMES
    t0 = time.time()
    real = atd.load(use_cache=not args.fresh)
    n = len(real["period"])
    if n != GAMES[GAME]["n_video_frames"]:
        raise RuntimeError("tracking length %d != video length" % n)
    idx, blocks = real_layout(real, use_anchor=not args.no_kickoff_anchor)
    key = "%s_%s_%d_%d" % (FEAT_REV, "anchor" if not args.no_kickoff_anchor else "noanchor", len(idx),
                           int(real["players"].shape[2]))
    x, xs = real_features(real, idx, args.cache, key)
    log("real tracking: %d frames, %d labelled by the model, %d features, players array %s, %.0fs" % (
        n, len(idx), x.shape[1], tuple(real["players"].shape), time.time() - t0))

    e = emissions(booster, prior, x, xs)
    period = real["period"][idx]
    codes = decode(e, period, args.switch_team, args.switch_dead)
    del x, xs

    letters = np.full(n, "X", dtype="<U1")
    graded = np.zeros(n, bool)
    for _half, first, last in blocks:
        letters[first:last + 1] = "D"          # graded frames before the kick-off: ball not in play
        graded[first:last + 1] = True
    letters[idx] = np.array(["H", "A", "D"])[codes]

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))

    gcodes = np.where(letters[graded] == "H", methods.H, np.where(letters[graded] == "A", methods.A, methods.D))
    gperiod = real["period"][graded]
    stats = sanity_stats(gcodes, gperiod)
    counts = {k: int((letters == k).sum()) for k in "HADX"}
    log("wrote %s: %d rows, label counts %s" % (out, n, counts))
    print(format_stats("BAR-ATH graded (our prediction)", stats), flush=True)
    sha = file_sha256(out)
    print("sha256 %s" % sha, flush=True)
    summary = {"out": out, "sha256": sha, "mixture": args.mixture, "switch_team": args.switch_team,
               "switch_dead": args.switch_dead, "label_counts": counts, "stats": stats,
               "seconds": round(time.time() - t_all, 1), "fresh": args.fresh}
    for path in [os.path.join(args.cache, "last_run.json")] + ([args.summary_json] if args.summary_json else []):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out],
                   check=True)
    log("total %.0fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()
