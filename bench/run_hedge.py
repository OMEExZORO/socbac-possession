"""Label-definition hedge: train on one (or pooled) label definition, score on all four.

    python -m bench.run_hedge --selftest                 # tiny synthetic tests, no data, no models
    python -m bench.run_hedge --check --deg v4 --seeds 0 # reports missing caches / label stats only
    python -m bench.run_hedge --summary                  # print matrices from the results file
    python -m bench.run_hedge --deg v4 --config v3 --seeds 0 1 2 --boosts 3 20 40 --build

Rows of the matrix are TRAIN definitions L0 L1 L2 L3 and L12 (pooled L1+L2 with sample weights);
columns are TEST definitions L0 L1 L2 L3. All scored with the official scorer (bench.run.score) on the
three TEST matches of run_v3cand, per seed. Features come from the v2/v3 cache of bench.run_v3cand
(feats3 columns, cached per degrader, seed and match); `--build` allows building missing caches
(slow, RAM heavy). Degraders v5, v5ball, v5player, v5stress are registered from bench.degrade_v5.

Pooled target L12: where L1 and L2 agree the frame is one row (weight 1, or the mean boundary weight);
where they differ it is two rows with weight 0.5 each, one per definition. No memory doubling.
Boundary weight b: frames within +-RADIUS (50 = 2 s) of a label change of the row's own definition get
weight b, others 1.

Costs: fixed (--costs team dead, default 10 40) or leave-one-match-out on TRAIN scored against the
train definition (--lomo; pooled: mean of L1 and L2), grids from run_v3cand. Nothing here selects
anything on the TEST matches. One process, BelowNormal priority, num_threads 1.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

RUN_DIR = r"D:\Barca_Hackathon\research\runs\degrade5"
LAB_CACHE = os.path.join(RUN_DIR, "labels")
ROW_CACHE = os.path.join(RUN_DIR, "rows")
RES = os.path.join(RUN_DIR, "hedge_results.jsonl")
LOG = os.path.join(RUN_DIR, "hedge.log")
TRAIN_DEFS = ("L0", "L1", "L2", "L3", "L12")
TEST_DEFS = ("L0", "L1", "L2", "L3")
RADIUS = 50
STRIDE, ROUNDS = 5, 300


def log(msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def set_low_priority():
    try:
        import ctypes
        BELOW_NORMAL = 0x00004000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    except Exception:
        pass


# ---- targets: labels, boundary weights, pooling ---------------------------------------------

def boundary_weight(lab, period, boost, radius=RADIUS):
    """boost within +-radius frames of a label change (inside a period), else 1. float32."""
    lab = np.asarray(lab)
    period = np.asarray(period)
    w = np.ones(len(lab), np.float32)
    if boost == 1:
        return w
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        y = lab[idx]
        ch = np.zeros(len(idx), np.int32)
        ch[1:] = (y[1:] != y[:-1])
        pos = np.flatnonzero(ch)
        if len(pos) == 0:
            continue
        near = np.zeros(len(idx), bool)
        for t in pos:
            near[max(0, t - radius):t + radius + 1] = True
        w[idx[near]] = boost
    return w


def targets(train_def, labs, rows, period, boost):
    """(row index into `rows`, labels, weights) for one match. labs: dict def -> int8 array."""
    per = np.asarray(period)[rows]
    if train_def != "L12":
        y = labs[train_def][rows]
        w = boundary_weight(labs[train_def], period, boost)[rows]
        return np.arange(len(rows)), y.astype(np.int8), w
    y1, y2 = labs["L1"][rows], labs["L2"][rows]
    w1 = boundary_weight(labs["L1"], period, boost)[rows]
    w2 = boundary_weight(labs["L2"], period, boost)[rows]
    same = y1 == y2
    diff = np.flatnonzero(~same)
    idx = np.concatenate([np.arange(len(rows)), diff])
    y = np.concatenate([y1, y2[diff]]).astype(np.int8)
    w = np.concatenate([np.where(same, 0.5 * (w1 + w2), 0.5 * w1), 0.5 * w2[diff]]).astype(np.float32)
    del per
    return idx, y, w


# ---- cached feature rows ---------------------------------------------------------------------

def rows_for(deg, seed, config, spec, pack, meta, match, stride):
    """(x_rows, xs_rows) of one TRAIN match, cached as npy under ROW_CACHE (float32)."""
    d = os.path.join(ROW_CACHE, "%s_%s_st%d_s%d" % (deg, config, stride, seed))
    os.makedirs(d, exist_ok=True)
    px, pxs = os.path.join(d, match + ".x.npy"), os.path.join(d, match + ".xs.npy")
    if not (os.path.isfile(px) and os.path.isfile(pxs)):
        dd = pack[match]
        rows = np.arange(0, len(dd["period"]), stride)
        out = []
        for key in ("x", "xs"):
            if spec.kind == "v3" or spec.robust:
                full = spec.view(dd[key], meta, dd["period"])
                out.append(full[rows])
                del full
            else:
                out.append(spec.view(dd[key], meta, dd["period"], rows))
        np.save(px, out[0])
        np.save(pxs, out[1])
    return np.load(px, mmap_mode="r"), np.load(pxs, mmap_mode="r")


def collect(deg, seed, config, spec, pack, meta, matches, labs, train_def, boost, stride=STRIDE):
    from bench.methods import SWAP
    xs, ys, ws = [], [], []
    for m in matches:
        n = len(pack[m]["period"])
        rows = np.arange(0, n, stride)
        idx, y, w = targets(train_def, labs[m], rows, pack[m]["period"], boost)
        x, xsw = rows_for(deg, seed, config, spec, pack, meta, m, stride)
        xs.append(np.asarray(x[idx]))
        ys.append(y)
        ws.append(w)
        xs.append(np.asarray(xsw[idx]))
        ys.append(SWAP[y])
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def fit(x, y, w, rounds=ROUNDS):
    import lightgbm as lgb
    from bench.methods import LGBMViterbi
    model = lgb.train(dict(LGBMViterbi.PARAMS), lgb.Dataset(x, y, weight=w), num_boost_round=rounds)
    prior = np.bincount(y, weights=w, minlength=3).astype(np.float64)
    return model, prior / prior.sum()


def score_vs(labs_m, pred, period, defs):
    from bench.run import score
    return {k: float(score(labs_m[k], pred, period)[0]) for k in defs}


def tune_costs_lomo(deg, seed, config, spec, pack, meta, labs, train_def, boost, rv):
    """LOMO on TRAIN: pick (team, dead) costs maximising mean spellF1 against the train definition."""
    from bench.feats3 import decode_periods_fast
    score_defs = ("L1", "L2") if train_def == "L12" else (train_def,)
    oof = []
    for hold in rv.TRAIN:
        rest = [m for m in rv.TRAIN if m != hold]
        x, y, w = collect(deg, seed, config, spec, pack, meta, rest, labs, train_def, boost)
        model, prior = fit(x, y, w)
        del x, y, w
        oof.append(rv.emissions(model, prior, pack[hold], meta, spec))
    best = None
    for st in rv.GRID_TEAM:
        for sd in rv.GRID_DEAD:
            s = []
            for e, m in zip(oof, rv.TRAIN):
                pred = decode_periods_fast(e, pack[m]["period"], st, sd)
                s.append(np.mean(list(score_vs(labs[m], pred, pack[m]["period"], score_defs).values())))
            if best is None or np.mean(s) > best[0]:
                best = (float(np.mean(s)), st, sd)
    return best


def done(key):
    if not os.path.isfile(RES):
        return False
    for line in open(RES, encoding="utf-8"):
        r = json.loads(line)
        if (r["deg"], r["seed"], r["config"], r["train_def"], r["boost"]) == key:
            return True
    return False


def run_cell(deg, seed, config, train_def, boost, costs, lomo, rv, ctx):
    from bench.feats3 import decode_periods_fast
    key = (deg, seed, config, train_def, boost)
    if done(key):
        log("skip %s" % (key,))
        return
    spec, pack, meta, labs = ctx
    t0 = time.time()
    if lomo:
        best = tune_costs_lomo(deg, seed, config, spec, pack, meta, labs, train_def, boost, rv)
        st, sd, oof = best[1], best[2], best[0]
    else:
        st, sd = costs
        oof = None
    x, y, w = collect(deg, seed, config, spec, pack, meta, rv.TRAIN, labs, train_def, boost)
    model, prior = fit(x, y, w)
    del x, y, w
    test = {k: {} for k in TEST_DEFS}
    for m in rv.TEST:
        e = rv.emissions(model, prior, pack[m], meta, spec)
        pred = decode_periods_fast(e, pack[m]["period"], st, sd)
        for k, v in score_vs(labs[m], pred, pack[m]["period"], TEST_DEFS).items():
            test[k][m] = v
    rec = {"deg": deg, "seed": seed, "config": config, "train_def": train_def, "boost": boost,
           "costs": [st, sd], "oof": oof, "test": test, "seconds": round(time.time() - t0)}
    with open(RES, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    log("%s: %s" % (key, {k: round(float(np.mean(list(v.values()))), 3) for k, v in test.items()}))


# ---- reporting -------------------------------------------------------------------------------

def matrices(records):
    """{(deg, config, boost): 5x4 array of mean spellF1 over seeds and TEST matches} plus seed counts."""
    acc = {}
    for r in records:
        k = (r["deg"], r["config"], r["boost"])
        row = TRAIN_DEFS.index(r["train_def"])
        cells = acc.setdefault(k, {})
        for j, t in enumerate(TEST_DEFS):
            cells.setdefault((row, j), []).append(float(np.mean(list(r["test"][t].values()))))
    out = {}
    for k, cells in acc.items():
        mat = np.full((len(TRAIN_DEFS), len(TEST_DEFS)), np.nan)
        cnt = np.zeros(mat.shape, int)
        for (i, j), v in cells.items():
            mat[i, j] = float(np.mean(v))
            cnt[i, j] = len(v)
        out[k] = (mat, cnt)
    return out


def regret(mat):
    """Per train row: max over test definitions of (best row value in that column - this row's value)."""
    best = np.nanmax(mat, axis=0)
    return np.nanmax(best[None, :] - mat, axis=1)


def print_summary(records):
    for (deg, config, boost), (mat, cnt) in sorted(matrices(records).items()):
        print("\n%s %s boundary x%s  (rows train, cols test; seeds per cell %s)" % (
            deg, config, boost, sorted(set(cnt.ravel().tolist()))))
        print("%-6s" % "" + "".join("%8s" % t for t in TEST_DEFS) + "   max regret")
        rg = regret(mat)
        for i, name in enumerate(TRAIN_DEFS):
            print("%-6s" % name + "".join("%8.3f" % v for v in mat[i]) + "   %.3f" % rg[i])


def load_records():
    if not os.path.isfile(RES):
        return []
    return [json.loads(line) for line in open(RES, encoding="utf-8")]


# ---- self test -------------------------------------------------------------------------------

def _expect(name, cond, detail=""):
    if not cond:
        raise AssertionError("%s (%s)" % (name, detail))
    print("PASS", name)


def selftest():
    n = 3000
    period = np.ones(n, np.int8)
    period[n // 2:] = 2
    rng = np.random.default_rng(0)
    base = (np.cumsum(rng.random(n) < 0.01) % 2).astype(np.int8)
    l1 = base.copy()
    l2 = base.copy()
    l2[500:560] = 1 - base[500:560]            # L2 differs from L1 on a block
    labs = {"L0": base, "L1": l1, "L2": l2, "L3": base}
    rows = np.arange(0, n, 5)
    w = boundary_weight(base, period, 20.0)
    ch = np.flatnonzero(base[1:] != base[:-1]) + 1
    far = np.array([np.abs(ch - t).min() for t in range(n)])
    _expect("boundary weight x20 exactly within +-radius of a change (same period)",
            np.all(w[(far <= RADIUS - 1) & (np.abs(np.arange(n) - n // 2) > RADIUS)] == 20.0)
            and np.all(w[far > RADIUS + 1] == 1.0))
    _expect("boost 1 is all ones", np.all(boundary_weight(base, period, 1) == 1.0))
    idx, y, ww = targets("L1", labs, rows, period, 3.0)
    _expect("single target shapes", len(idx) == len(rows) == len(y) == len(ww))
    idx, y, ww = targets("L12", labs, rows, period, 3.0)
    ndiff = int((l1[rows] != l2[rows]).sum())
    _expect("pooled adds one row per disagreement", len(idx) == len(rows) + ndiff and ndiff > 0, str(ndiff))
    # weight of each frame summed over its rows equals the (mean) single-label weight mass
    tot = np.zeros(len(rows))
    np.add.at(tot, idx, ww)
    w1 = boundary_weight(l1, period, 3.0)[rows]
    w2 = boundary_weight(l2, period, 3.0)[rows]
    _expect("pooled weights sum to the mean of the two definitions", np.allclose(tot, 0.5 * (w1 + w2)))
    _expect("pooled rows carry the right labels",
            np.array_equal(y[:len(rows)], l1[rows]) and np.array_equal(y[len(rows):], l2[rows][l1[rows] != l2[rows]]))
    mat = np.array([[.5, .45, .4, .4], [.45, .6, .55, .5], [.4, .55, .6, .5], [.4, .5, .5, .55], [.44, .58, .58, .5]])
    rg = regret(mat)
    _expect("regret of the best-in-column row is zero for that column", rg.shape == (5,) and rg.min() >= 0)
    recs = [{"deg": "v4", "config": "v3", "boost": 3.0, "seed": s, "train_def": td,
             "test": {t: {"a": float(mat[i, j]), "b": float(mat[i, j])} for j, t in enumerate(TEST_DEFS)}}
            for s in range(3) for i, td in enumerate(TRAIN_DEFS)]
    (k, (m2, c2)), = matrices(recs).items()
    _expect("matrix aggregation", np.allclose(m2, mat) and (c2 == 3).all())
    print("run_hedge selftest ok")


# ---- main ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--check", action="store_true", help="report missing caches and label stats, no models")
    ap.add_argument("--build", action="store_true", help="allow building missing feature caches")
    ap.add_argument("--deg", default="v4")
    ap.add_argument("--config", default="v3", help="run_v3cand.CONFIGS key: v1 v2 v3 ...")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--boosts", type=float, nargs="+", default=[3.0, 20.0, 40.0])
    ap.add_argument("--train-defs", nargs="+", default=list(TRAIN_DEFS), choices=TRAIN_DEFS)
    ap.add_argument("--costs", type=float, nargs=2, default=[10.0, 40.0])
    ap.add_argument("--lomo", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.summary:
        print_summary(load_records())
        return
    set_low_priority()
    from bench import degrade_v5
    degrade_v5.register()
    from bench import run_v3cand as rv
    from bench import labels5
    labs = {m: labels5.cached_labels(m, LAB_CACHE) for m in rv.ORDER}
    if a.check:
        from bench import labels_l1
        for m in rv.ORDER:
            per = np.load(os.path.join(rv.cdir(a.deg, a.seeds[0]), m + ".period.npy")) \
                if os.path.isfile(os.path.join(rv.cdir(a.deg, a.seeds[0]), m + ".period.npy")) else None
            st = {k: labels_l1.label_stats(v, per if per is not None else np.ones(len(v), np.int8))["n_possessions"]
                  for k, v in labs[m].items()}
            print(m, "possessions per definition", st, "feature cache", "yes" if per is not None else "MISSING")
        return
    spec = rv.CONFIGS[a.config]()
    for seed in a.seeds:
        if not a.build:
            miss = [m for m in rv.ORDER
                    if not os.path.isfile(os.path.join(rv.cdir(a.deg, seed), m + ".meta.json"))]
            if miss:
                raise SystemExit("missing feature cache for %s seed %d: %s (use --build)" % (a.deg, seed, miss))
        meta = rv.ensure(a.deg, seed)
        pack = rv.load_pack(a.deg, seed)
        for m in rv.ORDER:
            assert len(pack[m]["period"]) == len(labs[m]["L0"]), "label/cache length mismatch " + m
        ctx = (spec, pack, meta, labs)
        for boost in a.boosts:
            for td in a.train_defs:
                run_cell(a.deg, seed, a.config, td, boost, tuple(a.costs), a.lomo, rv, ctx)
    print_summary([r for r in load_records() if r["deg"] == a.deg and r["config"] == a.config])


if __name__ == "__main__":
    main()
