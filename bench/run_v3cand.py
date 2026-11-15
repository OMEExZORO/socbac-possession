"""v3 candidate evaluation on decoupled (v4) and old (v2, PLAYER_SMOOTH 12) degraders.

    python -m bench.run_v3cand --deg v4 --seed 0 --configs baseline v1 v2 v3
    python -m bench.run_v3cand --deg v4 --seed 0 --configs v2_k3 v3_h25 ...

Protocol: TRAIN = 4 matches, TEST = 3; costs by leave-one-match-out on TRAIN; TEST scored once.
v1 = feats3 v1 columns; v2 = v1 + context + lang_dead (groups=('context','lang_dead'));
v3 = v2 + look-ahead channels (bench.rivalideas.lookahead_from_context).
Features cached per (degrader, seed, match) as float32 under RUN_DIR/cache.
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

RUN_DIR = r"D:\Barca_Hackathon\research\runs\v3cand"
CACHE = os.path.join(RUN_DIR, "cache")
RES = os.path.join(RUN_DIR, "results.jsonl")
LOG = os.path.join(RUN_DIR, "run.log")
TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
TEST = ["J03WPY", "J03WQQ", "J03WR9"]
ORDER = TRAIN + TEST
GRID_TEAM = [5.0, 10.0, 20.0, 40.0, 80.0]
GRID_DEAD = [10.0, 20.0, 40.0, 80.0, 160.0]
GROUPS = ("context", "lang_dead")
STRIDE, ROUNDS = 5, 300


def log(msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def cdir(deg, seed, tag=""):
    return os.path.join(CACHE, deg + tag, "s%d" % seed)


def build_one(deg_name, seed, index, match, tag="", mask_extra=None):
    """Degrade one match and save both team-view feature matrices (v1+context+lang_dead)."""
    from bench.dfl import load
    from bench.degrade_v4 import DEGRADERS
    from bench.methods import swap_teams
    from bench.run import labels
    from bench.feats3 import build_features

    out = cdir(deg_name, seed, tag)
    stem = os.path.join(out, match)
    if os.path.isfile(stem + ".meta.json"):
        return
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    clean = load(match)
    deg = DEGRADERS[deg_name](clean, 1000 * seed + index)
    if mask_extra is not None:
        mask_extra(deg, 7000 + 1000 * seed + index)
    x, names, ball, sub, kind = build_features(deg, GROUPS)
    xs, names2, _, _, _ = build_features(swap_teams(deg), GROUPS)
    assert names == names2 and np.isfinite(x).all() and np.isfinite(xs).all()
    np.save(stem + ".x.npy", x.astype(np.float32))
    np.save(stem + ".xs.npy", xs.astype(np.float32))
    np.save(stem + ".y.npy", labels(clean))
    np.save(stem + ".period.npy", deg["period"])
    np.save(stem + ".vis.npy", ~np.isnan(deg["ball"][:, 0]))
    np.save(stem + ".alive.npy", clean["alive"])
    # the baseline needs only ball and players
    np.save(stem + ".base.npy", _baseline_pred(deg))
    meta = {"names": names, "sub": list(sub), "frames": int(len(x)), "hidden": float(deg["ball_hidden_share"]),
            "seconds": round(time.time() - t0, 1)}
    with open(stem + ".meta.json", "w") as f:
        json.dump(meta, f)
    log("built %s %s seed %d (%d cols, %.0fs)" % (deg_name + tag, match, seed, x.shape[1], meta["seconds"]))


def _baseline_pred(deg):
    from bench.methods import baseline_smooth50
    return baseline_smooth50(deg)


def ensure(deg_name, seed, tag="", mask_extra=None):
    for i, m in enumerate(ORDER):
        build_one(deg_name, seed, i, m, tag, mask_extra)
    with open(os.path.join(cdir(deg_name, seed, tag), TRAIN[0] + ".meta.json")) as f:
        return json.load(f)


# ---- feature views ---------------------------------------------------------------------------

def load_pack(deg, seed, tag=""):
    d = cdir(deg, seed, tag)
    pack = {}
    for m in ORDER:
        s = os.path.join(d, m)
        pack[m] = {"x": np.load(s + ".x.npy", mmap_mode="r"), "xs": np.load(s + ".xs.npy", mmap_mode="r"),
                   "y": np.load(s + ".y.npy"), "period": np.load(s + ".period.npy"),
                   "base": np.load(s + ".base.npy"), "vis": np.load(s + ".vis.npy")}
    return pack


class Spec:
    """Which columns / extra channels / transforms a config uses."""

    def __init__(self, kind, shifts=(25, 50, 125), robust=False):
        self.kind, self.shifts, self.robust = kind, tuple(shifts), robust

    def cols(self, meta):
        sub = meta["sub"]
        if self.kind == "v1":
            return np.array([i for i, s in enumerate(sub) if s == "v1"], np.int32)
        return np.arange(len(sub), dtype=np.int32)

    def view(self, mm, meta, period, rows=None):
        """Feature matrix (rows, ncols) for one team-view."""
        from bench.rivalideas import lookahead_from_context
        cols = self.cols(meta)
        la = None
        if self.kind == "v3":
            la, _ = lookahead_from_context(mm, meta["names"], period, shifts=self.shifts)
        if rows is None:
            x = np.asarray(mm[:, cols], np.float32)
            if la is not None:
                x = np.concatenate([x, la], axis=1)
        else:
            x = np.asarray(mm[rows][:, cols], np.float32)
            if la is not None:
                x = np.concatenate([x, la[rows]], axis=1)
        if self.robust:
            x = robust_std(x, mm, cols, la)
        return x


def robust_std(x_rows, mm, cols, la):
    """Per-match robust standardisation (median / IQR over the whole match of each column)."""
    full = np.asarray(mm[::10][:, cols], np.float32)
    if la is not None:
        full = np.concatenate([full, la[::10]], axis=1)
    med = np.median(full, axis=0)
    iqr = np.percentile(full, 75, axis=0) - np.percentile(full, 25, axis=0)
    iqr = np.where(iqr < 1e-6, 1.0, iqr)
    return ((x_rows - med) / iqr).astype(np.float32)


# ---- model -------------------------------------------------------------------------------------

def collect(packs_metas, spec, matches_by_pack, stride=STRIDE):
    from bench.methods import SWAP
    xs, ys = [], []
    for (pack, meta), matches in zip(packs_metas, matches_by_pack):
        for m in matches:
            d = pack[m]
            rows = np.arange(0, len(d["y"]), stride)
            for key, lab in (("x", d["y"]), ("xs", SWAP[d["y"]])):
                if spec.kind == "v3" or spec.robust:
                    full = spec.view(d[key], meta, d["period"])
                    xs.append(full[rows])
                    del full
                else:
                    xs.append(spec.view(d[key], meta, d["period"], rows))
                ys.append(lab[rows])
    return np.concatenate(xs), np.concatenate(ys)


def fit(x, y, rounds=ROUNDS):
    import lightgbm as lgb
    from bench.methods import LGBMViterbi
    prm = dict(LGBMViterbi.PARAMS)
    model = lgb.train(prm, lgb.Dataset(x, y), num_boost_round=rounds)
    prior = np.bincount(y, minlength=3).astype(np.float64)
    return model, prior / prior.sum()


def emissions(model, prior, d, meta, spec):
    from bench.methods import SWAP
    p = model.predict(spec.view(d["x"], meta, d["period"]))
    ps = model.predict(spec.view(d["xs"], meta, d["period"]))
    p = 0.5 * (p + ps[:, SWAP])
    return np.log(p + 1e-6) - np.log(prior)


def tune_costs(oof, pack):
    from bench.feats3 import decode_periods_fast
    from bench.run import score
    best = None
    for st in GRID_TEAM:
        for sd in GRID_DEAD:
            s = [score(pack[m]["y"], decode_periods_fast(e, pack[m]["period"], st, sd), pack[m]["period"])[0]
                 for e, m in zip(oof, TRAIN)]
            if best is None or np.mean(s) > best[0]:
                best = (float(np.mean(s)), st, sd)
    return best


def run_config(deg_name, seed, name, spec, train_sources=None, extra_train=None, test_deg=None):
    """train_sources: list of (deg, tag) whose TRAIN matches are used for training (default: deg_name).
    extra_train: list of (deg, tag) extra training copies (augmentation). test on test_deg or deg_name."""
    from bench.feats3 import decode_periods_fast
    from bench.run import score
    from bench.methods import H, D
    t0 = time.time()
    meta = ensure(deg_name, seed)
    pack = load_pack(deg_name, seed)
    sources = [(pack, meta)]
    for (dn, tg) in (train_sources or [])[1:] if train_sources else []:
        mt = ensure(dn, seed, tg)
        sources.append((load_pack(dn, seed, tg), mt))
    for (dn, tg) in (extra_train or []):
        mt = ensure(dn, seed, tg, AUG.get(tg))
        sources.append((load_pack(dn, seed, tg), mt))
    test_pack, test_meta = pack, meta
    if test_deg:
        test_meta = ensure(test_deg, seed)
        test_pack = load_pack(test_deg, seed)
    for p_, _ in sources:
        pass
    oof = []
    for k, hold in enumerate(TRAIN):
        rest = [m for m in TRAIN if m != hold]
        x, y = collect(sources, spec, [rest] * len(sources))
        model, prior = fit(x, y)
        del x, y
        oof.append(emissions(model, prior, pack[hold], meta, spec) if not test_deg else
                   emissions(model, prior, test_pack[hold], test_meta, spec))
        log("  %s fold %d done (%.0fs)" % (name, k, time.time() - t0))
    best = tune_costs(oof, test_pack)
    x, y = collect(sources, spec, [TRAIN] * len(sources))
    model, prior = fit(x, y)
    del x, y
    rows = []
    for m in TEST:
        e = emissions(model, prior, test_pack[m], test_meta, spec)
        pred = decode_periods_fast(e, test_pack[m]["period"], best[1], best[2])
        rows.append(finish_row(test_pack[m], pred, m))
    rec = {"deg": deg_name + ("->" + test_deg if test_deg else ""), "seed": seed, "config": name,
           "costs": [best[1], best[2]], "oof": best[0], "rows": rows, "seconds": round(time.time() - t0)}
    save(rec)
    log("%s %s seed %d: spellF1 %.3f (%.0fs)" % (rec["deg"], name, seed, np.mean([r["spell"] for r in rows]),
                                              rec["seconds"]))
    return rec


def finish_row(d, pred, m):
    from bench.run import score
    from bench.methods import H, D
    y = d["y"]
    sp, mf, npred, ngt = score(y, pred, d["period"])
    alive_t, alive_p = y != D, pred != D
    err = abs(np.mean(pred[alive_p] == H) - np.mean(y[alive_t] == H)) if alive_p.any() else 1.0
    return {"match": m, "spell": float(sp), "macro": float(mf), "npred": int(npred), "ngt": int(ngt),
            "share_err": float(err)}


def save(rec):
    with open(RES, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def done(deg, seed, name):
    if not os.path.isfile(RES):
        return False
    for line in open(RES, encoding="utf-8"):
        r = json.loads(line)
        if r["deg"] == deg and r["seed"] == seed and r["config"] == name:
            return True
    return False


# ---- augmentation (K2) -------------------------------------------------------------------------

def ball_mask_aug(deg, seed):
    """Extra random ball-hiding spans (5-30 s) until up to a random 0-60% of frames are hidden."""
    rng = np.random.default_rng(seed)
    n = len(deg["ball"])
    target = rng.uniform(0.0, 0.6)
    hid = np.isnan(deg["ball"][:, 0])
    guard = 0
    while hid.mean() < 1 - (1 - target) * (1 - 0.407) and guard < 200:
        a = int(rng.integers(0, n))
        b = min(n, a + int(rng.integers(125, 751)))
        hid[a:b] = True
        guard += 1
    deg["ball"][hid] = np.nan
    deg["ball_hidden_share"] = float(hid.mean())


AUG = {"_k2": ball_mask_aug}

CONFIGS = {
    "v1": lambda: Spec("v1"),
    "v2": lambda: Spec("v2"),
    "v3": lambda: Spec("v3"),
    "v2_k3": lambda: Spec("v2", robust=True),
    "v3_k3": lambda: Spec("v3", robust=True),
    "v3_h25": lambda: Spec("v3", shifts=(25,)),
    "v3_h25_50": lambda: Spec("v3", shifts=(25, 50)),
    "v3_h250": lambda: Spec("v3", shifts=(25, 50, 125, 250)),
    "v2_k2": lambda: Spec("v2"),
    "v3_k2": lambda: Spec("v3"),
}


def run_baseline(deg_name, seed):
    meta = ensure(deg_name, seed)
    pack = load_pack(deg_name, seed)
    rows = [finish_row(pack[m], pack[m]["base"], m) for m in TEST]
    save({"deg": deg_name, "seed": seed, "config": "baseline", "costs": None, "oof": None, "rows": rows,
          "seconds": 0})
    log("%s baseline seed %d: spellF1 %.3f" % (deg_name, seed, np.mean([r["spell"] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", default="v4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=["baseline", "v1", "v2", "v3"])
    ap.add_argument("--mix", action="store_true", help="v2 trained on v4+v2s mixture (name v2_mix)")
    ap.add_argument("--test-deg", default=None)
    args = ap.parse_args()
    for c in args.configs:
        if c == "baseline":
            if not done(args.deg, args.seed, "baseline"):
                run_baseline(args.deg, args.seed)
            continue
        tag_deg = args.deg + ("->" + args.test_deg if args.test_deg else "")
        if done(tag_deg, args.seed, c):
            log("skip %s %s %d" % (tag_deg, c, args.seed))
            continue
        if c.endswith("_k2"):
            run_config(args.deg, args.seed, c, CONFIGS[c](), extra_train=[(args.deg, "_k2")])
        elif c.endswith("_mix"):
            base = c[:-4]
            other = "v2s" if args.deg == "v4" else "v4"
            run_config(args.deg, args.seed, c, CONFIGS[base](), train_sources=[(args.deg, ""), (other, "")])
        else:
            run_config(args.deg, args.seed, c, CONFIGS[c](), test_deg=args.test_deg)


if __name__ == "__main__":
    main()
