"""External bench for the v2 insurance pipeline (socbac.pipeline_v2). Open IDSSE data only.

    python -m bench.run_pipe_bench --seeds 0 1 2 --lomo v2|l1|coupled v2|l1|mix
    python -m bench.run_pipe_bench --seeds 0 1 2 --summary
    python -m bench.run_pipe_bench --seeds 0 1 2 --rescore TEAM DEAD       # reporting only

TRAIN = J03WMX J03WN1 J03WOH J03WOY (fit, and LOMO for costs), TEST = J03WPY J03WQQ J03WR9 (scored once).
Per seed, every model is trained on the TRAIN matches and scored on the TEST matches with the official
scorer (bench.run.score -> evaluation/possession_metrics.evaluate, unmodified):

  models   v1 = methods.dense_features columns (49), v2 = v1 + context + lang_dead (351)
  labels   prov = provider flags (bench.run.labels), l1 = bench.labels_l1 (change needs >= 1 s)
  train    coupled = bench.degrade with PLAYER_SMOOTH 12; mix = coupled + label-free bench.degrade_alt
  test     coupled and alt degradations of the TEST matches, scored against prov and l1 labels

Degradation seeds follow bench.run (coupled 1000*seed + i, alt 1000*seed + 500 + i, i over TRAIN+TEST),
so seed 0 shares its cached rows with the pipeline. LightGBM is exactly LGBMViterbi's (one thread,
deterministic); independent fits run in a 2-thread pool of this one process while the main thread
builds features, predicts (row-parallel prediction gives the same numbers) and scores.
Costs: fixed team 5 / dead 20 for every config on the board (same protocol for all). --lomo picks
costs per config by leave-one-match-out on TRAIN (OOF on the held-out match's coupled and alt copies).
Prints and writes aggregates only (scores, counts).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from bench import methods  # noqa: E402
from bench.run import score  # noqa: E402
from socbac import pipeline_v2 as pv  # noqa: E402

TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
TEST = ["J03WPY", "J03WQQ", "J03WR9"]
ORDER = TRAIN + TEST
RUN_DIR = r"D:\Barca_Hackathon\research\runs\pipeline\bench"
EM_DIR = os.path.join(pv.CACHE, "em")
FIXED = (5.0, 20.0)
GRID_TEAM = [5.0, 10.0, 20.0, 40.0, 80.0]
GRID_DEAD = [10.0, 20.0, 40.0, 80.0, 160.0]
KINDS = ("coupled", "alt")
LABELS = ("prov", "l1")
PREDICT_THREADS = 3

# name -> (n_cols or None, label, training copies)
MODELS = {
    "v1|prov|coupled": (pv.N_V1, "prov", ("coupled",)),
    "v1|l1|coupled": (pv.N_V1, "l1", ("coupled",)),
    "v1|l1|mix": (pv.N_V1, "l1", ("coupled", "alt")),
    "v2|prov|coupled": (None, "prov", ("coupled",)),
    "v2|l1|coupled": (None, "l1", ("coupled",)),
    "v2|l1|mix": (None, "l1", ("coupled", "alt")),
}
FIT_ORDER = ["v2|l1|mix", "v2|l1|coupled", "v2|prov|coupled", "v1|l1|mix", "v1|l1|coupled", "v1|prov|coupled"]


def log(msg):
    pv.log(msg)


def seed_of(kind, i, seed):
    return pv.coupled_seed(i, seed) if kind == "coupled" else pv.alt_seed(i, seed)


def labels_of(match):
    prov, l1, period = pv.ext_labels(match)
    return {"prov": prov, "l1": l1}, period


def train_rows(matches, seed, kinds):
    rows = {}
    for m in matches:
        i = ORDER.index(m)
        for kind in kinds:
            t0 = time.time()
            rows[(m, kind)] = pv.ext_rows(kind, seed_of(kind, i, seed), m, mmap=True)
            log("rows %s %s seed %d ready (%.0fs)" % (m, kind, seed, time.time() - t0))
    return rows


def fit_model(name, rows, matches, tag):
    n_cols, label, copies = MODELS[name]
    blocks = []
    for m in matches:
        y = labels_of(m)[0][label]
        for kind in copies:
            r = rows[(m, kind)]
            blocks.append((r["x5"], r["xs5"], y))
    t0 = time.time()
    booster, prior = pv.fit(blocks, n_cols)
    log("fit %s [%s] on %s in %.0fs" % (name, tag, "+".join(matches), time.time() - t0))
    return booster, prior


def em_path(tag, model, match, kind):
    return os.path.join(EM_DIR, "%s_%s_%s_%s.npy" % (tag, model.replace("|", "-"), match, kind))


def predict_copy(models, match, kind, seed, tag):
    """Full-length features of one degraded copy (in memory only), emissions of every model saved."""
    todo = {n: mb for n, mb in models.items() if not os.path.isfile(em_path(tag, n, match, kind))}
    if not todo:
        return
    t0 = time.time()
    i = ORDER.index(match)
    # only seed 0's stride-5 rows are reused (by the pipeline); other seeds keep no rows of TEST copies
    r = pv.ext_rows(kind, seed_of(kind, i, seed), match, full=True, cache_rows=(seed == 0))
    t1 = time.time()
    os.makedirs(EM_DIR, exist_ok=True)
    for name, (booster, prior) in todo.items():
        e = pv.emissions(booster, prior, r["x"], r["xs"], MODELS[name][0], threads=PREDICT_THREADS)
        pv._save(em_path(tag, name, match, kind), e)
    log("predicted %d models on %s %s seed %d [%s] (features %.0fs, predict %.0fs)" % (
        len(todo), match, kind, seed, tag, t1 - t0, time.time() - t1))


def share_err(y, pred):
    at, ap = y != methods.D, pred != methods.D
    if not at.any() or not ap.any():
        return 1.0
    return float(abs(np.mean(pred[ap] == methods.H) - np.mean(y[at] == methods.H)))


def score_rows(tag, model_names, matches, kinds, costs):
    """Score stored emissions. Returns list of dicts (one per model x match x kind x label)."""
    rows = []
    for name in model_names:
        for m in matches:
            labs, period = labels_of(m)
            for kind in kinds:
                path = em_path(tag, name, m, kind)
                if not os.path.isfile(path):
                    continue
                e = np.load(path)
                pred = pv.decode(e, period, costs[0], costs[1])
                stats = pv.sanity_stats(pred, period)
                for lab in LABELS:
                    s, mf, npred, ngt = score(labs[lab], pred, period)
                    rows.append({"model": name, "match": m, "test": kind, "label": lab,
                                 "spell": float(s), "macro": float(mf), "n_pred": int(npred), "n_gt": int(ngt),
                                 "share_err": share_err(labs[lab], pred), "costs": list(costs),
                                 "pred_stats": stats})
    return rows


# -- one seed of the TEST board, in two halves so fits overlap with feature work -----------------------

def start_seed(seed, ex):
    tag = "s%d" % seed
    need = [n for n in FIT_ORDER if not all(os.path.isfile(em_path(tag, n, m, k)) for m in TEST for k in KINDS)]
    if not need:
        return {"seed": seed, "futs": {}, "t0": time.time()}
    kinds = sorted({k for n in need for k in MODELS[n][2]})
    rows = train_rows(TRAIN, seed, kinds)
    futs = {n: ex.submit(fit_model, n, rows, TRAIN, tag) for n in need}
    return {"seed": seed, "futs": futs, "t0": time.time()}


def finish_seed(job):
    seed, tag = job["seed"], "s%d" % job["seed"]
    models = {n: f.result() for n, f in job["futs"].items()}
    log("seed %d: %d models fitted" % (seed, len(models)))
    for m in TEST:
        for kind in KINDS:
            predict_copy(models, m, kind, seed, tag)
    models.clear()
    res = score_rows(tag, list(MODELS), TEST, KINDS, FIXED)
    truth = {}
    for m in TEST:
        labs, period = labels_of(m)
        truth[m] = {lab: pv.sanity_stats(labs[lab], period) for lab in LABELS}
    rec = {"seed": seed, "costs": list(FIXED), "rows": res, "truth_stats": truth,
           "seconds_after_start": round(time.time() - job["t0"], 1)}
    os.makedirs(RUN_DIR, exist_ok=True)
    out_path = os.path.join(RUN_DIR, "s%d.json" % seed)
    with open(out_path + ".tmp", "w") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(out_path + ".tmp", out_path)
    log("seed %d scored; disk free %.1f GB" % (seed, disk_free_gb()))
    print_board([rec])
    return rec


# -- leave-one-match-out costs on TRAIN ----------------------------------------------------------------

def start_lomo(seed, config, ex):
    tag = "lomo%d" % seed
    copies = MODELS[config][2]
    held_todo = [h for h in TRAIN if not all(os.path.isfile(em_path(tag, config, h, k)) for k in KINDS)]
    rows = train_rows(TRAIN, seed, sorted(set(copies))) if held_todo else {}
    futs = {}
    for h in held_todo:
        k = TRAIN.index(h)
        futs[h] = ex.submit(fit_model, config, rows, TRAIN[:k] + TRAIN[k + 1:], "%s held %s" % (tag, h))
    return {"seed": seed, "config": config, "futs": futs, "t0": time.time()}


def finish_lomo(job):
    seed, config, tag = job["seed"], job["config"], "lomo%d" % job["seed"]
    label = MODELS[config][1]
    for h, f in job["futs"].items():
        model = f.result()
        for kind in KINDS:
            predict_copy({config: model}, h, kind, seed, tag)
    grid, cache = [], {}
    for st in GRID_TEAM:
        for sd in GRID_DEAD:
            per = {}
            for kind in KINDS:
                vals = []
                for h in TRAIN:
                    labs, period = labels_of(h)
                    if (h, kind) not in cache:
                        cache[(h, kind)] = np.load(em_path(tag, config, h, kind))
                    pred = pv.decode(cache[(h, kind)], period, st, sd)
                    vals.append(float(score(labs[label], pred, period)[0]))
                per[kind] = float(np.mean(vals))
            grid.append({"team": st, "dead": sd, "coupled": per["coupled"], "alt": per["alt"],
                         "mean": 0.5 * (per["coupled"] + per["alt"])})
    best = {}
    for crit in ("coupled", "alt", "mean"):
        b = max(grid, key=lambda g: (g[crit], -g["team"], -g["dead"]))
        best[crit] = {"team": b["team"], "dead": b["dead"], "oof_spell": b[crit]}
    rec = {"seed": seed, "config": config, "label": label, "grid": grid, "best": best,
           "seconds_after_start": round(time.time() - job["t0"], 1)}
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, "lomo_s%d_%s.json" % (seed, config.replace("|", "-"))), "w") as fh:
        json.dump(rec, fh, indent=1)
    for crit, b in best.items():
        log("LOMO %s seed %d best by OOF on %s copies: team %.0f dead %.0f (OOF spellF1 %.3f)" % (
            config, seed, crit, b["team"], b["dead"], b["oof_spell"]))
    fx = [g for g in grid if (g["team"], g["dead"]) == FIXED][0]
    log("LOMO %s at fixed 5/20: coupled %.3f alt %.3f mean %.3f" % (config, fx["coupled"], fx["alt"], fx["mean"]))
    return rec


# -- summaries ---------------------------------------------------------------------------------------

def disk_free_gb():
    import shutil
    return shutil.disk_usage(pv.CACHE if os.path.isdir(pv.CACHE) else ROOT).free / 1e9


def aggregate(recs):
    """(model, test, label) -> per-seed mean spell/macro/poss over TEST matches."""
    out = {}
    for rec in recs:
        for r in rec["rows"]:
            key = (r["model"], r["test"], r["label"])
            out.setdefault(key, {}).setdefault(rec["seed"], []).append(r)
    table = {}
    for key, by_seed in out.items():
        seeds = sorted(by_seed)
        allr = [r for s in seeds for r in by_seed[s]]
        table[key] = {
            "seeds": seeds,
            "spell_per_seed": [float(np.mean([r["spell"] for r in by_seed[s]])) for s in seeds],
            "macro": float(np.mean([r["macro"] for r in allr])),
            "spell": float(np.mean([r["spell"] for r in allr])),
            "spell_sd": float(np.std([r["spell"] for r in allr])),
            "n_pred": float(np.mean([r["n_pred"] for r in allr])),
            "n_gt": float(np.mean([r["n_gt"] for r in allr])),
            "share_err": float(np.mean([r["share_err"] for r in allr])),
        }
    return table


def print_board(recs):
    t = aggregate(recs)
    print("\n%-18s %-8s %-5s %8s %6s %8s %9s %7s %6s  per-seed spell" % (
        "model", "test", "label", "spellF1", "sd", "macroF1", "poss pred", "poss gt", "share"))
    for key in sorted(t, key=lambda k: (k[2], k[1], k[0])):
        v = t[key]
        print("%-18s %-8s %-5s %8.3f %6.3f %8.3f %9.0f %7.0f %6.3f  %s" % (
            key[0], key[1], key[2], v["spell"], v["spell_sd"], v["macro"], v["n_pred"], v["n_gt"],
            v["share_err"], " ".join("%.3f" % s for s in v["spell_per_seed"])), flush=True)


def load_recs(seeds):
    recs = []
    for s in seeds:
        p = os.path.join(RUN_DIR, "s%d.json" % s)
        if os.path.isfile(p):
            with open(p) as fh:
                recs.append(json.load(fh))
    return recs


def rescore(seeds, costs, label="rescored"):
    """Re-decode the stored TEST emissions with other costs (reporting only, never for choosing)."""
    recs = []
    for s in seeds:
        recs.append({"seed": s, "costs": list(costs), "rows": score_rows("s%d" % s, list(MODELS), TEST, KINDS, costs)})
    path = os.path.join(RUN_DIR, "rescore_%s_%g_%g.json" % (label, costs[0], costs[1]))
    with open(path, "w") as fh:
        json.dump(recs, fh, indent=1)
    print("\nTEST board re-decoded at costs team %g dead %g (reporting only)" % tuple(costs))
    print_board(recs)
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lomo", nargs="*", default=[], help="configs to pick costs for by LOMO (first seed)")
    ap.add_argument("--rescore", type=float, nargs=2, default=None, metavar=("TEAM", "DEAD"))
    ap.add_argument("--summary", action="store_true", help="print the board from saved results only")
    args = ap.parse_args(argv)
    from bench import feats3
    feats3._warn_off()
    if args.summary:
        print_board(load_recs(args.seeds))
        return
    if args.rescore:
        rescore(args.seeds, tuple(args.rescore))
        return
    t0 = time.time()
    log("run_pipe_bench seeds %s, lomo %s, workers %d, disk free %.1f GB" % (
        args.seeds, args.lomo, args.workers, disk_free_gb()))
    for m in ORDER:          # label cache written here, before any pool thread reads it
        labels_of(m)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        pending = None
        for s in args.seeds:
            job = start_seed(s, ex)            # rows (main thread), fits queued in the pool
            if pending is not None:
                finish_seed(pending)           # previous seed: features, predictions, scores
            pending = job
        lomo_jobs = [start_lomo(args.seeds[0], c, ex) for c in args.lomo]
        if pending is not None:
            finish_seed(pending)
        lomo = [finish_lomo(j) for j in lomo_jobs]
    print("\nALL SEEDS (costs team 5 / dead 20)")
    print_board(load_recs(args.seeds))
    for rec in lomo:
        b = rec["best"]["mean"]
        if (b["team"], b["dead"]) != FIXED:
            rescore(args.seeds, (b["team"], b["dead"]), "lomo-" + rec["config"].replace("|", "-"))
    log("bench done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
