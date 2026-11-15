"""Ablate feats3 groups on the external bench.

Train and choose switch costs (and the split-expert blend) only on TRAIN,
by leave-one-match-out. TEST is scored once. Same LightGBM setup as v1.

    python -m bench.run_feats3 --seeds 0 1 2

Do not import numpy at module level: spawned workers set thread env first.
"""

import json
import os
import shutil
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = r"D:\Barca_Hackathon\research\runs\features3"
TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
TEST = ["J03WPY", "J03WQQ", "J03WR9"]
GRID_TEAM = [5.0, 10.0, 20.0, 40.0, 80.0]
GRID_DEAD = [10.0, 20.0, 40.0, 80.0, 160.0]
MIDS = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
SCALES = [5.0, 12.0, 25.0, 50.0, 100.0]


def _threads(n):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def _log(path, msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


def _configs():
    """name, groups, context_drop, kind."""
    rows = [
        ("v1", (), (), "single"),
        ("v1+context", ("context",), (), "single"),
        ("v1+lang_dead", ("lang_dead",), (), "single"),
        ("v1+context+lang_dead", ("context", "lang_dead"), (), "single"),
        ("v1+context+touch+restart", ("context", "touch", "restart"), (), "single"),
        ("v1+all", ("context", "lang_dead", "touch", "restart", "gap"), (), "single"),
        ("split-all", ("context", "lang_dead", "touch", "restart", "gap"), (), "split"),
    ]
    for sub in ("centroid", "spread", "line", "hull", "convergence", "velocity"):
        rows.append(("ctx-drop-%s" % sub, ("context",), (sub,), "single"))
    return rows


def _result_path(seed, name):
    safe = name.replace("+", "_")
    return os.path.join(RUN_DIR, "results", "s%d_%s.json" % (seed, safe))


def _cache_dir(seed, rev):
    return os.path.join(RUN_DIR, "cache", rev, "seed%d" % seed)


def persist_code():
    os.makedirs(RUN_DIR, exist_ok=True)
    for name in ("feats3.py", "run_feats3.py"):
        shutil.copy2(os.path.join(HERE, name), os.path.join(RUN_DIR, name))


def build_one(job):
    """Degrade one match and write both team-views of the full feature matrix."""
    seed, index, match, out_dir = job
    _threads(3)
    import numpy as np
    from bench.dfl import load
    from bench.degrade import degrade
    from bench.methods import swap_teams
    from bench.run import labels as labeller
    from bench.feats3 import ALL_GROUPS, build_features, clocks

    t0 = time.time()
    clean = load(match)
    deg = degrade(clean, seed=1000 * seed + index)
    x, names, ball, sub, kind = build_features(deg, ALL_GROUPS)
    xs, names2, _, _, _ = build_features(swap_teams(deg), ALL_GROUPS)
    if names != names2:
        raise RuntimeError("swap changed feature names for %s" % match)
    if not (np.isfinite(x).all() and np.isfinite(xs).all()):
        raise RuntimeError("non-finite features for %s" % match)
    vis, gap_back, _, _, _ = clocks(deg)
    y = labeller(clean)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, match)
    np.save(stem + ".x.npy", np.ascontiguousarray(x))
    np.save(stem + ".xs.npy", np.ascontiguousarray(xs))
    np.save(stem + ".y.npy", np.ascontiguousarray(y))
    np.save(stem + ".period.npy", np.ascontiguousarray(deg["period"]))
    np.save(stem + ".gap.npy", np.ascontiguousarray(gap_back))
    np.save(stem + ".vis.npy", np.ascontiguousarray(vis))
    return {
        "match": match,
        "names": names,
        "ball": np.asarray(ball, np.uint8).tolist(),
        "sub": sub,
        "kind": kind,
        "frames": int(len(y)),
        "feats": int(x.shape[1]),
        "seconds": round(time.time() - t0, 1),
    }


def _take(mm, cols, rows=None):
    import numpy as np
    cols = np.asarray(cols, np.int32)
    if rows is None:
        n = mm.shape[0]
        out = np.empty((n, len(cols)), np.float32)
        for i in range(0, n, 25000):
            out[i:i + 25000] = np.asarray(mm[i:i + 25000][:, cols], np.float32)
        return out
    rows = np.asarray(rows, np.int32)
    parts = []
    for i in range(0, len(rows), 20000):
        r = rows[i:i + 20000]
        parts.append(np.asarray(mm[r][:, cols], np.float32))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, len(cols)), np.float32)


def _load_pack(cache, matches):
    import numpy as np
    pack = {}
    for m in matches:
        stem = os.path.join(cache, m)
        pack[m] = {
            "x": np.load(stem + ".x.npy", mmap_mode="r"),
            "xs": np.load(stem + ".xs.npy", mmap_mode="r"),
            "y": np.load(stem + ".y.npy"),
            "period": np.load(stem + ".period.npy"),
            "gap": np.load(stem + ".gap.npy"),
            "vis": np.load(stem + ".vis.npy"),
        }
    return pack


def _collect(pack, matches, cols, stride, visible_only):
    import numpy as np
    from bench.methods import SWAP
    xs, ys = [], []
    for m in matches:
        d = pack[m]
        rows = np.arange(0, len(d["y"]), stride)
        if visible_only:
            rows = rows[np.asarray(d["vis"][rows], bool)]
        xs.append(_take(d["x"], cols, rows))
        ys.append(np.asarray(d["y"][rows]))
        xs.append(_take(d["xs"], cols, rows))
        ys.append(SWAP[d["y"][rows]])
    if not xs:
        raise RuntimeError("no training rows")
    return np.concatenate(xs), np.concatenate(ys)


def _fit(x, y, rounds):
    import numpy as np
    import lightgbm as lgb
    from bench.methods import LGBMViterbi
    params = dict(LGBMViterbi.PARAMS)
    params["num_threads"] = 1
    params["deterministic"] = True
    params["verbose"] = -1
    model = lgb.train(params, lgb.Dataset(x, y), num_boost_round=rounds)
    prior = np.bincount(y, minlength=3).astype(np.float64)
    prior /= prior.sum()
    return model, prior


def _full_prior(pack, matches, stride):
    """Class prior of every TRAIN frame, both team views. Not used as a feature."""
    import numpy as np
    from bench.methods import SWAP
    ys = []
    for m in matches:
        y = pack[m]["y"]
        ys.append(y[::stride])
        ys.append(SWAP[y][::stride])
    y = np.concatenate(ys)
    prior = np.bincount(y, minlength=3).astype(np.float64)
    prior /= prior.sum()
    return prior


def _proba_sym(model, pack_m, cols):
    import numpy as np
    from bench.methods import SWAP
    p = model.predict(_take(pack_m["x"], cols))
    ps = model.predict(_take(pack_m["xs"], cols))
    return (0.5 * (p + ps[:, SWAP])).astype(np.float64)


def _emissions(p, prior):
    import numpy as np
    return np.log(p + 1e-6) - np.log(prior)


def _share(y, pred):
    import numpy as np
    from bench.methods import H, D
    alive_t = y != D
    alive_p = pred != D
    if not np.any(alive_t) or not np.any(alive_p):
        return 1.0
    return float(abs(np.mean(pred[alive_p] == H) - np.mean(y[alive_t] == H)))


def _tune_costs(oof, pack, matches, grid_t, grid_d):
    """oof is a list of emission matrices aligned with matches. TRAIN only."""
    import numpy as np
    from bench.feats3 import decode_periods_fast
    from bench.run import score
    best = None
    for st in grid_t:
        for sd in grid_d:
            spells = []
            for e, m in zip(oof, matches):
                pred = decode_periods_fast(e, pack[m]["period"], st, sd)
                spells.append(score(pack[m]["y"], pred, pack[m]["period"])[0])
            mean = float(np.mean(spells))
            if best is None or mean > best[0]:
                best = (mean, float(st), float(sd))
    return best


def _blend(pp, pf, gap, mid, scale):
    import numpy as np
    g = np.minimum(np.asarray(gap, np.float64), 5000.0)
    w = 1.0 / (1.0 + np.exp(-(g - mid) / scale))
    p = (1.0 - w)[:, None] * pp + w[:, None] * pf
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
    return p


def _tune_split(oof, pack, matches, grid_t, grid_d):
    """Two-stage: costs at a neutral blend, then the logistic, then costs again.

    oof items are (proba_present, proba_free, prior) for the held-out TRAIN match.
    """
    import numpy as np
    from bench.feats3 import decode_periods_fast
    from bench.run import score

    def mean_spell(mid, scale, st, sd):
        spells = []
        for (pp, pf, prior), m in zip(oof, matches):
            p = _blend(pp, pf, pack[m]["gap"], mid, scale)
            pred = decode_periods_fast(_emissions(p, prior), pack[m]["period"], st, sd)
            spells.append(score(pack[m]["y"], pred, pack[m]["period"])[0])
        return float(np.mean(spells))

    best_c = None
    for st in grid_t:
        for sd in grid_d:
            mean = mean_spell(50.0, 25.0, st, sd)
            if best_c is None or mean > best_c[0]:
                best_c = (mean, float(st), float(sd))
    st, sd = best_c[1], best_c[2]
    best_w = None
    for mid in MIDS:
        for scale in SCALES:
            mean = mean_spell(mid, scale, st, sd)
            if best_w is None or mean > best_w[0]:
                best_w = (mean, float(mid), float(scale))
    mid, scale = best_w[1], best_w[2]
    best = None
    for st2 in grid_t:
        for sd2 in grid_d:
            mean = mean_spell(mid, scale, st2, sd2)
            if best is None or mean > best[0]:
                best = (mean, float(st2), float(sd2), float(mid), float(scale))
    return best


def _score_match(pack_m, pred):
    from bench.run import score
    spell, macro, npred, ngt = score(pack_m["y"], pred, pack_m["period"])
    return {
        "spell": float(spell), "macro": float(macro),
        "n_pred": int(npred), "n_gt": int(ngt),
        "share_err": _share(pack_m["y"], pred),
    }


def train_one(job):
    try:
        return _train_one(job)
    except Exception:
        err = traceback.format_exc()
        path = job["result"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"config": job["name"], "seed": job["seed"], "error": err}, f, indent=2)
        _log(job["log"], "FAILED %s seed %d\n%s" % (job["name"], job["seed"], err))
        return {"config": job["name"], "seed": job["seed"], "error": err}


def _train_one(job):
    _threads(1)
    import numpy as np
    from bench.feats3 import column_index, decode_periods_fast

    t0 = time.time()
    log = job["log"]
    _log(log, "start %s seed %d" % (job["name"], job["seed"]))
    with open(os.path.join(job["cache"], "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["names"]
    ball = np.asarray(meta["ball"], dtype=bool)
    sub = meta["sub"]
    cols, used = column_index(names, sub, ball, job["groups"], job["drop"], mask_ball=False)
    cols_free, _ = column_index(names, sub, ball, job["groups"], job["drop"], mask_ball=True)
    pack = _load_pack(job["cache"], job["train"] + job["test"])
    train = list(job["train"])
    fits = []

    def timed_fit(matches, cols_use, visible_only):
        t1 = time.time()
        x, y = _collect(pack, matches, cols_use, job["stride"], visible_only)
        model, prior = _fit(x, y, job["rounds"])
        fits.append(round(time.time() - t1, 1))
        _log(log, "fit %s rows %d feats %d in %.0fs" % (job["name"], len(y), x.shape[1], fits[-1]))
        return model, prior

    if job["kind"] == "single":
        oof = []
        for k in range(len(train)):
            model, prior = timed_fit(train[:k] + train[k + 1:], cols, False)
            oof.append(_emissions(_proba_sym(model, pack[train[k]], cols), prior))
        oof_spell, st, sd = _tune_costs(oof, pack, train, job["grid_team"], job["grid_dead"])
        _log(log, "oof %s spell %.3f costs %.0f %.0f" % (job["name"], oof_spell, st, sd))
        model, prior = timed_fit(train, cols, False)
        mid = scale = None
        predict_em = lambda m: _emissions(_proba_sym(model, pack[m], cols), prior)
    else:
        oof = []
        for k in range(len(train)):
            rest = train[:k] + train[k + 1:]
            present, _ = timed_fit(rest, cols, True)
            free, _ = timed_fit(rest, cols_free, False)
            prior = _full_prior(pack, rest, job["stride"])
            pp = _proba_sym(present, pack[train[k]], cols)
            pf = _proba_sym(free, pack[train[k]], cols_free)
            oof.append((pp, pf, prior))
        oof_spell, st, sd, mid, scale = _tune_split(oof, pack, train, job["grid_team"], job["grid_dead"])
        _log(log, "oof %s spell %.3f costs %.0f %.0f logistic m %.0f s %.0f" % (
            job["name"], oof_spell, st, sd, mid, scale))
        present, _ = timed_fit(train, cols, True)
        free, _ = timed_fit(train, cols_free, False)
        prior = _full_prior(pack, train, job["stride"])

        def predict_em(m, present=present, free=free, prior=prior, mid=mid, scale=scale):
            pp = _proba_sym(present, pack[m], cols)
            pf = _proba_sym(free, pack[m], cols_free)
            return _emissions(_blend(pp, pf, pack[m]["gap"], mid, scale), prior)

    rows = []
    for m in job["test"]:
        pred = decode_periods_fast(predict_em(m), pack[m]["period"], st, sd)
        rec = _score_match(pack[m], pred)
        rec["match"] = m
        rows.append(rec)
        _log(log, "TEST %s %s spell %.3f macro %.3f poss %d/%d share %.3f" % (
            job["name"], m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"], rec["share_err"]))
    out = {
        "config": job["name"],
        "seed": job["seed"],
        "kind": job["kind"],
        "groups": list(job["groups"]),
        "drop": list(job["drop"]),
        "n_features": int(len(used)),
        "n_features_free": int(len(cols_free)) if job["kind"] == "split" else None,
        "rounds": job["rounds"],
        "stride": job["stride"],
        "switch_team": st,
        "switch_dead": sd,
        "logistic_m": mid,
        "logistic_s": scale,
        "oof_spell": float(oof_spell),
        "fit_seconds": fits,
        "seconds": round(time.time() - t0, 1),
        "matches": rows,
    }
    os.makedirs(os.path.dirname(job["result"]), exist_ok=True)
    tmp = job["result"] + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, job["result"])
    _log(log, "done %s seed %d in %.0fs" % (job["name"], job["seed"], out["seconds"]))
    return out


def _done_result(path, rounds):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except json.JSONDecodeError:
        return None
    if rec.get("error") or rec.get("rounds") != rounds:
        return None
    return rec


def _ensure_cache(seed, workers, rev, log_path):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cache = _cache_dir(seed, rev)
    meta_path = os.path.join(cache, "meta.json")
    order = TRAIN + TEST
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("rev") == rev and all(
                os.path.isfile(os.path.join(cache, m + ".x.npy")) for m in order):
            _log(log_path, "cache hit seed %d (%d feats)" % (seed, meta["feats"]))
            return cache
    os.makedirs(cache, exist_ok=True)
    jobs = [(seed, i, m, cache) for i, m in enumerate(order)]
    _log(log_path, "building features seed %d (%d matches)" % (seed, len(jobs)))
    ctx = mp.get_context("spawn")
    got = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = [ex.submit(build_one, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            got.append(rec)
            _log(log_path, "cached %s %d frames %d feats in %.0fs" % (
                rec["match"], rec["frames"], rec["feats"], rec["seconds"]))
    got.sort(key=lambda r: order.index(r["match"]))
    names0 = got[0]["names"]
    for rec in got[1:]:
        if rec["names"] != names0 or rec["sub"] != got[0]["sub"]:
            raise RuntimeError("feature schema differs across matches")
    meta = {
        "rev": rev,
        "names": names0,
        "ball": got[0]["ball"],
        "sub": got[0]["sub"],
        "kind": got[0]["kind"],
        "feats": got[0]["feats"],
        "matches": {r["match"]: {"frames": r["frames"], "seconds": r["seconds"]} for r in got},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    _log(log_path, "cache seed %d ready, %d columns" % (seed, meta["feats"]))
    return cache


def _fmt(rows, seeds):
    """rows: list of result dicts. Adoption needs every requested seed."""
    import numpy as np
    by = {}
    for r in rows:
        if r.get("error"):
            continue
        by.setdefault(r["config"], []).append(r)
    v1 = {}
    if "v1" in by:
        for r in by["v1"]:
            v1[r["seed"]] = float(np.mean([m["spell"] for m in r["matches"]]))
    lines = []
    header = ("| config | spellF1 | sd | macroF1 | poss pred | poss gt | home-share | "
              "per-seed spell | per-seed gain | adopt |")
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    order = [c[0] for c in _configs()]
    seen = list(order) + [k for k in by if k not in order]
    for name in seen:
        if name not in by:
            continue
        recs = sorted(by[name], key=lambda r: r["seed"])
        spells, macros, preds, gts, shares = [], [], [], [], []
        seed_spell, seed_gain = [], []
        gains_ok = []
        for r in recs:
            ss = [m["spell"] for m in r["matches"]]
            spells.extend(ss)
            macros.extend(m["macro"] for m in r["matches"])
            preds.extend(m["n_pred"] for m in r["matches"])
            gts.extend(m["n_gt"] for m in r["matches"])
            shares.extend(m["share_err"] for m in r["matches"])
            sm = float(np.mean(ss))
            seed_spell.append(sm)
            if r["seed"] in v1 and name != "v1":
                g = sm - v1[r["seed"]]
                seed_gain.append(g)
                gains_ok.append(g >= 0.03 - 1e-12)
        have = {r["seed"] for r in recs}
        adopt = ""
        if (name != "v1" and seed_gain and have == set(seeds) and set(v1) >= set(seeds)
                and len(seed_gain) == len(seeds)):
            adopt = "yes" if all(gains_ok) else "no"
        gain_txt = " ".join("%+.3f" % g for g in seed_gain) if seed_gain else ""
        seed_txt = " ".join("%.3f" % s for s in seed_spell)
        lines.append("| %s | %.3f | %.3f | %.3f | %.0f | %.0f | %.3f | %s | %s | %s |" % (
            name, float(np.mean(spells)), float(np.std(spells)), float(np.mean(macros)),
            float(np.mean(preds)), float(np.mean(gts)), float(np.mean(shares)),
            seed_txt, gain_txt, adopt))
    return "\n".join(lines)


def _write_final(rows, seeds, rounds, notes):
    import numpy as np
    failed = [r for r in rows if r.get("error")]
    body = []
    body.append("# Features3 ablation")
    body.append("")
    body.append("Bench v2, official `possession_metrics.evaluate`. "
                "TRAIN = %s. TEST = %s. Seeds = %s. LightGBM rounds = %d, stride 5, "
                "one thread, seed 0, deterministic, the v1 hyperparameters." % (
                    ", ".join(TRAIN), ", ".join(TEST), ", ".join(str(s) for s in seeds), rounds))
    body.append("")
    body.append("Switch costs, and the split-expert logistic, are chosen by leave-one-match-out "
                "on TRAIN only. Each TEST match is scored once per seed. "
                "sd is the population standard deviation over match x seed rows, matching `bench.run`. "
                "A per-seed spellF1 is the mean of the three TEST matches. "
                "Adopt means the gain vs v1 on that same seed is at least +0.03 on every seed that finished.")
    body.append("")
    body.append("## Main tables")
    body.append("")
    body.append(_fmt(rows, seeds))
    body.append("")
    body.append("## Costs chosen on TRAIN")
    body.append("")
    body.append("| config | seed | oof spellF1 | switch team | switch dead | logistic m | logistic s | feats | seconds |")
    body.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted((x for x in rows if not x.get("error")), key=lambda z: (z["config"], z["seed"])):
        body.append("| %s | %d | %.3f | %.0f | %.0f | %s | %s | %d | %.0f |" % (
            r["config"], r["seed"], r["oof_spell"], r["switch_team"], r["switch_dead"],
            "" if r["logistic_m"] is None else "%.0f" % r["logistic_m"],
            "" if r["logistic_s"] is None else "%.0f" % r["logistic_s"],
            r["n_features"], r["seconds"]))
    body.append("")
    body.append("## Per match")
    body.append("")
    body.append("| config | seed | match | spellF1 | macroF1 | poss pred | poss gt | home-share |")
    body.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in sorted((x for x in rows if not x.get("error")), key=lambda z: (z["config"], z["seed"])):
        for m in r["matches"]:
            body.append("| %s | %d | %s | %.3f | %.3f | %d | %d | %.3f |" % (
                r["config"], r["seed"], m["match"], m["spell"], m["macro"], m["n_pred"], m["n_gt"], m["share_err"]))
    body.append("")
    body.append("## Reproduce")
    body.append("")
    body.append("From the repo root, with the bench npz cache available at `bench/cache/npz`:")
    body.append("")
    body.append("```")
    body.append(".venv\\Scripts\\python.exe -m bench.run_feats3 --seeds %s" % " ".join(str(s) for s in seeds))
    body.append("```")
    body.append("")
    body.append("Code copies: `feats3.py`, `run_feats3.py` in this folder. "
                "Per-config logs are under `logs/`, per-job JSON under `results/`.")
    body.append("")
    body.append("## What the groups are")
    body.append("")
    body.append("- **v1** is `methods.dense_features`, unchanged.")
    body.append("- **context** is team shape that does not need a visible ball at the current frame: "
                "centroid x (raw and attack-frame), width, depth, defensive-line height (mean of the "
                "four deepest outfielders, goalkeeper skipped), share of players in the opponent half, "
                "convex-hull area, centroid velocity toward each goal and toward the opponent goal, "
                "players within 10 m of the ball, players converging on it (closing speed > 1 m/s), "
                "and a pressure sum. Shape and velocity are sampled now, at the last visible ball "
                "frame, and at the next one, plus the next-minus-last span. Counts are taken both "
                "at those edge frames and from the current players toward those two ball positions. "
                "Attack direction is inferred per period from which team has a player living nearest "
                "x=0. Raw x flips between halves in this tracking; the attack frame does not.")
    body.append("- **lang_dead** is the Lang et al. 2022 block, permutation-invariant: per-team speed "
                "p50/p90, acceleration p50, centroid velocity (vx, vy), and minimum boundary distance, "
                "at shifts of +/-0.4, 0.8, 2, 4, 20, 40 and 80 s, edge-held inside a period. "
                "A 91-frame median of speed, acceleration and boundary distance is included as its own "
                "channel (the filter width Lang tuned). It is not applied to the decoded labels; "
                "Viterbi remains the only smoother.")
    body.append("- **touch** is the change in ball direction and speed over 0.5 s and 1 s "
                "(displacement over 12 and 25 frames), gated by a player within 2.5 m, plus time "
                "since and until that event. A stationary ball is not a touch.")
    body.append("- **restart** is kick-off split and a centre-spot taker, touchline / corner / penalty-spot "
                "counts, players in their own goal area and box, distance to their own goal, calm "
                "fraction, length of the current calm run, a free-kick wall count, and the last and "
                "next ball's distance to the touchline, goal line, corner and penalty spot.")
    body.append("- **gap** is log gap length both ways, fraction through the gap, last and next ball "
                "coordinates, and time within the period.")
    body.append("- **split-all** trains two LightGBMs on v1+all. The ball-present expert sees only "
                "frames where the ball is visible. The ball-free expert drops every column that reads "
                "the ball track (coordinates or visibility), including v1's distances and gaps, and is "
                "trained on all frames. Probabilities are blended with a logistic of frames since the "
                "last visible ball. The midpoint and scale are chosen on TRAIN leave-one-match-out, "
                "then the switch costs are re-chosen at that logistic.")
    body.append("- **ctx-drop-*** is v1+context with that subgroup removed.")
    body.append("")
    body.append("## Limits")
    body.append("")
    body.append("- No column reads `alive` or `owner`. A scramble of those two arrays is asserted not "
                "to change the matrix, on synthetic data and on one degraded match. Rolling the ball "
                "is asserted not to change columns tagged player-only.")
    body.append("- Edge samples and Lang's positive shifts look ahead at positions. That is the same "
                "offline setting as v1 (`gap_fwd`, centred windows). They do not look at the future "
                "of the owner flag.")
    body.append("- The public 5-minute window is not read and is not used to pick anything.")
    body.append("- Costs use the v1 grids (team 5..80, dead 10..160). The logistic grid is "
                "m in {0,10,25,50,100,200,400,800} frames and scale in {5,12,25,50,100}, searched in "
                "three passes rather than the full Cartesian product, so a combination off those "
                "passes can be missed.")
    body.append("- The 91-frame median is a feature. There is no second median filter on the decoded "
                "labels, so this is not a full copy of Lang's post-processing.")
    body.append("- Team symmetry, stride 5 and 300 trees are v1's. A subgroup that helps only at a "
                "different tree budget would not show up.")
    body.append("- Degrade seeds match `bench.run`: `1000 * seed + index` over TRAIN then TEST.")
    if notes:
        body.append("- " + notes)
    if failed:
        body.append("")
        body.append("## Failures")
        body.append("")
        for r in failed:
            body.append("```")
            body.append("%s seed %s\n%s" % (r.get("config"), r.get("seed"), r.get("error", "")))
            body.append("```")
    text = "\n".join(body) + "\n"
    path = os.path.join(RUN_DIR, "FINAL.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _load_rows():
    rows = []
    folder = os.path.join(RUN_DIR, "results")
    if not os.path.isdir(folder):
        return rows
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                rows.append(json.load(f))
        except json.JSONDecodeError:
            continue
    return rows


def main(argv=None):
    import argparse
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from bench.feats3 import REV

    if train_one.__module__ != "bench.run_feats3":
        raise RuntimeError("workers must be launched from bench.run_feats3, not __main__")

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="exact config names to run; default is the full ablation")
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--grid-team", type=float, nargs="+", default=GRID_TEAM)
    ap.add_argument("--grid-dead", type=float, nargs="+", default=GRID_DEAD)
    args = ap.parse_args(argv)

    persist_code()
    os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "results"), exist_ok=True)
    log_path = os.path.join(RUN_DIR, "logs", "main.log")
    _log(log_path, "run_feats3 rev %s seeds %s rounds %d workers %d" % (
        REV, args.seeds, args.rounds, args.workers))

    if not args.skip_self_test:
        _threads(4)
        from bench.dfl import load
        from bench.degrade import degrade
        from bench.feats3 import self_test
        _log(log_path, "self-test including one degraded TRAIN match")
        self_test(degrade(load(TRAIN[0]), seed=0))
        _log(log_path, "self-test passed")
        if args.self_test_only:
            persist_code()
            return

    wanted = set(args.configs) if args.configs else None
    specs = [c for c in _configs() if wanted is None or c[0] in wanted]
    if wanted is not None:
        missing = wanted - {c[0] for c in specs}
        if missing:
            raise SystemExit("unknown configs %s" % sorted(missing))

    ctx = mp.get_context("spawn")
    for seed in args.seeds:
        cache = _ensure_cache(seed, args.workers, REV, log_path)
        jobs = []
        for name, groups, drop, kind in specs:
            path = _result_path(seed, name)
            if _done_result(path, args.rounds):
                _log(log_path, "skip finished %s seed %d" % (name, seed))
                continue
            jobs.append({
                "name": name,
                "groups": list(groups),
                "drop": list(drop),
                "kind": kind,
                "seed": seed,
                "cache": cache,
                "rounds": args.rounds,
                "stride": args.stride,
                "grid_team": list(args.grid_team),
                "grid_dead": list(args.grid_dead),
                "train": list(TRAIN),
                "test": list(TEST),
                "result": path,
                "log": os.path.join(RUN_DIR, "logs", "s%d_%s.log" % (seed, name.replace("+", "_"))),
            })
        if not jobs:
            _write_final(_load_rows(), args.seeds, args.rounds, "")
            continue
        _log(log_path, "training seed %d: %d jobs" % (seed, len(jobs)))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, max_tasks_per_child=1) as ex:
            futs = {ex.submit(train_one, j): j for j in jobs}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec.get("error"):
                    _log(log_path, "FAILED %s seed %d" % (rec.get("config"), rec.get("seed")))
                else:
                    spells = [m["spell"] for m in rec["matches"]]
                    _log(log_path, "finished %s seed %d spell %.3f (matches %s) in %.0fs" % (
                        rec["config"], rec["seed"], sum(spells) / len(spells),
                        " ".join("%.3f" % s for s in spells), rec["seconds"]))
                _write_final(_load_rows(), args.seeds, args.rounds,
                             "Partial table: later seeds or configs may still be running.")
    note = "Finished seeds %s." % ", ".join(str(s) for s in args.seeds)
    path = _write_final(_load_rows(), args.seeds, args.rounds, note)
    persist_code()
    _log(log_path, "wrote %s" % path)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    import bench.run_feats3 as _mod
    _mod.main()
