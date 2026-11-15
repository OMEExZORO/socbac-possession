"""Measure rival-team ideas on top of v1+context.

TRAIN and TEST, the LightGBM setup, and the cost grids match bench v1 /
features3. Switch costs are chosen by leave-one-match-out on TRAIN. TEST is
scored once. Adoption is +0.03 spellF1 over v1+context on every requested seed.

    .venv\\Scripts\\python.exe -m bench.run_rivalideas --seeds 0 1 2

Prints aggregates only. Does not read competition data.
"""

import json
import math
import os
import shutil
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from bench.run import TEST, TRAIN, score

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = r"D:\Barca_Hackathon\research\runs\rivalideas"
F3_SEED0 = r"D:\Barca_Hackathon\research\runs\features3\cache\f3a\seed0"
CODE_REV = "ri1"
RUN_SEEDS = [0, 1, 2]
RUN_ROUNDS = 300
GRID_TEAM = [5.0, 10.0, 20.0, 40.0, 80.0]
GRID_DEAD = [10.0, 20.0, 40.0, 80.0, 160.0]
LOW_TEAM = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0]
LOW_DEAD = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
R2_SPECS = (
    ("r2_w3_r1", 3.0, 25),
    ("r2_w2_r1", 2.0, 25),
    ("r2_w5_r1", 5.0, 25),
    ("r2_w3_r2", 3.0, 50),
)
REF = {
    "v1": {0: 0.388},
    "v1+context": {0: 0.529},
}
ORDER = TRAIN + TEST
T0 = time.time()
BUDGET_S = 100 * 60
SKIPPED = []


def _threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _log(msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    path = os.path.join(RUN_DIR, "logs", "main.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


def _safe(name):
    return name.replace("+", "_")


def result_path(seed, name):
    return os.path.join(RUN_DIR, "results", "s%d_%s.json" % (seed, _safe(name)))


def preds_path(seed, name):
    return os.path.join(RUN_DIR, "preds", "s%d_%s.npz" % (seed, _safe(name)))


def model_stem(seed, name):
    return os.path.join(RUN_DIR, "models", "s%d_%s" % (seed, _safe(name)))


def persist_code():
    os.makedirs(RUN_DIR, exist_ok=True)
    for name in ("feats3.py", "degrade_v3.py", "rivalideas.py", "run_rivalideas.py", "test_rivalideas.py"):
        src = os.path.join(HERE, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(RUN_DIR, name))


def _json_dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=2)
    os.replace(tmp, path)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def allow(minutes, what):
    left = BUDGET_S - (time.time() - T0)
    if left < minutes * 60.0:
        msg = "SKIP %s (%.1f min left, need %.0f)" % (what, left / 60.0, minutes)
        _log(msg)
        SKIPPED.append(msg)
        return False
    return True


def _cache_ready(path):
    if not os.path.isfile(os.path.join(path, "meta.json")):
        return False
    for m in ORDER:
        for ext in (".x.npy", ".xs.npy", ".y.npy", ".period.npy"):
            if not os.path.isfile(os.path.join(path, m + ext)):
                return False
    return True


def v2_cache(seed):
    if seed == 0 and _cache_ready(F3_SEED0):
        return F3_SEED0
    return os.path.join(RUN_DIR, "cache", "v2", "seed%d" % seed)


def v3_cache(seed):
    return os.path.join(RUN_DIR, "cache", "v3", "seed%d" % seed)


def la_dir(seed, kind):
    return os.path.join(RUN_DIR, "cache", "la_" + kind, "seed%d" % seed)


def _save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(arr))
    os.replace(tmp, path)


def _build_cache(kind, seed):
    """Context-only features, one match at a time. kind is v2 or v3."""
    import gc
    from bench.dfl import load
    from bench.feats3 import build_features
    from bench.methods import swap_teams
    from bench.run import labels as frame_labels

    if kind == "v3":
        from bench.degrade_v3 import degrade
        path = v3_cache(seed)
    else:
        from bench.degrade import degrade
        path = v2_cache(seed)
        if seed == 0 and path == F3_SEED0:
            _log("cache hit v2 seed 0 (features3 f3a)")
            return path
    if _cache_ready(path):
        _log("cache hit %s seed %d" % (kind, seed))
        return path
    os.makedirs(path, exist_ok=True)
    _log("building %s features seed %d" % (kind, seed))
    meta_matches = {}
    names = ball = sub = kind_cols = None
    for index, match in enumerate(ORDER):
        stem = os.path.join(path, match)
        ready_m = all(os.path.isfile(stem + ext) for ext in (".x.npy", ".xs.npy", ".y.npy", ".period.npy"))
        if ready_m:
            _log("cache file exists %s" % match)
            continue
        t1 = time.time()
        clean = load(match)
        deg = degrade(clean, seed=1000 * seed + index)
        if kind == "v3":
            _log(
                "degrade %s dead_vis %.3f live_vis %.3f hidden %.3f"
                % (match, deg["ball_dead_visible"], deg["ball_live_visible"], deg["ball_hidden_share"])
            )
        x, names, ball, sub, kind_cols = build_features(deg, ("context",))
        if not np.isfinite(x).all():
            raise RuntimeError("non-finite features %s" % match)
        _save_npy(stem + ".x.npy", x)
        del x
        gc.collect()
        xs, names2, _, _, _ = build_features(swap_teams(deg), ("context",))
        if names2 != names:
            raise RuntimeError("swap changed names for %s" % match)
        if not np.isfinite(xs).all():
            raise RuntimeError("non-finite swapped features %s" % match)
        _save_npy(stem + ".xs.npy", xs)
        _save_npy(stem + ".y.npy", np.ascontiguousarray(frame_labels(clean)))
        _save_npy(stem + ".period.npy", np.ascontiguousarray(deg["period"]))
        meta_matches[match] = {"frames": int(len(deg["period"])), "seconds": round(time.time() - t1, 1)}
        _log("cached %s %d frames %d feats in %.0fs" % (match, meta_matches[match]["frames"], xs.shape[1], meta_matches[match]["seconds"]))
        del xs, deg, clean
        gc.collect()
    # Names come from the last built match, or from an existing file's sibling meta.
    if names is None:
        clean = load(ORDER[0])
        deg = degrade(clean, seed=1000 * seed)
        _x, names, ball, sub, kind_cols = build_features(deg, ("context",))
        del _x, deg, clean
    meta = {
        "rev": "v3ctx" if kind == "v3" else "v2ctx",
        "degrade": kind,
        "seed": seed,
        "names": names,
        "ball": [int(v) for v in np.asarray(ball).tolist()],
        "sub": list(sub),
        "kind": list(kind_cols),
        "feats": int(len(names)),
        "matches": meta_matches,
    }
    _json_dump(os.path.join(path, "meta.json"), meta)
    _log("cache %s seed %d ready, %d columns" % (kind, seed, meta["feats"]))
    return path


def _ensure_cache(kind, seed):
    path = v2_cache(seed) if kind == "v2" else v3_cache(seed)
    if kind == "v2" and seed == 0 and _cache_ready(F3_SEED0):
        return F3_SEED0
    if _cache_ready(path):
        return path
    return _build_cache(kind, seed)


def _meta(cache):
    with open(os.path.join(cache, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _resolve(names, want):
    pos = {n: i for i, n in enumerate(names)}
    missing = [n for n in want if n not in pos]
    if missing:
        raise RuntimeError("missing features %s" % missing[:4])
    return np.asarray([pos[n] for n in want], np.int32)


def _context_columns(cache, groups):
    from bench.feats3 import column_index

    meta = _meta(cache)
    ball = np.asarray(meta["ball"], dtype=bool)
    cols, used = column_index(meta["names"], meta["sub"], ball, tuple(groups), (), False)
    return cols, used


def _load_labels(cache, matches):
    out = {}
    for m in matches:
        stem = os.path.join(cache, m)
        out[m] = {"y": np.load(stem + ".y.npy"), "period": np.load(stem + ".period.npy")}
    return out


def _load_pack(cache, matches, base_cols, la_path=None):
    pack = {}
    for m in matches:
        stem = os.path.join(cache, m)
        d = {
            "x": np.load(stem + ".x.npy", mmap_mode="r"),
            "xs": np.load(stem + ".xs.npy", mmap_mode="r"),
            "y": np.load(stem + ".y.npy"),
            "period": np.load(stem + ".period.npy"),
            "cols": base_cols,
        }
        if d["x"].shape[0] != len(d["y"]):
            raise RuntimeError("length mismatch %s" % m)
        if la_path:
            d["la"] = np.load(os.path.join(la_path, m + ".la.npy"), mmap_mode="r")
            d["las"] = np.load(os.path.join(la_path, m + ".las.npy"), mmap_mode="r")
        pack[m] = d
    return pack


def _take(mm, cols, rows=None):
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
    if not parts:
        return np.zeros((0, len(cols)), np.float32)
    return np.concatenate(parts, axis=0)


def _view(pack_m, swapped, rows, la_cols):
    base = _take(pack_m["xs" if swapped else "x"], pack_m["cols"], rows)
    if la_cols is None:
        return base
    extra = _take(pack_m["las" if swapped else "la"], la_cols, rows)
    return np.concatenate([base, extra], axis=1)


def _collect(pack, matches, stride, weight_spec, la_cols):
    from bench.methods import SWAP
    from bench.rivalideas import boundary_weights

    xs, ys, ws = [], [], []
    for m in matches:
        d = pack[m]
        rows = np.arange(0, len(d["y"]), stride)
        y = np.asarray(d["y"])
        xs.append(_view(d, False, rows, la_cols))
        ys.append(y[rows])
        xs.append(_view(d, True, rows, la_cols))
        ys.append(SWAP[y[rows]])
        if weight_spec is not None:
            w = boundary_weights(y, d["period"], weight_spec["radius"], weight_spec["mult"])
            ws.append(w[rows])
            ws.append(w[rows])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws) if ws else None
    return x, y, w


def _fit(x, y, rounds, weight):
    import lightgbm as lgb
    from bench.methods import LGBMViterbi

    params = dict(LGBMViterbi.PARAMS)
    params["num_threads"] = 1
    params["deterministic"] = True
    params["verbose"] = -1
    if weight is None:
        ds = lgb.Dataset(x, y)
    else:
        if len(weight) != len(y):
            raise RuntimeError("weight length %d != %d" % (len(weight), len(y)))
        ds = lgb.Dataset(x, y, weight=np.ascontiguousarray(weight, np.float32))
    model = lgb.train(params, ds, num_boost_round=int(rounds))
    prior = np.bincount(y, minlength=3).astype(np.float64)
    prior /= max(float(prior.sum()), 1.0)
    return model, prior


def _proba(model, pack_m, la_cols):
    from bench.methods import SWAP

    p = model.predict(_view(pack_m, False, None, la_cols))
    ps = model.predict(_view(pack_m, True, None, la_cols))
    return (0.5 * (p + ps[:, SWAP])).astype(np.float64)


def _emissions(p, prior):
    return np.log(p + 1e-6) - np.log(prior)


def _share(y, pred):
    from bench.methods import D, H

    alive_t = y != D
    alive_p = pred != D
    if not np.any(alive_t) or not np.any(alive_p):
        return 1.0
    return float(abs(np.mean(pred[alive_p] == H) - np.mean(y[alive_t] == H)))


def _eval_pred(y, pred, period):
    spell, macro, npred, ngt = score(y, pred, period)
    return {
        "spell": float(spell),
        "macro": float(macro),
        "n_pred": int(npred),
        "n_gt": int(ngt),
        "share_err": _share(y, pred),
    }


def _decode(em, period, st, sd):
    from bench.feats3 import decode_periods_fast

    return decode_periods_fast(em, period, st, sd)


def _tune(pack, matches, emissions, grid_team, grid_dead):
    from bench.rivalideas import choose_costs_grid

    def score_fn(st, sd):
        spells = []
        for em, m in zip(emissions, matches):
            pred = _decode(em, pack[m]["period"], st, sd)
            spells.append(score(pack[m]["y"], pred, pack[m]["period"])[0])
        return float(np.mean(spells))

    return choose_costs_grid(score_fn, grid_team, grid_dead)


def _ensure_la(kind, seed, groups):
    from bench.rivalideas import LA_REV, lookahead_from_context

    src = _ensure_cache(kind, seed)
    dst = la_dir(seed, kind)
    meta_path = os.path.join(dst, "meta.json")
    if os.path.isfile(meta_path):
        meta = _load_json(meta_path)
        if meta and meta.get("rev") == LA_REV and all(os.path.isfile(os.path.join(dst, m + ".la.npy")) for m in ORDER):
            _log("lookahead cache hit %s seed %d" % (kind, seed))
            return dst
    os.makedirs(dst, exist_ok=True)
    src_meta = _meta(src)
    _log("building lookahead %s seed %d" % (kind, seed))
    names = None
    for match in ORDER:
        x = np.load(os.path.join(src, match + ".x.npy"), mmap_mode="r")
        period = np.load(os.path.join(src, match + ".period.npy"))
        la, names = lookahead_from_context(x, src_meta["names"], period)
        _save_npy(os.path.join(dst, match + ".la.npy"), la)
        del x, la
        xs = np.load(os.path.join(src, match + ".xs.npy"), mmap_mode="r")
        las, names2 = lookahead_from_context(xs, src_meta["names"], period)
        if names2 != names:
            raise RuntimeError("lookahead names changed on swap")
        _save_npy(os.path.join(dst, match + ".las.npy"), las)
        del xs, las
    _json_dump(os.path.join(dst, "meta.json"), {"rev": LA_REV, "names": names, "source": src})
    _log("lookahead ready %s seed %d (%d cols)" % (kind, seed, len(names)))
    return dst


def _save_preds(path, oof, test, em_oof=None, em_test=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for m, arr in oof.items():
        payload["oof_" + m] = np.asarray(arr, np.int8)
    for m, arr in test.items():
        payload["test_" + m] = np.asarray(arr, np.int8)
    if em_oof is not None:
        for m, arr in em_oof.items():
            payload["emoof_" + m] = np.asarray(arr, np.float64)
    if em_test is not None:
        for m, arr in em_test.items():
            payload["emtest_" + m] = np.asarray(arr, np.float64)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def _load_preds(seed, name):
    path = preds_path(seed, name)
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        files = set(z.files)
        oof = {m: np.asarray(z["oof_" + m], np.int8) for m in TRAIN}
        test = {m: np.asarray(z["test_" + m], np.int8) for m in TEST}
        em_oof = em_test = None
        if "emoof_" + TRAIN[0] in files:
            em_oof = {m: np.asarray(z["emoof_" + m], np.float64) for m in TRAIN}
            em_test = {m: np.asarray(z["emtest_" + m], np.float64) for m in TEST}
    return oof, test, em_oof, em_test


def _done(seed, name, rounds=None):
    rec = _load_json(result_path(seed, name))
    if not rec or rec.get("error") or rec.get("rev") != CODE_REV:
        return None
    if rounds is not None and rec.get("rounds") not in (None, rounds):
        return None
    if rec.get("needs_preds") and not os.path.isfile(preds_path(seed, name)):
        return None
    return rec


def _mean_spell(rec):
    return float(np.mean([m["spell"] for m in rec["matches"]]))


def _check_ref(rec):
    target = REF.get(rec["config"], {}).get(int(rec["seed"]))
    if target is None:
        return
    got = _mean_spell(rec)
    _log("reproduce %s seed %d spell %.3f (reference %.3f)" % (rec["config"], rec["seed"], got, target))
    if abs(got - target) > 0.008:
        write_final(RUN_SEEDS, RUN_ROUNDS)
        raise SystemExit(
            "reproduction of %s seed %d is %.3f, reference %.3f" % (rec["config"], rec["seed"], got, target)
        )


def _pooled(pack, preds, matches):
    from bench.rivalideas import pool_sanity

    return pool_sanity([preds[m] for m in matches], [pack[m]["period"] for m in matches])


def train_config(seed, name, groups, rounds, stride, weight=None, use_la=False,
                 train_kind="v2", test_kind="v2", save_emissions=False):
    """Fit on TRAIN of train_kind, score TEST of test_kind. Returns the result dict."""
    import gc

    rec = _done(seed, name, rounds)
    if rec:
        _log("skip finished %s seed %d" % (name, seed))
        return rec
    _threads()
    t0 = time.time()
    _log("start %s seed %d" % (name, seed))
    train_cache = _ensure_cache(train_kind, seed)
    test_cache = train_cache if test_kind == train_kind else _ensure_cache(test_kind, seed)
    cols, used = _context_columns(train_cache, groups)
    if tuple(groups) == ("context",) and len(used) != 165:
        raise RuntimeError("v1+context has %d columns, expected 165" % len(used))
    if tuple(groups) == () and len(used) != 49:
        raise RuntimeError("v1 has %d columns, expected 49" % len(used))
    la_path = None
    la_cols = None
    features = list(used)
    if use_la:
        la_path = _ensure_la(train_kind, seed, groups)
        la_meta = _load_json(os.path.join(la_path, "meta.json"))
        la_cols = np.arange(len(la_meta["names"]), dtype=np.int32)
        features = features + list(la_meta["names"])
        if test_kind != train_kind:
            _ensure_la(test_kind, seed, groups)
    # Column indices differ when the on-disk layout differs. Resolve per cache.
    train_cols = _resolve(_meta(train_cache)["names"], used)
    test_cols = train_cols if test_cache == train_cache else _resolve(_meta(test_cache)["names"], used)
    test_la = la_dir(seed, test_kind) if use_la else None
    pack = _load_pack(train_cache, TRAIN, train_cols, la_path)
    if test_cache != train_cache or test_cols is not train_cols:
        pack.update(_load_pack(test_cache, TEST, test_cols, test_la))
    elif use_la and test_la and test_la != la_path:
        for m in TEST:
            pack[m]["la"] = np.load(os.path.join(test_la, m + ".la.npy"), mmap_mode="r")
            pack[m]["las"] = np.load(os.path.join(test_la, m + ".las.npy"), mmap_mode="r")
    else:
        # TEST matches are not in pack yet when caches match.
        if TEST[0] not in pack:
            pack.update(_load_pack(test_cache, TEST, test_cols, la_path))

    weight_spec = None if weight is None else {"radius": int(weight[1]), "mult": float(weight[0])}
    oof_em = []
    oof_pred = {}
    fits = []
    for k, hold in enumerate(TRAIN):
        rest = TRAIN[:k] + TRAIN[k + 1:]
        t1 = time.time()
        x, y, w = _collect(pack, rest, stride, weight_spec, la_cols)
        model, prior = _fit(x, y, rounds, w)
        fits.append(round(time.time() - t1, 1))
        _log("fit %s rows %d feats %d in %.0fs" % (name, len(y), x.shape[1], fits[-1]))
        del x, y, w
        em = _emissions(_proba(model, pack[hold], la_cols), prior)
        oof_em.append(em)
        del model
        gc.collect()
    oof_spell, st, sd = _tune(pack, TRAIN, oof_em, GRID_TEAM, GRID_DEAD)
    _log("oof %s spell %.3f costs %.0f %.0f" % (name, oof_spell, st, sd))
    for em, m in zip(oof_em, TRAIN):
        oof_pred[m] = _decode(em, pack[m]["period"], st, sd)

    t1 = time.time()
    x, y, w = _collect(pack, TRAIN, stride, weight_spec, la_cols)
    model, prior = _fit(x, y, rounds, w)
    fits.append(round(time.time() - t1, 1))
    _log("fit %s rows %d feats %d in %.0fs" % (name, len(y), x.shape[1], fits[-1]))
    del x, y, w
    gc.collect()

    stem = model_stem(seed, name)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    model.save_model(stem + ".txt")
    _json_dump(stem + ".json", {
        "rev": CODE_REV, "config": name, "seed": seed, "prior": prior.tolist(),
        "features": features, "switch_team": st, "switch_dead": sd,
        "train_kind": train_kind, "groups": list(groups),
    })

    test_pred = {}
    test_em = {}
    rows = []
    for m in TEST:
        em = _emissions(_proba(model, pack[m], la_cols), prior)
        pred = _decode(em, pack[m]["period"], st, sd)
        test_pred[m] = pred
        test_em[m] = em
        rec_m = _eval_pred(pack[m]["y"], pred, pack[m]["period"])
        rec_m["match"] = m
        rows.append(rec_m)
        _log("TEST %s %s spell %.3f macro %.3f poss %d/%d share %.3f" % (
            name, m, rec_m["spell"], rec_m["macro"], rec_m["n_pred"], rec_m["n_gt"], rec_m["share_err"]))
    del model
    gc.collect()
    pooled = _pooled(pack, test_pred, TEST)
    out = {
        "rev": CODE_REV,
        "config": name,
        "seed": int(seed),
        "rounds": int(rounds),
        "stride": int(stride),
        "groups": list(groups),
        "weight": None if weight is None else [float(weight[0]), int(weight[1])],
        "lookahead": bool(use_la),
        "train_deg": train_kind,
        "test_deg": test_kind,
        "n_features": int(len(features)),
        "switch_team": float(st),
        "switch_dead": float(sd),
        "oof_spell": float(oof_spell),
        "fit_seconds": fits,
        "seconds": round(time.time() - t0, 1),
        "matches": rows,
        "sanity": pooled,
        "needs_preds": True,
    }
    em_oof = {m: oof_em[i] for i, m in enumerate(TRAIN)} if save_emissions else None
    em_test = test_em if save_emissions else None
    _save_preds(preds_path(seed, name), oof_pred, test_pred, em_oof, em_test)
    _json_dump(result_path(seed, name), out)
    _log("done %s seed %d spell %.3f in %.0fs" % (name, seed, _mean_spell(out), out["seconds"]))
    _check_ref(out)
    del pack, oof_em, test_em
    gc.collect()
    return out


def _search_cleanup(oof, labels, train_matches):
    from bench.rivalideas import cleanup_labels

    best = None
    table = []
    for td in (0, 1, 2, 3):
        for tm in (0, 1, 2, 3):
            spells = []
            for m in train_matches:
                pred = cleanup_labels(oof[m], labels[m]["period"], td, tm)
                spells.append(float(score(labels[m]["y"], pred, labels[m]["period"])[0]))
            mean = float(np.mean(spells))
            table.append({"t_d": td, "t_m": tm, "oof": mean})
            if best is None or mean > best[0]:
                best = (mean, td, tm)
    return best, table


def run_cleanup(seed, base, config_name):
    """Choose T_d, T_m on TRAIN out-of-fold preds of `base`, score TEST once."""
    rec = _done(seed, config_name)
    if rec:
        _log("skip finished %s seed %d" % (config_name, seed))
        return rec
    loaded = _load_preds(seed, base)
    if loaded is None:
        _log("skip %s seed %d, no preds for %s" % (config_name, seed, base))
        return None
    oof, _test, _eo, _et = loaded
    labels = _load_labels(v2_cache(seed), TRAIN + TEST)
    for m in TRAIN:
        if len(oof[m]) != len(labels[m]["y"]):
            raise RuntimeError("pred length %s" % m)
    best, table = _search_cleanup(oof, labels, TRAIN)
    raw = float(np.mean([
        score(labels[m]["y"], oof[m], labels[m]["period"])[0] for m in TRAIN
    ]))
    from bench.rivalideas import cleanup_labels

    td, tm = best[1], best[2]
    rows = []
    test_pred = {}
    base_rec = _load_json(result_path(seed, base))
    base_test = {m: None for m in TEST}
    # Reload test preds. _load_preds already returned them.
    test_src = loaded[1]
    for m in TEST:
        pred = cleanup_labels(test_src[m], labels[m]["period"], td, tm)
        test_pred[m] = pred
        rec_m = _eval_pred(labels[m]["y"], pred, labels[m]["period"])
        rec_m["match"] = m
        rows.append(rec_m)
        base_test[m] = rec_m
        _log("TEST %s %s spell %.3f macro %.3f poss %d/%d" % (
            config_name, m, rec_m["spell"], rec_m["macro"], rec_m["n_pred"], rec_m["n_gt"]))
    pack = {m: labels[m] for m in TEST}
    out = {
        "rev": CODE_REV,
        "config": config_name,
        "seed": int(seed),
        "base": base,
        "train_deg": "v2",
        "test_deg": "v2",
        "t_d": int(td),
        "t_m": int(tm),
        "oof_spell": float(best[0]),
        "oof_spell_raw": raw,
        "grid": table,
        "switch_team": None if not base_rec else base_rec.get("switch_team"),
        "switch_dead": None if not base_rec else base_rec.get("switch_dead"),
        "n_features": None if not base_rec else base_rec.get("n_features"),
        "matches": rows,
        "sanity": _pooled(pack, test_pred, TEST),
        "needs_preds": False,
    }
    _json_dump(result_path(seed, config_name), out)
    _log("done %s seed %d T_d=%d T_m=%d oof %.3f (raw %.3f) test %.3f" % (
        config_name, seed, td, tm, best[0], raw, _mean_spell(out)))
    return out


def run_lowcost(seed):
    """Retune switch costs on a grid that continues below v1's edge. Same emissions."""
    name = "context_lowcost"
    rec = _done(seed, name)
    if rec:
        _log("skip finished %s seed %d" % (name, seed))
        return rec
    loaded = _load_preds(seed, "v1+context")
    if loaded is None or loaded[2] is None:
        _log("skip lowcost seed %d, emissions missing" % seed)
        return None
    _oof, test_saved, em_oof, em_test = loaded
    labels = _load_labels(v2_cache(seed), TRAIN + TEST)
    from bench.rivalideas import choose_costs_grid

    def score_fn(st, sd, emissions, matches):
        spells = []
        for m in matches:
            pred = _decode(emissions[m], labels[m]["period"], st, sd)
            spells.append(float(score(labels[m]["y"], pred, labels[m]["period"])[0]))
        return float(np.mean(spells))

    base = _load_json(result_path(seed, "v1+context"))
    st0, sd0 = float(base["switch_team"]), float(base["switch_dead"])
    # Round-trip: decoding the saved emissions at the original costs must
    # reproduce the saved TEST predictions.
    differ = []
    for m in TEST:
        pred = _decode(em_test[m], labels[m]["period"], st0, sd0)
        differ.append(float(np.mean(pred != test_saved[m])))
    _log("lowcost seed %d emission roundtrip frame-differ %.6f" % (seed, float(np.mean(differ))))
    best = choose_costs_grid(
        lambda st, sd: score_fn(st, sd, em_oof, TRAIN), LOW_TEAM, LOW_DEAD)
    st, sd = best[1], best[2]
    rows = []
    test_pred = {}
    oof_pred = {}
    for m in TEST:
        pred = _decode(em_test[m], labels[m]["period"], st, sd)
        test_pred[m] = pred
        rec_m = _eval_pred(labels[m]["y"], pred, labels[m]["period"])
        rec_m["match"] = m
        rows.append(rec_m)
    for m in TRAIN:
        oof_pred[m] = _decode(em_oof[m], labels[m]["period"], st, sd)
    pack = {m: labels[m] for m in TEST}
    out = {
        "rev": CODE_REV,
        "config": name,
        "seed": int(seed),
        "base": "v1+context",
        "train_deg": "v2",
        "test_deg": "v2",
        "switch_team": float(st),
        "switch_dead": float(sd),
        "oof_spell": float(best[0]),
        "baseline_costs": [st0, sd0],
        "n_features": base.get("n_features"),
        "matches": rows,
        "sanity": _pooled(pack, test_pred, TEST),
        "needs_preds": True,
        "rounds": base.get("rounds"),
    }
    _save_preds(preds_path(seed, name), oof_pred, test_pred)
    _json_dump(result_path(seed, name), out)
    _log("done lowcost seed %d costs %.0f %.0f oof %.3f test %.3f" % (
        seed, st, sd, best[0], _mean_spell(out)))
    return out


def run_mismatch(seed, train_name, test_kind, config_name):
    """Apply a saved model to a different degradation. No refit."""
    import lightgbm as lgb

    rec = _done(seed, config_name)
    if rec:
        _log("skip finished %s seed %d" % (config_name, seed))
        return rec
    side = _load_json(model_stem(seed, train_name) + ".json")
    if not side or not os.path.isfile(model_stem(seed, train_name) + ".txt"):
        _log("skip %s, model %s seed %d missing" % (config_name, train_name, seed))
        return None
    _threads()
    cache = _ensure_cache(test_kind, seed)
    cols = _resolve(_meta(cache)["names"], side["features"])
    pack = _load_pack(cache, TEST, cols, None)
    model = lgb.Booster(model_file=model_stem(seed, train_name) + ".txt")
    prior = np.asarray(side["prior"], np.float64)
    st, sd = float(side["switch_team"]), float(side["switch_dead"])
    rows = []
    test_pred = {}
    for m in TEST:
        em = _emissions(_proba(model, pack[m], None), prior)
        pred = _decode(em, pack[m]["period"], st, sd)
        test_pred[m] = pred
        rec_m = _eval_pred(pack[m]["y"], pred, pack[m]["period"])
        rec_m["match"] = m
        rows.append(rec_m)
        _log("TEST %s %s spell %.3f macro %.3f poss %d/%d" % (
            config_name, m, rec_m["spell"], rec_m["macro"], rec_m["n_pred"], rec_m["n_gt"]))
    out = {
        "rev": CODE_REV,
        "config": config_name,
        "seed": int(seed),
        "base": train_name,
        "train_deg": side.get("train_kind", "v2"),
        "test_deg": test_kind,
        "switch_team": st,
        "switch_dead": sd,
        "oof_spell": None,
        "n_features": len(side["features"]),
        "matches": rows,
        "sanity": _pooled(pack, test_pred, TEST),
        "needs_preds": False,
        "rounds": None,
    }
    _json_dump(result_path(seed, config_name), out)
    _log("done %s seed %d spell %.3f" % (config_name, seed, _mean_spell(out)))
    return out


def run_calibrate():
    path = os.path.join(RUN_DIR, "results", "degrade_v3_calib.json")
    rec = _load_json(path)
    if rec and rec.get("rev") == "v3a" and rec.get("live_mask_match") is True:
        _log("degrade calib cached dead_vis %.3f" % rec["dead_visible"])
        return rec
    _log("calibrating degrade v3 on %d matches" % len(ORDER))
    from bench.degrade_v3 import aggregate

    rec = aggregate(ORDER, seed=0)
    rec["rev"] = "v3a"
    _json_dump(path, rec)
    _log(
        "degrade v3 dead_vis %.3f live_vis %.3f all_or_nothing %.3f lost_at_start %.3f "
        "live_mask_match %s violations %.0f"
        % (
            rec["dead_visible"], rec["live_visible"], rec["all_or_nothing"], rec["lost_at_start"],
            rec["live_mask_match"], rec["dropout_violations"],
        )
    )
    if not rec["live_mask_match"]:
        raise SystemExit("v3 live-ball mask does not match v2")
    if rec["dropout_violations"] > 0:
        raise SystemExit("v3 dropout min-length violations")
    return rec


def _gain(rec, base_rec):
    if rec is None or base_rec is None:
        return None
    if rec.get("test_deg") != base_rec.get("test_deg"):
        return None
    return _mean_spell(rec) - _mean_spell(base_rec)


def _load_rows():
    folder = os.path.join(RUN_DIR, "results")
    rows = []
    if not os.path.isdir(folder):
        return rows
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json") or name.startswith("degrade"):
            continue
        rec = _load_json(os.path.join(folder, name))
        if rec and rec.get("config") and isinstance(rec.get("matches"), list) and rec["matches"] and isinstance(rec["matches"][0], dict):
            rows.append(rec)
    return rows


def _fmt_table(rows, seeds):
    by = {}
    for r in rows:
        by.setdefault(r["config"], []).append(r)
    lines = []
    lines.append("| config | test | spellF1 | sd | macroF1 | poss pred | poss gt | per-seed spell | per-seed gain vs v1+context | adopt |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|---|")
    order = ["v1", "v1+context", "r1_cleanup", "context_lowcost",
             "r2_w3_r1", "r2_w2_r1", "r2_w5_r1", "r2_w3_r2", "r4_lookahead",
             "r3_matched", "r3_v2_on_v3", "r3_v3_on_v2"]
    extra = [k for k in by if k not in order]
    for name in order + extra:
        if name not in by:
            continue
        recs = sorted(by[name], key=lambda r: r["seed"])
        spells, macros, preds, gts = [], [], [], []
        seed_spell, seed_gain, gains_ok = [], [], []
        test = recs[0].get("test_deg", "v2")
        for r in recs:
            ss = [m["spell"] for m in r["matches"]]
            spells.extend(ss)
            macros.extend(m["macro"] for m in r["matches"])
            preds.extend(m["n_pred"] for m in r["matches"])
            gts.extend(m["n_gt"] for m in r["matches"])
            sm = float(np.mean(ss))
            seed_spell.append(sm)
            base = _done(r["seed"], "v1+context")
            g = _gain(r, base)
            if g is not None and name != "v1+context":
                seed_gain.append(g)
                gains_ok.append(g >= 0.03 - 1e-12)
            elif name != "v1+context" and name != "v1":
                seed_gain.append(None)
        have = {r["seed"] for r in recs}
        adopt = ""
        if name not in ("v1", "v1+context") and seed_gain and all(g is not None for g in seed_gain):
            if have == set(seeds) and len(seed_gain) == len(seeds):
                adopt = "yes" if all(gains_ok) else "no"
            else:
                adopt = "partial"
        elif name not in ("v1", "v1+context"):
            adopt = "n/a" if any(g is None for g in seed_gain) else ""
        gain_txt = " ".join("" if g is None else "%+.3f" % g for g in seed_gain)
        seed_txt = " ".join("%.3f" % s for s in seed_spell)
        lines.append("| %s | %s | %.3f | %.3f | %.3f | %.0f | %.0f | %s | %s | %s |" % (
            name, test, float(np.mean(spells)), float(np.std(spells)), float(np.mean(macros)),
            float(np.mean(preds)), float(np.mean(gts)), seed_txt, gain_txt, adopt))
    return "\n".join(lines)


def _cell(v, spec):
    if v is None:
        return ""
    return spec % float(v)


def _sanity_table(rows):
    from collections import OrderedDict

    lines = ["| config | seed | play share | play min | poss/90 | D p25 | D p50 | D p90 | poss<4s | home share |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    true = _load_json(os.path.join(RUN_DIR, "results", "true_test.json"))
    if true and true.get("sanity"):
        s = true["sanity"]
        lines.append("| TRUE test |  | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            _cell(s["play_share"], "%.3f"), _cell(s["play_min"], "%.2f"), _cell(s["poss_per_90"], "%.1f"),
            _cell(s["d_p25"], "%.2f"), _cell(s["d_p50"], "%.2f"), _cell(s["d_p90"], "%.2f"),
            _cell(s["poss_under_4s"], "%.3f"), _cell(s["home_share"], "%.3f")))
    seen = OrderedDict()
    for r in rows:
        if not r.get("sanity"):
            continue
        seen.setdefault(r["config"], []).append(r)
    for name, recs in seen.items():
        for r in sorted(recs, key=lambda z: z["seed"]):
            s = r["sanity"]
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                name, r["seed"],
                _cell(s["play_share"], "%.3f"), _cell(s["play_min"], "%.2f"), _cell(s["poss_per_90"], "%.1f"),
                _cell(s["d_p25"], "%.2f"), _cell(s["d_p50"], "%.2f"), _cell(s["d_p90"], "%.2f"),
                _cell(s["poss_under_4s"], "%.3f"), _cell(s["home_share"], "%.3f")))
    return "\n".join(lines)


def _fmt(v, nd=3):
    if v is None:
        return ""
    return ("%." + str(nd) + "f") % float(v)


def write_final(seeds, rounds):
    rows = _load_rows()
    calib = _load_json(os.path.join(RUN_DIR, "results", "degrade_v3_calib.json")) or {}
    body = []
    body.append("# Rival ideas on v1+context")
    body.append("")
    body.append("Bench protocol of `bench.run` / features3. Official `possession_metrics.evaluate`. "
                "TRAIN = %s. TEST = %s. Requested seeds = %s. LightGBM rounds = %d, stride 5, "
                "one thread, deterministic, v1 hyperparameters. Code rev %s."
                % (", ".join(TRAIN), ", ".join(TEST), ", ".join(str(s) for s in seeds), rounds, CODE_REV))
    body.append("")
    body.append("Switch costs are chosen by leave-one-match-out on TRAIN. Each TEST match is scored once. "
                "sd is the population standard deviation over match x seed rows. A per-seed spellF1 is the "
                "mean of the three TEST matches. Adopt means the gain versus v1+context on that same seed, "
                "on the same test degradation, is at least +0.03 on every requested seed.")
    body.append("")
    body.append("## Main table")
    body.append("")
    body.append(_fmt_table(rows, seeds))
    body.append("")
    body.append("## Costs and cleanup chosen on TRAIN")
    body.append("")
    body.append("| config | seed | oof spellF1 | switch team | switch dead | T_d | T_m | feats | seconds |")
    body.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda z: (z["config"], z["seed"])):
        body.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            r["config"], r["seed"], _fmt(r.get("oof_spell")),
            _fmt(r.get("switch_team"), 0), _fmt(r.get("switch_dead"), 0),
            "" if r.get("t_d") is None else str(r.get("t_d")),
            "" if r.get("t_m") is None else str(r.get("t_m")),
            "" if r.get("n_features") is None else str(r.get("n_features")),
            "" if r.get("seconds") is None else "%.0f" % r["seconds"]))
    body.append("")
    body.append("## Per match")
    body.append("")
    body.append("| config | seed | match | spellF1 | macroF1 | poss pred | poss gt | home-share err |")
    body.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda z: (z["config"], z["seed"])):
        for m in r["matches"]:
            body.append("| %s | %d | %s | %.3f | %.3f | %d | %d | %.3f |" % (
                r["config"], r["seed"], m["match"], m["spell"], m["macro"], m["n_pred"], m["n_gt"], m["share_err"]))
    body.append("")
    body.append("## Label-free sanity (R5)")
    body.append("")
    body.append("Pooled over the three TEST matches. TRUE uses the clean DFL labels (the same for every seed). "
                "play share is the fraction of frames labelled H or A. poss/90 is possessions per 90 minutes of "
                "tracking clock, dead frames included. D columns are run-length quantiles in seconds. "
                "poss<4s is the share of possessions shorter than 4 s. home share is P(H | not D).")
    body.append("")
    body.append(_sanity_table(rows))
    body.append("")
    body.append("## Degrade v3 calibration (seed 0, mean over 7 matches)")
    body.append("")
    if calib:
        body.append("| dead ball visible | live ball visible | all-or-nothing runs | lost at start | live mask = v2 | dropout violations |")
        body.append("|---:|---:|---:|---:|---|---:|")
        body.append("| %.3f | %.3f | %.3f | %.3f | %s | %.0f |" % (
            calib.get("dead_visible") or 0, calib.get("live_visible") or 0, calib.get("all_or_nothing") or 0,
            calib.get("lost_at_start") or 0, calib.get("live_mask_match"), calib.get("dropout_violations") or 0))
    else:
        body.append("Not run.")
    body.append("")
    body.append("## Reproduce")
    body.append("")
    body.append("From the repo root, with `bench/cache/npz` populated (this worktree junctions it to the SOCBAC cache) "
                "and the features3 seed-0 matrix cache left in place:")
    body.append("")
    body.append("```")
    body.append(".venv\\Scripts\\python.exe -m bench.run_rivalideas --seeds %s" % " ".join(str(s) for s in seeds))
    body.append("```")
    body.append("")
    body.append("Unit checks: `.venv\\Scripts\\python.exe -m bench.test_rivalideas`.")
    body.append("")
    body.append("## What was compared")
    body.append("")
    body.append("- **v1** and **v1+context** repeat the features3 protocol. Seed 0 reads the existing f3a cache. Later seeds rebuild context columns only, with `bench.degrade` and `1000 * seed + index`.")
    body.append("- **r1_cleanup** forbids D runs shorter than T_d seconds and merges same-team possessions separated by one D or opposing run shorter than T_m. T_d and T_m are in {0, 1, 2, 3}. 0 turns that rule off. A run of exactly T seconds is kept. Periods are cleaned separately. A short D with two different neighbours goes to the longer neighbour; a tie goes to the earlier run. T is chosen on TRAIN out-of-fold predictions at the costs already chosen for v1+context. A tie keeps the smaller thresholds, so (0, 0) wins when nothing helps.")
    body.append("- **context_lowcost** is the better-design check: v1+context's chosen team cost sat on the bottom of the grid (5) while possessions were under-counted (287 vs 397). The same saved emissions are re-decoded on team costs {1, 2, 3, 5, ...} and dead costs {5, 10, ...}. No refit.")
    body.append("- **r2_w\*** multiplies the LightGBM sample weight of frames within the radius of a true label change (25 frames = 1 s, 50 = 2 s). The class prior used in the emissions stays the unweighted prior.")
    body.append("- **r4_lookahead** appends team-shape channels sampled at t+25, t+50 and t+125 frames, plus the contrast (t+h) minus (t-h), edge-held inside the period. Channels are the attack-frame context features centx, centatt, width, depth, line, opp, hull and vattack, home and away.")
    body.append("- **r3_matched** trains and tests on degrade v3. **r3_v2_on_v3** applies the v2 v1+context model to v3 test features. **r3_v3_on_v2** applies the v3-trained model to v2 test features. Only the last of these is a gain on the same test distribution as v1+context.")
    body.append("- **R6** combines pieces that each cleared +0.03 on every requested seed. With fewer than two such pieces there is nothing to combine.")
    body.append("")
    body.append("## Negative results and adoption")
    body.append("")
    adopted = []
    negative = []
    for rname in sorted({r["config"] for r in rows}):
        if rname in ("v1", "v1+context"):
            continue
        gains = []
        complete = True
        for seed in seeds:
            rec = _done(seed, rname)
            base = _done(seed, "v1+context")
            g = _gain(rec, base)
            if rec is None or g is None:
                complete = False
                continue
            gains.append((seed, g))
        if complete and gains and all(g >= 0.03 - 1e-12 for _s, g in gains):
            adopted.append(rname)
        elif gains and all(g <= 0.0 for _s, g in gains):
            negative.append("%s: gain %s" % (rname, " ".join("%+.3f" % g for _s, g in gains)))
        elif gains:
            negative.append("%s: not adopted (%s)" % (rname, " ".join("s%d %+.3f" % (s, g) for s, g in gains)))
    if adopted:
        body.append("Adopted: " + ", ".join(adopted) + ".")
    else:
        body.append("No variant gained +0.03 spellF1 over v1+context on every requested seed.")
    if negative:
        body.append("")
        for line in negative:
            body.append("- " + line)
    body.append("")
    body.append("## Caveats")
    body.append("")
    body.append("- The public 5-minute window is not read and is not used to choose anything. No column reads `alive` or `owner` except inside the degrader, which v2 already uses to draw gaps.")
    body.append("- Look-ahead reads future positions, the same offline setting as v1's `gap_fwd` and centred windows. It does not read the future of the owner flag.")
    body.append("- Costs and T are not searched jointly. Cleanup is applied after the v1 cost grid. The low-cost grid is a separate decode of the same probabilities.")
    body.append("- v3 keeps v2's live-ball hidden mask bitwise (same seed stream). It does not move ball loss into the live frames before a stoppage. Player dropouts are v2's bursts extended to at least 2.04 s. Tracks that themselves last less than 2 s can still show a shorter observed gap.")
    body.append("- Dead-ball keep rates (0.80 / 0.35 / 0.18 by run length) were set from the DFL dead-run length mix so the frame-weighted visibility is about 25%, then checked on the seven matches. They were not fit to spellF1.")
    body.append("- Seed 0 v2 features are the features3 f3a cache. Rebuilding them would spend the same degradation seed but is not repeated here.")
    body.append("- Team symmetry, stride 5 and 300 trees are v1's. Boundary weights do not change which rows are sampled.")
    body.append("- Adoption on a v3 test set is not comparable to 0.529. Use r3_v2_on_v3 as the reference for that bench, and r3_v3_on_v2 for the transfer back onto v2.")
    if SKIPPED:
        body.append("- Stopped early for budget: " + "; ".join(SKIPPED))
    notes_path = os.path.join(RUN_DIR, "logs", "notes.txt")
    if os.path.isfile(notes_path):
        with open(notes_path, encoding="utf-8") as f:
            extra = f.read().strip()
        if extra:
            body.append("- " + extra.replace("\n", " "))
    text = "\n".join(body) + "\n"
    path = os.path.join(RUN_DIR, "FINAL.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _true_sanity(seed):
    path = os.path.join(RUN_DIR, "results", "true_test.json")
    if os.path.isfile(path):
        return
    labels = _load_labels(v2_cache(seed), TEST)
    from bench.rivalideas import pool_sanity
    sanity = pool_sanity([labels[m]["y"] for m in TEST], [labels[m]["period"] for m in TEST])
    _json_dump(path, {"sanity": sanity, "matches": list(TEST)})
    _log("true test play_share %.3f poss/90 %.1f home %.3f" % (
        sanity["play_share"], sanity["poss_per_90"], sanity["home_share"]))


def _first_gain(name, seed, baseline="v1+context"):
    return _gain(_done(seed, name), _done(seed, baseline))


def _r6(seeds):
    """Combine winners. A winner cleared +0.03 on every requested seed."""
    winners = []
    for name in ("r1_cleanup", "context_lowcost", "r2_w3_r1", "r2_w2_r1", "r2_w5_r1",
                 "r2_w3_r2", "r4_lookahead", "r3_v3_on_v2"):
        ok = True
        for seed in seeds:
            g = _first_gain(name, seed)
            if g is None or g < 0.03 - 1e-12:
                ok = False
                break
        if ok:
            winners.append(name)
    _log("R6 winners: %s" % (winners or "none"))
    note = os.path.join(RUN_DIR, "logs", "notes.txt")
    os.makedirs(os.path.dirname(note), exist_ok=True)
    with open(note, "a", encoding="utf-8") as f:
        f.write("R6 winners: %s\n" % (winners or "none"))
    if len(winners) < 2:
        return
    # Cheap combination: cleanup on top of another winning model's predictions.
    from_models = [w for w in winners if w != "r1_cleanup"]
    if "r1_cleanup" in winners:
        for base in from_models:
            if not allow(3, "r6 cleanup on %s" % base):
                return
            for seed in seeds:
                run_cleanup(seed, base, "r6_%s_cleanup" % _safe(base))
    # Weight + lookahead needs a new fit. Only when both families won.
    weights = [w for w in winners if w.startswith("r2_")]
    if "r4_lookahead" in winners and weights:
        spec = weights[0]
        mult_radius = {s[0]: (s[1], s[2]) for s in R2_SPECS}[spec]
        for seed in seeds:
            if not allow(18, "r6 %s + lookahead seed %d" % (spec, seed)):
                return
            train_config(seed, "r6_%s_lookahead" % _safe(spec), ("context",), 300, 5,
                         weight=mult_radius, use_la=True, save_emissions=False)


def main(argv=None):
    import argparse

    global BUDGET_S, T0, RUN_SEEDS, RUN_ROUNDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--budget-min", type=float, default=100)
    ap.add_argument("--self-test-only", action="store_true")
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--calibrate-degrade", action="store_true")
    ap.add_argument("--promote-bar", type=float, default=0.01,
                    help="minimum seed-0 gain required before repeating a variant on later seeds")
    args = ap.parse_args(argv)
    RUN_SEEDS = list(args.seeds)
    RUN_ROUNDS = int(args.rounds)
    BUDGET_S = float(args.budget_min) * 60.0
    T0 = time.time()
    os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "results"), exist_ok=True)
    persist_code()
    _log("run_rivalideas rev %s seeds %s rounds %d budget %.0f min" % (
        CODE_REV, args.seeds, args.rounds, args.budget_min))
    if not args.skip_self_test:
        from bench.rivalideas import self_test
        self_test()
        _log("self-test passed")
        if args.self_test_only:
            persist_code()
            return
    if args.calibrate_degrade:
        run_calibrate()
        persist_code()
        write_final(args.seeds, args.rounds)
        return

    run_calibrate()

    def _promising(name, bar):
        g = _first_gain(name, args.seeds[0])
        return g is not None and g >= bar

    def run_repro(seed):
        cache_ready = _cache_ready(v2_cache(seed))
        if not allow(12 if cache_ready else 22, "v1 seed %d" % seed):
            return False
        train_config(seed, "v1", (), args.rounds, args.stride, save_emissions=True)
        write_final(args.seeds, args.rounds)
        persist_code()
        if not allow(22, "v1+context seed %d" % seed):
            return False
        train_config(seed, "v1+context", ("context",), args.rounds, args.stride, save_emissions=True)
        _true_sanity(seed)
        if allow(3, "r1 seed %d" % seed):
            run_cleanup(seed, "v1+context", "r1_cleanup")
        if allow(6, "lowcost seed %d" % seed):
            run_lowcost(seed)
        write_final(args.seeds, args.rounds)
        persist_code()
        return True

    def run_r2(spec, mult, radius, seed):
        if seed != args.seeds[0] and not _promising(spec, args.promote_bar):
            _log("skip %s seed %d; seed 0 gain below %.3f" % (spec, seed, args.promote_bar))
            return
        if not allow(22, "%s seed %d" % (spec, seed)):
            return
        train_config(seed, spec, ("context",), args.rounds, args.stride, weight=(mult, radius))
        write_final(args.seeds, args.rounds)
        persist_code()

    def run_r4(seed):
        if seed != args.seeds[0] and not _promising("r4_lookahead", args.promote_bar):
            _log("skip r4 seed %d; seed 0 gain below %.3f" % (seed, args.promote_bar))
            return
        if not allow(18, "r4 seed %d" % seed):
            return
        train_config(seed, "r4_lookahead", ("context",), args.rounds, args.stride, use_la=True)
        write_final(args.seeds, args.rounds)
        persist_code()

    def run_r3(seed):
        if seed != args.seeds[0]:
            g_back = _first_gain("r3_v3_on_v2", args.seeds[0])
            g_match = _gain(_done(args.seeds[0], "r3_matched"), _done(args.seeds[0], "r3_v2_on_v3"))
            if not ((g_back is not None and g_back >= args.promote_bar) or
                    (g_match is not None and g_match >= args.promote_bar)):
                _log("skip r3 seed %d; seed 0 gains below %.3f" % (seed, args.promote_bar))
                return
        if not allow(30, "r3 seed %d" % seed):
            return
        train_config(seed, "r3_matched", ("context",), args.rounds, args.stride,
                     train_kind="v3", test_kind="v3", save_emissions=False)
        run_mismatch(seed, "v1+context", "v3", "r3_v2_on_v3")
        run_mismatch(seed, "r3_matched", "v2", "r3_v3_on_v2")
        write_final(args.seeds, args.rounds)
        persist_code()

    # Seed 0 of every idea first. Later seeds are a variance check, and only
    # for variants whose seed-0 gain is at least promote_bar. Finished jobs
    # are skipped, so a restart continues at the next missing piece.
    if not run_repro(args.seeds[0]):
        path = write_final(args.seeds, args.rounds)
        persist_code()
        _log("wrote %s" % path)
        return
    run_r2("r2_w3_r1", 3.0, 25, args.seeds[0])
    run_r4(args.seeds[0])
    run_r3(args.seeds[0])
    for spec, mult, radius in R2_SPECS[1:]:
        run_r2(spec, mult, radius, args.seeds[0])
    for seed in args.seeds[1:]:
        if not run_repro(seed):
            break
        run_r2("r2_w3_r1", 3.0, 25, seed)
        run_r4(seed)
        run_r3(seed)
        for spec, mult, radius in R2_SPECS[1:]:
            run_r2(spec, mult, radius, seed)

    if allow(5, "r6"):
        _r6(args.seeds)
    path = write_final(args.seeds, args.rounds)
    persist_code()
    _log("wrote %s" % path)


if __name__ == "__main__":
    main()
