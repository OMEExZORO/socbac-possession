"""Search and stack models on v2 = v1 + context + lang_dead.

    python -m bench.run_boost --seeds 0 1 2

TRAIN is scored leave-one-match-out. TEST is scored once per seed. Prints
aggregates only. Long fits run in a fresh process so the allocator can give
the memory back.
"""

import json
import os
import shutil
import time
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
PERSIST = r"D:\Barca_Hackathon\research\runs\boost"
SEED0_CACHE = r"D:\Barca_Hackathon\research\runs\features3\cache\f3a\seed0"
TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
TEST = ["J03WPY", "J03WQQ", "J03WR9"]
ORDER = TRAIN + TEST

# Seed-0 bench published by the features3 run. Used only as a reproduction check.
REF = {
    "v1": {"J03WPY": 0.392, "J03WQQ": 0.376, "J03WR9": 0.397},
    "v2": {"J03WPY": 0.546, "J03WQQ": 0.589, "J03WR9": 0.587},
}


class TooSlow(RuntimeError):
    pass


def _threads(n):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def _log(path, msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    print(line, flush=True)


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
    os.replace(tmp, path)


def _load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _result_path(method, seed):
    return os.path.join(PERSIST, "results", "s%d_%s.json" % (seed, method))


def _proba_dir(method, seed):
    return os.path.join(PERSIST, "proba", "s%d" % seed, method)


def _cache_for(seed):
    if int(seed) == 0 and os.path.isfile(os.path.join(SEED0_CACHE, "meta.json")):
        return SEED0_CACHE
    return os.path.join(PERSIST, "cache", "f3a", "seed%d" % int(seed))


def persist_code():
    os.makedirs(PERSIST, exist_ok=True)
    for name in ("feats3.py", "boost.py", "run_boost.py"):
        src = os.path.join(HERE, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(PERSIST, name))


def add_note(text):
    path = os.path.join(PERSIST, "notes.json")
    notes = _load_json(path) or []
    if text not in notes:
        notes.append(text)
        _atomic_json(path, notes)


def _labels(cache, match):
    import numpy as np
    y = np.load(os.path.join(cache, match + ".y.npy"))
    period = np.load(os.path.join(cache, match + ".period.npy"))
    return y, period


def _done_result(path):
    rec = _load_json(path)
    if not rec or rec.get("error") or not rec.get("matches"):
        return None
    return rec


def worker_build_seed(job):
    _threads(2)
    import gc
    from bench.boost import execute_build
    done = []
    for one in job["matches"]:
        if all(os.path.isfile(one[k]) for k in ("x_path", "xs_path", "y_path", "period_path")):
            continue
        _log(job["log"], "building %s seed %s" % (one["match"], one["seed"]))
        rec = execute_build(one)
        done.append(rec)
        _log(job["log"], "cached %s %d frames %d feats in %.0fs" % (
            rec["match"], rec["frames"], rec["feats"], rec["seconds"]))
        gc.collect()
    return done


def worker_protocol(job):
    _threads(1)
    try:
        return execute_protocol(job)
    except Exception:
        _log(job.get("log"), "FAILED %s seed %s\n%s" % (job.get("method"), job.get("seed"), traceback.format_exc()))
        raise


def worker_screen(job):
    _threads(1)
    try:
        return execute_screen(job)
    except Exception:
        _log(job.get("log"), "FAILED screen\n%s" % traceback.format_exc())
        raise


def worker_b6(job):
    _threads(1)
    try:
        return execute_b6(job)
    except Exception:
        _log(job.get("log"), "FAILED b6\n%s" % traceback.format_exc())
        raise


def worker_b9(job):
    _threads(1)
    try:
        return execute_b9(job)
    except Exception:
        _log(job.get("log"), "FAILED b9\n%s" % traceback.format_exc())
        raise


def _run_child(fn, job):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx, max_tasks_per_child=1) as ex:
        return ex.submit(fn, job).result()


def _schema():
    from bench.feats3 import ALL_GROUPS, _synthetic_shape, build_features
    _, names, ball, sub, kind = build_features(_synthetic_shape(), ALL_GROUPS)
    return list(names), [int(bool(v)) for v in ball], list(sub), list(kind)


def _write_meta(cache, names, ball, sub, kind):
    import numpy as np
    matches = {}
    for m in ORDER:
        y = np.load(os.path.join(cache, m + ".y.npy"), mmap_mode="r")
        matches[m] = {"frames": int(y.shape[0])}
        del y
    meta = {
        "rev": "f3a",
        "names": names,
        "ball": ball,
        "sub": sub,
        "kind": kind,
        "feats": len(names),
        "matches": matches,
    }
    _atomic_json(os.path.join(cache, "meta.json"), meta)
    return meta


def ensure_cache(seed, log):
    cache = _cache_for(seed)
    meta_path = os.path.join(cache, "meta.json")
    if os.path.isfile(meta_path) and all(
            os.path.isfile(os.path.join(cache, m + ".x.npy")) for m in ORDER):
        _log(log, "cache hit seed %d %s" % (seed, cache))
        return cache
    os.makedirs(cache, exist_ok=True)
    matches = []
    for i, m in enumerate(ORDER):
        stem = os.path.join(cache, m)
        matches.append({
            "seed": int(seed),
            "index": i,
            "match": m,
            "x_path": stem + ".x.npy",
            "xs_path": stem + ".xs.npy",
            "y_path": stem + ".y.npy",
            "period_path": stem + ".period.npy",
            "gap_path": stem + ".gap.npy",
            "vis_path": stem + ".vis.npy",
        })
    _log(log, "building feature cache seed %d" % seed)
    _run_child(worker_build_seed, {"matches": matches, "log": log})
    names, ball, sub, kind = _schema()
    _write_meta(cache, names, ball, sub, kind)
    _log(log, "cache seed %d ready, %d columns" % (seed, len(names)))
    return cache


def _groups_cols(meta, groups):
    import numpy as np
    from bench.feats3 import column_index
    ball = np.asarray(meta["ball"], dtype=bool)
    cols, names = column_index(meta["names"], meta["sub"], ball, groups)
    flags = [bool(ball[int(i)]) for i in cols]
    return [int(c) for c in cols], list(names), flags


def execute_protocol(job):
    """LOMO fits, cost tune on TRAIN, one refit, TEST scored once. Checkpoints each fold."""
    import gc
    import numpy as np
    from bench import boost

    log = job["log"]
    method = job["method"]
    seed = int(job["seed"])
    cache = job["cache"]
    train = list(job["train"])
    test = list(job["test"])
    proba_dir = job["proba_dir"]
    os.makedirs(proba_dir, exist_ok=True)
    state_path = os.path.join(proba_dir, "state.json")
    state = _load_json(state_path) or {"folds": {}}
    kind = job["kind"]
    cols = job["cols"]
    stride = int(job["stride"])
    scheme = job["weight"]
    params = job["params"]
    rounds = int(job["rounds"])
    limit = job.get("max_fold_seconds")
    t_all = time.time()

    for i, held in enumerate(train):
        path = os.path.join(proba_dir, held + ".npy")
        if held in state["folds"] and os.path.isfile(path):
            _log(log, "resume %s seed %d fold %s" % (method, seed, held))
            continue
        rest = [m for m in train if m != held]
        t0 = time.time()
        _log(log, "fit %s seed %d without %s rounds %d stride %d weight %s" % (
            method, seed, held, rounds, stride, scheme))
        info = boost._fit_and_predict(kind, cache, rest, [held], cols, stride, scheme, params, rounds)
        for m, arr in info["probas"]:
            boost.atomic_npy(os.path.join(proba_dir, m + ".npy"), arr)
        sec = round(time.time() - t0, 1)
        state["folds"][held] = {
            "prior": info["prior"], "gain": info["gain"], "seconds": sec, "rows": info["rows"],
        }
        _atomic_json(state_path, state)
        _log(log, "fit %s seed %d held-out %s rows %d feats %d in %.0fs" % (
            method, seed, held, info["rows"], info["feats"], sec))
        del info
        gc.collect()
        if limit and i == 0 and sec > float(limit):
            raise TooSlow("%s first fold %.0fs over limit %.0fs" % (method, sec, float(limit)))

    need_final = "final" not in state or any(
        not os.path.isfile(os.path.join(proba_dir, m + ".npy")) for m in test)
    if need_final:
        t0 = time.time()
        _log(log, "fit %s seed %d on all TRAIN" % (method, seed))
        info = boost._fit_and_predict(kind, cache, train, test, cols, stride, scheme, params, rounds)
        for m, arr in info["probas"]:
            boost.atomic_npy(os.path.join(proba_dir, m + ".npy"), arr)
        state["final"] = {
            "prior": info["prior"], "seconds": round(time.time() - t0, 1),
            "rows": info["rows"], "feats": info["feats"],
        }
        _atomic_json(state_path, state)
        _log(log, "final %s seed %d rows %d in %.0fs" % (
            method, seed, info["rows"], state["final"]["seconds"]))
        del info
        gc.collect()

    gains = np.vstack([np.asarray(state["folds"][m]["gain"], dtype=np.float64) for m in train])
    gain = gains.mean(axis=0)
    np.save(os.path.join(proba_dir, "gain.npy"), gain.astype(np.float32))
    ems, ys, periods = [], [], []
    for m in train:
        proba = np.load(os.path.join(proba_dir, m + ".npy"))
        y, period = _labels(cache, m)
        ems.append(boost.emissions(proba, state["folds"][m]["prior"]))
        ys.append(y)
        periods.append(period)
    oof_spell, st, sd, _ = boost.tune_costs(
        ems, ys, periods, job["grid_team"], job["grid_dead"])
    _log(log, "oof %s seed %d spell %.3f costs %.0f %.0f" % (method, seed, oof_spell, st, sd))
    rows = []
    test_prior = state["final"]["prior"]
    for m in test:
        proba = np.load(os.path.join(proba_dir, m + ".npy"))
        y, period = _labels(cache, m)
        pred = boost.decode_periods(boost.emissions(proba, test_prior), period, st, sd)
        rec = boost.frame_metrics(y, pred, period)
        rec["match"] = m
        rows.append(rec)
        _log(log, "TEST %s seed %d %s spell %.3f macro %.3f poss %d/%d share %.3f" % (
            method, seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"], rec["share_err"]))
    names = list(job.get("feat_names") or [])
    order = np.argsort(-gain, kind="mergesort")
    top = [names[int(i)] for i in order[:10]] if names else []
    result = {
        "method": method,
        "family": job.get("family", method),
        "seed": seed,
        "kind": kind,
        "weight": scheme,
        "rounds": rounds,
        "stride": stride,
        "n_features": int(gain.shape[0]),
        "oof_spell": float(oof_spell),
        "switch_team": st,
        "switch_dead": sd,
        "seconds": round(time.time() - t_all, 1),
        "fit_seconds": [state["folds"][m]["seconds"] for m in train] + [state["final"]["seconds"]],
        "top10": top,
        "matches": rows,
        "prior_test": test_prior,
        "params": {k: (float(v) if isinstance(v, float) else v) for k, v in params.items()
                   if isinstance(v, (int, float, str, bool))},
    }
    _atomic_json(job["result"], result)
    _log(log, "done %s seed %d in %.0fs" % (method, seed, result["seconds"]))
    return result


def _one_held_spell(cache, train, held, cols, stride, scheme, kind, params, rounds, grid_t, grid_d):
    import numpy as np
    from bench import boost
    rest = [m for m in train if m != held]
    info = boost._fit_and_predict(kind, cache, rest, [held], cols, stride, scheme, params, rounds)
    proba = info["probas"][0][1]
    y, period = _labels(cache, held)
    em = boost.emissions(proba, info["prior"])
    spell, st, sd, _ = boost.tune_costs([em], [y], [period], grid_t, grid_d)
    sec_rows = info["rows"]
    del info, proba
    return float(spell), float(st), float(sd), int(sec_rows)


def execute_screen(job):
    """Rank hyperparameters on one held-out TRAIN match, then confirm the leader on a second."""
    path = job["out"]
    done = _load_json(path) or {}
    train = list(job["train"])
    held_a, held_b = train[0], train[1]
    log = job["log"]
    for spec in job["specs"]:
        name = spec["name"]
        if name in done and "spell_a" in done[name]:
            _log(log, "screen resume %s spell_a %.3f" % (name, done[name]["spell_a"]))
            continue
        if job.get("deadline") and time.time() > float(job["deadline"]):
            _log(log, "screen stopped at the time budget")
            break
        t0 = time.time()
        params = _params_for(spec)
        try:
            spell, st, sd, rows = _one_held_spell(
                job["cache"], train, held_a, job["cols"], int(spec["stride"]), spec["weight"],
                "lgbm", params, int(spec["rounds"]), job["grid_team"], job["grid_dead"])
        except Exception as exc:
            _log(log, "screen %s failed: %s" % (name, exc.__class__.__name__))
            done[name] = {"spell_a": -1.0, "error": exc.__class__.__name__, "seconds": round(time.time() - t0, 1)}
            _atomic_json(path, done)
            continue
        done[name] = {
            "spell_a": spell, "cost_team": st, "cost_dead": sd, "rows": rows,
            "seconds": round(time.time() - t0, 1),
            "rounds": int(spec["rounds"]), "stride": int(spec["stride"]), "weight": spec["weight"],
        }
        _atomic_json(path, done)
        _log(log, "screen %s held %s spell %.3f costs %.0f %.0f in %.0fs" % (
            name, held_a, spell, st, sd, done[name]["seconds"]))
    if "base" not in done or "spell_a" not in done["base"]:
        _atomic_json(path, done)
        return done
    base = done["base"]["spell_a"]
    ranked = sorted(
        ((done[s["name"]]["spell_a"], s) for s in job["specs"]
         if s["name"] != "base" and s["name"] in done and "spell_a" in done[s["name"]]),
        key=lambda z: (-z[0], z[1]["name"]))
    winner = None
    for spell, spec in ranked:
        # rounds200 is trained longer than the screen base, so it must clear a wider gap.
        margin = 0.01 if spec["name"] == "rounds200" else 0.005
        if spell >= base + margin:
            winner = spec
            break
    done["_base_spell_a"] = base
    done["_winner"] = None if winner is None else winner["name"]
    if winner is not None and job.get("deadline") and time.time() + 400 > float(job["deadline"]):
        _log(log, "screen leader %s; second-match check skipped for time" % winner["name"])
        done["_winner_status"] = "unchecked"
    elif winner is not None:
        for spec, key in ((winner, winner["name"]), (_spec_by_name(job["specs"], "base"), "base")):
            if "spell_b" in done.get(key, {}):
                continue
            t0 = time.time()
            params = _params_for(spec)
            spell, _, _, _ = _one_held_spell(
                job["cache"], train, held_b, job["cols"], int(spec["stride"]), spec["weight"],
                "lgbm", params, int(spec["rounds"]), job["grid_team"], job["grid_dead"])
            done[key]["spell_b"] = spell
            done[key]["seconds_b"] = round(time.time() - t0, 1)
            _log(log, "screen %s held %s spell %.3f in %.0fs" % (key, held_b, spell, done[key]["seconds_b"]))
            _atomic_json(path, done)
        if done[winner["name"]].get("spell_b", -1) + 1e-9 < done["base"].get("spell_b", 0):
            _log(log, "screen leader %s lost the second match; not confirmed" % winner["name"])
            done["_winner"] = None
            done["_winner_status"] = "lost_second_match"
        else:
            done["_winner_status"] = "confirm"
            _log(log, "screen confirm %s" % winner["name"])
    else:
        done["_winner_status"] = "none"
        _log(log, "screen: no spec beat the 100-round base by the required margin")
    _atomic_json(path, done)
    return done


def _spec_by_name(specs, name):
    for s in specs:
        if s["name"] == name:
            return s
    raise KeyError(name)


def _base_spec(**kw):
    row = dict(num_leaves=63, min_child_samples=200, feature_fraction=0.8, learning_rate=0.05,
               rounds=100, stride=5, weight="none", confirm_rounds=300, confirm_stride=5)
    row.update(kw)
    return row


def screen_specs(full=True):
    specs = [_base_spec(name="base")]
    for v in (15, 31, 127):
        specs.append(_base_spec(name="leaves_%d" % v, num_leaves=v))
    if full:
        for v in (50, 500):
            specs.append(_base_spec(name="min_child_%d" % v, min_child_samples=v))
        for v in (0.3, 0.5):
            specs.append(_base_spec(name="ff_%s" % v, feature_fraction=v))
        specs.append(_base_spec(name="lr_0.03", learning_rate=0.03, confirm_rounds=400))
    for w in ("balanced", "alive2", "alive4", "dead2", "boundary"):
        specs.append(_base_spec(name="w_%s" % w, weight=w))
    specs.append(_base_spec(name="regularized", num_leaves=31, min_child_samples=500, feature_fraction=0.5))
    if full:
        specs.append(_base_spec(name="stride2", stride=2, rounds=70, confirm_stride=2, confirm_rounds=200))
        specs.append(_base_spec(name="rounds200", rounds=200, confirm_rounds=400))
    return specs


def _params_for(spec):
    from bench.boost import lgbm_params
    return lgbm_params(spec["num_leaves"], spec["min_child_samples"], spec["feature_fraction"], spec["learning_rate"])


def _v2_like(spec):
    return (int(spec["confirm_rounds"]) == 300 and int(spec["confirm_stride"]) == 5
            and spec["weight"] == "none" and int(spec["num_leaves"]) == 63
            and int(spec["min_child_samples"]) == 200 and float(spec["feature_fraction"]) == 0.8
            and float(spec["learning_rate"]) == 0.05)


def execute_b6(job):
    """Residual dilated TCN on v2 out-of-fold probabilities plus the 10 strongest features."""
    import gc
    import numpy as np
    from bench import boost

    log = job["log"]
    for seed in job["seeds"]:
        seed = int(seed)
        method = "b6_tcn"
        result_path = _result_path(method, seed)
        if _done_result(result_path):
            _log(log, "skip finished b6 seed %d" % seed)
            continue
        if job.get("deadline") and time.time() > float(job["deadline"]):
            _log(log, "b6 stopped at the time budget before seed %d" % seed)
            break
        cache = _cache_for(seed)
        meta = _load_json(os.path.join(cache, "meta.json"))
        cols, names, flags = _groups_cols(meta, ("context", "lang_dead"))
        gain = np.load(os.path.join(_proba_dir("v2", seed), "gain.npy"))
        order = np.argsort(-np.asarray(gain, dtype=np.float64), kind="mergesort")
        local = [int(i) for i in order[:10]]
        top_cols = [cols[i] for i in local]
        top_names = [names[i] for i in local]
        ball_idx = [6 + j for j, i in enumerate(local) if flags[i]]
        v2 = _load_json(_result_path("v2", seed))
        state = _load_json(os.path.join(_proba_dir("v2", seed), "state.json"))
        proba_dir = _proba_dir(method, seed)
        os.makedirs(proba_dir, exist_ok=True)
        _log(log, "b6 seed %d top %s ball-channels %d" % (seed, " ".join(top_names), len(ball_idx)))
        t_all = time.time()
        epochs = []
        for held in TRAIN:
            oof_path = os.path.join(proba_dir, held + ".npy")
            if os.path.isfile(oof_path):
                ep = _load_json(os.path.join(proba_dir, "epoch_%s.json" % held))
                epochs.append(int(ep["epoch"]) if ep else 4)
                _log(log, "b6 resume seed %d fold %s" % (seed, held))
                continue
            rest = [m for m in TRAIN if m != held]
            mu, sd = boost.feature_norm(cache, rest, top_cols, 5)
            train_views = []
            for m in rest:
                proba = np.load(os.path.join(_proba_dir("v2", seed), m + ".npy"))
                views, _, _ = boost.sequences_for_match(cache, m, top_cols, proba, mu, sd, 5)
                train_views.extend(views)
            held_p = np.load(os.path.join(_proba_dir("v2", seed), held + ".npy"))
            held_views, held_y, held_period = boost.sequences_for_match(
                cache, held, top_cols, held_p, mu, sd, 5)
            t0 = time.time()
            model, best_epoch, best_spell, history, device = boost.train_tcn_fold(
                train_views, held_views, held_y, held_period, state["folds"][held]["prior"],
                v2["switch_team"], v2["switch_dead"], ball_idx, seed=10 + seed,
            )
            full = boost.tcn_full_proba(model, held_views, len(held_y), device)
            boost.atomic_npy(oof_path, np.asarray(full, dtype=np.float32))
            _atomic_json(os.path.join(proba_dir, "epoch_%s.json" % held), {
                "epoch": int(best_epoch), "spell": float(best_spell),
                "history": history, "seconds": round(time.time() - t0, 1),
            })
            epochs.append(int(best_epoch))
            _log(log, "b6 seed %d fold %s best epoch %d spell %.3f in %.0fs" % (
                seed, held, best_epoch, best_spell, time.time() - t0))
            del model, train_views, held_views, full
            gc.collect()
        final_epochs = int(np.median(epochs)) if epochs else 4
        final_epochs = max(1, final_epochs)
        mu, sd = boost.feature_norm(cache, TRAIN, top_cols, 5)
        train_views = []
        for m in TRAIN:
            proba = np.load(os.path.join(_proba_dir("v2", seed), m + ".npy"))
            views, _, _ = boost.sequences_for_match(cache, m, top_cols, proba, mu, sd, 5)
            train_views.extend(views)
        _log(log, "b6 seed %d final train %d epochs" % (seed, final_epochs))
        model, _, _, _, device = boost.train_tcn_fold(
            train_views, None, None, None, None, None, None, ball_idx,
            seed=90 + seed, epochs=final_epochs,
        )
        ems, ys, periods = [], [], []
        for m in TRAIN:
            proba = np.load(os.path.join(proba_dir, m + ".npy"))
            y, period = _labels(cache, m)
            ems.append(boost.emissions(proba, state["folds"][m]["prior"]))
            ys.append(y)
            periods.append(period)
        oof_spell, st, sd, _ = boost.tune_costs(ems, ys, periods, job["grid_team"], job["grid_dead"])
        _log(log, "b6 seed %d oof %.3f costs %.0f %.0f" % (seed, oof_spell, st, sd))
        rows = []
        for m in TEST:
            proba_g = np.load(os.path.join(_proba_dir("v2", seed), m + ".npy"))
            views, y, period = boost.sequences_for_match(cache, m, top_cols, proba_g, mu, sd, 5)
            full = boost.tcn_full_proba(model, views, len(y), device)
            boost.atomic_npy(os.path.join(proba_dir, m + ".npy"), np.asarray(full, dtype=np.float32))
            pred = boost.decode_periods(boost.emissions(full, state["final"]["prior"]), period, st, sd)
            rec = boost.frame_metrics(y, pred, period)
            rec["match"] = m
            rows.append(rec)
            _log(log, "TEST b6_tcn seed %d %s spell %.3f macro %.3f poss %d/%d" % (
                seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"]))
        result = {
            "method": method, "family": "b6", "seed": seed, "kind": "tcn",
            "n_features": 10, "oof_spell": float(oof_spell), "switch_team": st, "switch_dead": sd,
            "seconds": round(time.time() - t_all, 1), "top10": top_names,
            "epochs": epochs, "final_epochs": final_epochs, "matches": rows,
            "device": str(device),
        }
        _atomic_json(result_path, result)
        _temperature_on(method, "b6_tcn_temp", seed, cache, proba_dir, state, log, job["grid_team"], job["grid_dead"])
        del model
        gc.collect()
    return True


def _temperature_on(base_method, out_method, seed, cache, proba_dir, v2_state, log, grid_t, grid_d):
    """Choose a temperature on the base model's TRAIN out-of-fold posteriors, then score TEST."""
    import numpy as np
    from bench import boost
    if _done_result(_result_path(out_method, seed)):
        return
    temps = (0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0)
    base = _load_json(_result_path(base_method, seed))
    st0, sd0 = base["switch_team"], base["switch_dead"]
    oof = {}
    ys, periods = {}, {}
    for m in TRAIN:
        oof[m] = np.load(os.path.join(proba_dir, m + ".npy"))
        ys[m], periods[m] = _labels(cache, m)
    frozen = []
    for T in temps:
        ems = [boost.emissions(boost.apply_temperature(oof[m], T), v2_state["folds"][m]["prior"]) for m in TRAIN]
        spells = []
        for e, m in zip(ems, TRAIN):
            pred = boost.decode_periods(e, periods[m], st0, sd0)
            spells.append(boost.frame_metrics(ys[m], pred, periods[m])["spell"])
        frozen.append((float(np.mean(spells)), float(T)))
    frozen.sort(key=lambda z: -z[0])
    best_T = frozen[0][1]
    _log(log, "%s seed %d temperature search %s pick %.2f" % (
        out_method, seed, " ".join("%.2f:%.3f" % (t, s) for s, t in sorted(frozen, key=lambda z: z[1])), best_T))
    ems = [boost.emissions(boost.apply_temperature(oof[m], best_T), v2_state["folds"][m]["prior"]) for m in TRAIN]
    oof_spell, st, sd, _ = boost.tune_costs(
        ems, [ys[m] for m in TRAIN], [periods[m] for m in TRAIN], grid_t, grid_d)
    rows = []
    prior = v2_state["final"]["prior"]
    for m in TEST:
        raw = np.load(os.path.join(proba_dir, m + ".npy"))
        cal = boost.apply_temperature(raw, best_T)
        y, period = _labels(cache, m)
        pred = boost.decode_periods(boost.emissions(cal, prior), period, st, sd)
        rec = boost.frame_metrics(y, pred, period)
        rec["match"] = m
        rows.append(rec)
        _log(log, "TEST %s seed %d %s spell %.3f macro %.3f poss %d/%d" % (
            out_method, seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"]))
    _atomic_json(_result_path(out_method, seed), {
        "method": out_method, "family": "b7" if out_method.startswith("b7") or out_method.startswith("v2_fine") else "b6",
        "seed": int(seed), "temperature": best_T, "oof_spell": float(oof_spell),
        "switch_team": st, "switch_dead": sd, "frozen": [{"T": t, "spell": s} for s, t in frozen],
        "matches": rows, "n_features": base.get("n_features"),
    })


def execute_b7(job):
    """Temperature, isotonic, and a finer switch-cost grid on v2 posteriors. No refit."""
    from bench import boost
    log = job["log"]
    for seed in job["seeds"]:
        seed = int(seed)
        cache = _cache_for(seed)
        state = _load_json(os.path.join(_proba_dir("v2", seed), "state.json"))
        proba_dir = _proba_dir("v2", seed)
        if not _done_result(_result_path("v2_fine_costs", seed)):
            _retune(seed, cache, proba_dir, state, "v2", "v2_fine_costs", "b7",
                    boost.GRID_TEAM_FINE, boost.GRID_DEAD_FINE, log, temperature=1.0)
        _temperature_on("v2", "b7_temperature", seed, cache, proba_dir, state, log,
                        job["grid_team"], job["grid_dead"])
        _isotonic(seed, cache, proba_dir, state, log, job["grid_team"], job["grid_dead"])
    return True


def _retune(seed, cache, proba_dir, state, base_method, out_method, family, grid_t, grid_d, log, temperature=1.0):
    import numpy as np
    from bench import boost
    ems, ys, periods = [], [], []
    for m in TRAIN:
        proba = np.load(os.path.join(proba_dir, m + ".npy"))
        if temperature != 1.0:
            proba = boost.apply_temperature(proba, temperature)
        y, period = _labels(cache, m)
        ems.append(boost.emissions(proba, state["folds"][m]["prior"]))
        ys.append(y)
        periods.append(period)
    oof_spell, st, sd, _ = boost.tune_costs(ems, ys, periods, grid_t, grid_d)
    _log(log, "oof %s seed %d spell %.3f costs %.0f %.0f" % (out_method, seed, oof_spell, st, sd))
    rows = []
    prior = state["final"]["prior"]
    for m in TEST:
        proba = np.load(os.path.join(proba_dir, m + ".npy"))
        if temperature != 1.0:
            proba = boost.apply_temperature(proba, temperature)
        y, period = _labels(cache, m)
        pred = boost.decode_periods(boost.emissions(proba, prior), period, st, sd)
        rec = boost.frame_metrics(y, pred, period)
        rec["match"] = m
        rows.append(rec)
        _log(log, "TEST %s seed %d %s spell %.3f macro %.3f poss %d/%d" % (
            out_method, seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"]))
    base = _load_json(_result_path(base_method, seed)) or {}
    _atomic_json(_result_path(out_method, seed), {
        "method": out_method, "family": family, "seed": int(seed),
        "oof_spell": float(oof_spell), "switch_team": st, "switch_dead": sd,
        "temperature": temperature, "n_features": base.get("n_features"),
        "matches": rows,
    })


def _isotonic(seed, cache, proba_dir, state, log, grid_t, grid_d):
    import numpy as np
    from bench import boost
    if _done_result(_result_path("b7_isotonic", seed)):
        return
    oof = {m: np.load(os.path.join(proba_dir, m + ".npy")) for m in TRAIN}
    ys, periods = {}, {}
    for m in TRAIN:
        ys[m], periods[m] = _labels(cache, m)
    cal = {}
    for held in TRAIN:
        rest = [m for m in TRAIN if m != held]
        models = boost.fit_isotonic([oof[m] for m in rest], [ys[m] for m in rest])
        cal[held] = boost.apply_isotonic(models, oof[held])
    ems = [boost.emissions(cal[m], state["folds"][m]["prior"]) for m in TRAIN]
    oof_spell, st, sd, _ = boost.tune_costs(
        ems, [ys[m] for m in TRAIN], [periods[m] for m in TRAIN], grid_t, grid_d)
    models = boost.fit_isotonic([oof[m] for m in TRAIN], [ys[m] for m in TRAIN])
    rows = []
    prior = state["final"]["prior"]
    for m in TEST:
        raw = np.load(os.path.join(proba_dir, m + ".npy"))
        proba = boost.apply_isotonic(models, raw)
        y, period = _labels(cache, m)
        pred = boost.decode_periods(boost.emissions(proba, prior), period, st, sd)
        rec = boost.frame_metrics(y, pred, period)
        rec["match"] = m
        rows.append(rec)
        _log(log, "TEST b7_isotonic seed %d %s spell %.3f macro %.3f poss %d/%d" % (
            seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"]))
    _atomic_json(_result_path("b7_isotonic", seed), {
        "method": "b7_isotonic", "family": "b7", "seed": int(seed),
        "oof_spell": float(oof_spell), "switch_team": st, "switch_dead": sd,
        "matches": rows,
    })


def _load_member_proba(method, seed, match):
    import numpy as np
    path = os.path.join(_proba_dir(method, seed), match + ".npy")
    if not os.path.isfile(path):
        return None
    return np.load(path)


def execute_ensemble(job):
    """Geometric mean of log probabilities, then a fresh cost tune. Needs two or more members."""
    import numpy as np
    from bench import boost
    log = job["log"]
    members = list(job["members"])
    for seed in job["seeds"]:
        seed = int(seed)
        if _done_result(_result_path("b4_ensemble", seed)):
            continue
        have = []
        for name in members:
            if all(_load_member_proba(name, seed, m) is not None for m in TRAIN + TEST):
                have.append(name)
        if len(have) < 2:
            _log(log, "b4 seed %d skipped, members %s" % (seed, " ".join(have) or "(none)"))
            add_note("B4 seed %d did not have two finished base models." % seed)
            continue
        state = _load_json(os.path.join(_proba_dir("v2", seed), "state.json"))
        cache = _cache_for(seed)
        ems, ys, periods = [], [], []
        for m in TRAIN:
            parts = [_load_member_proba(name, seed, m) for name in have]
            proba = boost.geo_mean(parts)
            y, period = _labels(cache, m)
            ems.append(boost.emissions(proba, state["folds"][m]["prior"]))
            ys.append(y)
            periods.append(period)
        oof_spell, st, sd, _ = boost.tune_costs(ems, ys, periods, job["grid_team"], job["grid_dead"])
        rows = []
        prior = state["final"]["prior"]
        for m in TEST:
            parts = [_load_member_proba(name, seed, m) for name in have]
            proba = boost.geo_mean(parts)
            y, period = _labels(cache, m)
            pred = boost.decode_periods(boost.emissions(proba, prior), period, st, sd)
            rec = boost.frame_metrics(y, pred, period)
            rec["match"] = m
            rows.append(rec)
            _log(log, "TEST b4_ensemble seed %d %s spell %.3f macro %.3f poss %d/%d members %s" % (
                seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"], " ".join(have)))
        _atomic_json(_result_path("b4_ensemble", seed), {
            "method": "b4_ensemble", "family": "b4", "seed": seed, "members": have,
            "oof_spell": float(oof_spell), "switch_team": st, "switch_dead": sd, "matches": rows,
        })
    return True


def execute_stack(job):
    """Multinomial logistic regression on out-of-fold probabilities. C chosen on TRAIN."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from bench import boost
    log = job["log"]
    members = list(job["members"])
    for seed in job["seeds"]:
        seed = int(seed)
        if _done_result(_result_path("b5_stack", seed)):
            continue
        have = [name for name in members if all(
            _load_member_proba(name, seed, m) is not None for m in TRAIN + TEST)]
        if len(have) < 2:
            _log(log, "b5 seed %d skipped" % seed)
            continue
        cache = _cache_for(seed)
        state = _load_json(os.path.join(_proba_dir("v2", seed), "state.json"))

        def design(match, names):
            return np.concatenate([_load_member_proba(name, seed, match) for name in names], axis=1)

        best = None
        best_pack = None
        for C in (0.1, 1.0, 10.0):
            cal = {}
            for held in TRAIN:
                rest = [m for m in TRAIN if m != held]
                x = np.concatenate([design(m, have)[::5] for m in rest], axis=0)
                y = np.concatenate([_labels(cache, m)[0][::5] for m in rest], axis=0)
                clf = LogisticRegression(C=C, max_iter=200, solver="lbfgs")
                clf.fit(x, y)
                cal[held] = clf.predict_proba(design(held, have))
                # Remap if the solver dropped a class. All three are present in these matches.
                cal[held] = _align_proba(clf, cal[held])
            ems, ys, periods = [], [], []
            for m in TRAIN:
                y, period = _labels(cache, m)
                ems.append(boost.emissions(cal[m], state["folds"][m]["prior"]))
                ys.append(y)
                periods.append(period)
            # Frozen v2 costs pick C; the winning C is retuned below. Decoding here is the selection score.
            spells = []
            v2 = _load_json(_result_path("v2", seed))
            for e, y, period in zip(ems, ys, periods):
                pred = boost.decode_periods(e, period, v2["switch_team"], v2["switch_dead"])
                spells.append(boost.frame_metrics(y, pred, period)["spell"])
            mean = float(np.mean(spells))
            _log(log, "b5 seed %d C %s frozen spell %.3f" % (seed, C, mean))
            if best is None or mean > best[0]:
                best = (mean, C)
                best_pack = cal
        C = best[1]
        ems, ys, periods = [], [], []
        for m in TRAIN:
            y, period = _labels(cache, m)
            ems.append(boost.emissions(best_pack[m], state["folds"][m]["prior"]))
            ys.append(y)
            periods.append(period)
        oof_spell, st, sd, _ = boost.tune_costs(ems, ys, periods, job["grid_team"], job["grid_dead"])
        x = np.concatenate([design(m, have)[::5] for m in TRAIN], axis=0)
        y = np.concatenate([_labels(cache, m)[0][::5] for m in TRAIN], axis=0)
        clf = LogisticRegression(C=C, max_iter=200, solver="lbfgs")
        clf.fit(x, y)
        rows = []
        prior = state["final"]["prior"]
        for m in TEST:
            proba = _align_proba(clf, clf.predict_proba(design(m, have)))
            yy, period = _labels(cache, m)
            pred = boost.decode_periods(boost.emissions(proba, prior), period, st, sd)
            rec = boost.frame_metrics(yy, pred, period)
            rec["match"] = m
            rows.append(rec)
            _log(log, "TEST b5_stack seed %d %s spell %.3f macro %.3f poss %d/%d C %s" % (
                seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"], C))
        _atomic_json(_result_path("b5_stack", seed), {
            "method": "b5_stack", "family": "b5", "seed": seed, "members": have, "C": C,
            "oof_spell": float(oof_spell), "switch_team": st, "switch_dead": sd, "matches": rows,
        })
    return True


def _align_proba(clf, proba):
    import numpy as np
    out = np.zeros((len(proba), 3), np.float64)
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = proba[:, j]
    out = np.clip(out, 1e-8, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def execute_b9(job):
    """Small set transformer over players, ensembled with v2. One match of tracking at a time."""
    import gc
    import numpy as np
    import torch
    from bench.dfl import load
    from bench.degrade import degrade
    from bench.methods import swap_teams
    from bench.run import labels as labeller
    from bench import boost

    log = job["log"]
    torch.manual_seed(0)
    device = boost._tcn_device()
    torch.set_num_threads(2)
    _log(log, "b9 device %s" % device)
    stride_train = 10
    stride_pred = 5
    for seed in job["seeds"]:
        seed = int(seed)
        if _done_result(_result_path("b9_set", seed)) and _done_result(_result_path("b9_ensemble", seed)):
            continue
        if job.get("deadline") and time.time() > float(job["deadline"]):
            _log(log, "b9 stopped before seed %d" % seed)
            break
        cache = _cache_for(seed)
        v2_state = _load_json(os.path.join(_proba_dir("v2", seed), "state.json"))
        bags = {}
        for index, match in enumerate(ORDER):
            clean = load(match)
            deg = degrade(clean, seed=1000 * seed + index)
            y = labeller(clean)
            del clean
            tok, pad = boost.build_tokens(deg)
            tok_s, pad_s = boost.build_tokens(swap_teams(deg))
            del deg
            n = len(y)
            bags[match] = {
                "y": y,
                "frames_tr": np.arange(0, n, stride_train, dtype=np.int32),
                "frames_pr": np.arange(0, n, stride_pred, dtype=np.int32),
            }
            fr, fp = bags[match]["frames_tr"], bags[match]["frames_pr"]
            bags[match]["tok"] = tok[fr]
            bags[match]["pad"] = pad[fr]
            bags[match]["tok_s"] = tok_s[fr]
            bags[match]["pad_s"] = pad_s[fr]
            bags[match]["tok_p"] = tok[fp]
            bags[match]["pad_p"] = pad[fp]
            bags[match]["tok_sp"] = tok_s[fp]
            bags[match]["pad_sp"] = pad_s[fp]
            del tok, pad, tok_s, pad_s
            gc.collect()
        _log(log, "b9 seed %d tokens ready" % seed)
        model = boost.SetEncoder(d=32, heads=4, layers=1, dropout=0.1).to(device)
        # LOMO: two epochs, fixed, deterministic given the seed.
        oof = {}
        for k, held in enumerate(TRAIN):
            torch.manual_seed(20 + seed * 3 + k)
            fold = boost.SetEncoder(d=32, heads=4, layers=1, dropout=0.1).to(device)
            opt = torch.optim.AdamW(fold.parameters(), lr=1e-3, weight_decay=1e-4)
            rest = [m for m in TRAIN if m != held]
            tokens, pads, labels = [], [], []
            for m in rest:
                b = bags[m]
                tokens += [b["tok"], b["tok_s"]]
                pads += [b["pad"], b["pad_s"]]
                labels += [b["y"][b["frames_tr"]], boost.SWAP[b["y"][b["frames_tr"]]]]
            for epoch in range(2):
                loss = boost.train_set_epoch(fold, opt, tokens, pads, labels, device, 32, seed + k, epoch)
                _log(log, "b9 seed %d fold %s epoch %d loss %.4f" % (seed, held, epoch + 1, loss))
            oof[held] = _set_predict_match(fold, bags[held], device)
            del fold, opt, tokens, pads, labels
            gc.collect()
        ems, ys, periods = [], [], []
        for m in TRAIN:
            y, period = _labels(cache, m)
            ems.append(boost.emissions(oof[m], v2_state["folds"][m]["prior"]))
            ys.append(y)
            periods.append(period)
        oof_spell, st, sd, _ = boost.tune_costs(ems, ys, periods, job["grid_team"], job["grid_dead"])
        _log(log, "b9 seed %d set oof %.3f costs %.0f %.0f" % (seed, oof_spell, st, sd))
        # Final model on all TRAIN, same two epochs.
        torch.manual_seed(50 + seed)
        final = boost.SetEncoder(d=32, heads=4, layers=1, dropout=0.1).to(device)
        opt = torch.optim.AdamW(final.parameters(), lr=1e-3, weight_decay=1e-4)
        tokens, pads, labels = [], [], []
        for m in TRAIN:
            b = bags[m]
            tokens += [b["tok"], b["tok_s"]]
            pads += [b["pad"], b["pad_s"]]
            labels += [b["y"][b["frames_tr"]], boost.SWAP[b["y"][b["frames_tr"]]]]
        for epoch in range(2):
            loss = boost.train_set_epoch(final, opt, tokens, pads, labels, device, 32, 50 + seed, epoch)
            _log(log, "b9 seed %d final epoch %d loss %.4f" % (seed, epoch + 1, loss))
        rows = []
        test_p = {}
        for m in TEST:
            proba = _set_predict_match(final, bags[m], device)
            test_p[m] = proba
            y, period = _labels(cache, m)
            pred = boost.decode_periods(boost.emissions(proba, v2_state["final"]["prior"]), period, st, sd)
            rec = boost.frame_metrics(y, pred, period)
            rec["match"] = m
            rows.append(rec)
            _log(log, "TEST b9_set seed %d %s spell %.3f macro %.3f poss %d/%d" % (
                seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"]))
        _atomic_json(_result_path("b9_set", seed), {
            "method": "b9_set", "family": "b9", "seed": seed, "oof_spell": float(oof_spell),
            "switch_team": st, "switch_dead": sd, "matches": rows, "device": str(device),
        })
        # Ensemble with v2: geometric mean, costs retuned on TRAIN OOF.
        ems = []
        for m in TRAIN:
            v2p = np.load(os.path.join(_proba_dir("v2", seed), m + ".npy"))
            ems.append(boost.emissions(boost.geo_mean([oof[m], v2p]), v2_state["folds"][m]["prior"]))
        spell2, st2, sd2, _ = boost.tune_costs(ems, ys, periods, job["grid_team"], job["grid_dead"])
        rows2 = []
        for m in TEST:
            v2p = np.load(os.path.join(_proba_dir("v2", seed), m + ".npy"))
            proba = boost.geo_mean([test_p[m], v2p])
            y, period = _labels(cache, m)
            pred = boost.decode_periods(boost.emissions(proba, v2_state["final"]["prior"]), period, st2, sd2)
            rec = boost.frame_metrics(y, pred, period)
            rec["match"] = m
            rows2.append(rec)
            _log(log, "TEST b9_ensemble seed %d %s spell %.3f macro %.3f poss %d/%d" % (
                seed, m, rec["spell"], rec["macro"], rec["n_pred"], rec["n_gt"]))
        _atomic_json(_result_path("b9_ensemble", seed), {
            "method": "b9_ensemble", "family": "b9", "seed": seed, "oof_spell": float(spell2),
            "switch_team": st2, "switch_dead": sd2, "matches": rows2,
        })
        del bags, final, model
        gc.collect()
    return True


def _set_predict_match(model, bag, device):
    from bench import boost
    p0 = boost.predict_set(model, bag["tok_p"], bag["pad_p"], device, batch=64)
    p1 = boost.predict_set(model, bag["tok_sp"], bag["pad_sp"], device, batch=64)[:, boost.SWAP]
    mixed = 0.5 * (p0 + p1)
    return boost.hold_fill(len(bag["y"]), bag["frames_pr"], mixed)


def _check_ref(method, seed, rec, log):
    if int(seed) != 0 or method not in REF:
        return
    errs = []
    for row in rec["matches"]:
        err = abs(row["spell"] - REF[method][row["match"]])
        errs.append(err)
        _log(log, "ref %s %s got %.3f published %.3f delta %+.3f" % (
            method, row["match"], row["spell"], REF[method][row["match"]], row["spell"] - REF[method][row["match"]]))
    if max(errs) > 0.04:
        raise SystemExit("reproduction of %s seed 0 drifted by %.3f; stopping before more fits" % (method, max(errs)))


def _protocol_job(method, family, seed, cache, cols, names, kind, params, rounds, stride, weight,
                  grid_t, grid_d, log, max_fold_seconds=None):
    return {
        "method": method, "family": family, "seed": int(seed), "cache": cache,
        "cols": list(cols), "feat_names": list(names), "kind": kind, "params": params,
        "rounds": int(rounds), "stride": int(stride), "weight": weight,
        "train": list(TRAIN), "test": list(TEST),
        "grid_team": list(grid_t), "grid_dead": list(grid_d),
        "proba_dir": _proba_dir(method, seed), "result": _result_path(method, seed),
        "log": log, "max_fold_seconds": max_fold_seconds,
    }


def _launch_protocol(job, log, critical=False):
    if _done_result(job["result"]):
        _log(log, "skip finished %s seed %s" % (job["method"], job["seed"]))
        return _load_json(job["result"])
    try:
        rec = _run_child(worker_protocol, job)
    except TooSlow as exc:
        _log(log, str(exc))
        add_note(str(exc))
        return None
    except Exception as exc:
        _log(log, "error %s seed %s: %s" % (job["method"], job["seed"], exc))
        if critical:
            raise
        add_note("Failed %s seed %s: %s" % (job["method"], job["seed"], exc.__class__.__name__))
        return None
    return rec


def _time_left(t0, budget):
    return budget - (time.time() - t0)


def _load_rows():
    rows = []
    folder = os.path.join(PERSIST, "results")
    if not os.path.isdir(folder):
        return rows
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            rec = _load_json(os.path.join(folder, name))
            if rec and rec.get("matches") and not rec.get("error"):
                rows.append(rec)
    return rows


def write_final(seeds, budget_note):
    import numpy as np
    rows = _load_rows()
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    v2 = {}
    for r in by.get("v2", []):
        v2[int(r["seed"])] = float(np.mean([m["spell"] for m in r["matches"]]))
    order = ["v1", "v2", "v2_fine_costs", "b7_temperature", "b7_isotonic",
             "b6_tcn", "b6_tcn_temp", "b2_xgb", "b2_catboost", "b3_extratrees",
             "b4_ensemble", "b5_stack", "b8_gain", "b9_set", "b9_ensemble"]
    names = [n for n in order if n in by] + sorted(k for k in by if k not in order)
    lines = []
    lines.append("| method | spellF1 | sd_seeds | sd_rows | macroF1 | poss pred | poss gt | per-seed spell | per-seed gain vs v2 | adopt |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    summary = []
    for name in names:
        recs = sorted(by[name], key=lambda r: int(r["seed"]))
        spells, macros, preds, gts = [], [], [], []
        seed_spell, seed_gain = [], []
        have = set()
        for r in recs:
            ss = [m["spell"] for m in r["matches"]]
            spells.extend(ss)
            macros.extend(m["macro"] for m in r["matches"])
            preds.extend(m["n_pred"] for m in r["matches"])
            gts.extend(m["n_gt"] for m in r["matches"])
            sm = float(np.mean(ss))
            seed_spell.append(sm)
            have.add(int(r["seed"]))
            if name != "v2" and int(r["seed"]) in v2:
                seed_gain.append(sm - v2[int(r["seed"])])
        adopt = "baseline" if name == "v2" else ""
        if name != "v2" and seed_gain and have == set(int(s) for s in seeds) and set(v2) >= set(int(s) for s in seeds):
            adopt = "yes" if all(g >= 0.03 - 1e-12 for g in seed_gain) else "no"
        elif name != "v2":
            adopt = "incomplete"
        sd_seeds = float(np.std(seed_spell)) if len(seed_spell) > 1 else 0.0
        sd_rows = float(np.std(spells)) if spells else 0.0
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.0f | %.0f | %s | %s | %s |" % (
            name, float(np.mean(spells)), sd_seeds, sd_rows, float(np.mean(macros)),
            float(np.mean(preds)), float(np.mean(gts)),
            " ".join("%.3f" % s for s in seed_spell),
            " ".join("%+.3f" % g for g in seed_gain),
            adopt))
        summary.append((name, float(np.mean(spells)), adopt, seed_gain, have))
    complete = [s for s in summary if s[2] in ("yes", "no", "baseline") or (
        s[0] != "v2" and s[4] == set(int(x) for x in seeds))]
    adopted = [s for s in summary if s[2] == "yes"]
    best = max(summary, key=lambda s: s[1]) if summary else None
    body = []
    body.append("# Boost: model class, ensemble, sequence")
    body.append("")
    body.append("Bench v2 degradation, official `possession_metrics.evaluate`. "
                "TRAIN = %s. TEST = %s. Requested seeds = %s." % (
                    ", ".join(TRAIN), ", ".join(TEST), ", ".join(str(s) for s in seeds)))
    body.append("")
    body.append("v2 is LightGBM on v1 + context + lang_dead, team-symmetric, 300 rounds, "
                "stride 5, the v1 hyperparameters. Switch costs are chosen by leave-one-match-out "
                "on TRAIN. sd_seeds is the population standard deviation of the per-seed means. "
                "sd_rows is the population standard deviation over match x seed rows, the figure "
                "`bench.run` prints. Adopt means the gain versus v2 on that same seed is at least "
                "+0.03 on every requested seed.")
    body.append("")
    body.append("## Results")
    body.append("")
    body.append("\n".join(lines))
    body.append("")
    if best:
        body.append("Highest mean spellF1 in this table: **%s** at %.3f (adopt: %s)." % (best[0], best[1], best[2]))
        body.append("")
    if adopted:
        body.append("Cleared +0.03 on every requested seed: %s." % ", ".join(s[0] for s in adopted))
    else:
        body.append("No variant cleared +0.03 spellF1 over v2 on every requested seed. v2 stays the baseline.")
    body.append("")
    body.append("## Per seed")
    body.append("")
    body.append("| method | seed | spellF1 | macroF1 | poss pred | poss gt | gain vs v2 | oof spell | team | dead |")
    body.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in names:
        for r in sorted(by[name], key=lambda z: int(z["seed"])):
            ss = [m["spell"] for m in r["matches"]]
            sm = float(np.mean(ss))
            gain = "" if name == "v2" or int(r["seed"]) not in v2 else "%+.3f" % (sm - v2[int(r["seed"])])
            body.append("| %s | %d | %.3f | %.3f | %.0f | %.0f | %s | %s | %s | %s |" % (
                name, int(r["seed"]), sm, float(np.mean([m["macro"] for m in r["matches"]])),
                float(np.mean([m["n_pred"] for m in r["matches"]])),
                float(np.mean([m["n_gt"] for m in r["matches"]])),
                gain,
                "" if r.get("oof_spell") is None else "%.3f" % r["oof_spell"],
                "" if r.get("switch_team") is None else "%.0f" % r["switch_team"],
                "" if r.get("switch_dead") is None else "%.0f" % r["switch_dead"]))
    body.append("")
    body.append("## Per match")
    body.append("")
    body.append("| method | seed | match | spellF1 | macroF1 | poss pred | poss gt | home-share |")
    body.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for name in names:
        for r in sorted(by[name], key=lambda z: int(z["seed"])):
            for m in r["matches"]:
                body.append("| %s | %d | %s | %.3f | %.3f | %d | %d | %.3f |" % (
                    name, int(r["seed"]), m["match"], m["spell"], m["macro"], m["n_pred"], m["n_gt"], m["share_err"]))
    body.append("")
    screen = _load_json(os.path.join(PERSIST, "results", "b1_screen.json"))
    if screen:
        body.append("## B1 screen (TRAIN only)")
        body.append("")
        body.append("Held-out match A is %s, match B is %s. Specs share a 100-round budget except "
                    "where the name says otherwise. The leader is confirmed at full round count only "
                    "if it beats the screen base on match A and does not lose on match B." % (TRAIN[0], TRAIN[1]))
        body.append("")
        body.append("| spec | spell A | spell B | seconds |")
        body.append("|---|---:|---:|---:|")
        for key in sorted(k for k in screen if not k.startswith("_")):
            row = screen[key]
            if not isinstance(row, dict) or "spell_a" not in row:
                continue
            body.append("| %s | %.3f | %s | %.0f |" % (
                key, row["spell_a"],
                "" if row.get("spell_b") is None else "%.3f" % row["spell_b"],
                row.get("seconds", 0)))
        body.append("")
        body.append("Winner status: %s. Winner: %s." % (screen.get("_winner_status"), screen.get("_winner")))
        body.append("")
    body.append("## Costs and fit time")
    body.append("")
    body.append("| method | seed | oof spell | team | dead | feats | seconds |")
    body.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in names:
        for r in sorted(by[name], key=lambda z: int(z["seed"])):
            body.append("| %s | %d | %s | %s | %s | %s | %s |" % (
                name, int(r["seed"]),
                "" if r.get("oof_spell") is None else "%.3f" % r["oof_spell"],
                "" if r.get("switch_team") is None else "%.0f" % r["switch_team"],
                "" if r.get("switch_dead") is None else "%.0f" % r["switch_dead"],
                "" if r.get("n_features") is None else "%s" % r["n_features"],
                "" if r.get("seconds") is None else "%.0f" % r["seconds"]))
    body.append("")
    body.append("## Reproduce")
    body.append("")
    body.append("From the repo root, with `bench/cache/npz` populated (this run symlinked the open DFL cache) "
                "and the seed-0 feature cache at `research/runs/features3/cache/f3a/seed0`:")
    body.append("")
    body.append("```")
    body.append(".venv\\Scripts\\python.exe -m bench.run_boost --seeds %s" % " ".join(str(s) for s in seeds))
    body.append("```")
    body.append("")
    body.append("Code copies: `feats3.py`, `boost.py`, `run_boost.py`. Logs: `logs/`. "
                "Per-seed JSON: `results/`. Probabilities: `proba/`.")
    body.append("")
    body.append("## Design choices")
    body.append("")
    body.append("- v2 reuses the features3 seed-0 matrices. Seeds 1 and 2 are rebuilt with the same "
                "`degrade` seed rule (`1000 * seed + index` over TRAIN then TEST) and the same feature code.")
    body.append("- The TCN is residual: its last layer is zero at init, so it starts as the GBDT posterior "
                "and is trained only on TRAIN out-of-fold probabilities. Inputs are those probabilities, "
                "their one-step deltas, and the 10 features with the highest leave-one-match-out gain. "
                "Team-swap is a second view, not a random flip. Ball-linked channels among those 10 are "
                "zeroed on random spans. The epoch kept for each fold is the one with the best held-out "
                "spellF1 at v2's frozen switch costs; the final net is trained for the median of those epochs.")
    body.append("- Temperature and isotonic maps are fit on TRAIN out-of-fold probabilities. Isotonic is "
                "itself leave-one-match-out. A finer switch-cost grid is a separate row (`v2_fine_costs`) "
                "so a cost-only change is not confused with a new model.")
    body.append("- B1 is a coordinate screen at 100 rounds on one TRAIN match, checked on a second TRAIN "
                "match, then the winner is refit at the v2 round budget (400 for the lower learning rate, "
                "200 for stride 2). The screen does not look at TEST.")
    body.append("- B4 averages log probabilities (a normalised geometric mean) of whichever of v2, the "
                "confirmed B1 model, XGBoost, CatBoost and ExtraTrees finished for that seed. B5 is "
                "logistic regression on the same out-of-fold probabilities. B9 is a one-layer set "
                "transformer with a CLS token; players are positions and velocities relative to the "
                "last-known ball, plus a team flag. It is ensembled with v2 the same way.")
    body.append("- Boundary sample weights (frames within 0.5 s of a label change, weight 4) are one arm "
                "of the LightGBM screen. They use TRAIN labels only as a training weight.")
    body.append("- Boosters are single-threaded and seeded. Each fit is a fresh process because this "
                "machine had about 1.4 GB free. Torch uses the RTX 3050 when CUDA is visible, otherwise CPU.")
    body.append("")
    body.append("## What was not adopted, and why a gain can fail")
    body.append("")
    body.append("A change is kept only when every requested seed gains at least 0.03 spellF1 over v2. "
                "The published v2 already reaches macroF1 about 0.91 and under-counts possessions "
                "(about 295 predicted versus 397 true on seed 0). Methods that move macroF1 without "
                "splitting spells, or that help one seed and not another, stay in the table as negative results.")
    body.append("")
    negatives = [s[0] for s in summary if s[2] == "no"]
    if negatives:
        body.append("Ran on every requested seed and missed the bar: %s." % ", ".join(negatives))
        body.append("")
    incomplete = [s[0] for s in summary if s[2] == "incomplete"]
    if incomplete:
        body.append("Incomplete across seeds (not eligible to adopt): %s." % ", ".join(incomplete))
        body.append("")
    body.append("## Caveats")
    body.append("")
    body.append("- The public 5-minute window is not read and is not used to choose anything.")
    body.append("- Model seed is 0 for every booster. The seed in the table is the degradation seed.")
    body.append("- v2's published seed-0 scores are a reproduction check. A drift above 0.04 stops the run.")
    body.append("- ExtraTrees is intentionally small (20 trees, depth 8, 12k bootstrap rows, stride 10). "
                "XGBoost uses 120 rounds, depth 6, one thread. CatBoost uses 80 iterations, depth 6, "
                "a 1.2 GB RAM cap, one thread.")
    body.append("- Feature-cache columns for seed 0 are the features3 `f3a` matrices. Rebuilding them "
                "would spend the memory budget twice and would not change the protocol.")
    body.append("- %s" % budget_note)
    notes = _load_json(os.path.join(PERSIST, "notes.json")) or []
    for note in notes:
        body.append("- " + note)
    body.append("")
    text = "\n".join(body) + "\n"
    path = os.path.join(PERSIST, "FINAL.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _check_schema(log):
    meta = _load_json(os.path.join(SEED0_CACHE, "meta.json"))
    if not meta:
        raise SystemExit("seed 0 feature cache is missing")
    names, ball, sub, kind = _schema()
    if list(meta["names"]) != names or list(meta["sub"]) != sub or list(meta["kind"]) != kind:
        raise SystemExit("feats3 schema does not match the seed 0 cache; refusing to train")
    if int(meta["feats"]) != 414:
        raise SystemExit("unexpected feature width %s" % meta.get("feats"))
    _log(log, "schema matches seed 0 cache (%d columns)" % len(names))


def _versions(log):
    import numpy
    import lightgbm
    import scipy
    text = "numpy %s lightgbm %s scipy %s" % (numpy.__version__, lightgbm.__version__, scipy.__version__)
    try:
        import torch
        text += " torch %s cuda %s" % (torch.__version__, torch.cuda.is_available())
    except Exception:
        text += " torch unavailable"
    _log(log, text)


def main(argv=None):
    import argparse
    if worker_protocol.__module__ != "bench.run_boost":
        raise RuntimeError("workers must be launched as bench.run_boost, not __main__")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--stage", default="all",
                    choices=["all", "test", "cache", "b0", "b7", "b6", "b1", "rest", "report"])
    ap.add_argument("--budget-min", type=float, default=95.0)
    ap.add_argument("--skip-self-test", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(os.path.join(PERSIST, "logs"), exist_ok=True)
    os.makedirs(os.path.join(PERSIST, "results"), exist_ok=True)
    log = os.path.join(PERSIST, "logs", "main.log")
    persist_code()
    if args.stage == "test":
        _threads(1)
        from bench.feats3 import self_test as feat_test
        from bench.boost import self_test as boost_test
        feat_test()
        boost_test()
        persist_code()
        return

    clock_path = os.path.join(PERSIST, "started_at.txt")
    if not os.path.isfile(clock_path):
        with open(clock_path, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    with open(clock_path, encoding="utf-8") as f:
        t0 = float(f.read().strip())
    budget = float(args.budget_min) * 60.0
    _log(log, "run_boost stage %s seeds %s budget %.0f min, elapsed %.0f min" % (
        args.stage, args.seeds, args.budget_min, (time.time() - t0) / 60.0))
    _versions(log)

    run_all = args.stage == "all"
    if (run_all or args.stage == "b0") and not args.skip_self_test:
        _threads(1)
        from bench.feats3 import self_test as feat_test
        from bench.boost import self_test as boost_test
        _log(log, "self-test")
        feat_test()
        boost_test()
        _log(log, "self-test passed")
    if run_all or args.stage in ("cache", "b0", "b6", "b1", "rest"):
        _check_schema(log)

    from bench import boost

    def cache(seed):
        return ensure_cache(seed, log)

    def cols_for(seed, groups):
        meta = _load_json(os.path.join(cache(seed), "meta.json"))
        return _groups_cols(meta, groups)

    if run_all or args.stage == "b0":
        for seed in args.seeds:
            c = cache(seed)
            v1_cols, v1_names, _ = cols_for(seed, ())
            v2_cols, v2_names, _ = cols_for(seed, ("context", "lang_dead"))
            if len(v1_cols) != 49 or len(v2_cols) != 351:
                raise SystemExit("column counts v1=%d v2=%d" % (len(v1_cols), len(v2_cols)))
            job = _protocol_job(
                "v1", "b0", seed, c, v1_cols, v1_names, "lgbm", boost.lgbm_params(),
                300, 5, "none", boost.GRID_TEAM, boost.GRID_DEAD, log)
            rec = _launch_protocol(job, log, critical=True)
            _check_ref("v1", seed, rec, log)
            write_final(args.seeds, "partial: B0 still running")
            persist_code()
            job = _protocol_job(
                "v2", "b0", seed, c, v2_cols, v2_names, "lgbm", boost.lgbm_params(),
                300, 5, "none", boost.GRID_TEAM, boost.GRID_DEAD, log)
            rec = _launch_protocol(job, log, critical=True)
            _check_ref("v2", seed, rec, log)
            write_final(args.seeds, "partial: B0 still running")
            persist_code()
        add_note("B0 finished for seeds %s." % ", ".join(str(s) for s in args.seeds))
        write_final(args.seeds, "B0 complete; later stages may still be running.")

    if run_all or args.stage == "b7":
        if _time_left(t0, budget) > 90 or args.stage == "b7":
            _log(log, "B7 calibration")
            execute_b7({
                "seeds": list(args.seeds), "log": log,
                "grid_team": list(boost.GRID_TEAM), "grid_dead": list(boost.GRID_DEAD),
            })
            write_final(args.seeds, "partial: B7 done")
            persist_code()
        else:
            add_note("B7 skipped: under 90 seconds of budget remained.")

    if run_all or args.stage == "b6":
        left = _time_left(t0, budget)
        seeds = list(args.seeds)
        if args.stage != "b6" and left < 240:
            add_note("B6 skipped: under 4 minutes remained.")
            seeds = []
        elif args.stage != "b6" and left < 900:
            seeds = [s for s in seeds if s in (0, 1)]
            add_note("B6 limited to seeds %s because the remaining budget was under 15 minutes." % seeds)
        if seeds:
            _log(log, "B6 seeds %s" % seeds)
            _run_child(worker_b6, {
                "seeds": seeds, "log": log,
                "grid_team": list(boost.GRID_TEAM), "grid_dead": list(boost.GRID_DEAD),
                "deadline": t0 + budget,
            })
            write_final(args.seeds, "partial: B6 done")
            persist_code()

    if run_all or args.stage == "b1":
        left = _time_left(t0, budget)
        if args.stage != "b1" and left < 200:
            add_note("B1 skipped: under 200 seconds remained after B6.")
        else:
            full = left > 1800 or args.stage == "b1"
            specs = screen_specs(full=full)
            if not full:
                add_note("B1 screen used the reduced grid (leaves, class weights, regularized) because the remaining budget was under 30 minutes.")
            c = cache(args.seeds[0])
            cols, names, _ = cols_for(args.seeds[0], ("context", "lang_dead"))
            _log(log, "B1 screen on seed %d (%d specs)" % (args.seeds[0], len(specs)))
            screen = _run_child(worker_screen, {
                "cache": c, "cols": cols, "train": list(TRAIN), "specs": specs,
                "grid_team": list(boost.GRID_TEAM), "grid_dead": list(boost.GRID_DEAD),
                "out": os.path.join(PERSIST, "results", "b1_screen.json"),
                "log": log, "deadline": t0 + budget - 60,
            })
            winner_name = screen.get("_winner") if screen else None
            status = screen.get("_winner_status") if screen else None
            if winner_name and status == "confirm":
                spec = _spec_by_name(specs, winner_name)
                if _v2_like(spec):
                    add_note("B1 winner %s confirms to the v2 hyperparameters, so it was not refit." % winner_name)
                else:
                    method = "b1_" + winner_name
                    add_note("B1 confirmed %s at rounds %d stride %d weight %s." % (
                        winner_name, spec["confirm_rounds"], spec["confirm_stride"], spec["weight"]))
                    for seed in args.seeds:
                        if args.stage != "b1" and _time_left(t0, budget) < 400 and int(seed) != int(args.seeds[0]):
                            add_note("B1 confirm skipped seed %s for time." % seed)
                            break
                        cc = cache(seed)
                        cc_cols, cc_names, _ = cols_for(seed, ("context", "lang_dead"))
                        job = _protocol_job(
                            method, "b1", seed, cc, cc_cols, cc_names, "lgbm", _params_for(spec),
                            spec["confirm_rounds"], spec["confirm_stride"], spec["weight"],
                            boost.GRID_TEAM, boost.GRID_DEAD, log)
                        _launch_protocol(job, log, critical=False)
                        write_final(args.seeds, "partial: B1 running")
            elif winner_name and status == "unchecked":
                add_note("B1 leader %s was not checked on the second TRAIN match because time ran out." % winner_name)
            else:
                add_note("B1 screen did not produce a confirmable winner. v2's hyperparameters stay.")
            write_final(args.seeds, "partial: B1 done")
            persist_code()

    if run_all or args.stage == "rest":
        _run_rest(args.seeds, t0, budget, log)

    note = "Finished stage %s for seeds %s. Elapsed %.0f min." % (
        args.stage, ", ".join(str(s) for s in args.seeds), (time.time() - t0) / 60.0)
    path = write_final(args.seeds, note)
    persist_code()
    _log(log, "wrote %s" % path)


def _run_rest(seeds, t0, budget, log):
    from bench import boost
    members = ["v2"]
    # Discover a confirmed B1 method from the results directory.
    folder = os.path.join(PERSIST, "results")
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.startswith("s") and "_b1_" in name:
                rec = _load_json(os.path.join(folder, name))
                if rec and rec.get("method") and rec["method"] not in members:
                    members.append(rec["method"])

    def left():
        return _time_left(t0, budget)

    def launch_kind(method, family, kind, params, rounds, stride, weight, limit):
        if left() < 180:
            add_note("%s skipped for time." % method)
            return
        use = list(seeds)
        if left() < 1200:
            use = [use[0]]
            add_note("%s ran on seed %s only; the budget could not cover every seed." % (method, use[0]))
        for seed in use:
            if left() < 120:
                add_note("%s stopped before seed %s." % (method, seed))
                break
            c = ensure_cache(seed, log)
            meta = _load_json(os.path.join(c, "meta.json"))
            if method == "b8_gain":
                gain = __import__("numpy").load(os.path.join(_proba_dir("v2", seed), "gain.npy"))
                keep_local = boost.gain_keep(gain)
                cols, names, _ = _groups_cols(meta, ("context", "lang_dead"))
                if len(keep_local) >= len(cols) - 5:
                    add_note("B8 seed %d kept %d of %d features; no refit." % (seed, len(keep_local), len(cols)))
                    continue
                cols = [cols[int(i)] for i in keep_local]
                names = [names[int(i)] for i in keep_local]
                _log(log, "b8 seed %d keeps %d features" % (seed, len(cols)))
            else:
                cols, names, _ = _groups_cols(meta, ("context", "lang_dead"))
            job = _protocol_job(
                method, family, seed, c, cols, names, kind, params, rounds, stride, weight,
                boost.GRID_TEAM, boost.GRID_DEAD, log, max_fold_seconds=limit)
            rec = _launch_protocol(job, log, critical=False)
            if rec:
                members.append(method) if method not in members else None
            write_final(seeds, "partial: rest running")

    if left() > 200:
        launch_kind("b2_xgb", "b2", "xgb", boost.xgb_params(), 120, 5, "none", 500)
        launch_kind("b2_catboost", "b2", "cat", boost.cat_params(), 80, 5, "none", 500)
        launch_kind("b3_extratrees", "b3", "et", boost.et_params(), 1, 10, "none", 400)
    else:
        add_note("B2/B3 skipped for time.")

    if left() > 60:
        _log(log, "B4/B5 members %s" % " ".join(dict.fromkeys(members)))
        try:
            execute_ensemble({
                "seeds": list(seeds), "members": list(dict.fromkeys(members)),
                "log": log, "grid_team": list(boost.GRID_TEAM), "grid_dead": list(boost.GRID_DEAD),
            })
            execute_stack({
                "seeds": list(seeds), "members": list(dict.fromkeys(members)),
                "log": log, "grid_team": list(boost.GRID_TEAM), "grid_dead": list(boost.GRID_DEAD),
            })
        except Exception:
            _log(log, "ensemble/stack failed\n" + traceback.format_exc())
            add_note("B4/B5 failed; see logs/main.log.")
    if left() > 300:
        launch_kind("b8_gain", "b8", "lgbm", boost.lgbm_params(), 300, 5, "none", None)
    else:
        add_note("B8 skipped for time.")
    if left() > 500:
        try:
            _run_child(worker_b9, {
                "seeds": [seeds[0]], "log": log, "deadline": t0 + budget - 30,
                "grid_team": list(boost.GRID_TEAM), "grid_dead": list(boost.GRID_DEAD),
            })
            add_note("B9 ran on seed %s only." % seeds[0])
        except Exception:
            _log(log, "b9 failed\n" + traceback.format_exc())
            add_note("B9 failed; see logs/main.log.")
    else:
        add_note("B9 skipped for time.")
    persist_code()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    import bench.run_boost as _mod
    _mod.main()
