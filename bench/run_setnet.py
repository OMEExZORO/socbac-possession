"""SetNet experiments on the external bench (docs/team/BENCH.md protocol).

    python -m bench.run_setnet v2    --seeds 0 1 2          # CPU: v2 LightGBM emissions cache
    python -m bench.run_setnet train --seeds 0 1 2 --variant main   # GPU: LOMO + final SetNet
    python -m bench.run_setnet eval  --seeds 0 1 2 --variants main noball ...  # tables, FINAL data

TRAIN = J03WMX J03WN1 J03WOH J03WOY, TEST = J03WPY J03WQQ J03WR9, degradation seed rule
1000*seed + index (TRAIN then TEST), bench.degrade with PLAYER_SMOOTH = 12. Switch costs, epochs
and ensemble weights are chosen by leave-one-match-out on TRAIN only; TEST is scored once.
Outputs go to D:\\Barca_Hackathon\\research\\runs\\setnet (code copy, logs, cache, results).

Do not import numpy/torch at module level: thread limits are set first.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = r"D:\Barca_Hackathon\research\runs\setnet"
CACHE = os.path.join(RUN_DIR, "cache")
V2_GROUPS = ("context", "lang_dead")       # v2 = v1 + context + Lang dead-ball block (351 columns)
V2_GRID_T = [5.0, 10.0, 20.0, 40.0, 80.0]  # same grids as run_feats3 for v2 alone
V2_GRID_D = [10.0, 20.0, 40.0, 80.0, 160.0]
GRID_T = [2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]   # SetNet and ensembles (wider)
GRID_D = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
COARSE_T = [3.0, 8.0, 20.0]                # epoch screen
COARSE_D = [10.0, 40.0, 160.0]
ENS_W = [0.25, 0.5, 0.75]                  # weight on SetNet in the log-likelihood average
ADOPT = 0.03

VARIANTS = {
    "main": {},
    "noball": {"use_ball": False},
    "notemp": {"temporal": "none"},
    "nomask": {"p_mask": 0.0},
    "tcn": {"temporal": "tcn"},
    "small": {"d": 32, "hidden": 32},
    "bw3": {"boundary_w": 3.0},
}


def _threads(n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        os.environ[k] = str(n)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _nice():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _avail_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return float("nan")


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return float("nan")


class Log:
    def __init__(self, name):
        os.makedirs(os.path.join(RUN_DIR, "logs"), exist_ok=True)
        self.path = os.path.join(RUN_DIR, "logs", name + ".log")

    def __call__(self, msg):
        line = time.strftime("%H:%M:%S") + " " + msg
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


def persist_code():
    dst = os.path.join(RUN_DIR, "code")
    os.makedirs(dst, exist_ok=True)
    for name in ("setnet.py", "run_setnet.py"):
        shutil.copy2(os.path.join(HERE, name), os.path.join(dst, name))


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_npz(path, **arrays):
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_npz(path):
    import numpy as np
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# -- v2 (LightGBM on v1 + context + Lang) emissions --------------------------------------------

def v2_path(seed, match):
    return os.path.join(CACHE, "v2", "s%d_%s.npz" % (seed, match))


def _v2_collect(featdir, matches, stride):
    import numpy as np
    from bench.methods import SWAP
    xs, ys = [], []
    for m in matches:
        y = np.load(os.path.join(featdir, m + ".y.npy"))
        rows = np.arange(0, len(y), stride)
        for key, yy in (("x", y), ("xs", SWAP[y])):
            mm = np.load(os.path.join(featdir, "%s.%s.npy" % (m, key)), mmap_mode="r")
            xs.append(np.asarray(mm[rows], np.float32))
            ys.append(yy[rows])
            del mm
    return np.concatenate(xs), np.concatenate(ys)


def _v2_predict(model, featdir, m):
    import numpy as np
    from bench.methods import SWAP
    out = []
    for key in ("x", "xs"):
        mm = np.load(os.path.join(featdir, "%s.%s.npy" % (m, key)), mmap_mode="r")
        parts = [model.predict(np.asarray(mm[i:i + 20000], np.float32)) for i in range(0, mm.shape[0], 20000)]
        out.append(np.concatenate(parts))
        del mm
    return 0.5 * (out[0] + out[1][:, SWAP])


def cmd_v2(args):
    _threads(1)
    _nice()
    import numpy as np
    import lightgbm as lgb
    from bench import setnet
    from bench.feats3 import build_features
    from bench.methods import LGBMViterbi, swap_teams
    log = Log("v2")
    persist_code()
    log("v2 builder seeds %s groups %s PLAYER_SMOOTH %d avail %.1f GB" % (
        args.seeds, V2_GROUPS, setnet.PLAYER_SMOOTH, _avail_gb()))
    params = dict(LGBMViterbi.PARAMS)
    params.update(num_threads=1, deterministic=True, verbose=-1)
    for seed in args.seeds:
        if all(os.path.isfile(v2_path(seed, m)) for m in setnet.ORDER) and os.path.isfile(
                os.path.join(RUN_DIR, "results", "v2_s%d.json" % seed)):
            log("seed %d: v2 cache complete, skip" % seed)
            continue
        t0 = time.time()
        featdir = os.path.join(CACHE, "v2feat", "s%d" % seed)
        os.makedirs(featdir, exist_ok=True)
        digests = {}
        for m in setnet.ORDER:
            meta = os.path.join(featdir, m + ".meta.json")
            if os.path.isfile(meta):
                digests[m] = _load_json(meta)["digest"]
                continue
            t1 = time.time()
            deg, y = setnet.degraded(m, seed)
            digests[m] = setnet.digest(deg)
            x, names, _, _, _ = build_features(deg, V2_GROUPS)
            if not np.isfinite(x).all():
                raise RuntimeError("non-finite v2 features")
            np.save(os.path.join(featdir, m + ".x.npy"), x)
            nf = x.shape[1]
            del x
            gc.collect()
            xs, names2, _, _, _ = build_features(swap_teams(deg), V2_GROUPS)
            if names2 != names:
                raise RuntimeError("swap changed v2 schema")
            np.save(os.path.join(featdir, m + ".xs.npy"), xs)
            del xs
            np.save(os.path.join(featdir, m + ".y.npy"), np.asarray(y, np.int8))
            np.save(os.path.join(featdir, m + ".period.npy"), np.asarray(deg["period"], np.int8))
            del deg
            gc.collect()
            _save_json(meta, {"digest": digests[m], "feats": nf})
            log("seed %d %s features %d cols in %.0fs (rss %.2f GB, avail %.1f GB)" % (
                seed, m, nf, time.time() - t1, _rss_gb(), _avail_gb()))

        def fit(matches):
            t1 = time.time()
            x, y = _v2_collect(featdir, matches, 5)
            prior = np.bincount(y, minlength=3).astype(np.float64)
            prior /= prior.sum()
            ds = lgb.Dataset(x, y, params=params, free_raw_data=True)
            ds.construct()
            rows = len(y)
            del x, y
            gc.collect()
            model = lgb.train(params, ds, num_boost_round=300)
            del ds
            gc.collect()
            log("seed %d fit on %s rows %d in %.0fs (rss %.2f GB)" % (
                seed, ",".join(matches), rows, time.time() - t1, _rss_gb()))
            return model, prior

        def yp(m):
            return (np.load(os.path.join(featdir, m + ".y.npy")),
                    np.load(os.path.join(featdir, m + ".period.npy")))

        for k, m in enumerate(setnet.TRAIN):
            if os.path.isfile(v2_path(seed, m)):
                continue
            rest = [t for t in setnet.TRAIN if t != m]
            model, prior = fit(rest)
            p = _v2_predict(model, featdir, m)
            y, per = yp(m)
            _save_npz(v2_path(seed, m), proba=p.astype(np.float32), prior=prior,
                      em=setnet.emissions_from_proba(p, prior), kind=np.array("oof"),
                      digest=np.array(digests[m]), y=y, period=per)
            del model
            gc.collect()
        if not all(os.path.isfile(v2_path(seed, m)) for m in setnet.TEST):
            model, prior = fit(setnet.TRAIN)
            for m in setnet.TEST:
                p = _v2_predict(model, featdir, m)
                y, per = yp(m)
                _save_npz(v2_path(seed, m), proba=p.astype(np.float32), prior=prior,
                          em=setnet.emissions_from_proba(p, prior), kind=np.array("test"),
                          digest=np.array(digests[m]), y=y, period=per)
            del model
            gc.collect()
        # v2 alone: costs by LOMO on TRAIN, TEST once
        res = evaluate_combo(seed, {"v2": 1.0}, V2_GRID_T, V2_GRID_D, None, log)
        _save_json(os.path.join(RUN_DIR, "results", "v2_s%d.json" % seed), res)
        log("seed %d v2 TEST spell %.3f macro %.3f poss %.0f/%.0f costs %s/%s (oof %.3f) in %.0fs" % (
            seed, res["spell"], res["macro"], res["n_pred"], res["n_gt"], res["switch_team"],
            res["switch_dead"], res["oof_spell"], time.time() - t0))
        # the full matrices are only needed to predict; keep emissions only
        for f in os.listdir(featdir):
            if f.endswith(".npy"):
                os.remove(os.path.join(featdir, f))
    log("v2 builder done")


# -- evaluation of emissions (single model or log-likelihood ensembles) -------------------------

def _labels_period(seed, match):
    """Labels and periods stored next to the v2 emissions (same degraded copy, checked by digest)."""
    import numpy as np
    from bench import setnet
    for path in (v2_path(seed, match), setnet.prep_path(CACHE, seed, match)):
        if os.path.isfile(path):
            with np.load(path) as z:
                if "y" in z.files and "period" in z.files:
                    return z["y"], z["period"], str(z["digest"])
    raise FileNotFoundError("no labels for %s seed %d" % (match, seed))


def _em(name, seed, match, oof):
    """Emissions of a source: 'v2' or 'set:<variant>'. oof=True for TRAIN matches."""
    if name == "v2":
        z = _load_npz(v2_path(seed, match))
        return z["em"].astype("float64"), str(z["digest"])
    variant = name.split(":", 1)[1]
    z = _load_npz(os.path.join(CACHE, "emis", "%s_s%d_%s.npz" % (variant, seed, match)))
    return z["em"].astype("float64"), str(z["digest"])


def evaluate_combo(seed, weights, grid_t, grid_d, ens_grid, log):
    """weights: {source: w}. If ens_grid is given (list of w for the first non-v2 source), the
    weight is tuned with the costs on TRAIN out-of-fold emissions. Returns a result dict."""
    import numpy as np
    from bench import setnet
    names = list(weights)

    def combine(match, oof, w_override=None):
        tot = None
        digs = set()
        ws = dict(weights)
        if w_override is not None:
            first = [k for k in names if k != "v2"][0]
            ws = {k: (w_override if k == first else 1.0 - w_override) for k in names}
        for k in names:
            e, dg = _em(k, seed, match, oof)
            digs.add(dg)
            tot = ws[k] * e if tot is None else tot + ws[k] * e
        if len(digs) != 1:
            raise RuntimeError("sources saw different degraded copies of %s" % match)
        return tot

    ys, ps = [], []
    for m in setnet.TRAIN:
        y, p, _ = _labels_period(seed, m)
        ys.append(y)
        ps.append(p)
    best = None
    for w in (ens_grid or [None]):
        ems = [combine(m, True, w) for m in setnet.TRAIN]
        oof_spell, st, sd = setnet.tune_costs(ems, ys, ps, grid_t, grid_d)
        if best is None or oof_spell > best[0]:
            best = (oof_spell, st, sd, w)
    oof_spell, st, sd, w = best
    rows = []
    for m in setnet.TEST:
        y, p, _ = _labels_period(seed, m)
        pred = setnet.decode_periods(combine(m, False, w), p, st, sd)
        s, mf, npred, ngt = setnet.score(y, pred, p)
        rows.append({"match": m, "spell": s, "macro": mf, "n_pred": npred, "n_gt": ngt})
    return {
        "seed": seed, "sources": weights, "w_tuned": w, "switch_team": st, "switch_dead": sd,
        "oof_spell": oof_spell,
        "spell": float(np.mean([r["spell"] for r in rows])),
        "macro": float(np.mean([r["macro"] for r in rows])),
        "n_pred": float(np.sum([r["n_pred"] for r in rows])),
        "n_gt": float(np.sum([r["n_gt"] for r in rows])),
        "matches": rows,
    }


# -- SetNet training (GPU) ------------------------------------------------------------------------

def cmd_train(args):
    _threads(1)
    _nice()
    import numpy as np
    import torch
    from bench import setnet
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = args.variant
    cfg = dict(setnet.DEFAULT_CFG)
    cfg.update(VARIANTS[variant])
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.ckpt_every:
        cfg["ckpt_every"] = args.ckpt_every
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        cfg[k] = type(setnet.DEFAULT_CFG[k])(v) if not isinstance(setnet.DEFAULT_CFG[k], bool) else v.lower() in ("1", "true", "yes")
    log = Log("train_" + variant)
    persist_code()
    log("train %s seeds %s device %s cfg %s avail %.1f GB" % (variant, args.seeds, dev, json.dumps(cfg), _avail_gb()))
    setnet.check_viterbi()
    for seed in args.seeds:
        res_path = os.path.join(RUN_DIR, "results", "set_%s_s%d.json" % (variant, seed))
        if os.path.isfile(res_path) and not args.force:
            log("seed %d done already, skip" % seed)
            continue
        t0 = time.time()
        preps = [setnet.load_prep(CACHE, seed, m, log) for m in setnet.ORDER]
        digests = {m: str(p["digest"]) for m, p in zip(setnet.ORDER, preps)}
        ys = {m: p["y"] for m, p in zip(setnet.ORDER, preps)}
        pers = {m: p["period"] for m, p in zip(setnet.ORDER, preps)}
        bank = setnet.Bank(preps, setnet.ORDER, dev)
        del preps
        gc.collect()
        log("seed %d bank on %s (rss %.2f GB, gpu %.0f MB)" % (
            seed, dev, _rss_gb(), torch.cuda.memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0))

        def prior_of(matches):
            y = np.concatenate([ys[m] for m in matches])
            c = np.bincount(y, minlength=3).astype(np.float64)
            c = np.array([0.5 * (c[0] + c[1]), 0.5 * (c[0] + c[1]), c[2]])
            return c / c.sum()

        # LOMO folds: OOF probabilities of each held-out TRAIN match at every checkpoint
        oof = {}
        fold_prior = {}
        fold_times = []
        for k, m_out in enumerate(setnet.TRAIN):
            rest = [m for m in setnet.TRAIN if m != m_out]
            fold_prior[m_out] = prior_of(rest)
            t1 = time.time()

            def on_ckpt(model, epoch, m_out=m_out):
                p = setnet.predict_match(model, bank, m_out, cfg)
                oof.setdefault(epoch, {})[m_out] = p
                yy = ys[m_out]
                log("  fold %d epoch %d held-out acc %.3f nll %.4f" % (
                    k, epoch, float((p.argmax(1) == yy).mean()),
                    float(-np.mean(np.log(p[np.arange(len(yy)), yy] + 1e-9)))))

            setnet.train(bank, rest, cfg, seed=100 * seed + k, epochs=cfg["epochs"], on_ckpt=on_ckpt,
                         log=log if k == 0 else None)
            fold_times.append(round(time.time() - t1, 1))
            log("seed %d fold %d (%s held out) trained+predicted in %.0fs" % (seed, k, m_out, fold_times[-1]))
        # epoch screen on a coarse grid, then the full grid at the chosen epoch
        curve = {}
        for epoch in sorted(oof):
            ems = [setnet.emissions_from_proba(oof[epoch][m], fold_prior[m]) for m in setnet.TRAIN]
            s, st, sd = setnet.tune_costs(ems, [ys[m] for m in setnet.TRAIN], [pers[m] for m in setnet.TRAIN],
                                          COARSE_T, COARSE_D)
            nll = float(np.mean([-np.mean(np.log(oof[epoch][m][np.arange(len(ys[m])), ys[m]] + 1e-9))
                                 for m in setnet.TRAIN]))
            curve[epoch] = {"oof_spell_coarse": s, "team": st, "dead": sd, "oof_nll": nll}
            log("seed %d epoch %d OOF spell %.3f (coarse costs %.0f/%.0f) nll %.4f" % (seed, epoch, s, st, sd, nll))
        best_epoch = max(sorted(curve), key=lambda e: (curve[e]["oof_spell_coarse"], -e))
        ems = [setnet.emissions_from_proba(oof[best_epoch][m], fold_prior[m]) for m in setnet.TRAIN]
        oof_spell, st, sd = setnet.tune_costs(ems, [ys[m] for m in setnet.TRAIN], [pers[m] for m in setnet.TRAIN],
                                              GRID_T, GRID_D)
        log("seed %d chosen epoch %d, costs %.0f/%.0f, OOF spell %.3f" % (seed, best_epoch, st, sd, oof_spell))
        for m in setnet.TRAIN:
            p = oof[best_epoch][m]
            _save_npz(os.path.join(CACHE, "emis", "%s_s%d_%s.npz" % (variant, seed, m)),
                      proba=p.astype(np.float32), prior=fold_prior[m],
                      em=setnet.emissions_from_proba(p, fold_prior[m]), kind=np.array("oof"),
                      digest=np.array(digests[m]), epoch=np.int64(best_epoch))
        del oof
        gc.collect()
        # final model on all TRAIN for the chosen number of epochs
        t1 = time.time()
        model = setnet.train(bank, setnet.TRAIN, cfg, seed=100 * seed + 4, epochs=best_epoch)
        prior = prior_of(setnet.TRAIN)
        rows = []
        for m in setnet.TEST:
            p = setnet.predict_match(model, bank, m, cfg)
            em = setnet.emissions_from_proba(p, prior)
            _save_npz(os.path.join(CACHE, "emis", "%s_s%d_%s.npz" % (variant, seed, m)),
                      proba=p.astype(np.float32), prior=prior, em=em, kind=np.array("test"),
                      digest=np.array(digests[m]), epoch=np.int64(best_epoch))
            pred = setnet.decode_periods(em.astype(np.float64), pers[m], st, sd)
            s, mf, npred, ngt = setnet.score(ys[m], pred, pers[m])
            rows.append({"match": m, "spell": s, "macro": mf, "n_pred": npred, "n_gt": ngt})
            log("seed %d TEST %s %s spell %.3f macro %.3f poss %d/%d" % (seed, variant, m, s, mf, npred, ngt))
        if args.save_model:
            os.makedirs(os.path.join(CACHE, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(CACHE, "models", "%s_s%d.pt" % (variant, seed)))
        final_time = round(time.time() - t1, 1)
        res = {
            "variant": variant, "seed": seed, "cfg": cfg, "epoch": best_epoch,
            "switch_team": st, "switch_dead": sd, "oof_spell": oof_spell,
            "curve": {str(k): v for k, v in curve.items()}, "fold_seconds": fold_times,
            "final_seconds": final_time, "seconds": round(time.time() - t0, 1),
            "spell": float(np.mean([r["spell"] for r in rows])),
            "macro": float(np.mean([r["macro"] for r in rows])),
            "n_pred": float(np.sum([r["n_pred"] for r in rows])),
            "n_gt": float(np.sum([r["n_gt"] for r in rows])),
            "matches": rows,
            "n_params": int(sum(p.numel() for p in model.parameters())),
        }
        _save_json(res_path, res)
        log("seed %d %s TEST spell %.3f macro %.3f poss %.0f/%.0f in %.0fs" % (
            seed, variant, res["spell"], res["macro"], res["n_pred"], res["n_gt"], res["seconds"]))
        if args.probe_drop and dev.type == "cuda":
            probe = {}
            for share in args.probe_drop:
                srow = []
                for m in setnet.TEST:
                    p = setnet.predict_match(model, bank, m, cfg, drop_players=share, seed=seed)
                    em = setnet.emissions_from_proba(p, prior).astype(np.float64)
                    pred = setnet.decode_periods(em, pers[m], st, sd)
                    srow.append(setnet.score(ys[m], pred, pers[m]))
                probe[str(share)] = {"spell": float(np.mean([r[0] for r in srow])),
                                     "macro": float(np.mean([r[1] for r in srow])),
                                     "n_pred": float(np.sum([r[2] for r in srow]))}
                log("seed %d probe drop %.2f of players: spell %.3f macro %.3f" % (
                    seed, share, probe[str(share)]["spell"], probe[str(share)]["macro"]))
            res["probe_drop"] = probe
            _save_json(res_path, res)
        del model, bank
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    log("train %s done" % variant)


# -- evaluation / tables -------------------------------------------------------------------------

def cmd_eval(args):
    _threads(1)
    _nice()
    import numpy as np
    log = Log("eval")
    persist_code()
    table = {}
    for seed in args.seeds:
        if not os.path.isfile(os.path.join(RUN_DIR, "results", "v2_s%d.json" % seed)):
            log("seed %d: v2 missing, skip" % seed)
            continue
        v2 = _load_json(os.path.join(RUN_DIR, "results", "v2_s%d.json" % seed))
        table.setdefault("v2", {})[seed] = v2
        # control: v2 alone decoded with the wider grid the ensembles use
        wp = os.path.join(RUN_DIR, "results", "v2wide_s%d.json" % seed)
        if os.path.isfile(wp) and not args.force:
            table.setdefault("v2:widegrid", {})[seed] = _load_json(wp)
        else:
            res = evaluate_combo(seed, {"v2": 1.0}, GRID_T, GRID_D, None, log)
            _save_json(wp, res)
            table.setdefault("v2:widegrid", {})[seed] = res
            log("seed %d v2 wide grid TEST spell %.3f costs %s/%s (oof %.3f)" % (
                seed, res["spell"], res["switch_team"], res["switch_dead"], res["oof_spell"]))
        for variant in args.variants:
            rp = os.path.join(RUN_DIR, "results", "set_%s_s%d.json" % (variant, seed))
            if not os.path.isfile(rp):
                continue
            table.setdefault("set:" + variant, {})[seed] = _load_json(rp)
            for tag, wgrid in (("ens50", None), ("ensW", ENS_W)):
                key = "%s+v2:%s" % (variant, tag)
                cp = os.path.join(RUN_DIR, "results", "%s_s%d.json" % (key.replace(":", "_").replace("+", "_"), seed))
                if os.path.isfile(cp) and not args.force:
                    table.setdefault(key, {})[seed] = _load_json(cp)
                    continue
                res = evaluate_combo(seed, {"set:" + variant: 0.5, "v2": 0.5}, GRID_T, GRID_D, wgrid, log)
                _save_json(cp, res)
                table.setdefault(key, {})[seed] = res
                log("seed %d %s TEST spell %.3f macro %.3f poss %.0f/%.0f w %s costs %s/%s (oof %.3f)" % (
                    seed, key, res["spell"], res["macro"], res["n_pred"], res["n_gt"], res["w_tuned"],
                    res["switch_team"], res["switch_dead"], res["oof_spell"]))
    _save_json(os.path.join(RUN_DIR, "results", "table.json"),
               {k: {str(s): v for s, v in d.items()} for k, d in table.items()})
    # per-frame diagnostics on TEST: argmax accuracy with the ball visible vs hidden, and the
    # team call on truly alive frames (H vs A only), for v2, each SetNet variant and the
    # geometric-mean ensemble of the two posteriors. Aggregates only.
    from bench import setnet
    diag = {}
    for seed in args.seeds:
        if "v2" not in table or seed not in table["v2"]:
            continue
        for m in setnet.TEST:
            pv = _load_npz(v2_path(seed, m))
            prep = _load_npz(setnet.prep_path(CACHE, seed, m))
            if str(prep["digest"]) != str(pv["digest"]):
                raise RuntimeError("digest mismatch %s seed %d" % (m, seed))
            y, vis = prep["y"], prep["vis"]
            srcs = {"v2": pv["proba"].astype(np.float64)}
            for variant in args.variants:
                ep = os.path.join(CACHE, "emis", "%s_s%d_%s.npz" % (variant, seed, m))
                if os.path.isfile(ep):
                    ps = _load_npz(ep)["proba"].astype(np.float64)
                    srcs["set:" + variant] = ps
                    g = 0.5 * np.log(ps + 1e-6) + 0.5 * np.log(srcs["v2"] + 1e-6)
                    g = np.exp(g - g.max(1, keepdims=True))
                    srcs["geo:" + variant] = g / g.sum(1, keepdims=True)
            alive = y != setnet.D
            for name, p in srcs.items():
                am = p.argmax(1)
                team = np.where(p[:, 0] >= p[:, 1], setnet.H, setnet.A)
                rec = diag.setdefault(name, {}).setdefault(str(seed), {"vis": [], "hid": [], "team_vis": [], "team_hid": []})
                rec["vis"].append(float((am[vis] == y[vis]).mean()))
                rec["hid"].append(float((am[~vis] == y[~vis]).mean()))
                rec["team_vis"].append(float((team[vis & alive] == y[vis & alive]).mean()))
                rec["team_hid"].append(float((team[~vis & alive] == y[~vis & alive]).mean()))
    for name, d in diag.items():
        vals = {k: float(np.mean([np.mean(d[s][k]) for s in d])) for k in ("vis", "hid", "team_vis", "team_hid")}
        d["mean"] = vals
        log("diag %s: frame acc ball visible %.3f hidden %.3f | team acc (alive) visible %.3f hidden %.3f" % (
            name, vals["vis"], vals["hid"], vals["team_vis"], vals["team_hid"]))
    _save_json(os.path.join(RUN_DIR, "results", "diag.json"), diag)
    # summary lines
    seeds_v2 = sorted(table.get("v2", {}))
    log("method | seeds | spellF1 mean+-sd | per-seed | gain vs v2 per seed | macroF1 | poss pred/gt | adopt")
    for key, d in table.items():
        ss = sorted(d)
        s = np.array([d[x]["spell"] for x in ss])
        mf = np.mean([d[x]["macro"] for x in ss])
        gains = [d[x]["spell"] - table["v2"][x]["spell"] for x in ss if x in table["v2"]]
        adopt = (key != "v2" and len(gains) == len(seeds_v2) and len(gains) > 0 and min(gains) >= ADOPT)
        log("%s | %s | %.3f +- %.3f | %s | %s | %.3f | %.0f/%.0f | %s" % (
            key, ss, s.mean(), s.std(), " ".join("%.3f" % v for v in s),
            " ".join("%+.3f" % g for g in gains), mf,
            np.mean([d[x]["n_pred"] for x in ss]), np.mean([d[x]["n_gt"] for x in ss]),
            "YES" if adopt else "no"))


def _label(key):
    if key == "v2":
        return "v2 (LightGBM, v1+context+Lang)"
    if key == "v2:widegrid":
        return "v2, wide cost grid (control)"
    if key.startswith("set:"):
        return "SetNet %s alone" % key[4:]
    variant, tag = key.split("+v2:")
    return "SetNet %s + v2, %s" % (variant, "w=0.5" if tag == "ens50" else "w by LOMO")


def cmd_report(args):
    import numpy as np
    table = _load_json(os.path.join(RUN_DIR, "results", "table.json"))
    v2 = table["v2"]
    seeds_v2 = sorted(v2, key=int)
    order = ["v2"] + (["v2:widegrid"] if "v2:widegrid" in table else [])
    for variant in VARIANTS:
        for key in ("set:" + variant, variant + "+v2:ens50", variant + "+v2:ensW"):
            if key in table:
                order.append(key)
    out = ["| method | seeds | spellF1 mean +- sd | per-seed spellF1 | gain vs v2 per seed | macroF1 | poss pred / true (3 matches) | adopt |",
           "|---|---|---|---|---|---:|---|---|"]
    for key in order:
        d = table[key]
        ss = sorted(d, key=int)
        s = np.array([d[x]["spell"] for x in ss])
        gains = [d[x]["spell"] - v2[x]["spell"] for x in ss if x in v2]
        full = len(ss) == len(seeds_v2)
        adopt = "baseline" if key == "v2" else "control" if key == "v2:widegrid" else (
            ("YES" if min(gains) >= ADOPT else "no") if full else "incomplete")
        out.append("| %s | %s | %.3f +- %.3f | %s | %s | %.3f | %.0f / %.0f | %s |" % (
            _label(key), ",".join(ss), s.mean(), s.std(), " / ".join("%.3f" % v for v in s),
            "" if key == "v2" else " / ".join("%+.3f" % g for g in gains),
            np.mean([d[x]["macro"] for x in ss]),
            np.mean([d[x]["n_pred"] for x in ss]), np.mean([d[x]["n_gt"] for x in ss]), adopt))
    out += ["", "Per seed (TEST = mean of 3 matches; costs and epochs chosen by LOMO on TRAIN):", "",
            "| method | seed | spellF1 | macroF1 | poss pred / true | OOF spellF1 (TRAIN) | team / dead cost | epoch | w(SetNet) |",
            "|---|---:|---:|---:|---|---:|---|---:|---:|"]
    for key in order:
        for x in sorted(table[key], key=int):
            r = table[key][x]
            w = r.get("w_tuned")
            if w is None and key.endswith("ens50"):
                w = 0.5
            out.append("| %s | %s | %.3f | %.3f | %.0f / %.0f | %.3f | %s / %s | %s | %s |" % (
                _label(key), x, r["spell"], r["macro"], r["n_pred"], r["n_gt"], r["oof_spell"],
                r["switch_team"], r["switch_dead"], r.get("epoch", ""), "" if w is None else w))
    out += ["", "Per match:", "", "| method | seed | match | spellF1 | macroF1 | poss pred / true |",
            "|---|---:|---|---:|---:|---|"]
    for key in order:
        for x in sorted(table[key], key=int):
            for r in table[key][x]["matches"]:
                out.append("| %s | %s | %s | %.3f | %.3f | %d / %d |" % (
                    _label(key), x, r["match"], r["spell"], r["macro"], r["n_pred"], r["n_gt"]))
    out += ["", "SetNet out-of-fold curves (TRAIN, LOMO, coarse cost grid), OOF spellF1 by epoch:", ""]
    for key in order:
        if not key.startswith("set:"):
            continue
        for x in sorted(table[key], key=int):
            c = table[key][x].get("curve", {})
            pts = " ".join("%s:%.3f" % (e, c[e]["oof_spell_coarse"]) for e in sorted(c, key=int))
            out.append("- %s seed %s: %s (chosen %s, %s params, %.0f s)" % (
                key[4:], x, pts, table[key][x].get("epoch"), table[key][x].get("n_params"),
                table[key][x].get("seconds", 0)))
            pr = table[key][x].get("probe_drop")
            if pr:
                out.append("  - test-time player-token drop probe (same costs): " + "; ".join(
                    "%s of players hidden in 2 s blocks -> spellF1 %.3f, macroF1 %.3f, poss %.0f" % (
                        k, v["spell"], v["macro"], v["n_pred"]) for k, v in sorted(pr.items())))
    dp = os.path.join(RUN_DIR, "results", "diag.json")
    if os.path.isfile(dp):
        diag = _load_json(dp)
        out += ["", "Per-frame diagnostics on TEST (argmax of the posterior, before decoding; mean over seeds and matches):", "",
                "| source | frame acc, ball visible | frame acc, ball hidden | team acc on alive frames, visible | team acc on alive frames, hidden |",
                "|---|---:|---:|---:|---:|"]
        for name, d in diag.items():
            v = d["mean"]
            label = {"v2": "v2"}.get(name, name.replace("set:", "SetNet ").replace("geo:", "geo-mean SetNet+v2 "))
            out.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (label, v["vis"], v["hid"], v["team_vis"], v["team_hid"]))
    path = os.path.join(RUN_DIR, "TABLES.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[:2 + len(order)]))


def cmd_selftest(args):
    _threads(1)
    from bench import setnet
    setnet.self_test()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["v2", "train", "eval", "report", "selftest"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variant", default="main", choices=sorted(VARIANTS))
    ap.add_argument("--variants", nargs="+", default=["main"])
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--set", nargs="*", help="cfg overrides key=value")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--probe-drop", type=float, nargs="*", default=[])
    args = ap.parse_args(argv)
    if args.cmd == "v2":
        cmd_v2(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "report":
        cmd_report(args)
    else:
        cmd_selftest(args)


if __name__ == "__main__":
    main()
