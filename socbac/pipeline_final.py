"""Final SOCBAC submission pipeline: features v3 LightGBM + set-transformer ("setnet big") blend.

    python -m socbac.pipeline_final --out evaluation/results/socbac_final.csv            # submission
    python -m socbac.pipeline_final --out evaluation/results/socbac_final.csv --fresh    # no cache read
    python -m socbac.pipeline_final --bench        # same code path on open data: TRAIN 4 -> TEST 3
    python -m socbac.pipeline_final --det-test     # determinism probes (setnet train / inference)

Frozen recipe (bench evidence: research/runs/final_recipe/FINAL.md, this pipeline:
research/runs/pipeline_final/FINAL.md):
  1. Training data: the 7 open IDSSE/DFL matches (bench.dfl, CC-BY 4.0) only. Each is degraded
     once with bench.degrade_v4 (label-independent ball hiding, PLAYER_SMOOTH 12), seed
     1000*0 + match index (bench seed 0). No label, frame or statistic of BAR-ATH is used for
     training or for choosing anything.
  2. Targets: pooled L1 + L2 (bench.labels5: opposing runs shorter than 1 s / 2 s merged).
     LightGBM: frames where L1 and L2 differ are two rows of weight 0.5; weight x20 within +-2 s
     of a label change. Setnet: mean cross-entropy over both definitions.
  3. Features v3 = feats3 v1 + context + lang_dead (351 columns) + look-ahead t+25/50/125
     (bench.run_v3cand.Spec("v3")), both team views; team-symmetric LightGBM (LGBMViterbi
     params, deterministic, fixed thread count, 300 rounds, stride 5).
  4. Setnet big (socbac.setnet_pool, d 96, 3 attention blocks), swap+mirror test-time views.
  5. Blend: 0.5 * GBM log-likelihood + 0.5 * setnet log-likelihood (fixed), 3-state Viterbi per
     half. The setnet epoch count and the two switch costs are chosen by leave-one-match-out on
     the 4 bench TRAIN matches only (exactly research/runs/final_recipe/code/run_final.py); the
     final models are then trained on all 7 matches with those settings.
  6. BAR-ATH (real data handling copied from socbac.pipeline_v2): socbac.atd.load, graded halves
     from games.half_video_blocks, graded frames up to the published kick-off anchor are D, X
     outside the graded halves, CSV frame,label, then evaluation/validate_submission.py.

Determinism: all seeds fixed; torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG,
cudnn deterministic, one torch CPU thread, fixed LightGBM / BLAS thread counts.
Prints aggregates only (shapes, timings, hashes, label shares of OUR output). Per-frame
intermediates and model weights live under data/derived/pipeline_final/ (gitignored).
"""

import os

# fixed thread counts and deterministic cuBLAS, before numpy / torch are imported
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from socbac import setnet_pool as sp  # noqa: E402

CACHE = os.path.join(ROOT, "data", "derived", "pipeline_final")
REV = "pf1"
SEED = 0                                   # bench degradation seed (1000*SEED + match index)
TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
TEST = ["J03WPY", "J03WQQ", "J03WR9"]
ORDER = TRAIN + TEST
GROUPS = ("context", "lang_dead")
N_FEATURES = 351
STRIDE, ROUNDS = 5, 300
BOOST = 20.0                               # GBM boundary weight within +-2 s
NT = 4                                     # LightGBM threads (fixed: part of the recipe)
W_SET = 0.5                                # fixed setnet weight in the log-likelihood blend
SET_VARIANT = {"d": 96, "n_sab": 3}        # "big"
SEED_TRAIN4 = 100 * SEED + 4               # run_final's TRAIN-4 setnet seed (bench mode)
SEED_FINAL = 100 * SEED + 7                # final setnet on all 7 matches
GRID_T = [2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
GRID_D = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]
COARSE_T = [3.0, 8.0, 20.0]
COARSE_D = [10.0, 40.0, 160.0]
DEFS = ("L0", "L1", "L2", "L3")
FPS = 25


def log(msg):
    print(time.strftime("%H:%M:%S") + " " + msg, flush=True)


def nice():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def wait_ram(need=2.5):
    import psutil
    while psutil.virtual_memory().available / 2 ** 30 < need:
        log("free RAM %.1f GB < %.1f, waiting" % (psutil.virtual_memory().available / 2 ** 30, need))
        time.sleep(60)
    if shutil.disk_usage(ROOT).free / 2 ** 30 < 6:
        raise SystemExit("free disk < 6 GB, stop")


def sha(arr_or_bytes):
    h = hashlib.sha256()
    if isinstance(arr_or_bytes, (bytes, bytearray)):
        h.update(arr_or_bytes)
    else:
        a = np.ascontiguousarray(arr_or_bytes)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(arr))
    os.replace(tmp, path)


def _save_npz(path, **a):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **a)
    os.replace(tmp, path)


# -- torch determinism ------------------------------------------------------------------------------

def torch_setup(device):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_cfg():
    cfg = dict(sp.DEFAULT_CFG)
    cfg.update(SET_VARIANT)
    return cfg


def new_setnet(cfg, dev):
    tp = sp.torch_parts()
    return tp.SetNet(d=cfg["d"], heads=cfg["heads"], n_sab=cfg["n_sab"], temporal=cfg["temporal"],
                     hidden=cfg["hidden"], rnn_layers=cfg["rnn_layers"], drop=cfg["drop"]).to(dev)


def save_setnet(model, path):
    import torch
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)


def load_setnet(path, cfg, dev):
    import torch
    model = new_setnet(cfg, dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.eval()
    return model


# -- open data: degradation, labels, features ----------------------------------------------------

def odir(cache):
    return os.path.join(cache, "open_s%d" % SEED)


def build_open(cache):
    """Degraded copy (degrade_v4, seed 1000*SEED + index), labels L0-L3, setnet prep and the 351
    v2 feature columns of both team views, for each of the 7 open matches (cached)."""
    from bench import feats3, labels5, methods
    from bench.dfl import load
    from bench.degrade_v4 import DEGRADERS
    d = odir(cache)
    os.makedirs(d, exist_ok=True)
    for i, m in enumerate(ORDER):
        stem = os.path.join(d, m)
        if os.path.isfile(stem + ".meta.json"):
            continue
        wait_ram()
        t0 = time.time()
        clean = load(m)
        labs = labels5.all_labels(clean)
        deg = DEGRADERS["v4"](clean, 1000 * SEED + i)
        n = len(deg["period"])
        if not all(len(v) == n for v in labs.values()):
            raise RuntimeError("label length mismatch " + m)
        del clean
        dig = sp.digest(deg)
        prep = sp.prepare(deg, labs["L1"])
        prep["y2"] = labs["L2"].astype(np.int8)
        prep["digest"] = np.array(dig)
        _save_npz(stem + ".prep.npz", **prep)
        del prep
        _save_npz(stem + ".labs.npz", **labs)
        x, names, _, sub, _ = feats3.build_features(deg, GROUPS)
        if len(names) != N_FEATURES or not np.isfinite(x).all():
            raise RuntimeError("unexpected feature schema / non-finite features")
        _save_npy(stem + ".x.npy", x.astype(np.float32))
        del x
        xs, n2, _, _, _ = feats3.build_features(methods.swap_teams(deg), GROUPS)
        if n2 != names or not np.isfinite(xs).all():
            raise RuntimeError("swapped view schema mismatch")
        _save_npy(stem + ".xs.npy", xs.astype(np.float32))
        del xs
        _save_npy(stem + ".period.npy", np.asarray(deg["period"], np.int8))
        meta = {"rev": REV, "names": names, "sub": list(sub), "digest": dig, "frames": n,
                "hidden": float(deg["ball_hidden_share"])}
        del deg
        with open(stem + ".meta.json.tmp", "w") as fh:
            json.dump(meta, fh)
        os.replace(stem + ".meta.json.tmp", stem + ".meta.json")
        gc.collect()
        log("open %s: %d frames, ball hidden %.3f, digest %s (%.0fs)" % (m, n, meta["hidden"], dig, time.time() - t0))


class Open:
    """Lazy access to the cached open-match arrays."""

    def __init__(self, cache):
        self.d = odir(cache)
        self.meta = {}
        for m in ORDER:
            with open(os.path.join(self.d, m + ".meta.json")) as fh:
                self.meta[m] = json.load(fh)
            if self.meta[m].get("rev") != REV:
                raise SystemExit("open cache of another recipe revision: run with --fresh")
        self.names = self.meta[ORDER[0]]["names"]
        self.fmeta = {"names": self.names, "sub": self.meta[ORDER[0]]["sub"]}
        self._labs, self._rows = {}, {}

    def period(self, m):
        return np.load(os.path.join(self.d, m + ".period.npy"))

    def labs(self, m):
        if m not in self._labs:
            with np.load(os.path.join(self.d, m + ".labs.npz")) as z:
                self._labs[m] = {k: z[k] for k in z.files}
        return self._labs[m]

    def full(self, m):
        return {"x": np.load(os.path.join(self.d, m + ".x.npy"), mmap_mode="r"),
                "xs": np.load(os.path.join(self.d, m + ".xs.npy"), mmap_mode="r"),
                "period": self.period(m)}

    def rows_v3(self, m, spec, keep=True):
        """Stride-5 rows of the v3 view of both team views (kept in memory if keep)."""
        if not keep:
            self._rows.clear()
            gc.collect()
        if m not in self._rows:
            f = self.full(m)
            rows = np.arange(0, len(f["period"]), STRIDE)
            out = []
            for key in ("x", "xs"):
                v = spec.view(f[key], self.fmeta, f["period"])
                out.append(np.ascontiguousarray(v[rows]))
                del v
            self._rows[m] = tuple(out)
        return self._rows[m]

    def prep(self, m):
        with np.load(os.path.join(self.d, m + ".prep.npz")) as z:
            return {k: z[k] for k in z.files}


# -- LightGBM v3 ----------------------------------------------------------------------------------

def gbm_params():
    from bench.methods import LGBMViterbi
    p = dict(LGBMViterbi.PARAMS)
    p.update(num_threads=NT, deterministic=True, verbose=-1)
    return p


def gbm_fit(op, spec, ms):
    """Pooled L1+L2 rows (bench.run_hedge.targets, boundary x20) of matches ms, both views."""
    import lightgbm as lgb
    from bench.methods import SWAP
    from bench.run_hedge import targets
    # one preallocated matrix, filled in the same row order as before (match by match, view x then
    # xs); rows of the v3 view are recomputed per match instead of kept for all 7 (lower peak RAM)
    tg = {}
    for m in ms:
        per = op.period(m)
        tg[m] = targets("L12", op.labs(m), np.arange(0, len(per), STRIDE), per, BOOST)
    total = sum(2 * len(tg[m][0]) for m in ms)
    x = None
    ys_, ws_ = [], []
    at = 0
    for m in ms:
        idx, y, w = tg[m]
        xr, xsr = op.rows_v3(m, spec, keep=len(ms) < len(ORDER))
        if x is None:
            x = np.empty((total, xr.shape[1]), np.float32)
        for part, lab in ((xr, y), (xsr, SWAP[y])):
            x[at:at + len(idx)] = part[idx]
            at += len(idx)
            ys_.append(lab)
            ws_.append(w)
        del xr, xsr
        gc.collect()
    y, w = np.concatenate(ys_), np.concatenate(ws_)
    wait_ram()
    model = lgb.train(gbm_params(), lgb.Dataset(x, y, weight=w), num_boost_round=ROUNDS)
    prior = np.bincount(y, weights=w, minlength=3).astype(np.float64)
    del x, y, w
    gc.collect()
    return model, prior / prior.sum()


def gbm_em(model, prior, data, fmeta, spec):
    from bench import run_v3cand as rv
    return rv.emissions(model, prior, data, fmeta, spec)


def gbm_stage_oof(op, spec, cache):
    """LOMO on TRAIN: OOF emissions of the GBM for each TRAIN match (cached)."""
    out = {}
    for hold in TRAIN:
        path = os.path.join(cache, "gbm_oof_%s.npy" % hold)
        if not os.path.isfile(path):
            t0 = time.time()
            model, prior = gbm_fit(op, spec, [t for t in TRAIN if t != hold])
            _save_npy(path, gbm_em(model, prior, op.full(hold), op.fmeta, spec).astype(np.float32))
            del model
            log("gbm LOMO fold %s out (%.0fs)" % (hold, time.time() - t0))
        out[hold] = np.load(path).astype(np.float64)
    return out


def gbm_model(op, spec, cache, ms, name):
    """GBM trained on matches ms, saved as text (cached). Returns (booster, prior, sha)."""
    import lightgbm as lgb
    path = os.path.join(cache, "gbm_%s.txt" % name)
    ppath = os.path.join(cache, "gbm_%s_prior.npy" % name)
    if not (os.path.isfile(path) and os.path.isfile(ppath)):
        t0 = time.time()
        model, prior = gbm_fit(op, spec, ms)
        model.save_model(path + ".tmp")
        os.replace(path + ".tmp", path)
        _save_npy(ppath, prior)
        log("gbm %s fitted on %d matches (%.0fs)" % (name, len(ms), time.time() - t0))
    model = lgb.Booster(model_file=path)
    model.params["num_threads"] = NT
    return model, np.load(ppath), file_sha256(path)


# -- setnet ---------------------------------------------------------------------------------------

def prior_of(op, matches):
    """Pooled labels: mean of the L1 and L2 class counts, H/A symmetrised (as run_final)."""
    c = sum(np.bincount(op.labs(m)[k], minlength=3).astype(np.float64) for m in matches for k in ("L1", "L2")) / 2
    c = np.array([0.5 * (c[0] + c[1]), 0.5 * (c[0] + c[1]), c[2]])
    return c / c.sum()


def score_defs(labs_m, pred, period, defs):
    from bench.run import score
    return {k: float(score(labs_m[k], pred, period)[0]) for k in defs}


def tune(ems, periods, labs_list, grid_t, grid_d, defs=("L1", "L2")):
    """Switch costs maximising mean over matches and over defs of spellF1 (run_final.tune)."""
    best = None
    for st in grid_t:
        for sd in grid_d:
            s = []
            for e, p, lb in zip(ems, periods, labs_list):
                pred = sp.decode_periods(np.asarray(e, np.float64), p, st, sd)
                s.append(np.mean(list(score_defs(lb, pred, p, defs).values())))
            if best is None or np.mean(s) > best[0]:
                best = (float(np.mean(s)), st, sd)
    return best


def bank_of(op, ms, dev):
    preps = [op.prep(m) for m in ms]
    bank = sp.Bank(preps, ms, dev)
    del preps
    gc.collect()
    return bank


def set_stage_lomo(op, bank, cfg, cache):
    """run_final.setnet_stage, LOMO part: per-checkpoint OOF on TRAIN -> epoch; OOF emissions."""
    path = os.path.join(cache, "set_lomo.npz")
    if not os.path.isfile(path):
        t0 = time.time()
        oof, fprior = {}, {}
        for k, m_out in enumerate(TRAIN):
            rest = [m for m in TRAIN if m != m_out]
            fprior[m_out] = prior_of(op, rest)
            t1 = time.time()

            def on_ckpt(model, epoch, m_out=m_out):
                oof.setdefault(epoch, {})[m_out] = sp.predict_match(model, bank, m_out, cfg)

            sp.train(bank, rest, cfg, seed=100 * SEED + k, epochs=cfg["epochs"], on_ckpt=on_ckpt)
            log("setnet LOMO fold %d (%s out) %.0fs" % (k, m_out, time.time() - t1))
        curve = {}
        for epoch in sorted(oof):
            ems = [sp.emissions_from_proba(oof[epoch][m], fprior[m]) for m in TRAIN]
            s, st, sd = tune(ems, [op.period(m) for m in TRAIN], [op.labs(m) for m in TRAIN], COARSE_T, COARSE_D)
            curve[epoch] = s
            log("setnet epoch %d OOF pooled spellF1 %.3f (costs %.0f/%.0f)" % (epoch, s, st, sd))
        best_epoch = max(sorted(curve), key=lambda e: (curve[e], -e))
        arrays = {"oof_" + m: sp.emissions_from_proba(oof[best_epoch][m], fprior[m]) for m in TRAIN}
        _save_npz(path, epoch=np.int64(best_epoch), curve_e=np.array(sorted(curve)),
                  curve_s=np.array([curve[e] for e in sorted(curve)]), **arrays)
        log("setnet chosen epoch %d (LOMO %.0fs)" % (best_epoch, time.time() - t0))
    with np.load(path) as z:
        return int(z["epoch"]), {m: z["oof_" + m].astype(np.float64) for m in TRAIN}


def set_model(op, bank, cfg, cache, ms, name, seed, epochs, dev):
    path = os.path.join(cache, "set_%s.pt" % name)
    if not os.path.isfile(path):
        t0 = time.time()
        model = sp.train(bank, ms, cfg, seed=seed, epochs=epochs)
        save_setnet(model, path)
        log("setnet %s trained on %d matches, %d epochs (%.0fs), weights sha256 %s" % (
            name, len(ms), epochs, time.time() - t0, state_sha(model)))
        del model
    model = load_setnet(path, cfg, dev)
    return model, state_sha(model)


# -- shared TRAIN-only choices --------------------------------------------------------------------

def choose(op, spec, cache, dev):
    """Epoch and switch costs from LOMO on the 4 TRAIN matches only (cached in choice.json)."""
    path = os.path.join(cache, "choice.json")
    g_oof = gbm_stage_oof(op, spec, cache)
    cfg = set_cfg()
    if os.path.isfile(path):
        with open(path) as fh:
            ch = json.load(fh)
        return ch, g_oof, None
    bank = bank_of(op, TRAIN, dev)
    epoch, s_oof = set_stage_lomo(op, bank, cfg, cache)
    del bank
    gc.collect()
    ems = [(1 - W_SET) * g_oof[m] + W_SET * s_oof[m] for m in TRAIN]
    s, st, sd = tune(ems, [op.period(m) for m in TRAIN], [op.labs(m) for m in TRAIN], GRID_T, GRID_D)
    ch = {"rev": REV, "epoch": epoch, "switch_team": st, "switch_dead": sd, "oof_pooled": s}
    with open(path, "w") as fh:
        json.dump(ch, fh, indent=1)
    log("TRAIN-only LOMO choice: setnet epoch %d, switch costs team %.0f dead %.0f (OOF pooled %.3f)" % (
        epoch, st, sd, s))
    return ch, g_oof, s_oof


# -- bench mode -----------------------------------------------------------------------------------

def run_bench(cache, dev, report):
    from bench import run_v3cand as rv
    spec = rv.CONFIGS["v3"]()
    op = Open(cache)
    ch, g_oof, _ = choose(op, spec, cache, dev)
    with np.load(os.path.join(cache, "set_lomo.npz")) as z:
        s_oof = {m: z["oof_" + m].astype(np.float64) for m in TRAIN}
    cfg = set_cfg()
    gm, gprior, gsha = gbm_model(op, spec, cache, TRAIN, "train4")
    em_g = {m: gbm_em(gm, gprior, op.full(m), op.fmeta, spec) for m in TEST}
    bank = bank_of(op, ORDER, dev)
    model, ssha = set_model(op, bank, cfg, cache, TRAIN, "train4", SEED_TRAIN4, ch["epoch"], dev)
    prior = prior_of(op, TRAIN)
    em_s = {m: sp.emissions_from_proba(sp.predict_match(model, bank, m, cfg), prior).astype(np.float64) for m in TEST}
    del bank, model
    out = {"seed": SEED, "choice": ch, "gbm_train4_sha256": gsha, "set_train4_sha256": ssha, "methods": {}}
    ptr, ltr = [op.period(m) for m in TRAIN], [op.labs(m) for m in TRAIN]
    methods = {"v3": (1.0, 0.0), "setnet big": (0.0, 1.0), "v3+big w=.5": (1 - W_SET, W_SET)}
    for name, (wg, ws) in methods.items():
        if name == "v3+big w=.5":
            st, sd, oofs = ch["switch_team"], ch["switch_dead"], ch["oof_pooled"]
        else:
            oofs, st, sd = tune([wg * g_oof[m] + ws * s_oof[m] for m in TRAIN], ptr, ltr, GRID_T, GRID_D)
        per = {k: {} for k in DEFS}
        for m in TEST:
            e = wg * em_g[m] + ws * em_s[m]
            pred = sp.decode_periods(e, op.period(m), st, sd)
            for k, v in score_defs(op.labs(m), pred, op.period(m), DEFS).items():
                per[k][m] = v
        res = {k: float(np.mean(list(v.values()))) for k, v in per.items()}
        out["methods"][name] = {"costs": [st, sd], "oof_pooled": oofs, "test": res, "test_per_match": per}
        log("bench s%d %-12s costs %.0f/%.0f TEST %s" % (SEED, name, st, sd,
                                                        " ".join("%s %.3f" % (k, res[k]) for k in DEFS)))
    if report:
        os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
        with open(report, "w") as fh:
            json.dump(out, fh, indent=1)
    return out


# -- determinism probe ----------------------------------------------------------------------------

def run_det_test(cache, dev, epochs, cpu_probe):
    import torch
    op = Open(cache)
    cfg = set_cfg()
    ms = TRAIN[:2]
    bank = bank_of(op, ms, dev)
    res = {}
    shas, times = [], []
    for rep in range(2):
        t0 = time.time()
        model = sp.train(bank, ms[:1], cfg, seed=123, epochs=epochs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        shas.append(state_sha(model))
        if rep == 0:
            m0 = model
    res["train_bitwise"] = shas[0] == shas[1]
    log("det-test %s: train %d epochs on 1 match twice: %.1fs / %.1fs, weights equal %s" % (
        dev.type, epochs, times[0], times[1], res["train_bitwise"]))
    p1 = sp.predict_match(m0, bank, ms[1], cfg)
    p2 = sp.predict_match(m0, bank, ms[1], cfg)
    res["infer_bitwise"] = bool(np.array_equal(p1, p2))
    path = os.path.join(cache, "det_probe.pt")
    save_setnet(m0, path)
    p3 = sp.predict_match(load_setnet(path, cfg, dev), bank, ms[1], cfg)
    res["infer_reload_bitwise"] = bool(np.array_equal(p1, p3))
    os.remove(path)
    log("det-test %s: inference twice equal %s, after save/reload equal %s" % (
        dev.type, res["infer_bitwise"], res["infer_reload_bitwise"]))
    del bank, m0
    if cpu_probe:
        cpu = torch.device("cpu")
        bank = bank_of(op, ms, cpu)
        for nt in (1, 6):
            torch.set_num_threads(nt)
            t0 = time.time()
            sp.train(bank, ms[:1], cfg, seed=123, epochs=1)
            log("det-test cpu (%d threads): 1 epoch on 1 match %.1fs" % (nt, time.time() - t0))
        torch.set_num_threads(1)
    return res


# -- the real match -------------------------------------------------------------------------------

def real_inputs(real, idx, cache, key, fresh):
    """v2 feature columns (both views) and setnet prep of the model frames, cached."""
    from bench import feats3, methods
    px, pxs, pp = (os.path.join(cache, "real_%s.%s" % (key, s)) for s in ("x.npy", "xs.npy", "prep.npz"))
    m = {"ball": real["ball"][idx], "players": real["players"][idx], "period": real["period"][idx]}
    if not fresh and all(os.path.isfile(p) for p in (px, pxs, pp)):
        x, xs = np.load(px, mmap_mode="r"), np.load(pxs, mmap_mode="r")
        with np.load(pp) as z:
            prep = {k: z[k] for k in z.files}
        return x, xs, prep
    x, names, _, _, _ = feats3.build_features(m, GROUPS)
    xs, names2, _, _, _ = feats3.build_features(methods.swap_teams(m), GROUPS)
    if len(names) != N_FEATURES or names != names2:
        raise RuntimeError("unexpected feature schema on the real tracking")
    if not (np.isfinite(x).all() and np.isfinite(xs).all()):
        raise RuntimeError("non-finite features on the real tracking")
    prep = sp.prepare(m, np.zeros(len(idx), np.int8))
    _save_npy(px, x.astype(np.float32))
    _save_npy(pxs, xs.astype(np.float32))
    _save_npz(pp, **prep)
    return x.astype(np.float32), xs.astype(np.float32), prep


def run_final(args, dev):
    from bench import run_v3cand as rv
    from socbac.pipeline_v2 import format_stats, real_layout, sanity_stats
    from bench import methods
    t_all = time.time()
    spec = rv.CONFIGS["v3"]()
    cache = args.cache
    op = Open(cache)
    ch, _, _ = choose(op, spec, cache, dev)
    st, sd, epoch = ch["switch_team"], ch["switch_dead"], ch["epoch"]
    cfg = set_cfg()
    gm, gprior, gsha = gbm_model(op, spec, cache, ORDER, "final7")
    log("gbm final7 model sha256 %s (prior H %.3f A %.3f D %.3f)" % (gsha, gprior[0], gprior[1], gprior[2]))
    wait_ram()
    bank = bank_of(op, ORDER, dev)
    smodel, ssha = set_model(op, bank, cfg, cache, ORDER, "final7", SEED_FINAL, epoch, dev)
    del bank
    gc.collect()
    log("setnet final7 weights sha256 %s" % ssha)
    sprior = prior_of(op, ORDER)

    from socbac import atd
    from games import GAME, GAMES
    t0 = time.time()
    real = atd.load(use_cache=not args.fresh)
    n = len(real["period"])
    if n != GAMES[GAME]["n_video_frames"]:
        raise RuntimeError("tracking length %d != video length" % n)
    idx, blocks = real_layout(real, use_anchor=True)
    key = "%s_%d_%d" % (REV, len(idx), int(real["players"].shape[2]))
    x, xs, prep = real_inputs(real, idx, cache, key, args.fresh)
    period = np.asarray(real["period"][idx])
    log("real tracking: %d frames, %d labelled by the model, %d feature columns, players %s (%.0fs)" % (
        n, len(idx), x.shape[1], tuple(real["players"].shape), time.time() - t0))

    t0 = time.time()
    e_g = gbm_em(gm, gprior, {"x": x, "xs": xs, "period": period}, op.fmeta, spec)
    del x, xs
    rbank = sp.Bank([prep], ["real"], dev)
    p = sp.predict_match(smodel, rbank, "real", cfg)
    e_s = sp.emissions_from_proba(p, sprior).astype(np.float64)
    del rbank, smodel
    e = (1 - W_SET) * e_g + W_SET * e_s
    codes = sp.decode_periods(e, period, st, sd)
    log("emissions: gbm sha256 %s, setnet sha256 %s, blend sha256 %s (%.0fs)" % (
        sha(e_g.astype(np.float64))[:16], sha(e_s)[:16], sha(e)[:16], time.time() - t0))

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
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(letters))

    gcodes = np.where(letters[graded] == "H", methods.H, np.where(letters[graded] == "A", methods.A, methods.D))
    stats = sanity_stats(gcodes, real["period"][graded])
    counts = {k: int((letters == k).sum()) for k in "HADX"}
    log("wrote %s: %d rows, label counts %s" % (out, n, counts))
    print(format_stats("BAR-ATH graded (our prediction)", stats), flush=True)
    digest = file_sha256(out)
    print("sha256 %s" % digest, flush=True)
    summary = {"rev": REV, "out": out, "sha256": digest, "device": dev.type, "fresh": args.fresh,
               "choice": ch, "gbm_final7_sha256": gsha, "set_final7_sha256": ssha,
               "label_counts": counts, "stats": stats, "seconds": round(time.time() - t_all, 1)}
    for path in [os.path.join(cache, "last_run.json")] + ([args.summary_json] if args.summary_json else []):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out], check=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="submission CSV (required unless --bench / --det-test)")
    ap.add_argument("--fresh", action="store_true", help="delete this pipeline's cache, re-parse the tracking")
    ap.add_argument("--bench", action="store_true", help="open-data check: TRAIN 4 -> TEST 3 (seed 0)")
    ap.add_argument("--bench-report", default=None, help="bench mode: also write the result JSON here")
    ap.add_argument("--det-test", action="store_true", help="setnet determinism probes")
    ap.add_argument("--det-epochs", type=int, default=2)
    ap.add_argument("--cpu-probe", action="store_true", help="det-test: also time one CPU epoch")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args(argv)
    if not (args.bench or args.det_test or args.out):
        ap.error("--out is required")
    nice()
    t0 = time.time()
    from bench import feats3
    feats3._warn_off()
    if args.fresh and os.path.isdir(args.cache):
        shutil.rmtree(args.cache)
        log("fresh run: removed this pipeline's cache")
    os.makedirs(args.cache, exist_ok=True)
    dev = torch_setup(args.device)
    log("pipeline_final %s: v3 GBM + setnet big (w %.2f), degrade_v4 seed %d, device %s" % (REV, W_SET, SEED, dev))
    build_open(args.cache)
    if args.det_test:
        run_det_test(args.cache, dev, args.det_epochs, args.cpu_probe)
    if args.bench:
        run_bench(args.cache, dev, args.bench_report)
    if args.out:
        run_final(args, dev)
    log("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
