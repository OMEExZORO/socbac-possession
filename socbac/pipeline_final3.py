"""pipeline_final3: provenance-clean P1 (research/19_prov_clean.md, research/runs/prov_clean/LOG.md).

    python -m socbac.pipeline_final3 --bench                    # open data: TRAIN 4 -> TEST 3, seed 0
    python -m socbac.pipeline_final3 --train-final              # shipped weights on the 7 open matches
    python -m socbac.pipeline_final3 --variant pcf --out evaluation/results/socbac_final3_pcf_cpu1.csv --device cpu
    python -m socbac.pipeline_final3 --variant pct --out evaluation/results/socbac_final3_pct_cpu1.csv --device cpu

Training data: ONLY the 7 open IDSSE matches, each degraded twice by the provenance-clean degrader
(socbac.pc_degrade = research/runs/prov_clean/code/pc_degrade.py, byte-identical copy): copy k in {1, 2} of the
match with index i uses regime pc_degrade.draw_regimes(0, i)[k-1] and seed 60000*k + i. No v4 or v5ball copy is
used for training (in bench mode the degrade_v4 copies are only the evaluation proxy / OOF of the PCT costs, exactly
as in prov_clean). Two variants share the same networks:
  * PCF: v3 LightGBM retrained on the pc copies (same recipe as the frozen v3: L12 target, boost 20, 300 rounds)
         + touchnet-RK, blend w = .5, costs from LOMO on the pc OOF (regimes p, q) of bench seed 0.
  * PCT: the frozen shipped v3 GBM (pipeline_final2 ship/gbm_final7.txt, manifest-checked; v4-trained)
         + the same touchnet-RK, costs from LOMO on the v4 OOF of bench seed 0.
Networks: reconstructor (setnet big + aux heads, 45 epochs) cross-fitted; clean-ball teacher (56 epochs, p1 copies,
cross-fitted, training only); touchnet-RK (63 inputs, kd .5) on all pc copies, epoch from the bench seed-0 LOMO.
Inference reads the shipped files and the real tracking only; CPU float32; prints aggregates only.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from socbac import pipeline_final as pf  # noqa: E402  (read only)
from socbac import pipeline_final2 as pf2  # noqa: E402  (read only: load_rec/load_tn/save_pt/tn_inputs)
from socbac import p2_nets as nets  # noqa: E402
from socbac import setnet_pool as sp  # noqa: E402

REV = "pf3"
CACHE = os.path.join(ROOT, "data", "derived", "pipeline_final3")
PF2_SHIP = os.path.join(ROOT, "data", "derived", "pipeline_final2", "ship")
PF2_BENCH = os.path.join(ROOT, "data", "derived", "pipeline_final2", "bench_s0")
PC_RUN = r"D:\Barca_Hackathon\research\runs\prov_clean"    # research run, read only (bench comparison)
SEED = 0
TRAIN, TEST, ORDER = pf.TRAIN, pf.TEST, pf.ORDER
SUF = {"p": "~p1", "q": "~p2", "4": ""}
W_TN = 0.5
GBM7_SHA = "10255001b5cf750b13b098ff7a7d0a36c0716defe9aa6b6eafeff3d8ec76f57e"
GBM7_PRIOR_SHA = "dc3451cd56cf13847157e098f4f762a8e27ebe47724e6c86cb8b96ffbde1b767"
FMETA_SHA = "90daab8be3575e8c020fa6b488f80f8a85c3a3cd4daf7c08450f0696dfdfbd9d"
DEFS = pf.DEFS
log = pf.log
STRIDE = 5
SHIP_FILES = ("gbm_pc7.txt", "gbm_pc7_prior.npy", "gbm_fmeta.json", "rec_final7.pt", "tn_final7.pt", "choice.json")


def P(*a):
    return os.path.join(CACHE, *a)


def B(*a):
    return P("bench_s%d" % SEED, *a)


def F(*a):
    return P("final", *a)


def S(*a):
    return P("ship", *a)


def pcn(ms):
    return [m + s for m in ms for s in ("~p1", "~p2")]


BENCH_BANK = ORDER + pcn(TRAIN)          # = prov_clean BANK (same order: bitwise-identical training)
FINAL_BANK = pcn(ORDER)


def gate(ram=3.0, disk=7.0):
    """Wait until free RAM >= ram GB and D: free >= disk GB (one heavy process at a time on this machine)."""
    import psutil
    while True:
        r = psutil.virtual_memory().available / 2 ** 30
        d = shutil.disk_usage(ROOT).free / 2 ** 30
        if r >= ram and d >= disk:
            return
        log("waiting: free RAM %.1f GB, disk %.1f GB" % (r, d))
        time.sleep(60)


def free_gpu():
    pf2.free_gpu()


# -- preps ------------------------------------------------------------------------------------------

def degraded(m, kind, clean=None):
    """kind 'v4' or 'p1'/'p2'. Deterministic degraded copy of open match m."""
    from bench.dfl import load
    from socbac import pc_degrade as pcd
    i = ORDER.index(m)
    clean = load(m) if clean is None else clean
    if kind == "v4":
        from bench.degrade_v4 import DEGRADERS
        return DEGRADERS["v4"](clean, 1000 * SEED + i), None
    k = int(kind[1])
    return pcd.degrade(clean, 60000 * k + 1000 * SEED + i, pcd.draw_regimes(SEED, i)[k - 1])


def build_preps(names):
    from bench import labels5
    from bench.dfl import load
    from bench.feats3 import _attack_sign
    d = P("prep_s%d" % SEED)
    os.makedirs(d, exist_ok=True)
    for name in names:
        m = name.split("~")[0]
        path = os.path.join(d, name + ".npz")
        lpath = os.path.join(d, m + ".labs.npz")
        if os.path.isfile(path) and os.path.isfile(lpath):
            continue
        gate()
        t0 = time.time()
        clean = load(m)
        labs = labels5.all_labels(clean)
        if not os.path.isfile(lpath):
            pf._save_npz(lpath, **labs)
        deg, info = degraded(m, name.split("~")[1] if "~" in name else "v4", clean)
        prep = sp.prepare(deg, labs["L1"])
        prep["y2"] = labs["L2"].astype(np.int8)
        prep["y0"] = labs["L0"].astype(np.int8)
        prep["digest"] = np.array(sp.digest(deg))
        sign = _attack_sign(np.asarray(deg["players"], np.float32), np.asarray(deg["period"]))
        flip = sign[:, 0] < 0
        cb = np.asarray(clean["ball"][:, :2], np.float32).copy()
        cb[flip, 0] = sp.PL - cb[flip, 0]
        cb[flip, 1] = sp.PW - cb[flip, 1]
        prep["cok"] = np.isfinite(cb).all(axis=1)
        prep["cball"] = np.nan_to_num(cb, nan=0.0).astype(np.float32)
        pf._save_npz(path, **prep)
        log("prep %s: ball hidden %.3f, digest %s%s (%.0fs)" % (
            name, float(deg["ball_hidden_share"]), str(prep["digest"])[:16],
            "" if info is None else ", mean objects %.2f" % info["mean_objects"], time.time() - t0))
        del clean, deg, prep
        gc.collect()


def load_prep(name, privileged=True):
    with np.load(P("prep_s%d" % SEED, name + ".npz")) as z:
        p = {k: z[k] for k in z.files}
    if not privileged:
        for k in ("cball", "cok", "y0", "y2"):
            p.pop(k, None)
    return p


def labs_of(m):
    with np.load(P("prep_s%d" % SEED, m.split("~")[0] + ".labs.npz")) as z:
        return {k: z[k] for k in z.files}


def period_of(name):
    with np.load(P("prep_s%d" % SEED, name + ".npz")) as z:
        return z["period"]


def bank_of(names, dev, privileged=True):
    preps = [load_prep(n, privileged) for n in names]
    bank = nets.make_bank(preps, names, dev)
    del preps
    gc.collect()
    return bank


def prior_of(ms):
    c = sum(np.bincount(labs_of(m)[k], minlength=3).astype(np.float64) for m in ms for k in ("L1", "L2")) / 2
    c = np.array([0.5 * (c[0] + c[1]), 0.5 * (c[0] + c[1]), c[2]])
    return c / c.sum()


def load_R(stage_dir, bank):
    return pf2.load_R(stage_dir, bank)


# -- networks -------------------------------------------------------------------------------------------

def rec_stage(stage_dir, bank_names, folds, seed_base, dev, save_fold=None, ship=None):
    """folds: [(train names, predicted names)]; GPU bf16-autocast predictions saved per name (training inputs).
    save_fold: (fold index, path) whose model is saved. ship: (train names, path, seed) model saved, not predicted."""
    cfg = nets.rec_cfg(45)
    os.makedirs(stage_dir, exist_ok=True)
    todo = [k for k, (_, te) in enumerate(folds)
            if not all(os.path.isfile(os.path.join(stage_dir, n + ".npz")) for n in te)
            or (save_fold and save_fold[0] == k and not os.path.isfile(save_fold[1]))]
    need_ship = ship is not None and not os.path.isfile(ship[1])
    if not todo and not need_ship:
        return
    gate()
    bank = bank_of(bank_names, dev)
    for k in todo:
        tr, te = folds[k]
        t0 = time.time()
        model = nets.train_rec(bank, tr, cfg, nets.REC_AUX, seed_base + k, cfg["epochs"], log if k == todo[0] else None)
        if save_fold and save_fold[0] == k:
            pf2.save_pt(model, save_fold[1])
        for n in te:
            o = nets.predict_aux(model, bank, n, cfg, amp=True)
            pf._save_npz(os.path.join(stage_dir, n + ".npz"), r=o.cpu().numpy().astype(np.float32))
        del model
        free_gpu()
        log("rec fold %d (%s) %.0fs" % (k, ",".join(te), time.time() - t0))
    if need_ship:
        t0 = time.time()
        model = nets.train_rec(bank, ship[0], cfg, nets.REC_AUX, ship[2], cfg["epochs"])
        pf2.save_pt(model, ship[1])
        log("rec model on %d copies saved (%.0fs), weights sha256 %s" % (len(ship[0]), time.time() - t0,
                                                                        pf.state_sha(model)))
        del model
    del bank
    free_gpu()


def rec_report(stage_dir, bank, groups):
    import torch
    R = load_R(stage_dir, bank)
    est, lin, _, _ = nets.estimates(bank, R, "bh")
    out = {}
    for grp, names in groups:
        sel = torch.zeros(bank.n, dtype=torch.bool, device=bank.device)
        for n in names:
            sel[bank.offset[n]:bank.offset[n] + bank.length[n]] = True
        hid = sel & ~bank.vis & bank.cok
        d = torch.linalg.norm(est - bank.cball, dim=-1)[hid]
        dl = torch.linalg.norm(lin - bank.cball, dim=-1)[hid]
        out[grp] = [round(float(d.mean()), 3), round(float(d.median()), 3), round(float(dl.mean()), 3)]
    log("rec hidden-ball error [mean, median, straight-line mean] m: %s" % json.dumps(out))
    del R, est, lin
    return out


def teacher_stage(path, bank_names, folds, seed_base, dev):
    """Cross-fitted clean-ball teacher log-probabilities over the bank (zeros where not predicted)."""
    if os.path.isfile(path):
        return
    gate()
    bank = bank_of(bank_names, dev)
    XT, AUX = nets.clean_ball_inputs(bank)
    sym = nets.sym_vectors(dev)
    cfg = dict(nets.TN_CFG, p_mask=0.0)
    TL = np.zeros((bank.n, 3), np.float32)
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        model = nets.train_tn(bank, XT, AUX, sym, tr, cfg, seed_base + k, 56)
        for nm in te:
            p = nets.predict_tn(model, bank, XT, sym, nm)
            TL[bank.offset[nm]:bank.offset[nm] + bank.length[nm]] = np.log(np.clip(p, 1e-6, 1.0))
        del model
        log("teacher fold %d (%s) %.0fs" % (k, ",".join(te), time.time() - t0))
    del XT, AUX, bank
    free_gpu()
    pf._save_npy(path, TL)


TN_CFG = dict(nets.TN_CFG, p_mask=0.0, kd=0.5)


def bench_tn(dev):
    """LOMO on TRAIN (pc copies) -> epoch on p/q OOF; OOF emissions on p, q and v4; TRAIN-4 model + TEST (v4)."""
    path = B("tn_em.npz")
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    import torch
    gate()
    cfg = dict(TN_CFG)
    bank = bank_of(BENCH_BANK, dev)
    TL = torch.from_numpy(np.load(B("teach.npy"))).to(dev)
    R = load_R(B("rec"), bank)
    X, AUX, sym = pf2.tn_inputs(bank, R)
    del R
    oof, fprior = {}, {}
    t0 = time.time()
    for k, m_out in enumerate(TRAIN):
        rest = [m for m in TRAIN if m != m_out]
        fprior[m_out] = prior_of(rest)
        t1 = time.time()

        def on_ckpt(model, epoch, m_out=m_out):
            for te in "pq4":
                oof.setdefault(epoch, {})[(m_out, te)] = nets.predict_tn(model, bank, X, sym, m_out + SUF[te])

        nets.train_tn(bank, X, AUX, sym, pcn(rest), cfg, 100 * SEED + k, cfg["epochs"], on_ckpt, None, TL)
        log("touchnet-RK LOMO fold %d (%s out) %.0fs" % (k, m_out, time.time() - t1))
    pers = [period_of(m) for m in TRAIN]
    labs = [labs_of(m) for m in TRAIN]
    curve = {}
    for epoch in sorted(oof):
        s = []
        for te in "pq4":
            ems = [sp.emissions_from_proba(oof[epoch][(m, te)], fprior[m]) for m in TRAIN]
            s.append(pf.tune(ems, pers, labs, pf.COARSE_T, pf.COARSE_D)[0])
        curve[epoch] = float(np.mean(s[:2]))
        log("touchnet-RK epoch %d OOF pooled spellF1 %.3f (p q v4 %s)" % (epoch, curve[epoch], np.round(s, 3)))
    best = max(sorted(curve), key=lambda e: (curve[e], -e))
    log("touchnet-RK chosen epoch %d on pq (LOMO %.0fs)" % (best, time.time() - t0))
    out = {"epoch": np.int64(best), "curve_e": np.array(sorted(curve)), "curve_s": np.array([curve[e] for e in sorted(curve)])}
    for m in TRAIN:
        for te in "pq4":
            out["oof%s_%s" % (te, m)] = sp.emissions_from_proba(oof[best][(m, te)], fprior[m])
    del oof
    gc.collect()
    t1 = time.time()
    model = nets.train_tn(bank, X, AUX, sym, pcn(TRAIN), cfg, 100 * SEED + 4, best, None, None, TL)
    pf2.save_pt(model, B("tn_train4.pt"))
    prior = prior_of(TRAIN)
    for m in TEST:
        out["test_" + m] = sp.emissions_from_proba(nets.predict_tn(model, bank, X, sym, m), prior)
    log("touchnet-RK TRAIN-4 model %d epochs (%.0fs)" % (best, time.time() - t1))
    del model, bank, X, AUX, TL
    free_gpu()
    pf._save_npz(path, **out)
    return out


# -- v3 GBM on the pc copies (streamed: strided rows cached per copy, full matrices never written) -------

def features(name):
    """v3 feature matrices (both team views, float32) of a degraded copy, in memory, digest-checked."""
    from bench.feats3 import build_features
    from bench.methods import swap_teams
    m = name.split("~")[0]
    deg, _ = degraded(m, name.split("~")[1] if "~" in name else "v4")
    with np.load(P("prep_s%d" % SEED, name + ".npz")) as z:
        if str(z["digest"]) != sp.digest(deg):
            raise RuntimeError("degraded copy differs from its prep copy: " + name)
    x, names, _, sub, _ = build_features(deg, pf.GROUPS)
    x = x.astype(np.float32)
    xs, n2, _, _, _ = build_features(swap_teams(deg), pf.GROUPS)
    if n2 != names:
        raise RuntimeError("swapped view schema mismatch")
    d = {"x": x, "xs": xs.astype(np.float32), "period": np.asarray(deg["period"], np.int8)}
    del deg, xs
    gc.collect()
    return d, json.loads(json.dumps({"names": names, "sub": list(sub)}))


def check_fmeta(meta):
    with open(os.path.join(PF2_SHIP, "gbm_fmeta.json")) as fh:
        ref = json.load(fh)
    if meta != ref:
        raise RuntimeError("pc feature schema differs from the shipped v3 feature meta")


def gbm_rows(names):
    """Strided v3 rows of each copy, cached under rows/ (bench.run_hedge.rows_for layout)."""
    from bench import run_hedge as hg
    from bench import run_v3cand as rv
    hg.ROW_CACHE = P("rows")
    spec = rv.CONFIGS["v3"]()
    d0 = os.path.join(hg.ROW_CACHE, "pc_v3_st%d_s%d" % (STRIDE, SEED))
    for nm in names:
        if all(os.path.isfile(os.path.join(d0, nm + e)) for e in (".x.npy", ".xs.npy")):
            continue
        gate(2.5, 7.0)
        t0 = time.time()
        d, meta = features(nm)
        check_fmeta(meta)
        hg.rows_for("pc", SEED, "v3", spec, {nm: d}, meta, nm, STRIDE)
        del d
        gc.collect()
        log("gbm rows %s (%.0fs)" % (nm, time.time() - t0))


def gbm_fit_save(names, path):
    """v3 LightGBM (L12, boost 20, 300 rounds, 4 threads, deterministic) on the cached rows of names, saved with
    its class prior. Row order = bench.run_hedge.collect (per copy: view x then swapped view), preallocated."""
    import lightgbm as lgb
    from bench import run_hedge as hg
    from bench import run_v3cand as rv
    from bench.methods import SWAP
    if os.path.isfile(path) and os.path.isfile(path + ".prior.npy"):
        return
    hg.ROW_CACHE = P("rows")
    spec = rv.CONFIGS["v3"]()
    gate()
    t0 = time.time()
    tg, total = {}, 0
    for nm in names:
        per = period_of(nm)
        tg[nm] = hg.targets("L12", labs_of(nm), np.arange(0, len(per), STRIDE), per, pf.BOOST)
        total += 2 * len(tg[nm][0])
    x, ys, ws, at = None, [], [], 0
    for nm in names:
        idx, y, w = tg[nm]
        xr, xsr = hg.rows_for("pc", SEED, "v3", spec, None, None, nm, STRIDE)
        if x is None:
            x = np.empty((total, xr.shape[1]), np.float32)
        for part, lab in ((xr, y), (xsr, SWAP[y])):
            x[at:at + len(idx)] = np.asarray(part[idx])
            at += len(idx)
            ys.append(lab)
            ws.append(w)
        del xr, xsr
    y, w = np.concatenate(ys), np.concatenate(ws)
    mdl = lgb.train(pf.gbm_params(), lgb.Dataset(x, y, weight=w), num_boost_round=300)
    prior = np.bincount(y, weights=w, minlength=3).astype(np.float64)
    np.save(path + ".prior.npy", prior / prior.sum())
    mdl.save_model(path + ".tmp")
    os.replace(path + ".tmp", path)
    log("gbm fitted on %d copies, %d rows (%.0fs) -> %s" % (len(names), total, time.time() - t0, os.path.basename(path)))
    del x, y, w, mdl
    gc.collect()


def gbm_predict(path, name):
    import lightgbm as lgb
    from bench import run_v3cand as rv
    gate(2.5, 7.0)
    d, meta = features(name)
    mdl = lgb.Booster(model_file=path)
    mdl.params["num_threads"] = pf.NT
    e = rv.emissions(mdl, np.load(path + ".prior.npy"), d, meta, rv.CONFIGS["v3"]())
    del d, mdl
    gc.collect()
    return e.astype(np.float32)


def bench_gbm():
    """PC GBM: 4 LOMO models (OOF on p1 / p2 copies) + TRAIN-4 model (TEST v4 copies); frozen v3: pipeline_final2."""
    path = B("gbm_em.npz")
    if not os.path.isfile(path):
        t0 = time.time()
        gbm_rows(pcn(TRAIN))
        os.makedirs(B("gbm"), exist_ok=True)
        for hold in TRAIN + [None]:
            gbm_fit_save(pcn([m for m in TRAIN if m != hold]), B("gbm", "%s.txt" % (hold or "all")))
        arr = {}
        for h in TRAIN:
            for te in "pq":
                arr["oof%s_%s" % (te, h)] = gbm_predict(B("gbm", h + ".txt"), h + SUF[te])
        for m in TEST:
            arr["test_" + m] = gbm_predict(B("gbm", "all.txt"), m)
        pf._save_npz(path, **arr)
        log("bench pc gbm done (%.0fs)" % (time.time() - t0))
    with np.load(path) as z:
        out = {k: z[k].astype(np.float64) for k in z.files}
    with np.load(os.path.join(PF2_BENCH, "gbm_em.npz")) as z:      # frozen v3 (v4-trained), pipeline_final2 bench
        for m in TRAIN:
            out["g4oof_" + m] = z["oof_" + m].astype(np.float64)
        for m in TEST:
            out["g4test_" + m] = z["test_" + m].astype(np.float64)
    return out


# -- bench ----------------------------------------------------------------------------------------------

def score_test(em, costs):
    per = {k: {} for k in DEFS}
    for m in TEST:
        pred = sp.decode_periods(np.asarray(em[m], np.float64), period_of(m), costs[0], costs[1])
        for k, v in pf.score_defs(labs_of(m), pred, period_of(m), DEFS).items():
            per[k][m] = v
    return [float(np.mean(list(per[k].values()))) for k in DEFS]


def tune_on(oofs_by_reg):
    ems, pers, labs = [], [], []
    for reg, d in oofs_by_reg.items():
        for m in TRAIN:
            ems.append(d[m])
            pers.append(period_of(m))
            labs.append(labs_of(m))
    return pf.tune(ems, pers, labs, pf.GRID_T, pf.GRID_D)


def compare_research(t, g):
    """Max |difference| vs the prov_clean research files (seed 0), when present (read only)."""
    out = {}
    try:
        for nm in BENCH_BANK:
            a = os.path.join(PC_RUN, "rec", "s0", nm + ".npz")
            b = B("rec", nm + ".npz")
            if os.path.isfile(a) and os.path.isfile(b):
                with np.load(a) as za, np.load(b) as zb:
                    out["rec"] = max(out.get("rec", 0.0), float(np.abs(za["r"] - zb["r"]).max()))
        a = os.path.join(PC_RUN, "teach", "s0.npy")
        if os.path.isfile(a):
            out["teacher"] = float(np.abs(np.load(a) - np.load(B("teach.npy"))).max())
        for tag, src, key in [("xrk", t, "oof%s_%s"), ("gp", g, "oof%s_%s")]:
            for te in ("pq4" if tag == "xrk" else "pq"):
                for m in TRAIN:
                    a = os.path.join(PC_RUN, "emis", "%s_on%s_s0_%s.npz" % (tag, te, m))
                    if os.path.isfile(a):
                        with np.load(a) as z:
                            out[tag + "_oof"] = max(out.get(tag + "_oof", 0.0),
                                                    float(np.abs(z["em"] - src[key % (te, m)]).max()))
            for m in TEST:
                a = os.path.join(PC_RUN, "emis", "%s_on4_s0_%s.npz" % (tag, m))
                if os.path.isfile(a):
                    with np.load(a) as z:
                        out[tag + "_test"] = max(out.get(tag + "_test", 0.0),
                                                 float(np.abs(z["em"] - src["test_" + m]).max()))
        for m in ORDER:                   # frozen v3 emissions used by prov_clean (round2 emis)
            a = os.path.join(os.path.dirname(PC_RUN), "round2", "emis", "gbm_v3_s0_%s.npz" % m)
            if os.path.isfile(a):
                with np.load(a) as z:
                    k = ("g4oof_" if m in TRAIN else "g4test_") + m
                    out["g4"] = max(out.get("g4", 0.0), float(np.abs(z["em"] - g[k]).max()))
    except Exception as exc:              # comparison only; never fatal
        out["error"] = type(exc).__name__
    log("bench vs prov_clean research files, max |diff|: %s" % json.dumps(out))
    return out


def cpu_test_emissions(rec_path, tn_path):
    """Shipped inference path on the TEST matches: each match alone in a CPU bank without privileged arrays,
    weights reloaded from disk, float32."""
    import torch
    cpu = torch.device("cpu")
    cfg_rec = nets.rec_cfg()
    rec = pf2.load_rec(rec_path, cfg_rec, cpu)
    tnm = pf2.load_tn(tn_path, TN_CFG, cpu)
    prior = prior_of(TRAIN)
    out, times = {}, {}
    for m in TEST:
        t0 = time.time()
        bank = bank_of([m], cpu, privileged=False)
        with torch.no_grad():
            R = nets.predict_aux(rec, bank, m, cfg_rec, amp=False)
        t1 = time.time()
        X, _, sym = pf2.tn_inputs(bank, R)
        out[m] = sp.emissions_from_proba(nets.predict_tn(tnm, bank, X, sym, m), prior).astype(np.float64)
        times[m] = (round(t1 - t0, 1), round(time.time() - t1, 1))
        log("cpu float32 TEST %s: reconstructor %.0fs, touchnet %.0fs" % (m, times[m][0], times[m][1]))
        del bank, R, X
        gc.collect()
    return out, times


LOG_ROWS = {"PCF": (.660, .715, .777, .559), "PCT": (.648, .707, .773, .574)}


def run_bench(dev, report):
    t_all = time.time()
    tm = {}
    build_preps(BENCH_BANK)
    t0 = time.time()
    folds = [(pcn([m for m in TRAIN if m != h]), [h + "~p1", h + "~p2", h]) for h in TRAIN] + [(pcn(TRAIN), TEST)]
    rec_stage(B("rec"), BENCH_BANK, folds, 100 * SEED + 7000, dev, save_fold=(4, B("rec_train4.pt")))
    tm["rec"] = time.time() - t0
    bank = bank_of(BENCH_BANK, dev)
    rrep = rec_report(B("rec"), bank, [("OOF~p", [m + "~p1" for m in TRAIN] + [m + "~p2" for m in TRAIN]),
                                       ("OOF~v4", TRAIN), ("TEST~v4", TEST)])
    del bank
    free_gpu()
    t0 = time.time()
    teacher_stage(B("teach.npy"), BENCH_BANK,
                  [([m + "~p1" for m in TRAIN if m != h], [h + "~p1", h + "~p2"]) for h in TRAIN], 100 * SEED + 9000, dev)
    tm["teacher"] = time.time() - t0
    t0 = time.time()
    t = bench_tn(dev)
    tm["touchnet"] = time.time() - t0
    t0 = time.time()
    g = bench_gbm()
    tm["gbm"] = time.time() - t0
    cmp_ = compare_research(t, g)
    epoch = int(t["epoch"])
    xo = {te: {m: t["oof%s_%s" % (te, m)].astype(np.float64) for m in TRAIN} for te in "pq4"}
    xt = {m: t["test_" + m].astype(np.float64) for m in TEST}
    go = {te: {m: g["oof%s_%s" % (te, m)] for m in TRAIN} for te in "pq"}
    go["4"] = {m: g["g4oof_" + m] for m in TRAIN}
    gt = {"PCF": {m: g["test_" + m] for m in TEST}, "PCT": {m: g["g4test_" + m] for m in TEST}}
    sel = {"PCF": "pq", "PCT": "4"}
    res = {"seed": SEED, "tn_epoch": epoch, "tn_curve": dict(zip(map(int, t["curve_e"]), map(float, t["curve_s"]))),
           "rec_error": rrep, "vs_research": cmp_, "methods": {}}
    choice = {"rev": REV, "tn_epoch": epoch}
    for v in ("PCF", "PCT"):
        oofs = {te: {m: 0.5 * go[te][m] + 0.5 * xo[te][m] for m in TRAIN} for te in sel[v]}
        oof, st, sd = tune_on(oofs)
        sc = score_test({m: 0.5 * gt[v][m] + 0.5 * xt[m] for m in TEST}, (st, sd))
        res["methods"][v] = {"costs": [st, sd], "oof": oof, "gpu_research_path": sc, "log": LOG_ROWS[v]}
        choice[v.lower()] = {"switch_team": st, "switch_dead": sd, "oof": oof, "sel": sel[v]}
        log("bench %s costs %g/%g OOF %.3f TEST (GPU research path) %s | LOG %s" % (
            v, st, sd, oof, " ".join("%.3f" % s for s in sc), LOG_ROWS[v]))
    with open(B("choice.json"), "w") as fh:
        json.dump(choice, fh, indent=1)
    t0 = time.time()
    x_cpu, times = cpu_test_emissions(B("rec_train4.pt"), B("tn_train4.pt"))
    tm["cpu_check"] = time.time() - t0
    res["cpu_max_abs_emission_diff"] = max(float(np.abs(x_cpu[m] - xt[m]).max()) for m in TEST)
    res["cpu_seconds_per_match"] = times
    ok = True
    for v in ("PCF", "PCT"):
        c = res["methods"][v]["costs"]
        sc = score_test({m: 0.5 * gt[v][m] + 0.5 * x_cpu[m] for m in TEST}, c)
        dl = [s - r for s, r in zip(sc, LOG_ROWS[v])]
        res["methods"][v]["shipped_path"] = sc
        res["methods"][v]["shipped_minus_log"] = dl
        ok &= max(abs(d) for d in dl) <= 0.005
        log("bench %s shipped path (CPU float32, reloaded weights, match alone): TEST %s | minus LOG %s" % (
            v, " ".join("%.3f" % s for s in sc), " ".join("%+.4f" % d for d in dl)))
    res["within_0.005"] = bool(ok)
    res["seconds"] = {k: round(v, 1) for k, v in tm.items()}
    res["seconds"]["total"] = round(time.time() - t_all, 1)
    log("bench reproduction within .005 of LOG: %s" % ok)
    if report:
        os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
        with open(report, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


# -- final training on all 7 open matches -----------------------------------------------------------------

def run_train_final(dev, report=None):
    t_all = time.time()
    if not os.path.isfile(B("choice.json")):
        raise SystemExit("run --bench first: the epoch and switch costs come from the bench seed-0 LOMO")
    with open(B("choice.json")) as fh:
        choice = json.load(fh)
    build_preps(FINAL_BANK)
    os.makedirs(S(), exist_ok=True)
    tm = {}
    t0 = time.time()
    rec_stage(F("rec"), FINAL_BANK, [(pcn([m for m in ORDER if m != h]), pcn([h])) for h in ORDER],
              100 * SEED + 7100, dev, ship=(FINAL_BANK, S("rec_final7.pt"), 100 * SEED + 7107))
    tm["rec"] = time.time() - t0
    t0 = time.time()
    teacher_stage(F("teach.npy"), FINAL_BANK,
                  [([m + "~p1" for m in ORDER if m != h], pcn([h])) for h in ORDER], 100 * SEED + 9100, dev)
    tm["teacher"] = time.time() - t0
    if not os.path.isfile(S("tn_final7.pt")):
        import torch
        gate()
        t0 = time.time()
        bank = bank_of(FINAL_BANK, dev)
        rep = rec_report(F("rec"), bank, [("OOF7~p", FINAL_BANK)])
        with open(F("rec_error.json"), "w") as fh:
            json.dump(rep, fh)
        TL = torch.from_numpy(np.load(F("teach.npy"))).to(dev)
        R = load_R(F("rec"), bank)
        X, AUX, sym = pf2.tn_inputs(bank, R)
        del R
        model = nets.train_tn(bank, X, AUX, sym, FINAL_BANK, TN_CFG, 100 * SEED + 7, choice["tn_epoch"], None, log, TL)
        pf2.save_pt(model, S("tn_final7.pt"))
        log("touchnet-RK final7 %d epochs (%.0fs), weights sha256 %s" % (
            choice["tn_epoch"], time.time() - t0, pf.state_sha(model)))
        tm["touchnet"] = time.time() - t0
        del model, bank, X, AUX, TL
        free_gpu()
    t0 = time.time()
    if not os.path.isfile(S("gbm_pc7.txt")):
        gbm_rows(FINAL_BANK)
        gbm_fit_save(FINAL_BANK, F("gbm_pc7.txt"))
        shutil.copyfile(F("gbm_pc7.txt.prior.npy"), S("gbm_pc7_prior.npy"))
        shutil.copyfile(F("gbm_pc7.txt"), S("gbm_pc7.txt"))
    tm["gbm"] = time.time() - t0
    if not os.path.isfile(S("gbm_fmeta.json")):
        if pf.file_sha256(os.path.join(PF2_SHIP, "gbm_fmeta.json")) != FMETA_SHA:
            raise SystemExit("pipeline_final2 gbm_fmeta.json changed")
        shutil.copyfile(os.path.join(PF2_SHIP, "gbm_fmeta.json"), S("gbm_fmeta.json"))
    ship = {"rev": REV, "tn_epoch": choice["tn_epoch"], "w_tn": W_TN,
            "variants": {
                "pcf": {"gbm": "gbm_pc7.txt", "gbm_prior": "gbm_pc7_prior.npy",
                        "switch_team": choice["pcf"]["switch_team"], "switch_dead": choice["pcf"]["switch_dead"]},
                "pct": {"gbm": "../../pipeline_final2/ship/gbm_final7.txt", "gbm_sha256": GBM7_SHA,
                        "gbm_prior": "../../pipeline_final2/ship/gbm_final7_prior.npy", "gbm_prior_sha256": GBM7_PRIOR_SHA,
                        "switch_team": choice["pct"]["switch_team"], "switch_dead": choice["pct"]["switch_dead"]}},
            "tn_prior": list(prior_of(ORDER)), "tn_cfg": TN_CFG, "rec_cfg": nets.rec_cfg(),
            "training_copies": "pc_degrade (rev %s) x2 per open match, seed 60000*k + 1000*%d + index" % ("pc1", SEED),
            "seeds": {"rec_lomo7": 100 * SEED + 7100, "rec_final7": 100 * SEED + 7107, "teacher_lomo7": 100 * SEED + 9100,
                      "tn_final7": 100 * SEED + 7}}
    with open(S("choice.json"), "w") as fh:
        json.dump(ship, fh, indent=1)
    write_manifest()
    tm["total"] = time.time() - t_all
    log("train-final done (%.0fs)" % tm["total"])
    if report:
        with open(report, "w") as fh:
            json.dump({k: round(v, 1) for k, v in tm.items()}, fh, indent=1)


def write_manifest():
    man = {f: {"bytes": os.path.getsize(S(f)), "sha256": pf.file_sha256(S(f))} for f in SHIP_FILES}
    with open(S("manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    for f, v in man.items():
        log("ship %-20s %9d bytes sha256 %s" % (f, v["bytes"], v["sha256"]))


def check_manifest():
    with open(S("manifest.json")) as fh:
        man = json.load(fh)
    for f in SHIP_FILES:
        if pf.file_sha256(S(f)) != man[f]["sha256"]:
            raise SystemExit("shipped file %s does not match manifest.json" % f)
    return man


# -- the real match (inference from the shipped files only) ----------------------------------------------

def run_infer(args, dev):
    import lightgbm as lgb
    import torch
    from bench import run_v3cand as rv
    from bench import methods
    from socbac.pipeline_v2 import format_stats, real_layout, sanity_stats
    t_all = time.time()
    man = check_manifest()
    with open(S("choice.json")) as fh:
        ch = json.load(fh)
    var = ch["variants"][args.variant]
    gpath, ppath = os.path.normpath(S(var["gbm"])), os.path.normpath(S(var["gbm_prior"]))
    if args.variant == "pct":
        with open(os.path.join(PF2_SHIP, "manifest.json")) as fh:
            man2 = json.load(fh)
        for p, sha, key in ((gpath, var["gbm_sha256"], "gbm_final7.txt"), (ppath, var["gbm_prior_sha256"], "gbm_final7_prior.npy")):
            if pf.file_sha256(p) != sha or man2[key]["sha256"] != sha:
                raise SystemExit("frozen v3 GBM file %s does not match its manifest" % key)
    with open(S("gbm_fmeta.json")) as fh:
        fmeta = json.load(fh)
    spec = rv.CONFIGS["v3"]()
    gm = lgb.Booster(model_file=gpath)
    gm.params["num_threads"] = pf.NT
    gprior = np.load(ppath)

    from socbac import atd
    from games import GAME, GAMES
    t0 = time.time()
    real = atd.load(use_cache=not args.fresh)
    n = len(real["period"])
    if n != GAMES[GAME]["n_video_frames"]:
        raise RuntimeError("tracking length %d != video length" % n)
    idx, blocks = real_layout(real, use_anchor=True)
    key = "%s_%d_%d" % (REV, len(idx), int(real["players"].shape[2]))
    if args.fresh:
        for s in ("x.npy", "xs.npy", "prep.npz"):
            p = P("real_%s.%s" % (key, s))
            if os.path.isfile(p):
                os.remove(p)
    x, xs, prep = pf.real_inputs(real, idx, CACHE, key, args.fresh)
    period = np.asarray(real["period"][idx])
    t_load = time.time() - t0
    log("real tracking: %d frames, %d labelled by the model (%.0fs)" % (n, len(idx), t_load))

    t0 = time.time()
    e_g = pf.gbm_em(gm, gprior, {"x": x, "xs": xs, "period": period}, fmeta, spec).astype(np.float64)
    del x, xs
    t_gbm = time.time() - t0
    t0 = time.time()
    bank = nets.make_bank([prep], ["real"], dev)
    rec = pf2.load_rec(S("rec_final7.pt"), ch["rec_cfg"], dev)
    with torch.no_grad():
        R = nets.predict_aux(rec, bank, "real", ch["rec_cfg"], amp=False)
    del rec
    t_rec = time.time() - t0
    t0 = time.time()
    X, _, sym = pf2.tn_inputs(bank, R)
    tnm = pf2.load_tn(S("tn_final7.pt"), ch["tn_cfg"], dev)
    e_t = sp.emissions_from_proba(nets.predict_tn(tnm, bank, X, sym, "real"), np.array(ch["tn_prior"])).astype(np.float64)
    t_tn = time.time() - t0
    del bank, X, R, tnm
    e = (1 - ch["w_tn"]) * e_g + ch["w_tn"] * e_t
    codes = sp.decode_periods(e, period, var["switch_team"], var["switch_dead"])
    log("emissions sha256: gbm %s touchnet %s blend %s | gbm %.0fs, reconstructor %.0fs, touchnet %.0fs" % (
        pf.sha(e_g)[:16], pf.sha(e_t)[:16], pf.sha(e)[:16], t_gbm, t_rec, t_tn))

    letters = np.full(n, "X", dtype="<U1")
    graded = np.zeros(n, bool)
    for _half, first, last in blocks:
        letters[first:last + 1] = "D"
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
    digest = pf.file_sha256(out)
    print("sha256 %s" % digest, flush=True)
    summary = {"rev": REV, "variant": args.variant, "out": out, "sha256": digest, "device": dev.type, "fresh": args.fresh,
               "choice": {"tn_epoch": ch["tn_epoch"], "w_tn": ch["w_tn"], **var}, "ship": man, "label_counts": counts,
               "stats": stats, "emission_sha": {"gbm": pf.sha(e_g)[:16], "touchnet": pf.sha(e_t)[:16], "blend": pf.sha(e)[:16]},
               "seconds": {"load": round(t_load, 1), "gbm": round(t_gbm, 1), "rec": round(t_rec, 1),
                           "touchnet": round(t_tn, 1), "total": round(time.time() - t_all, 1)}}
    for path in [P("last_run_%s.json" % args.variant)] + ([args.summary_json] if args.summary_json else []):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out], check=True)
    return summary


def compare_csv(a, b):
    """Agreement of two of OUR submission CSVs: share of equal labels on frames graded (non-X) in b, X identical."""
    la = np.loadtxt(a, delimiter=",", skiprows=1, dtype=str)[:, 1]
    lb = np.loadtxt(b, delimiter=",", skiprows=1, dtype=str)[:, 1]
    g = lb != "X"
    print("agreement %s vs %s: graded frames %.4f, X frames identical %s, rows %d/%d" % (
        os.path.basename(a), os.path.basename(b), float((la[g] == lb[g]).mean()), bool(((la == "X") == (lb == "X")).all()),
        len(la), len(lb)), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="submission CSV, from the shipped weights")
    ap.add_argument("--variant", default="pcf", choices=["pcf", "pct"])
    ap.add_argument("--fresh", action="store_true", help="re-parse the real tracking and rebuild its inputs")
    ap.add_argument("--bench", action="store_true", help="open-data check: TRAIN 4 -> TEST 3 (seed 0)")
    ap.add_argument("--bench-report", default=None)
    ap.add_argument("--train-final", action="store_true", help="train the shipped weights on all 7 open matches")
    ap.add_argument("--train-report", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--summary-json", default=None)
    ap.add_argument("--compare", nargs=2, default=None, help="agreement of two of our CSVs")
    args = ap.parse_args(argv)
    if args.compare:
        compare_csv(*args.compare)
        return
    if not (args.bench or args.train_final or args.out):
        ap.error("one of --bench, --train-final, --out is required")
    pf.nice()
    t0 = time.time()
    from bench import feats3
    feats3._warn_off()
    os.makedirs(CACHE, exist_ok=True)
    dev = pf.torch_setup(args.device)
    log("pipeline_final3 %s: provenance-clean P1, device %s" % (REV, dev))
    if args.bench:
        run_bench(dev, args.bench_report)
    if args.train_final:
        run_train_final(dev, args.train_report)
    if args.out:
        run_infer(args, dev)
    log("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
