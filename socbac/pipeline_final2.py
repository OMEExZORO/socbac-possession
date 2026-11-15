"""pipeline_final2: frozen v3 GBM + touchnet-RK (Fable round 3, candidate P1), blend w = .5.

    python -m socbac.pipeline_final2 --bench                       # open data: TRAIN 4 -> TEST 3, seed 0
    python -m socbac.pipeline_final2 --train-final                 # train shipped weights on all 7 open matches
    python -m socbac.pipeline_final2 --out evaluation/results/socbac_final2_cpu1.csv --device cpu
                                                                   # CSV from the shipped weights only

Recipe (research/17_fable_round3.md section 4; bench evidence research/runs/round3/LOG.md):
  * GBM: the frozen v3 LightGBM of socbac.pipeline_final (same features, pooled L1+L2, degrade_v4 seed 0).
    Its 7-match model file is copied from pipeline_final's cache after a SHA-256 check (refitted with the
    same code if absent); in bench mode its TRAIN-4 model is refitted with pipeline_final's code.
  * Ball reconstructor: setnet big + ball / nearest-player / L0 heads (socbac.p2_nets), 45 epochs, aux
    weights 3 / 1 / .3, trained on degrade_v4 + degrade_v5 v5ball(ball_stress) copies (seed 1000*0 + index)
    of the open matches. Cross-fitted: the reconstruction of each training match comes from a model that
    never saw it. The real match is reconstructed by the model trained on all 7 (float32, CPU).
  * Teacher: touchnet on the clean, never-hidden open ball, 56 epochs, v4 copies, cross-fitted (training only).
  * Touchnet-RK: 63 inputs (observed ball where visible, anchored reconstruction where hidden, visible flag,
    P(home/away/none nearest), time to the visible ball on each side), loss .5 CE(L1,L2) + .5 KL(teacher),
    trained on v4 + v5ball copies; epoch count from LOMO on the 4 bench TRAIN matches (both regimes).
  * Blend .5 GBM + .5 touchnet log-likelihoods, 3-state Viterbi per half; switch costs from LOMO on the
    v4 OOF of the 4 bench TRAIN matches. All choices are made on the open bench only.
  * Real match handling as socbac.pipeline_final (atd.load, graded halves, kick-off anchor, X outside).

Shipped files (data/derived/pipeline_final2/ship): gbm_final7.txt (+ prior, feature meta),
rec_final7.pt, tn_final7.pt (state dicts), choice.json, manifest.json. Inference reads only those and
the real tracking; it runs in float32 with deterministic settings. Prints aggregates only.
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

from socbac import pipeline_final as pf  # noqa: E402  (imported, never modified)
from socbac import p2_nets as nets  # noqa: E402
from socbac import setnet_pool as sp  # noqa: E402

REV = "pf2"
CACHE = os.path.join(ROOT, "data", "derived", "pipeline_final2")
SEED = 0
TRAIN, TEST, ORDER = pf.TRAIN, pf.TEST, pf.ORDER
SUF = {"4": "", "b": "~b"}
REGS = ("4", "b")
DEGR = {"4": "v4", "b": "v5ball"}
W_TN = 0.5
GBM7_SHA = "10255001b5cf750b13b098ff7a7d0a36c0716defe9aa6b6eafeff3d8ec76f57e"   # pipeline_final gbm_final7
DEFS = pf.DEFS
log = pf.log
SMOKE = os.environ.get("P2_SMOKE") == "1"       # plumbing test only: tiny epochs, separate stage folders
TAG = "_smoke" if SMOKE else ""


def P(*a):
    return os.path.join(CACHE, *a)


def names_of(ms, regs=REGS):
    return [m + SUF[r] for m in ms for r in regs]


ALL14 = ORDER + [m + "~b" for m in ORDER]          # bank order of the research run


def save_pt(model, path):
    import torch
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(sd, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_rec(path, cfg, dev):
    import torch
    m = nets.new_rec(cfg, dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    return m


def load_tn(path, cfg, dev, nf=63):
    import torch
    m = nets.make_tn(cfg, dev, nf)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    return m


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# -- open data: v4 + v5ball copies with privileged targets ---------------------------------------

def build_preps():
    """Per open match: labels L0-L3 and two degraded copies (v4, v5ball; seed 1000*SEED + index) with the
    setnet prep arrays, L0/L1/L2 targets and the clean ball in the canonical frame (training only)."""
    from bench import labels5, degrade_v5
    from bench.dfl import load
    from bench.degrade_v4 import DEGRADERS
    from bench.feats3 import _attack_sign
    degrade_v5.register()
    d = P("prep_s%d" % SEED)
    os.makedirs(d, exist_ok=True)
    for i, m in enumerate(ORDER):
        for r in REGS:
            path = os.path.join(d, m + SUF[r] + ".npz")
            if os.path.isfile(path):
                continue
            pf.wait_ram()
            t0 = time.time()
            clean = load(m)
            lpath = os.path.join(d, m + ".labs.npz")
            labs = labels5.all_labels(clean)
            if not os.path.isfile(lpath):
                pf._save_npz(lpath, **labs)
            deg = DEGRADERS[DEGR[r]](clean, 1000 * SEED + i)
            n = len(deg["period"])
            if not all(len(v) == n for v in labs.values()):
                raise RuntimeError("label length mismatch")
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
            log("prep %s%s (%s seed %d): %d frames, ball hidden %.3f, digest %s (%.0fs)" % (
                m, SUF[r], DEGR[r], 1000 * SEED + i, n, float(deg["ball_hidden_share"]), str(prep["digest"]),
                time.time() - t0))
            del clean, deg, prep
            gc.collect()


def load_prep(name, privileged=True):
    with np.load(P("prep_s%d" % SEED, name + ".npz")) as z:
        p = {k: z[k] for k in z.files}
    if not privileged:                      # inference view: what a real match provides
        for k in ("cball", "cok", "y0", "y2"):
            p.pop(k, None)
    return p


def labs_of(m):
    with np.load(P("prep_s%d" % SEED, m + ".labs.npz")) as z:
        return {k: z[k] for k in z.files}


def period_of(m):
    with np.load(P("prep_s%d" % SEED, m + ".npz")) as z:
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


# -- reconstructor ----------------------------------------------------------------------------------

def rec_stage(stage_dir, folds, seed_base, dev, ship=None):
    """folds: [(train matches, predicted matches)]; predictions (in-memory EMA model, bf16 autocast on the
    GPU as in the research run) saved per name. ship: (train matches, path) of a model saved, not predicted."""
    cfg = nets.rec_cfg(1 if SMOKE else 45)
    os.makedirs(stage_dir, exist_ok=True)
    todo = [k for k, (_, te) in enumerate(folds)
            if not all(os.path.isfile(os.path.join(stage_dir, n + ".npz")) for n in names_of(te))]
    need_ship = ship is not None and not os.path.isfile(ship[1])
    if not todo and not need_ship:
        return
    bank = bank_of(ALL14, dev)
    for k in todo:
        tr, te = folds[k]
        t0 = time.time()
        model = nets.train_rec(bank, names_of(tr), cfg, nets.REC_AUX, seed_base + k, cfg["epochs"],
                               log if k == todo[0] else None)
        if len(tr) == len(TRAIN) and te == TEST:
            save_pt(model, os.path.join(stage_dir, "rec_train4.pt"))
        for n in names_of(te):
            o = nets.predict_aux(model, bank, n, cfg, amp=True)
            pf._save_npz(os.path.join(stage_dir, n + ".npz"), r=o.cpu().numpy().astype(np.float32))
        del model
        free_gpu()
        log("rec fold %d (%s out) %.0fs" % (k, ",".join(te), time.time() - t0))
    if need_ship:
        t0 = time.time()
        model = nets.train_rec(bank, names_of(ship[0]), cfg, nets.REC_AUX, seed_base + len(folds), cfg["epochs"])
        save_pt(model, ship[1])
        log("rec model on %d matches saved (%.0fs), weights sha256 %s" % (
            len(ship[0]), time.time() - t0, pf.state_sha(model)))
        del model
    del bank
    free_gpu()


def load_R(stage_dir, bank):
    import torch
    R = torch.zeros(bank.n, 7, device=bank.device)
    for nm in bank.names:
        with np.load(os.path.join(stage_dir, nm + ".npz")) as z:
            R[bank.offset[nm]:bank.offset[nm] + bank.length[nm]] = torch.from_numpy(z["r"]).to(bank.device)
    return R


def rec_report(stage_dir, bank, groups):
    """Hidden-frame error of the anchored estimate vs the true ball (open data), mean / median metres."""
    import torch
    R = load_R(stage_dir, bank)
    est, lin, _, _ = nets.estimates(bank, R, "bh")
    out = {}
    for grp, ms in groups:
        for r in REGS:
            sel = torch.zeros(bank.n, dtype=torch.bool, device=bank.device)
            for m in ms:
                n = m + SUF[r]
                sel[bank.offset[n]:bank.offset[n] + bank.length[n]] = True
            hid = sel & ~bank.vis & bank.cok
            d = torch.linalg.norm(est - bank.cball, dim=-1)[hid]
            dl = torch.linalg.norm(lin - bank.cball, dim=-1)[hid]
            out["%s~%s" % (grp, r)] = [round(float(d.mean()), 3), round(float(d.median()), 3), round(float(dl.mean()), 3)]
    log("rec hidden-ball error [mean, median, straight-line mean] m: %s" % json.dumps(out))
    del R, est, lin
    return out


# -- teacher ----------------------------------------------------------------------------------------

def teacher_stage(path, folds, seed_base, dev):
    """Cross-fitted clean-ball teacher log-probabilities over the ALL14 bank (zeros where not predicted)."""
    if os.path.isfile(path):
        return
    import torch
    bank = bank_of(ALL14, dev)
    XT, AUX = nets.clean_ball_inputs(bank)
    sym = nets.sym_vectors(dev)
    cfg = dict(nets.TN_CFG, p_mask=0.0)
    TL = np.zeros((bank.n, 3), np.float32)
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        model = nets.train_tn(bank, XT, AUX, sym, tr, cfg, seed_base + k, 1 if SMOKE else 56)
        for nm in names_of(te):
            p = nets.predict_tn(model, bank, XT, sym, nm)
            TL[bank.offset[nm]:bank.offset[nm] + bank.length[nm]] = np.log(np.clip(p, 1e-6, 1.0))
        del model
        log("teacher fold %d (%s) %.0fs" % (k, ",".join(te), time.time() - t0))
    del XT, AUX, bank
    free_gpu()
    pf._save_npy(path, TL)
    del torch


def tn_inputs(bank, R):
    X, AUX = nets.fill_inputs(bank, R)
    for k in ("pos", "vel", "vok"):
        setattr(bank, k, None)
    free_gpu()
    return X, AUX, nets.sym_vectors(bank.device, extended=True)


# -- bench ------------------------------------------------------------------------------------------

def B(*a):
    return P("bench_s%d%s" % (SEED, TAG), *a)


def bench_gbm():
    """Frozen v3 GBM emissions: OOF of the 4 TRAIN matches (pipeline_final LOMO) and TEST (TRAIN-4 model)."""
    from bench import run_v3cand as rv
    path = B("gbm_em.npz")
    if not os.path.isfile(path):
        spec = rv.CONFIGS["v3"]()
        op = pf.Open(pf.CACHE)            # pipeline_final's open-match feature cache, read only
        have = all(os.path.isfile(os.path.join(pf.CACHE, "gbm_oof_%s.npy" % m)) for m in TRAIN)
        g_oof = pf.gbm_stage_oof(op, spec, pf.CACHE if have else B())   # pipeline_final LOMO (cached there)
        t0 = time.time()
        gm, gprior, gsha = pf.gbm_model(op, spec, B(), TRAIN, "train4")
        arr = {"oof_" + m: g_oof[m].astype(np.float32) for m in TRAIN}
        for m in TEST:
            arr["test_" + m] = pf.gbm_em(gm, gprior, op.full(m), op.fmeta, spec).astype(np.float32)
        pf._save_npz(path, **arr)
        log("bench gbm: TRAIN-4 model sha256 %s (%.0fs)" % (gsha, time.time() - t0))
        del gm, op
        gc.collect()
    with np.load(path) as z:
        return ({m: z["oof_" + m].astype(np.float64) for m in TRAIN},
                {m: z["test_" + m].astype(np.float64) for m in TEST})


def bench_tn(dev):
    """LOMO on TRAIN (both regimes) -> epoch; OOF emissions; TRAIN-4 model + TEST emissions (research path)."""
    path = B("tn_em.npz")
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    import torch
    cfg = dict(nets.TN_CFG, p_mask=0.0, kd=0.5)
    if SMOKE:
        cfg.update(epochs=4, ckpt_every=2)
    bank = bank_of(ALL14, dev)
    TL = torch.from_numpy(np.load(B("teach.npy"))).to(dev)
    R = load_R(B("rec"), bank)
    X, AUX, sym = tn_inputs(bank, R)
    del R
    oof, fprior = {}, {}
    t0 = time.time()
    for k, m_out in enumerate(TRAIN):
        rest = [m for m in TRAIN if m != m_out]
        fprior[m_out] = prior_of(rest)
        t1 = time.time()

        def on_ckpt(model, epoch, m_out=m_out):
            for r in REGS:
                oof.setdefault(epoch, {})[(m_out, r)] = nets.predict_tn(model, bank, X, sym, m_out + SUF[r])

        nets.train_tn(bank, X, AUX, sym, names_of(rest), cfg, 100 * SEED + k, cfg["epochs"], on_ckpt,
                      log if k == 0 else None, TL)
        log("touchnet-RK LOMO fold %d (%s out) %.0fs" % (k, m_out, time.time() - t1))
    pers = {m: period_of(m) for m in ORDER}
    labs = {m: labs_of(m) for m in ORDER}
    curve = {}
    for epoch in sorted(oof):
        s = []
        for r in REGS:
            ems = [sp.emissions_from_proba(oof[epoch][(m, r)], fprior[m]) for m in TRAIN]
            s.append(pf.tune(ems, [pers[m] for m in TRAIN], [labs[m] for m in TRAIN], pf.COARSE_T, pf.COARSE_D)[0])
        curve[epoch] = float(np.mean(s))
        log("touchnet-RK epoch %d OOF pooled spellF1 %.3f %s" % (epoch, curve[epoch], np.round(s, 3)))
    best = max(sorted(curve), key=lambda e: (curve[e], -e))
    log("touchnet-RK chosen epoch %d (LOMO %.0fs)" % (best, time.time() - t0))
    out = {"epoch": np.int64(best), "curve_e": np.array(sorted(curve)), "curve_s": np.array([curve[e] for e in sorted(curve)])}
    for m in TRAIN:
        for r in REGS:
            out["oof%s_%s" % (r, m)] = sp.emissions_from_proba(oof[best][(m, r)], fprior[m])
    del oof
    gc.collect()
    t1 = time.time()
    model = nets.train_tn(bank, X, AUX, sym, names_of(TRAIN), cfg, 100 * SEED + 4, best, None, None, TL)
    save_pt(model, B("tn_train4.pt"))
    prior = prior_of(TRAIN)
    for m in TEST:
        for r in REGS:
            out["test%s_%s" % (r, m)] = sp.emissions_from_proba(nets.predict_tn(model, bank, X, sym, m + SUF[r]), prior)
    log("touchnet-RK TRAIN-4 model %d epochs (%.0fs)" % (best, time.time() - t1))
    del model, bank, X, AUX, TL
    free_gpu()
    pf._save_npz(path, **out)
    return out


def score_test(em, costs):
    per = {k: {} for k in DEFS}
    for m in TEST:
        pred = sp.decode_periods(np.asarray(em[m], np.float64), period_of(m), costs[0], costs[1])
        for k, v in pf.score_defs(labs_of(m), pred, period_of(m), DEFS).items():
            per[k][m] = v
    return {k: float(np.mean(list(v.values()))) for k, v in per.items()}, per


def cpu_test_emissions(cfg_rec, rec_path, tn_path, epoch_cfg):
    """Shipped inference path on the bench TEST matches: each match alone in a CPU bank without any
    privileged array, weights reloaded from disk, float32 everywhere."""
    import torch
    cpu = torch.device("cpu")
    rec = load_rec(rec_path, cfg_rec, cpu)
    tnm = load_tn(tn_path, epoch_cfg, cpu)
    prior = prior_of(TRAIN)
    out, times = {}, {}
    for m in TEST:
        t0 = time.time()
        bank = bank_of([m], cpu, privileged=False)
        with torch.no_grad():
            R = nets.predict_aux(rec, bank, m, cfg_rec, amp=False)
        t1 = time.time()
        X, _, sym = tn_inputs(bank, R)
        out[m] = sp.emissions_from_proba(nets.predict_tn(tnm, bank, X, sym, m), prior).astype(np.float64)
        times[m] = (round(t1 - t0, 1), round(time.time() - t1, 1))
        log("cpu float32 TEST %s: reconstructor %.0fs, touchnet %.0fs" % (m, times[m][0], times[m][1]))
        del bank, R, X
        gc.collect()
    return out, times


def run_bench(dev, report, cpu_check=True):
    t_all = time.time()
    rec_stage(B("rec"), [([m for m in TRAIN if m != h], [h]) for h in TRAIN] + [(TRAIN, TEST)],
              100 * SEED + 7000, dev)
    bank = bank_of(ALL14, dev)
    rrep = rec_report(B("rec"), bank, (("OOF", TRAIN), ("TEST", TEST)))
    del bank
    free_gpu()
    teacher_stage(B("teach.npy"), [(names_of([m for m in TRAIN if m != h], ("4",)), [h]) for h in TRAIN],
                  100 * SEED + 9000, dev)
    t = bench_tn(dev)
    g_oof, g_test = bench_gbm()
    epoch = int(t["epoch"])
    x_oof = {m: t["oof4_" + m].astype(np.float64) for m in TRAIN}
    x_test = {m: t["test4_" + m].astype(np.float64) for m in TEST}
    ptr, ltr = [period_of(m) for m in TRAIN], [labs_of(m) for m in TRAIN]
    res = {"seed": SEED, "tn_epoch": epoch, "rec_error": rrep, "methods": {}}
    fable = {"P1 v3+touchnet-RK w=.5": (.657, .715, .774, .566), "touchnet-RK": (.652, .710, .768, .552),
             "v3": (.614, .670, .727, .556)}
    choice = None
    for name, wg in (("v3", 1.0), ("touchnet-RK", 0.0), ("P1 v3+touchnet-RK w=.5", 1 - W_TN)):
        oofs, st, sd = pf.tune([wg * g_oof[m] + (1 - wg) * x_oof[m] for m in TRAIN], ptr, ltr, pf.GRID_T, pf.GRID_D)
        sc, per = score_test({m: wg * g_test[m] + (1 - wg) * x_test[m] for m in TEST}, (st, sd))
        res["methods"][name] = {"costs": [st, sd], "oof_pooled": oofs, "test": sc, "test_per_match": per,
                                "fable_s0": fable[name]}
        log("bench s%d %-24s costs %g/%g OOF %.3f TEST %s | Fable %s" % (
            SEED, name, st, sd, oofs, " ".join("%s %.3f" % (k, sc[k]) for k in DEFS), fable[name]))
        if name.startswith("P1"):
            choice = {"rev": REV, "tn_epoch": epoch, "switch_team": st, "switch_dead": sd, "oof_pooled": oofs}
    with open(B("choice.json"), "w") as fh:
        json.dump(choice, fh, indent=1)
    if cpu_check:
        cfgt = dict(nets.TN_CFG, p_mask=0.0, kd=0.5)
        x_cpu, times = cpu_test_emissions(nets.rec_cfg(), B("rec", "rec_train4.pt"), B("tn_train4.pt"), cfgt)
        diff = max(float(np.abs(x_cpu[m] - x_test[m]).max()) for m in TEST)
        costs = (choice["switch_team"], choice["switch_dead"])
        sc, per = score_test({m: (1 - W_TN) * g_test[m] + W_TN * x_cpu[m] for m in TEST}, costs)
        sc1, _ = score_test(x_cpu, tuple(res["methods"]["touchnet-RK"]["costs"]))
        res["cpu_fp32"] = {"P1": sc, "P1_per_match": per, "touchnet-RK": sc1, "max_abs_emission_diff": diff,
                           "seconds_per_match": times}
        d = {k: sc[k] - res["methods"]["P1 v3+touchnet-RK w=.5"]["test"][k] for k in DEFS}
        log("bench s%d P1 shipped path (CPU float32, reloaded weights, match alone): TEST %s | change vs GPU bf16 %s"
            " | touchnet max |d emission| %.4f" % (SEED, " ".join("%s %.3f" % (k, sc[k]) for k in DEFS),
                                                  " ".join("%+.4f" % d[k] for k in DEFS), diff))
    res["seconds"] = round(time.time() - t_all, 1)
    if report:
        os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
        with open(report, "w") as fh:
            json.dump(res, fh, indent=1)
    return res


# -- final training on all 7 open matches ------------------------------------------------------------

def S(*a):
    return P("ship" + TAG, *a)


def run_train_final(dev):
    t_all = time.time()
    if not os.path.isfile(B("choice.json")):
        raise SystemExit("run --bench first: the epoch and switch costs come from the bench TRAIN-4 LOMO")
    with open(B("choice.json")) as fh:
        choice = json.load(fh)
    os.makedirs(S(), exist_ok=True)
    F = lambda *a: P("final" + TAG, *a)
    # (b) 7 leave-one-match-out reconstructors (OOF inputs of the touchnet) + the 7-match reconstructor
    rec_stage(F("rec"), [([m for m in ORDER if m != h], [h]) for h in ORDER], 100 * SEED + 7100, dev,
              ship=(ORDER, S("rec_final7.pt")))
    # (c) 7 cross-fitted clean-ball teachers
    teacher_stage(F("teach.npy"), [(names_of([m for m in ORDER if m != h], ("4",)), [h]) for h in ORDER],
                  100 * SEED + 9100, dev)
    # (d) touchnet-RK on 7 matches x 2 copies
    cfgt = dict(nets.TN_CFG, p_mask=0.0, kd=0.5)
    if not os.path.isfile(S("tn_final7.pt")):
        import torch
        t0 = time.time()
        bank = bank_of(ALL14, dev)
        TL = torch.from_numpy(np.load(F("teach.npy"))).to(dev)
        R = load_R(F("rec"), bank)
        X, AUX, sym = tn_inputs(bank, R)
        del R
        model = nets.train_tn(bank, X, AUX, sym, names_of(ORDER), cfgt, 100 * SEED + 7, choice["tn_epoch"],
                              None, log, TL)
        save_pt(model, S("tn_final7.pt"))
        log("touchnet-RK final7 %d epochs (%.0fs), weights sha256 %s" % (
            choice["tn_epoch"], time.time() - t0, pf.state_sha(model)))
        del model, bank, X, AUX, TL
        free_gpu()
    # (e) frozen v3 GBM on 7 matches (pipeline_final)
    if not os.path.isfile(S("gbm_final7.txt")):
        src = os.path.join(pf.CACHE, "gbm_final7.txt")
        if os.path.isfile(src) and pf.file_sha256(src) == GBM7_SHA:
            shutil.copyfile(os.path.join(pf.CACHE, "gbm_final7_prior.npy"), S("gbm_final7_prior.npy"))
            shutil.copyfile(src, S("gbm_final7.txt"))
            log("gbm final7 copied from pipeline_final (sha256 checked)")
        else:
            from bench import run_v3cand as rv
            op = pf.Open(pf.CACHE)
            pf.gbm_model(op, rv.CONFIGS["v3"](), S(), ORDER, "final7")
    if not os.path.isfile(S("gbm_fmeta.json")):
        op = pf.Open(pf.CACHE)
        with open(S("gbm_fmeta.json"), "w") as fh:
            json.dump(op.fmeta, fh)
    ship = dict(choice, w_tn=W_TN, tn_prior=list(prior_of(ORDER)), tn_cfg=cfgt, rec_cfg=nets.rec_cfg(),
                seeds={"degrade": "1000*%d+index" % SEED, "rec_lomo7": 100 * SEED + 7100, "rec_final7": 100 * SEED + 7107,
                       "teacher_lomo7": 100 * SEED + 9100, "tn_final7": 100 * SEED + 7})
    with open(S("choice.json"), "w") as fh:
        json.dump(ship, fh, indent=1)
    write_manifest()
    log("train-final done (%.0fs)" % (time.time() - t_all))


SHIP_FILES = ("gbm_final7.txt", "gbm_final7_prior.npy", "gbm_fmeta.json", "rec_final7.pt", "tn_final7.pt", "choice.json")


def write_manifest():
    man = {f: {"bytes": os.path.getsize(S(f)), "sha256": pf.file_sha256(S(f))} for f in SHIP_FILES}
    with open(S("manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    for f, v in man.items():
        log("ship %-22s %9d bytes sha256 %s" % (f, v["bytes"], v["sha256"]))


def check_manifest():
    with open(S("manifest.json")) as fh:
        man = json.load(fh)
    for f in SHIP_FILES:
        if pf.file_sha256(S(f)) != man[f]["sha256"]:
            raise SystemExit("shipped file %s does not match manifest.json" % f)
    return man


# -- the real match (inference from the shipped files only) -----------------------------------------

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
    with open(S("gbm_fmeta.json")) as fh:
        fmeta = json.load(fh)
    spec = rv.CONFIGS["v3"]()
    gm = lgb.Booster(model_file=S("gbm_final7.txt"))
    gm.params["num_threads"] = pf.NT
    gprior = np.load(S("gbm_final7_prior.npy"))

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
    rec = load_rec(S("rec_final7.pt"), ch["rec_cfg"], dev)
    with torch.no_grad():
        R = nets.predict_aux(rec, bank, "real", ch["rec_cfg"], amp=False)
    del rec
    t_rec = time.time() - t0
    t0 = time.time()
    X, _, sym = tn_inputs(bank, R)
    tnm = load_tn(S("tn_final7.pt"), ch["tn_cfg"], dev)
    e_t = sp.emissions_from_proba(nets.predict_tn(tnm, bank, X, sym, "real"), np.array(ch["tn_prior"])).astype(np.float64)
    t_tn = time.time() - t0
    del bank, X, R, tnm
    e = (1 - ch["w_tn"]) * e_g + ch["w_tn"] * e_t
    codes = sp.decode_periods(e, period, ch["switch_team"], ch["switch_dead"])
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
    summary = {"rev": REV, "out": out, "sha256": digest, "device": dev.type, "fresh": args.fresh,
               "choice": {k: ch[k] for k in ("tn_epoch", "switch_team", "switch_dead", "w_tn")},
               "ship": man, "label_counts": counts, "stats": stats,
               "seconds": {"load": round(t_load, 1), "gbm": round(t_gbm, 1), "rec": round(t_rec, 1),
                           "touchnet": round(t_tn, 1), "total": round(time.time() - t_all, 1)}}
    for path in [P("last_run.json")] + ([args.summary_json] if args.summary_json else []):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out], check=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="submission CSV, from the shipped weights")
    ap.add_argument("--fresh", action="store_true", help="re-parse the real tracking and rebuild its inputs")
    ap.add_argument("--bench", action="store_true", help="open-data check: TRAIN 4 -> TEST 3 (seed 0)")
    ap.add_argument("--bench-report", default=None)
    ap.add_argument("--no-cpu-check", action="store_true")
    ap.add_argument("--train-final", action="store_true", help="train the shipped weights on all 7 open matches")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args(argv)
    if not (args.bench or args.train_final or args.out):
        ap.error("one of --bench, --train-final, --out is required")
    pf.nice()
    t0 = time.time()
    from bench import feats3
    feats3._warn_off()
    os.makedirs(CACHE, exist_ok=True)
    dev = pf.torch_setup(args.device)
    log("pipeline_final2 %s: frozen v3 GBM + touchnet-RK (w %.2f), device %s" % (REV, W_TN, dev))
    if args.bench or args.train_final:
        build_preps()
    if args.bench:
        run_bench(dev, args.bench_report, not args.no_cpu_check)
    if args.train_final:
        run_train_final(dev)
    if args.out:
        run_infer(args, dev)
    log("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
