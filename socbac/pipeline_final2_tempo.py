"""P1 (pipeline_final2) with the tempo wrapper around its whole predict path. pipeline_final2.py is
imported and not edited; the shipped files and their manifest check are P1's.

    python -m socbac.pipeline_final2_tempo --out evaluation/results/<name>.csv --device cpu
    python -m socbac.pipeline_final2_tempo --out ... --device cpu --tempo-off    # identical to P1

Wrapper (socbac/tempo.py; rule agreed before the real-match check):
  pace = p75, over the graded frames, of the per-frame median tracked-player speed (label-free)
  k    = PACE_REF / pace with PACE_REF = 2.23 (mean of the 4 TRAIN degrade_v4 copies, open data),
         clipped to [0.75, 1.33]; k = 1 (wrapper off) if |k - 1| <= DEADBAND (0.03) or k < 1
         (one-sided: only a match slower than the training pace is resampled).
  If k != 1, the model frames (graded halves) are resampled by k inside each half (positions linearly
  interpolated in float32, NaN kept) and ALL THREE models run on the resampled frames: the GBM
  (features rebuilt), the ball reconstructor and the touchnet. Each model's emissions are mapped back
  to the source frames (nearest resampled frame), blended with P1's weight and decoded on the source
  timeline with P1's shipped costs. Deterministic: no randomness, fixed order, CPU float32.
  With k = 1 the code path, inputs and output are exactly pipeline_final2's.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from socbac import pipeline_final2 as p2   # sets threads/env and sys.path exactly as P1
from socbac import tempo

import numpy as np  # noqa: E402

pf, nets, sp, log = p2.pf, p2.nets, p2.sp, p2.log
REV = "pf2t"


def load_models(dev):
    """P1's shipped models (manifest-checked)."""
    import lightgbm as lgb
    man = p2.check_manifest()
    with open(p2.S("choice.json")) as fh:
        ch = json.load(fh)
    with open(p2.S("gbm_fmeta.json")) as fh:
        fmeta = json.load(fh)
    gm = lgb.Booster(model_file=p2.S("gbm_final7.txt"))
    gm.params["num_threads"] = pf.NT
    gprior = np.load(p2.S("gbm_final7_prior.npy"))
    rec = p2.load_rec(p2.S("rec_final7.pt"), ch["rec_cfg"], dev)
    tnm = p2.load_tn(p2.S("tn_final7.pt"), ch["tn_cfg"], dev)
    return dict(man=man, ch=ch, fmeta=fmeta, gm=gm, gprior=gprior, rec=rec, tnm=tnm,
                rec_cfg=ch["rec_cfg"], tn_prior=np.array(ch["tn_prior"]), w_tn=ch["w_tn"])


def p1_emissions(x, xs, prep, period, M, dev, name="real"):
    """GBM and touchnet emissions of one match on its own timeline (P1's run_infer steps)."""
    import torch
    from bench import run_v3cand as rv
    spec = rv.CONFIGS["v3"]()
    t0 = time.time()
    e_g = pf.gbm_em(M["gm"], M["gprior"], {"x": x, "xs": xs, "period": period}, M["fmeta"], spec).astype(np.float64)
    t_g = time.time() - t0
    t0 = time.time()
    bank = nets.make_bank([prep], [name], dev)
    with torch.no_grad():
        R = nets.predict_aux(M["rec"], bank, name, M["rec_cfg"], amp=False)
    t_r = time.time() - t0
    t0 = time.time()
    X, _, sym = p2.tn_inputs(bank, R)
    e_t = sp.emissions_from_proba(nets.predict_tn(M["tnm"], bank, X, sym, name), M["tn_prior"]).astype(np.float64)
    t_t = time.time() - t0
    del bank, X, R
    return e_g, e_t, (t_g, t_r, t_t)


def wrapped_emissions(sub, k, inputs, M, dev, name="real"):
    """sub: {ball, players, period} of the model frames. inputs(match) -> (x, xs, prep).
    Returns GBM and touchnet emissions on sub's timeline, the models run on sub resampled by k."""
    n = len(sub["period"])
    if k == 1.0:
        x, xs, prep = inputs(sub)
        return p1_emissions(x, xs, prep, np.asarray(sub["period"]), M, dev, name)
    w, _, blocks = tempo.warp(sub, k)
    x, xs, prep = inputs(w)
    e_g, e_t, t = p1_emissions(x, xs, prep, np.asarray(w["period"]), M, dev, name)
    return tempo.to_source(e_g, blocks, k, n), tempo.to_source(e_t, blocks, k, n), t


def run_infer(args, dev):
    from bench import methods
    from socbac import atd
    from socbac.pipeline_v2 import format_stats, real_layout, sanity_stats
    from games import GAME, GAMES
    t_all = time.time()
    M = load_models(dev)
    ch = M["ch"]
    t0 = time.time()
    real = atd.load(use_cache=not args.fresh)
    n = len(real["period"])
    if n != GAMES[GAME]["n_video_frames"]:
        raise RuntimeError("tracking length %d != video length" % n)
    idx, blocks = real_layout(real, use_anchor=True)
    period = np.asarray(real["period"][idx])
    k = 1.0 if args.tempo_off else tempo.factor(tempo.pace(real), tempo.PACE_REF)
    log("tempo wrapper: %s (factor %.2f; PACE_REF %.2f, deadband %.2f, one-sided)" % (
        "ON" if k != 1.0 else "off", k, tempo.PACE_REF, tempo.DEADBAND))
    sub = {"ball": real["ball"][idx], "players": real["players"][idx], "period": period}
    base = "%s_%d_%d" % (p2.REV, len(idx), int(real["players"].shape[2]))
    key = base if k == 1.0 else "%s_k%.4f" % (base, k)

    def inputs(m):        # P1's cached real-input builder; the resampled frames get their own cache key
        nm = len(m["period"])
        if args.fresh:
            for s in ("x.npy", "xs.npy", "prep.npz"):
                p = p2.P("real_%s.%s" % (key, s))
                if os.path.isfile(p):
                    os.remove(p)
        return pf.real_inputs(m, np.arange(nm), p2.CACHE, key, args.fresh)

    e_g, e_t, (t_g, t_r, t_t) = wrapped_emissions(sub, k, inputs, M, dev)
    t_load = time.time() - t0 - t_g - t_r - t_t
    e = (1 - M["w_tn"]) * e_g + M["w_tn"] * e_t
    codes = sp.decode_periods(e, period, ch["switch_team"], ch["switch_dead"])
    log("emissions sha256: gbm %s touchnet %s blend %s | inputs %.0fs, gbm %.0fs, reconstructor %.0fs, touchnet %.0fs" % (
        pf.sha(e_g)[:16], pf.sha(e_t)[:16], pf.sha(e)[:16], t_load, t_g, t_r, t_t))

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
    counts = {c: int((letters == c).sum()) for c in "HADX"}
    log("wrote %s: %d rows, label counts %s" % (out, n, counts))
    print(format_stats("BAR-ATH graded (our prediction)", stats), flush=True)
    digest = pf.file_sha256(out)
    print("sha256 %s" % digest, flush=True)
    summary = {"rev": REV, "out": out, "sha256": digest, "device": dev.type, "fresh": args.fresh,
               "tempo": {"factor": k, "pace_ref": tempo.PACE_REF, "deadband": tempo.DEADBAND, "off": args.tempo_off},
               "choice": {c: ch[c] for c in ("tn_epoch", "switch_team", "switch_dead", "w_tn")},
               "ship": M["man"], "label_counts": counts, "stats": stats,
               "seconds": {"inputs": round(t_load, 1), "gbm": round(t_g, 1), "rec": round(t_r, 1),
                           "touchnet": round(t_t, 1), "total": round(time.time() - t_all, 1)}}
    for path in [p2.P("last_run_tempo.json")] + ([args.summary_json] if args.summary_json else []):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
    subprocess.run([sys.executable, os.path.join(p2.ROOT, "evaluation", "validate_submission.py"), out], check=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="submission CSV, from P1's shipped weights")
    ap.add_argument("--fresh", action="store_true", help="re-parse the real tracking and rebuild its inputs")
    ap.add_argument("--tempo-off", action="store_true", help="force factor 1 (must reproduce pipeline_final2)")
    ap.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args(argv)
    pf.nice()
    t0 = time.time()
    from bench import feats3
    feats3._warn_off()
    os.makedirs(p2.CACHE, exist_ok=True)
    dev = pf.torch_setup(args.device)
    log("pipeline_final2_tempo %s: P1 + tempo wrapper, device %s" % (REV, dev))
    run_infer(args, dev)
    log("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
