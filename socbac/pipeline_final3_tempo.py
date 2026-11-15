"""final3 (provenance-clean P1, variants PCF / PCT) with the tempo wrapper around its whole predict path.
pipeline_final3.py is imported and not edited; shipped files, manifest checks and per-variant costs are final3's.

    python -m socbac.pipeline_final3_tempo --variant pcf --out evaluation/results/<name>.csv --device cpu
    python -m socbac.pipeline_final3_tempo --variant pcf --out ... --device cpu --tempo-off   # = pipeline_final3

The wrapper is the same rule and code as pipeline_final2_tempo (socbac/tempo.py: PACE_REF 2.23, deadband 0.03,
one-sided, k in [0.75, 1.33]); the GBM, ball reconstructor and touchnet all run on the resampled frames; each
model's emissions are mapped back to the source frames, blended (w_tn) and decoded with the variant's costs.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from socbac import pipeline_final3 as p3          # sets threads/env and sys.path exactly as final3
from socbac import pipeline_final2_tempo as t2    # shared wrapper core (p1_emissions / wrapped_emissions)
from socbac import tempo

import numpy as np  # noqa: E402

pf, pf2, nets, sp, log = p3.pf, p3.pf2, p3.nets, p3.sp, p3.log
REV = "pf3t"
p2 = t2.p2   # wrapped_emissions uses p2.tn_inputs (= pf2.tn_inputs, the function final3 calls)


def load_models(variant, dev):
    """final3's shipped models for a variant, with final3's own checks (copied from pipeline_final3.run_infer)."""
    import lightgbm as lgb
    man = p3.check_manifest()
    with open(p3.S("choice.json")) as fh:
        ch = json.load(fh)
    var = ch["variants"][variant]
    gpath, ppath = os.path.normpath(p3.S(var["gbm"])), os.path.normpath(p3.S(var["gbm_prior"]))
    if variant == "pct":
        with open(os.path.join(p3.PF2_SHIP, "manifest.json")) as fh:
            man2 = json.load(fh)
        for p, sha, key in ((gpath, var["gbm_sha256"], "gbm_final7.txt"),
                            (ppath, var["gbm_prior_sha256"], "gbm_final7_prior.npy")):
            if pf.file_sha256(p) != sha or man2[key]["sha256"] != sha:
                raise SystemExit("frozen v3 GBM file %s does not match its manifest" % key)
    with open(p3.S("gbm_fmeta.json")) as fh:
        fmeta = json.load(fh)
    gm = lgb.Booster(model_file=gpath)
    gm.params["num_threads"] = pf.NT
    gprior = np.load(ppath)
    rec = pf2.load_rec(p3.S("rec_final7.pt"), ch["rec_cfg"], dev)
    tnm = pf2.load_tn(p3.S("tn_final7.pt"), ch["tn_cfg"], dev)
    return dict(man=man, ch=ch, var=var, fmeta=fmeta, gm=gm, gprior=gprior, rec=rec, tnm=tnm,
                rec_cfg=ch["rec_cfg"], tn_prior=np.array(ch["tn_prior"]), w_tn=ch["w_tn"])


def run_infer(args, dev):
    from bench import methods
    from socbac import atd
    from socbac.pipeline_v2 import format_stats, real_layout, sanity_stats
    from games import GAME, GAMES
    t_all = time.time()
    M = load_models(args.variant, dev)
    ch, var = M["ch"], M["var"]
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
    base = "%s_%d_%d" % (p3.REV, len(idx), int(real["players"].shape[2]))
    key = base if k == 1.0 else "%s_k%.4f" % (base, k)

    def inputs(m):        # final3's cached real-input builder; resampled frames get their own cache key
        if args.fresh:
            for s in ("x.npy", "xs.npy", "prep.npz"):
                p = p3.P("real_%s.%s" % (key, s))
                if os.path.isfile(p):
                    os.remove(p)
        return pf.real_inputs(m, np.arange(len(m["period"])), p3.CACHE, key, args.fresh)

    e_g, e_t, (t_g, t_r, t_t) = t2.wrapped_emissions(sub, k, inputs, M, dev)
    t_load = time.time() - t0 - t_g - t_r - t_t
    e = (1 - M["w_tn"]) * e_g + M["w_tn"] * e_t
    codes = sp.decode_periods(e, period, var["switch_team"], var["switch_dead"])
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
    summary = {"rev": REV, "variant": args.variant, "out": out, "sha256": digest, "device": dev.type, "fresh": args.fresh,
               "tempo": {"factor": k, "pace_ref": tempo.PACE_REF, "deadband": tempo.DEADBAND, "off": args.tempo_off},
               "choice": {"tn_epoch": ch["tn_epoch"], "w_tn": ch["w_tn"], **var}, "ship": M["man"],
               "label_counts": counts, "stats": stats,
               "emission_sha": {"gbm": pf.sha(e_g)[:16], "touchnet": pf.sha(e_t)[:16], "blend": pf.sha(e)[:16]},
               "seconds": {"inputs": round(t_load, 1), "gbm": round(t_g, 1), "rec": round(t_r, 1),
                           "touchnet": round(t_t, 1), "total": round(time.time() - t_all, 1)}}
    for path in [p3.P("last_run_tempo_%s.json" % args.variant)] + ([args.summary_json] if args.summary_json else []):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
    subprocess.run([sys.executable, os.path.join(p3.ROOT, "evaluation", "validate_submission.py"), out], check=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="submission CSV, from final3's shipped weights")
    ap.add_argument("--variant", default="pcf", choices=["pcf", "pct"])
    ap.add_argument("--fresh", action="store_true", help="re-parse the real tracking and rebuild its inputs")
    ap.add_argument("--tempo-off", action="store_true", help="force factor 1 (must reproduce pipeline_final3)")
    ap.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args(argv)
    pf.nice()
    t0 = time.time()
    from bench import feats3
    feats3._warn_off()
    os.makedirs(p3.CACHE, exist_ok=True)
    dev = pf.torch_setup(args.device)
    log("pipeline_final3_tempo %s: final3 %s + tempo wrapper, device %s" % (REV, args.variant.upper(), dev))
    run_infer(args, dev)
    log("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
