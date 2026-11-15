"""final3 PCF + tempo wrapper, exactly as pipeline_final3_tempo, but with Viterbi team-switch cost 30 (dead 20 unchanged).

    python -m socbac.pipeline_final3_c30 --out evaluation/results/socbac_final3_pcf_c30_tempo_cpu1.csv --device cpu
    python -m socbac.pipeline_final3_c30 --out ... --device cpu --fresh
    python -m socbac.pipeline_final3_c30 --out ... --device cpu --tempo-off      # final3 PCF at cost 30, no wrapper

Why 30: pre-declared label-definition hedge (handover/CHECKPOINT.md, 2026-11-14 ~16:00), decided on open IDSSE data only.
The team cost minimising worst-case regret over the label variants {L1, L2, L4, L4 with opposing runs < 5 s merged} on
TRAIN-match OOF PCF emissions (grid 4..60, dead 20) is 30; on TEST (degrade_v4, seeds 0-2) it raises mean(L1,L2,L4)
by +.008 and lowers worst-case regret .090 -> .032 vs the shipped 8. See research/runs/cost_hedge/RESULT.md.
The real match and the public window were never used for this choice.

Shipped files, manifest checks, weights, blend weight and tempo rule are final3 / pipeline_final3_tempo's, unchanged
(pipeline_final3, pipeline_final3_tempo, pipeline_final2_tempo and tempo are imported read only). Only the decoder's
team-switch cost differs. Real-input caches use their own key (prefix pf3c30) and the summary its own file.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from socbac import pipeline_final3 as p3          # sets threads/env and sys.path exactly as final3
from socbac import pipeline_final3_tempo as t3    # load_models with final3's manifest checks
from socbac import pipeline_final2_tempo as t2    # shared wrapper core
from socbac import tempo

import numpy as np  # noqa: E402

pf, sp, log = p3.pf, p3.sp, p3.log
REV = "pf3c30"
SWITCH_TEAM = 30.0       # pre-declared hedge rule result (shipped final3 PCF: 8)
SWITCH_DEAD = 20.0       # unchanged


def run_infer(args, dev):
    from bench import methods
    from socbac import atd
    from socbac.pipeline_v2 import format_stats, real_layout, sanity_stats
    from games import GAME, GAMES
    t_all = time.time()
    M = t3.load_models("pcf", dev)
    ch, var = M["ch"], M["var"]
    if float(var["switch_team"]) != 8.0 or float(var["switch_dead"]) != SWITCH_DEAD:
        raise SystemExit("shipped PCF costs are not 8/20; refusing to run")
    log("decoder costs: team %g (shipped %g), dead %g (unchanged)" % (SWITCH_TEAM, var["switch_team"], SWITCH_DEAD))
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
    base = "%s_%d_%d" % (REV, len(idx), int(real["players"].shape[2]))
    key = base if k == 1.0 else "%s_k%.4f" % (base, k)

    def inputs(m):
        if args.fresh:
            for s in ("x.npy", "xs.npy", "prep.npz"):
                p = p3.P("real_%s.%s" % (key, s))
                if os.path.isfile(p):
                    os.remove(p)
        return pf.real_inputs(m, np.arange(len(m["period"])), p3.CACHE, key, args.fresh)

    e_g, e_t, (t_g, t_r, t_t) = t2.wrapped_emissions(sub, k, inputs, M, dev)
    t_load = time.time() - t0 - t_g - t_r - t_t
    e = (1 - M["w_tn"]) * e_g + M["w_tn"] * e_t
    codes = sp.decode_periods(e, period, SWITCH_TEAM, SWITCH_DEAD)
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
    summary = {"rev": REV, "variant": "pcf", "out": out, "sha256": digest, "device": dev.type, "fresh": args.fresh,
               "decoder": {"switch_team": SWITCH_TEAM, "switch_dead": SWITCH_DEAD,
                           "shipped_switch_team": var["switch_team"], "rule": "research/runs/cost_hedge/RESULT.md"},
               "tempo": {"factor": k, "pace_ref": tempo.PACE_REF, "deadband": tempo.DEADBAND, "off": args.tempo_off},
               "choice": {"tn_epoch": ch["tn_epoch"], "w_tn": ch["w_tn"], **var}, "ship": M["man"],
               "label_counts": counts, "stats": stats,
               "emission_sha": {"gbm": pf.sha(e_g)[:16], "touchnet": pf.sha(e_t)[:16], "blend": pf.sha(e)[:16]},
               "seconds": {"inputs": round(t_load, 1), "gbm": round(t_g, 1), "rec": round(t_r, 1),
                           "touchnet": round(t_t, 1), "total": round(time.time() - t_all, 1)}}
    for path in [p3.P("last_run_%s%s.json" % (REV, "_off" if args.tempo_off else ""))] + \
            ([args.summary_json] if args.summary_json else []):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)
    subprocess.run([sys.executable, os.path.join(p3.ROOT, "evaluation", "validate_submission.py"), out], check=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="submission CSV, from final3's shipped weights")
    ap.add_argument("--fresh", action="store_true", help="re-parse the real tracking and rebuild its inputs")
    ap.add_argument("--tempo-off", action="store_true", help="force factor 1 (final3 PCF at team cost 30)")
    ap.add_argument("--device", default="cpu", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args(argv)
    pf.nice()
    t0 = time.time()
    from bench import feats3
    feats3._warn_off()
    os.makedirs(p3.CACHE, exist_ok=True)
    dev = pf.torch_setup(args.device)
    log("pipeline_final3_c30 %s: final3 PCF + tempo wrapper, team-switch cost %g / dead %g, device %s" % (
        REV, SWITCH_TEAM, SWITCH_DEAD, dev))
    run_infer(args, dev)
    log("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
