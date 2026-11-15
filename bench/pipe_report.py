"""Markdown tables from bench.run_pipe_bench results (aggregates only).

    python -m bench.pipe_report --seeds 0 1 2 [--real research/runs/pipeline/real_stats.json]
"""

import argparse
import glob
import json
import os

import numpy as np

from bench.run_pipe_bench import RUN_DIR, aggregate, load_recs

STAT_KEYS = ("share_H", "share_A", "share_D", "poss_per_90", "poss_mean_s", "poss_median_s",
             "poss_share_under_4s", "home_share_alive", "play_share")


def board_md(recs, title):
    t = aggregate(recs)
    lines = ["#### " + title, "",
             "| model | test degr. | labels | spellF1 | sd | per-seed spellF1 | macroF1 | poss pred | poss true | home-share err |",
             "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for key in sorted(t, key=lambda k: (k[2], k[1], k[0])):
        v = t[key]
        lines.append("| %s | %s | %s | %.3f | %.3f | %s | %.3f | %.0f | %.0f | %.3f |" % (
            key[0], key[1], key[2], v["spell"], v["spell_sd"], " ".join("%.3f" % s for s in v["spell_per_seed"]),
            v["macro"], v["n_pred"], v["n_gt"], v["share_err"]))
    return "\n".join(lines)


def gains_md(recs):
    t = aggregate(recs)
    lines = ["| comparison | test degr. | labels | per-seed gain | mean gain | +0.03 on every seed |",
             "|---|---|---|---|---:|---|"]
    pairs = [("v2|prov|coupled", "v1|prov|coupled", "prov"),
             ("v2|l1|coupled", "v1|l1|coupled", "l1"), ("v2|l1|coupled", "v1|l1|coupled", "prov"),
             ("v2|l1|mix", "v1|l1|mix", "l1"),
             ("v2|l1|mix", "v2|l1|coupled", "l1"), ("v2|l1|mix", "v2|l1|coupled", "prov"),
             ("v1|l1|mix", "v1|l1|coupled", "l1"),
             ("v2|l1|coupled", "v2|prov|coupled", "l1"), ("v2|l1|coupled", "v2|prov|coupled", "prov")]
    for a, b, lab in pairs:
        for test in ("coupled", "alt"):
            ka, kb = (a, test, lab), (b, test, lab)
            if ka not in t or kb not in t:
                continue
            sa, sb = t[ka], t[kb]
            common = [s for s in sa["seeds"] if s in sb["seeds"]]
            g = [sa["spell_per_seed"][sa["seeds"].index(s)] - sb["spell_per_seed"][sb["seeds"].index(s)]
                 for s in common]
            lines.append("| %s vs %s | %s | %s | %s | %+.3f | %s |" % (
                a, b, test, lab, " ".join("%+.3f" % x for x in g), float(np.mean(g)),
                "yes" if all(x >= 0.03 for x in g) else "no"))
    return "\n".join(lines)


def mean_stats(dicts):
    return {k: float(np.mean([d[k] for d in dicts])) for k in STAT_KEYS}


def stats_md(recs, model, real=None):
    lines = ["| sequence | H | A | D | poss / 90 min | mean poss s | median poss s | share < 4 s | P(H given alive) | play share |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def row(name, s):
        return "| %s | %.3f | %.3f | %.3f | %.0f | %.1f | %.1f | %.2f | %.3f | %.3f |" % (
            name, s["share_H"], s["share_A"], s["share_D"], s["poss_per_90"], s["poss_mean_s"],
            s["poss_median_s"], s["poss_share_under_4s"], s["home_share_alive"], s["play_share"])
    if real is not None:
        lines.append(row("BAR-ATH graded, our prediction", real))
    for test in ("coupled", "alt"):
        preds = [r["pred_stats"] for rec in recs for r in rec["rows"]
                 if r["model"] == model and r["test"] == test and r["label"] == "l1"]
        if preds:
            lines.append(row("bench TEST, %s prediction (%s degr.)" % (model, test), mean_stats(preds)))
    for lab, name in (("prov", "provider"), ("l1", "L1")):
        tr = [rec["truth_stats"][m][lab] for rec in recs[:1] for m in rec["truth_stats"]]
        if tr:
            lines.append(row("bench TEST, true %s labels" % name, mean_stats(tr)))
    return "\n".join(lines)


def lomo_md():
    lines = []
    for path in sorted(glob.glob(os.path.join(RUN_DIR, "lomo_s*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        lines.append("LOMO %s (seed %d, labels %s, %.0f s):" % (rec["config"], rec["seed"], rec["label"], rec["seconds"]))
        for crit, b in rec["best"].items():
            lines.append("- best by OOF spellF1 on %s copies: team %g, dead %g (%.3f)" % (
                crit, b["team"], b["dead"], b["oof_spell"]))
        top = sorted(rec["grid"], key=lambda g: -g["mean"])[:6]
        lines.append("- top of grid by mean: " + "; ".join(
            "%g/%g %.3f (c %.3f a %.3f)" % (g["team"], g["dead"], g["mean"], g["coupled"], g["alt"]) for g in top))
        fx = [g for g in rec["grid"] if g["team"] == 5 and g["dead"] == 20]
        if fx:
            g = fx[0]
            lines.append("- at 5/20: mean %.3f (coupled %.3f, alt %.3f)" % (g["mean"], g["coupled"], g["alt"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--real", default=None)
    ap.add_argument("--model", default="v2|l1|coupled")
    args = ap.parse_args()
    recs = load_recs(args.seeds)
    print(board_md(recs, "TEST board, costs team 5 / dead 20 (seeds %s)" % " ".join(str(r["seed"]) for r in recs)))
    print()
    for path in sorted(glob.glob(os.path.join(RUN_DIR, "rescore_*.json"))):
        with open(path) as fh:
            rr = json.load(fh)
        print(board_md(rr, "TEST board re-decoded at %s (reporting only)" % os.path.basename(path)))
        print()
    print(gains_md(recs))
    print()
    real = None
    if args.real and os.path.isfile(args.real):
        with open(args.real) as fh:
            real = json.load(fh)["stats"]
    print(stats_md(recs, args.model, real))
    print()
    print(lomo_md())


if __name__ == "__main__":
    main()
