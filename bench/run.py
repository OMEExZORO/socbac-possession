"""Score possession methods on degraded open DFL matches with the official scorer.

    python -m bench.run                       # all methods, held-out matches, 3 seeds
    python -m bench.run --seeds 0 1 2 3 4

Split: learned/tuned methods only ever see TRAIN; the board is computed on TEST.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as pm  # noqa: E402  the official scorer, unmodified

from bench import dfl, methods  # noqa: E402
from bench.degrade import degrade  # noqa: E402

TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
TEST = ["J03WPY", "J03WQQ", "J03WR9"]
GAP = 500  # X frames between periods so the scorer treats each half as its own run


def labels(match):
    return np.where(~match["alive"], methods.D, np.where(match["owner"] == 1, methods.A, methods.H)).astype(np.int8)


def with_gaps(codes, period):
    """Concatenate periods with X gaps, as the scorer expects for separate halves."""
    parts = []
    for p in np.unique(period):
        parts.append(codes[period == p])
        parts.append(np.full(GAP, pm.X, np.int8))
    return np.concatenate(parts[:-1])


def score(gt, pred, period):
    r = pm.evaluate(with_gaps(gt, period), with_gaps(pred, period))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def tune_nb(train_deg, train_lab, grid_team, grid_dead):
    """Fit the evidence on TRAIN, then pick the two switch costs by mean spellF1 on TRAIN."""
    model = methods.NBViterbi().fit(train_deg, train_lab)
    ems = [model.emissions(m) for m in train_deg]
    best = None
    for st in grid_team:
        for sd in grid_dead:
            model.switch_team, model.switch_dead = st, sd
            s = []
            for m, e, y in zip(train_deg, ems, train_lab):
                pred = np.full(len(e), methods.D, np.int8)
                for p in np.unique(m["period"]):
                    idx = np.flatnonzero(m["period"] == p)
                    pred[idx] = model.decode(e[idx])
                s.append(score(y, pred, m["period"])[0])
            if best is None or np.mean(s) > best[0]:
                best = (np.mean(s), st, sd)
    model.switch_team, model.switch_dead = best[1], best[2]
    print("nb_viterbi tuned on TRAIN: switch_team={} switch_dead={} (train spellF1 {:.3f})".format(
        best[1], best[2], best[0]))
    return model


def tune_lgbm(train_deg, train_lab, grid_team, grid_dead, symmetric=True):
    """Leave-one-match-out within TRAIN gives honest emissions to tune the two costs on; the final
    model is then refit on all of TRAIN."""
    oof = []
    for k in range(len(train_deg)):
        rest = [i for i in range(len(train_deg)) if i != k]
        model = methods.LGBMViterbi(symmetric=symmetric).fit([train_deg[i] for i in rest],
                                                              [train_lab[i] for i in rest])
        oof.append(model.emissions(train_deg[k]))
    best = None
    for st in grid_team:
        for sd in grid_dead:
            s = [score(y, methods.decode_periods(e, m["period"], st, sd), m["period"])[0]
                 for m, e, y in zip(train_deg, oof, train_lab)]
            if best is None or np.mean(s) > best[0]:
                best = (np.mean(s), st, sd)
    final = methods.LGBMViterbi(best[1], best[2], symmetric=symmetric).fit(train_deg, train_lab)
    print("lgbm_viterbi tuned out-of-fold on TRAIN: switch_team={} switch_dead={} (oof spellF1 {:.3f})".format(
        best[1], best[2], best[0]))
    return final


ESPADAS = os.path.join(HERE, "cache", "third_party", "espadas")


def espadas_runner(version):
    """Team Espadas' labelers (MIT, github.com/MoreThanAHackEspadas/espadas), run as published with
    their own settings. Cloned into bench/cache/third_party (gitignored), never vendored."""
    if ESPADAS not in sys.path:
        sys.path.insert(0, ESPADAS)
    from espadas import labelers

    def run(match):
        n = len(match["ball"])
        blocks, kicks = [], []
        for p in np.unique(match["period"]):
            idx = np.flatnonzero(match["period"] == p)
            blocks.append((int(p), int(idx[0]), int(idx[-1])))
            kicks.append(int(idx[0]))
        tracking = {"players": match["players"].reshape(n, -1, 2).astype(float),
                    "ball": match["ball"][:, :2].astype(float),
                    "team_of_slot": np.repeat([0, 1], match["players"].shape[2]).astype(np.int8),
                    "half_blocks": blocks, "kickoffs": tuple(kicks), "fps": 25, "pitch": (105.0, 68.0)}
        lab = np.asarray(labelers.LABELERS[version](tracking))
        return np.where(lab == "H", methods.H, np.where(lab == "A", methods.A, methods.D)).astype(np.int8)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grid-team", type=float, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--grid-dead", type=float, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--espadas", nargs="*", default=[],
                    help="opt-in: Espadas labelers to score (third-party code; needs the team's OK and the clone)")
    ap.add_argument("--no-lgbm", action="store_true")
    ap.add_argument("--with-v1-asym", action="store_true", help="also run v1 without team symmetry")
    ap.add_argument("--lgbm-grid-team", type=float, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--lgbm-grid-dead", type=float, nargs="+", default=[10, 20, 40, 80, 160])
    args = ap.parse_args()

    clean = {m: dfl.load(m) for m in TRAIN + TEST}
    board = {}
    for seed in args.seeds:
        t0 = time.time()
        deg = {m: degrade(clean[m], seed=1000 * seed + i) for i, m in enumerate(clean)}
        hidden = np.mean([deg[m]["ball_hidden_share"] for m in TEST])
        lab = {m: labels(clean[m]) for m in clean}
        nb = tune_nb([deg[m] for m in TRAIN], [lab[m] for m in TRAIN], args.grid_team, args.grid_dead)
        runners = {"baseline_smooth50": methods.baseline_smooth50, "nb_viterbi (ours v0)": nb.predict}
        if not args.no_lgbm:
            lg = tune_lgbm([deg[m] for m in TRAIN], [lab[m] for m in TRAIN],
                           args.lgbm_grid_team, args.lgbm_grid_dead)
            runners["lgbm_viterbi (ours v1)"] = lg.predict
            if args.with_v1_asym:
                lg0 = tune_lgbm([deg[m] for m in TRAIN], [lab[m] for m in TRAIN],
                                args.lgbm_grid_team, args.lgbm_grid_dead, symmetric=False)
                runners["lgbm_viterbi asym (v1.0)"] = lg0.predict
        if args.espadas and os.path.isdir(ESPADAS):
            for v in args.espadas:
                runners["espadas_" + v] = espadas_runner(v)
        for name, fn in runners.items():
            for m in TEST:
                pred = fn(deg[m])
                alive_t, alive_p = lab[m] != methods.D, pred != methods.D
                share_err = abs(np.mean(pred[alive_p] == methods.H) - np.mean(lab[m][alive_t] == methods.H))
                board.setdefault(name, []).append(score(lab[m], pred, deg[m]["period"]) + (share_err, m, seed))
        print("seed {}: ball hidden {:.1%} on test matches, {:.0f}s".format(seed, hidden, time.time() - t0))

    print("\nTEST matches {} x seeds {}  (official possession_metrics.evaluate)".format(TEST, args.seeds))
    print("{:<24} {:>8} {:>8} {:>8} {:>10} {:>8} {:>11}".format(
        "method", "spellF1", "+/-sd", "macroF1", "poss pred", "poss gt", "home-share err"))
    for name, rows in board.items():
        s = np.array([r[0] for r in rows]); mf = np.array([r[1] for r in rows])
        print("{:<24} {:>8.3f} {:>8.3f} {:>8.3f} {:>10.0f} {:>8.0f} {:>11.3f}".format(
            name, s.mean(), s.std(), mf.mean(), np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]),
            np.mean([r[4] for r in rows])))


if __name__ == "__main__":
    main()
