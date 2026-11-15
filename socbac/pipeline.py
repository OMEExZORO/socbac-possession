"""One command from the data to the submission.

    python -m bench.dfl fetch && python -m bench.dfl convert     # open training data, once
    python -m socbac.pipeline --out evaluation/results/<team_name>.csv

Method (ours v1, see docs/team/BENCH.md):
  1. Train on the 7 open DFL matches, each degraded twice to look like the ATD (fixed seeds).
     No label of the competition match is used anywhere.
  2. Pick the two decoder costs by leave-one-match-out spellF1 over those DFL matches.
  3. Per-frame LightGBM evidence on the BAR-ATH ATD -> 3-state Viterbi per half.
  4. X outside the graded halves; validate with the organisers' checks.

Prints only aggregates: label counts and possession counts of our own output.
"""

import argparse
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from bench import dfl, methods  # noqa: E402
from bench.degrade import degrade  # noqa: E402
from bench.run import labels as dfl_labels, score  # noqa: E402
from socbac import atd  # noqa: E402

SEEDS = (0, 1)
GRID_TEAM = (5, 10, 20, 40)
GRID_DEAD = (20, 40, 80, 160)


def train():
    deg, lab = [], []
    for i, m in enumerate(dfl.MATCHES):
        clean = dfl.load(m)
        y = dfl_labels(clean)
        for s in SEEDS:
            deg.append(degrade(clean, seed=10000 + 100 * s + i))
            lab.append(y)

    # leave-one-match-out emissions (both degraded copies of a match held out together)
    groups = np.repeat(np.arange(len(dfl.MATCHES)), len(SEEDS))
    oof = [None] * len(deg)
    for g in range(len(dfl.MATCHES)):
        rest = np.flatnonzero(groups != g)
        model = methods.LGBMViterbi().fit([deg[i] for i in rest], [lab[i] for i in rest])
        for i in np.flatnonzero(groups == g):
            oof[i] = model.emissions(deg[i])
    best = None
    for st in GRID_TEAM:
        for sd in GRID_DEAD:
            s = np.mean([score(y, methods.decode_periods(e, m["period"], st, sd), m["period"])[0]
                         for m, e, y in zip(deg, oof, lab)])
            if best is None or s > best[0]:
                best = (s, st, sd)
    print("decoder costs: switch_team={} switch_dead={} (leave-one-match-out spellF1 {:.3f})".format(
        best[1], best[2], best[0]))
    return methods.LGBMViterbi(best[1], best[2]).fit(deg, lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = train()
    match = atd.load()
    codes = model.predict(match)
    letters = np.array(["H", "A", "D"])[codes]
    letters[match["period"] == 0] = "X"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))

    graded = match["period"] > 0
    counts = {k: int((letters[graded] == k).sum()) for k in "HAD"}
    print("wrote {}: graded label counts {}".format(args.out, counts))
    with open(args.out, "rb") as fh:
        print("sha256", hashlib.sha256(fh.read()).hexdigest())

    import subprocess  # organisers' checks, exactly as they run them
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                    args.out], check=True)


if __name__ == "__main__":
    main()
