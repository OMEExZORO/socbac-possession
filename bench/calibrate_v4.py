"""Calibrate degrade_v4 ball hiding to label-free real-tracking aggregates and print the probe.
Targets: visible 59.3%, gap median 268, mean 431, p90 845 frames."""
import itertools
import sys

import numpy as np

from bench import dfl, degrade_v4 as d4
from bench.run import TRAIN, TEST

TARGET = dict(vis=0.593, med=268, mean=431, p90=845)


def stats(ms, **kw):
    hid, gaps = [], []
    for i, m in enumerate(ms):
        rng = np.random.default_rng(i)
        h = d4.ball_hidden_mask(rng, m["ball"], m["players"], **kw)
        hid.append(h.mean())
        gaps.append(d4.gap_lengths(np.where(h, np.nan, 0.0), m["period"]))
    g = np.concatenate(gaps)
    return dict(vis=1 - float(np.mean(hid)), med=float(np.median(g)), mean=float(g.mean()),
                p90=float(np.percentile(g, 90)), n=len(g))


def err(s):
    return sum(((s[k] - TARGET[k]) / TARGET[k]) ** 2 for k in TARGET)


if __name__ == "__main__":
    ms = [dfl.load(m) for m in TRAIN + TEST]
    best = None
    for rate, dur, sig in itertools.product([4e-4, 6e-4, 8e-4, 1.0e-3, 1.4e-3], [140, 180, 220, 260],
                                            [0.6, 0.8, 1.0]):
        s = stats(ms, base_rate=rate, dur_median=dur, sigma=sig)
        e = err(s)
        if best is None or e < best[0]:
            best = (e, rate, dur, sig, s)
            print("best", ["%.5g" % v for v in best[:4]], {k: round(v, 3) for k, v in s.items()}, flush=True)
    print("FINAL", best[1:4])
