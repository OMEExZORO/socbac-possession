"""Label variant L1 for the external bench: a change of possession counts only if the new team keeps
the ball for at least 1 s.

The provider flag of the open DFL/IDSSE data changes owner on touches rather than on control (83% of
opponent touches flip it within 1 s), so its possessions are much more fragmented than the written
rule of the challenge ("a change of possession occurs when a player from the opposing team gains
controlled possession"; a simple touch or a deflection does not count). L1 is the simplest proxy of
"controlled" from that rule: the new team has to keep the ball for MIN_RUN frames.

Rule, applied inside every period and every alive stretch (maximal run of alive frames) separately:
  * dead-ball frames stay D, exactly as in the provider labels (bench.run.labels);
  * the first team of an alive stretch (the restart team) is kept whatever its length;
  * walking forward, a run of the other team's provider-flag frames lasting >= MIN_RUN frames is a
    change of possession from its first frame; a shorter run is merged into the current possession;
  * an unknown owner on an alive frame carries the current team (the first known team at the start
    of a stretch; H if a stretch has no known owner, as bench.run.labels does). This never happens
    on the 7 IDSSE matches (0 alive frames with an unknown owner).

Nothing here reads tracking positions; it only relabels the provider's two flags.

    python -m bench.labels_l1          # self-test + per-match label statistics (open data only)
"""

import numpy as np

from bench import methods

H, A, D = methods.H, methods.A, methods.D
MIN_RUN = 25  # frames at 25 fps = 1 s


def _runs(values):
    """(starts, stops, values) of maximal constant runs of a 1-D array."""
    n = len(values)
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int), values[:0]
    br = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], br))
    stops = np.concatenate((br, [n]))
    return starts, stops, values[starts]


def _stretch_l1(owner, min_run):
    """L1 team codes (H/A) for one alive stretch given the provider owner flags (0, 1, -1)."""
    own = np.asarray(owner, np.int8).copy()
    known = np.flatnonzero(own >= 0)
    if len(known) == 0:
        return np.full(len(own), H, np.int8)
    # unknown -> carry the last known team; leading unknown -> first known team
    carry = np.maximum.accumulate(np.where(own >= 0, np.arange(len(own)), -1))
    carry[carry < 0] = known[0]
    own = own[carry]
    team = np.where(own == 1, A, H).astype(np.int8)
    starts, stops, vals = _runs(team)
    out = np.empty(len(team), np.int8)
    current = vals[0]
    for a, b, v in zip(starts, stops, vals):
        if v != current and (b - a) >= min_run:
            current = v
        out[a:b] = current
    return out


def labels_l1(match, min_run=MIN_RUN):
    """int8 codes (0 H, 1 A, 2 D) per frame of a bench.dfl.load() dict, L1 variant."""
    alive = np.asarray(match["alive"], bool)
    owner = np.asarray(match["owner"], np.int8)
    period = np.asarray(match["period"])
    n = len(alive)
    out = np.full(n, D, np.int8)
    # alive stretches never cross a period boundary
    key = np.where(alive, period.astype(np.int32), -1)
    starts, stops, vals = _runs(key)
    for a, b, v in zip(starts, stops, vals):
        if v < 0:
            continue
        out[a:b] = _stretch_l1(owner[a:b], min_run)
    return out


def labels_provider(match):
    """bench.run.labels, re-exported so callers can import both variants from one place."""
    from bench.run import labels
    return labels(match)


def label_stats(codes, period, fps=25):
    """Match-level aggregates of a label sequence (possessions = maximal H or A runs per period)."""
    lengths = []
    for p in np.unique(period):
        c = codes[period == p]
        s, e, v = _runs(c)
        keep = v != D
        lengths.extend((e - s)[keep].tolist())
    lengths = np.asarray(lengths, float) / fps
    alive = codes != D
    return {
        "n_possessions": int(len(lengths)),
        "mean_s": float(lengths.mean()) if len(lengths) else 0.0,
        "median_s": float(np.median(lengths)) if len(lengths) else 0.0,
        "share_under_4s": float((lengths < 4).mean()) if len(lengths) else 0.0,
        "alive_share": float(alive.mean()),
        "home_share": float((codes[alive] == H).mean()) if alive.any() else 0.0,
    }


def self_test():
    """Synthetic cases for every rule above. Prints pass/fail only."""
    def mk(owner, alive, period=None):
        owner = np.asarray(owner, np.int8)
        alive = np.asarray(alive, bool)
        period = np.ones(len(owner), np.int8) if period is None else np.asarray(period, np.int8)
        return {"owner": owner, "alive": alive, "period": period}

    r = 3  # min_run for the synthetic cases
    # short opposing run inside a possession is merged
    m = mk([0, 0, 0, 1, 1, 0, 0, 0], [1] * 8)
    assert labels_l1(m, r).tolist() == [H] * 8
    # long opposing run is a change from its first frame; the short run back to H is merged into A
    m = mk([0, 0, 1, 1, 1, 1, 0, 0], [1] * 8)
    assert labels_l1(m, r).tolist() == [H, H, A, A, A, A, A, A]
    # short run before a stoppage is merged into the previous team; D unchanged
    m = mk([0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0])
    assert labels_l1(m, r).tolist() == [H, H, H, H, H, D, D]
    # restart team kept even if short, then a long run of the other team
    m = mk([1, 0, 0, 0, 0], [1] * 5)
    assert labels_l1(m, r).tolist() == [A, H, H, H, H]
    # stretches are independent across D and across periods
    m = mk([0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1])
    assert labels_l1(m, r).tolist() == [H, H, H, D, A, A, A]
    m = mk([0, 0, 0, 1, 1], [1] * 5, [1, 1, 1, 2, 2])
    assert labels_l1(m, r).tolist() == [H, H, H, A, A]
    # unknown owner carries the current team; leading unknown takes the first known team
    m = mk([-1, 1, 1, -1, 1, 0], [1] * 6)
    assert labels_l1(m, r).tolist() == [A, A, A, A, A, A]
    # min_run 1 reproduces the provider labels when no owner is unknown
    rng = np.random.default_rng(0)
    own = rng.integers(0, 2, 500).astype(np.int8)
    alv = rng.random(500) > 0.2
    m = mk(own, alv)
    prov = np.where(~alv, D, np.where(own == 1, A, H)).astype(np.int8)
    assert np.array_equal(labels_l1(m, 1), prov)
    print("labels_l1 self-test ok")


def main():
    from bench import dfl
    from bench.run import labels
    self_test()
    print("{:<8} {:>14} {:>14} {:>16} {:>16}".format("match", "poss prov", "poss L1", "mean s prov/L1",
                                                    "<4s share prov/L1"))
    for m in dfl.MATCHES:
        clean = dfl.load(m)
        a = label_stats(labels(clean), clean["period"])
        b = label_stats(labels_l1(clean), clean["period"])
        print("{:<8} {:>14d} {:>14d} {:>8.1f}/{:<7.1f} {:>8.2f}/{:<7.2f}".format(
            m, a["n_possessions"], b["n_possessions"], a["mean_s"], b["mean_s"],
            a["share_under_4s"], b["share_under_4s"]))


if __name__ == "__main__":
    main()
