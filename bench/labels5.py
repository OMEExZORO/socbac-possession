"""Four label definitions as functions of the DFL event labels / clean tracking (open IDSSE only).

  L0  provider flag (bench.run.labels): owner flips on touches.
  L1  merge opposing runs shorter than 1 s into the current possession (bench.labels_l1, 25 frames).
  L2  same, 2 s (50 frames).
  L3  mechanical owner (bench.mechlabel rule: r_p 1.0 m, T_c 1.0 s, touch confirmation, duel and
      loose-ball stickiness, D = provider alive flag) with every steal moved back from the frame where
      T_c is reached to the first contact frame of the confirming attempt.

L1 and L2 only merge: D frames are untouched, every H<->A transition they keep is a transition of
L0, and the number of H/A runs never grows.

    python -m bench.labels5      # unit tests (synthetic, a few thousand frames)
"""
import os

import numpy as np

from bench import labels_l1
from bench import mechlabel as ml

H, A, D = 0, 1, 2
NAMES = ("L0", "L1", "L2", "L3")
MIN_RUN = {"L1": 25, "L2": 50}


def labels_l0(match):
    return np.where(~np.asarray(match["alive"], bool), D,
                    np.where(np.asarray(match["owner"]) == 1, A, H)).astype(np.int8)


def labels_merge(match, frames):
    return labels_l1.labels_l1(match, min_run=frames)


# ---- L3: mechanical with flips moved back to first contact ----------------------------------

def _possess_backdated(control, alive, t_c):
    """mechlabel._possess, but when the owner flips at frame t the whole confirming attempt
    [first confirmed contact, t) is written as the new owner (also for steals from a known owner)."""
    n = len(control)
    need = ml.tc_frames(t_c)
    gap_max = int(ml.GAP_S * ml.FPS)
    owners = np.full(n, -1, np.int16)
    owner = att = -1
    contact = gap = start = 0
    for t in range(n):
        if not alive[t]:
            owners[t] = -2
            owner = att = -1
            contact = gap = 0
            continue
        ct = int(control[t])
        if ct >= 0:
            if att != ct:
                att, contact, gap, start = ct, 1, 0, t
            else:
                contact += 1
                gap = 0
            if owner != att and contact >= need:
                owner = att
                if start < t:
                    seg = owners[start:t]
                    seg[seg > -2] = owner          # alive frames of the attempt
                owners[t] = owner
            else:
                owners[t] = owner if owner >= 0 else -1
        else:
            if att >= 0:
                gap += 1
                if gap > gap_max:
                    att, contact, gap = -1, 0, 0
            owners[t] = owner if owner >= 0 else -1
    return owners


def labels_l3(match, r_p=ml.DEFAULT_R_P, t_c=ml.DEFAULT_T_C):
    parts = ml.prepare(match, impute=False)
    n = len(match["ball"])
    out = np.full(n, D, np.int8)
    for part in parts:
        zone = ml._zone(part, r_p)
        control = ml._confirm(part, zone, None)
        owners = _possess_backdated(control, part["alive"], t_c)
        ml._close_owners(owners, zone)
        out[part["idx"]] = ml._to_codes(owners)
    return out


def all_labels(match):
    return {"L0": labels_l0(match), "L1": labels_merge(match, MIN_RUN["L1"]),
            "L2": labels_merge(match, MIN_RUN["L2"]), "L3": labels_l3(match)}


def cached_labels(match_id, cache_dir):
    """All four label arrays for an open IDSSE match, cached as npy under cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {k: os.path.join(cache_dir, "%s.%s.npy" % (match_id, k)) for k in NAMES}
    if all(os.path.isfile(p) for p in paths.values()):
        return {k: np.load(p) for k, p in paths.items()}
    from bench import dfl
    labs = all_labels(dfl.load(match_id))
    for k, p in paths.items():
        np.save(p, labs[k])
    return labs


# ---- statistics / tests ---------------------------------------------------------------------

def run_count(codes, period):
    n = 0
    for p in np.unique(period):
        c = codes[period == p]
        if len(c) == 0:
            continue
        new = np.ones(len(c), bool)
        new[1:] = c[1:] != c[:-1]
        n += int((new & (c != D)).sum())
    return n


def transitions(codes, period=None):
    """Frame indices t where codes[t-1], codes[t] are both H/A and differ (same period)."""
    hit = (codes[1:] != codes[:-1]) & (codes[1:] != D) & (codes[:-1] != D)
    if period is not None:
        hit &= period[1:] == period[:-1]
    return set((np.flatnonzero(hit) + 1).tolist())


def _synth_provider(n, seed):
    rng = np.random.default_rng(seed)
    owner = np.zeros(n, np.int8)
    t, cur = 0, 0
    while t < n:
        L = int(rng.integers(3, 90))
        owner[t:t + L] = cur
        cur = 1 - cur if rng.random() < 0.8 else cur
        t += L
    alive = np.ones(n, bool)
    t = 0
    while t < n:
        t += int(rng.integers(200, 700))
        L = int(rng.integers(20, 120))
        alive[t:t + L] = False
        t += L
    period = np.ones(n, np.int8)
    period[n // 2:] = 2
    return {"owner": owner, "alive": alive, "period": period}


def _expect(name, cond, detail=""):
    if not cond:
        raise AssertionError("{} ({})".format(name, detail))
    print("PASS", name)


def _selftest():
    n = 4000
    for seed in range(5):
        m = _synth_provider(n, seed)
        l0, l1, l2 = labels_l0(m), labels_merge(m, 25), labels_merge(m, 50)
        for nm, lab in (("L0", l0), ("L1", l1), ("L2", l2)):
            assert lab.shape == (n,) and lab.dtype == np.int8, nm
        assert np.array_equal(l0 == D, l1 == D) and np.array_equal(l0 == D, l2 == D), "D changed"
        pr = m["period"]
        assert transitions(l1, pr) <= transitions(l0, pr) and transitions(l2, pr) <= transitions(l0, pr), "new transition"
        c0, c1, c2 = (run_count(x, m["period"]) for x in (l0, l1, l2))
        assert c1 <= c0 and c2 <= c0, "runs grew"
        assert (l1 != l0).any(), "L1 merged nothing on this seed"
    _expect("L1/L2 keep D, shape, dtype and only merge (5 random streams)", True)

    m = _synth_provider(n, 0)
    for fr in (25, 50):
        lab = labels_merge(m, fr)
        bad = 0
        i = 0
        while i < n:
            if lab[i] == D:
                i += 1
                continue
            j = i
            while j < n and lab[j] == lab[i] and m["period"][j] == m["period"][i]:
                j += 1
            first = i == 0 or lab[i - 1] == D or m["period"][i - 1] != m["period"][i]
            if (j - i) < fr and not first and j < n and lab[j] != D:
                bad += 1
            i = j
        _expect("no interior run under %d frames" % fr, bad == 0, "bad %d" % bad)
    _expect("L2 has no more runs than L1 on this stream",
            run_count(labels_merge(m, 50), m["period"]) <= run_count(labels_merge(m, 25), m["period"]))

    # L3 on mechlabel's synthetic geometry: owner changes at first contact, not at confirmation
    control = np.array([H] * 40 + [A] * 40, np.int8)
    alive = np.ones(80, bool)
    own = _possess_backdated(control, alive, 1.0)
    ml._close_owners(own, control)
    c = ml._to_codes(own)
    _expect("steal is back-dated to first contact", np.all(c[:40] == H) and np.all(c[40:] == A))
    base = ml._possess(control, alive, 1.0)
    ml._close_owners(base, control)
    _expect("mechlabel original flips later", int(np.argmax(ml._to_codes(base) == A)) == 64)
    control = np.array([H] * 40 + [A] * 8 + [-1] * 20, np.int8)
    own = _possess_backdated(control, np.ones(len(control), bool), 1.0)
    ml._close_owners(own, control)
    _expect("short contact still does not flip", np.all(ml._to_codes(own) == H))
    control = np.array([H] * 30 + [-1] * 10 + [A] * 40, np.int8)
    alive = np.array([True] * 30 + [False] * 10 + [True] * 40)
    own = _possess_backdated(control, alive, 1.0)
    ml._close_owners(own, control)
    c = ml._to_codes(own)
    _expect("dead ball stays D, restart team kept",
            np.all(c[:30] == H) and np.all(c[30:40] == D) and np.all(c[40:] == A))

    mm = ml._blank(120)
    mm["ball"][:, 0] = 40.0
    mm["ball"][:, 1] = 34.0
    mm["players"][:60, 0, 0] = (40.2, 34.0)
    mm["players"][:, 1, 0] = (90.0, 10.0)
    mm["players"][60:, 0, 0] = (80.0, 10.0)
    mm["players"][60:, 1, 0] = (40.2, 34.0)
    mm["alive"][100:105] = False
    a, b = labels_l3(mm), ml.label_match(mm)
    _expect("L3 shape/dtype and D identical to mechlabel",
            a.shape == (120,) and a.dtype == np.int8 and np.array_equal(a == D, b == D))
    fa, fb = transitions(a), transitions(b)
    _expect("L3 flips are not later than mechlabel's",
            all(any(0 <= f - g <= 40 for g in fa) for f in fb) if fb else True)
    print("labels5 selftest ok")


if __name__ == "__main__":
    _selftest()
