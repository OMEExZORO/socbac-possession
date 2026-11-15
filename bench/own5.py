"""own5: ball-control ownership block, an optional extra column group for the feats3 pipeline.

    python -m bench.own5                                    # synthetic unit tests only (seconds)
    python -m bench.own5 build --deg v4 v5ball --seeds 0 1 2    # sidecar cache (full matches, later)
    python -m bench.own5 hedge --deg v4 v5ball --seeds 0 1 2    # v3 vs v3own5, L12 labels, boundary x20
    python -m bench.own5 verdict                            # per-seed gains and the adoption rule

Why this block (research/13 item 1.4 and 4.1): v1/v3 know how FAR each team is from the ball, not
whether anybody CONTROLS it, and they carry nothing through a ball gap except distances at the gap
edges. The written rule is controlled possession, so the block states control explicitly:

 1. control runs      ball within R_IN of a team's nearest player AND velocity-matched to him
                      (a ball flying past a player is not control), with hysteresis (R_OUT, looser
                      velocity) and a hold through ball gaps of <= 1 s. Team level, never slot level:
                      slot ids are tracklets, so everything is a min/sum/count over slots.
 2. brackets          last and next CONFIRMED control (exclusive run >= 0.5 s) of each team. Same team
                      on both sides = retained through a pass or a ball gap; different = a turnover
                      candidate, with its span, the frame's position in it, and a 3 s "fast" flag.
                      This is the W2 turnover checker as columns, and it is what survives missing ball.
 3. impulses          ball velocity change over +-0.2 s given to the nearest unchallenged player;
                      split into release (speed up) and reception (slow down). Displacement over 5
                      frames still bends at a kick after the ~17 frame ball smoothing; v1's per-frame
                      acceleration does not.
 4. cross-team        pressure on each team's ball-nearest player (nearest opponent, opponents within
                      4 m, closing intensity, nearest team-mate), duel flag, and the interception
                      time of the ball's current path per team (in-flight ball: who gets there first).
 5. ball-free         goal-side marking share (defenders stand between their own goal and their
                      mark), its 2 s past/future contrast (flips at a turnover), and the contest
                      centre (players weighted by opponents within 5 m) relative to each team's
                      attack direction. No ball imputation anywhere: nothing is interpolated.

Rolling shares are divided by the number of VISIBLE frames in the window, so a window that is 50%
hidden is not read as 50% less control. Offline windows look back and ahead like v1/v3. Nothing reads
alive/owner. Deterministic (no random numbers), float64 inside, float32 out.

Symmetry: every column is kind h, a, diff or sym, built from the player array, so exchanging the
teams exchanges _h/_a, negates _diff and leaves sym alone; mirroring the pitch changes nothing (only
distances and attack-frame coordinates are used). The swapped team-view is therefore produced from
the stored view by a column permutation (swap_view), which halves the cache.

v3 columns are untouched: the block is stored as a sidecar next to the run_v3cand cache and joined
by the 'v3own5' config (Own5Spec). feats3.py, run_v3cand.py and run_hedge.py are not modified.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from bench import feats3, methods

PITCH_L, PITCH_W = methods.PITCH
CX, CY = PITCH_L / 2.0, PITCH_W / 2.0
FPS = 25.0
REV = "o5a"                    # bump when a formula changes: the sidecar cache is keyed on it
SUB = "own5"

RUN_DIR = r"D:\Barca_Hackathon\research\runs\own5"
CACHE = os.path.join(RUN_DIR, "cache")
RES = os.path.join(RUN_DIR, "hedge_results.jsonl")
LOG = os.path.join(RUN_DIR, "hedge.log")
MIN_FREE_GB = 8.0

R_IN, R_OUT = 1.5, 2.5         # m: enter / keep control (13_fable_design 1.4: 1.5 m)
V_IN, V_OUT = 2.5, 5.0         # m/s: ball-player relative speed to enter / keep
HOLD_GAP = 25                  # frames: control is held through a ball gap of at most 1 s
EXCL = 0.75                    # m: in a duel the nearer team must lead by this to be exclusive
CONFIRM = 13                   # frames (0.5 s) of exclusive control that make a "confirmed" run
CAP = 500                      # frames, cap of since/until (v1's cap)
FAST_TURN = 75                 # frames (3 s): a fast turnover bracket
BALL_W = 3                     # frames each side for the ball velocity
IMP_W = 5                      # frames each side (0.2 s) for the impulse
IMP_GATE, IMP_LEAD = 2.5, 1.0  # m: toucher within, and nearer than the opponent by
IMP_MIN = 3.0                  # m/s change that counts as a release or a reception
SHARE_W = (12, 50, 150)        # +-0.5, 2, 6 s
T_MAX, T_STEPS = 3.0, 16       # interception horizon (s) and samples
RUN_SPEED, REACH, REACT = 6.0, 1.0, 0.4
BALL_DRAG = 0.3                # 1/s
MARK_R, DUEL_R, PC_R = 10.0, 5.0, 15.0
CHUNK = 4096


# ---- small period-aware helpers --------------------------------------------------------------

def _shift(a, k, period):
    """a[t + k] if that frame is in the same period, else NaN. float64."""
    a = np.asarray(a, np.float64)
    out = np.full(a.shape, np.nan, np.float64)
    n = len(a)
    if k == 0:
        return a.copy()
    j = abs(int(k))
    if j >= n:
        return out
    same = (period[j:] == period[:-j]).reshape((-1,) + (1,) * (a.ndim - 1))
    if k > 0:
        out[:-j] = np.where(same, a[j:], np.nan)
    else:
        out[j:] = np.where(same, a[:-j], np.nan)
    return out


def _wsum(x, back, fwd, period):
    """Sum of x over frames t-back..t+fwd, clipped to the period."""
    x = np.asarray(x, np.float64)
    out = np.empty(len(x), np.float64)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        c = np.concatenate(([0.0], np.cumsum(x[idx])))
        i = np.arange(len(idx))
        lo, hi = np.maximum(0, i - back), np.minimum(len(idx), i + fwd + 1)
        out[idx] = c[hi] - c[lo]
    return out


def _latch(enter, stay, boundary):
    """state[t] = enter[t] or (state[t-1] and stay[t]); nothing carries over a period start."""
    t = np.arange(len(enter))
    stay = stay | enter
    brk = (~stay) | (boundary & ~enter)
    last_brk = np.maximum.accumulate(np.where(brk, t, -1))
    last_ent = np.maximum.accumulate(np.where(enter, t, -1))
    return last_ent > last_brk


def _runs(mask, boundary):
    """(elapsed, remaining, total) frames of the run of True a frame sits in; 0 off a run."""
    n = len(mask)
    t = np.arange(n)
    start = np.maximum.accumulate(np.where(~mask, t + 1, np.where(boundary, t, 0)))
    last_of_period = np.ones(n, bool)
    last_of_period[:-1] = boundary[1:]
    end = np.minimum.accumulate(np.where(~mask, t - 1, np.where(last_of_period, t, n - 1))[::-1])[::-1]
    el = np.where(mask, t - start + 1, 0)
    rem = np.where(mask, end - t, 0)
    return el, rem, np.where(mask, end - start + 1, 0)


def _norm(v):
    return np.sqrt(np.sum(v * v, axis=-1))


# ---- pieces ----------------------------------------------------------------------------------

def _kinematics(match):
    ball = np.asarray(match["ball"], np.float64)[:, :2]
    xy = np.asarray(match["players"], np.float64)
    period = np.asarray(match["period"])
    n = len(period)
    boundary = np.zeros(n, bool)
    boundary[0] = True
    boundary[1:] = period[1:] != period[:-1]
    vis = np.isfinite(ball[:, 0])
    fin = np.isfinite(xy[..., 0])
    # players: central difference over +-2 frames; unknown or absurd -> 0 (standing)
    pv = (_shift(xy, 2, period) - _shift(xy, -2, period)) * (FPS / 4.0)
    bad = ~np.isfinite(pv[..., 0]) | (_norm(pv) > 12.0)
    pv[bad] = 0.0
    # ball: central over +-BALL_W, else one-sided, else unknown
    fw, bw = _shift(ball, BALL_W, period), _shift(ball, -BALL_W, period)
    cands = ((fw - bw) * (FPS / (2.0 * BALL_W)), (fw - ball) * (FPS / BALL_W), (ball - bw) * (FPS / BALL_W))
    bv = np.full((n, 2), np.nan)
    for c in cands:
        use = ~np.isfinite(bv[:, 0]) & np.isfinite(c[:, 0])
        bv[use] = c[use]
    bv_ok = np.isfinite(bv[:, 0]) & (_norm(np.nan_to_num(bv)) <= 45.0)
    bv[~bv_ok] = 0.0
    return {"ball": ball, "xy": xy, "period": period, "boundary": boundary, "vis": vis, "fin": fin,
            "pv": pv, "bv": bv, "bv_ok": bv_ok, "n": n}


def _nearest(k):
    """Per team: distance of the ball-nearest player, his position and velocity."""
    xy, ball, n = k["xy"], k["ball"], k["n"]
    d = _norm(xy - ball[:, None, None, :])
    d = np.where(k["fin"] & k["vis"][:, None, None], d, np.inf)
    arg = np.argmin(d, axis=2)
    dmin = np.min(d, axis=2)
    ix = np.broadcast_to(arg[:, :, None, None], (n, 2, 1, 2))
    pos = np.take_along_axis(xy, ix, axis=2)[:, :, 0]
    vel = np.take_along_axis(k["pv"], ix, axis=2)[:, :, 0]
    has = np.isfinite(dmin)
    rel = _norm(k["bv"][:, None, :] - vel)
    return dmin, has, pos, vel, rel


def _control(k, dmin, has, rel, gap_total):
    """Team control with hysteresis, and the exclusive version (one team clearly on the ball)."""
    free = ~k["bv_ok"][:, None]            # unknown ball velocity: distance alone decides
    enter = has & (dmin < R_IN) & (free | (rel < V_IN))
    hold = (~k["vis"]) & (gap_total <= HOLD_GAP)
    stay = (has & (dmin < R_OUT) & (free | (rel < V_OUT))) | hold[:, None]
    ctl = np.stack([_latch(enter[:, s], stay[:, s], k["boundary"]) for s in (0, 1)], axis=1)
    other = ctl[:, ::-1]
    lead = (dmin + EXCL) < dmin[:, ::-1]
    return ctl, ctl & (~other | lead)


def _intercept(k):
    """Earliest time (s) at which each team can meet the ball on its current path. T_MAX+ if never.

    Ball: current velocity with a mild drag. Player: keeps his velocity for REACT s, then covers
    RUN_SPEED m/s, and reaches REACH m. A crude potential-pitch-control model; the GBM only needs
    which team is earlier and by how much.
    """
    n = k["n"]
    none = T_MAX + 0.2
    best = np.full((n, 2), none)
    xy, pv, ball, bv = k["xy"], k["pv"], k["ball"], k["bv"]
    for tau in np.linspace(0.0, T_MAX, T_STEPS):
        travel = (1.0 - np.exp(-BALL_DRAG * tau)) / BALL_DRAG
        target = ball + bv * travel
        origin = xy + pv * min(tau, REACT)
        d = _norm(origin - target[:, None, None, :])
        can = k["fin"] & np.isfinite(d) & (d <= REACH + RUN_SPEED * tau)
        hit = can.any(axis=2) & (best >= none)
        best[hit] = tau
    best[~k["vis"]] = none
    return best


def _pressure(k, pos, has):
    """Around each team's ball-nearest player: nearest opponent, opponents within 4 m, closing
    intensity of opponents within 8 m, nearest team-mate."""
    n = k["n"]
    gap = np.full((n, 2), 30.0)
    n4 = np.zeros((n, 2))
    press = np.zeros((n, 2))
    mate = np.full((n, 2), 30.0)
    for s in (0, 1):
        o = 1 - s
        delta = pos[:, s, None, :] - k["xy"][:, o]
        d = _norm(delta)
        ok = k["fin"][:, o] & np.isfinite(d) & has[:, s, None]
        dd = np.where(ok, d, np.inf)
        g = dd.min(axis=1)
        gap[:, s] = np.where(np.isfinite(g), np.minimum(g, 30.0), 30.0)
        n4[:, s] = (dd <= 4.0).sum(axis=1)
        closing = np.sum(k["pv"][:, o] * delta, axis=-1) / np.maximum(d, 1e-3)
        inten = np.clip(closing, 0.0, 8.0) / np.maximum(d, 1.0)
        press[:, s] = np.where(ok & (dd <= 8.0), inten, 0.0).sum(axis=1)
        dm = _norm(pos[:, s, None, :] - k["xy"][:, s])
        dm = np.where(k["fin"][:, s] & np.isfinite(dm) & (dm > 1e-6) & has[:, s, None], dm, np.inf)
        m = dm.min(axis=1)
        mate[:, s] = np.where(np.isfinite(m), np.minimum(m, 30.0), 30.0)
    return gap, n4, press, mate


def _marking(k, sign):
    """Ball-free cross-team geometry, in frame chunks so no (n, S, S) array is ever held.

    goal-side share: of a team's players with an opponent within MARK_R, the share standing nearer
    to their OWN goal than that opponent. The defending side marks goal-side, so this is high for
    the team out of possession. contest centre: mean position of all players weighted by their
    number of opponents within DUEL_R; where the two teams are locked together is where the ball is.
    """
    xy, fin, n = k["xy"], k["fin"], k["n"]
    s = xy.shape[2]
    goal_x = np.where(sign > 0, 0.0, PITCH_L)
    dg = np.hypot(xy[..., 0] - goal_x[:, :, None], xy[..., 1] - CY)
    marked = np.zeros((n, 2, s), bool)
    goalside = np.zeros((n, 2, s), bool)
    weight = np.zeros((n, 2, s))
    tight = np.full((n, 2), 30.0)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        dd = _norm(xy[a:b, 0][:, :, None, :] - xy[a:b, 1][:, None, :, :])
        dd = np.where(fin[a:b, 0][:, :, None] & fin[a:b, 1][:, None, :], dd, np.inf)
        for side, axis in ((0, 2), (1, 1)):
            dn = dd.min(axis=axis)
            q = dd.argmin(axis=axis)
            dg_opp = np.take_along_axis(dg[a:b, 1 - side], q, axis=1)
            mk = np.isfinite(dn) & (dn <= MARK_R)
            marked[a:b, side] = mk
            goalside[a:b, side] = mk & (dg[a:b, side] < dg_opp - 0.5)
            weight[a:b, side] = (dd <= DUEL_R).sum(axis=axis)
            cnt = np.isfinite(dn).sum(axis=1)
            tot = np.where(np.isfinite(dn), np.minimum(dn, 30.0), 0.0).sum(axis=1)
            tight[a:b, side] = np.where(cnt > 0, tot / np.maximum(cnt, 1), 30.0)
    xyz = np.where(fin[..., None], xy, 0.0)
    wsum = weight.sum(axis=(1, 2))
    pc = (xyz * weight[..., None]).sum(axis=(1, 2)) / np.maximum(wsum, 1.0)[:, None]
    cnt = fin.sum(axis=2)
    cent = xyz.sum(axis=2) / np.maximum(cnt, 1)[..., None]
    cent = np.where(cnt[..., None] > 0, cent, np.array([CX, CY]))
    pc = np.where(wsum[:, None] > 0, pc, 0.5 * (cent[:, 0] + cent[:, 1]))
    dpc = np.where(fin, _norm(xy - pc[:, None, None, :]), np.inf)

    def share(mask_num, mask_den):
        den = mask_den.sum(axis=2)
        return np.where(den > 0, mask_num.sum(axis=2) / np.maximum(den, 1), 0.5)

    local = dpc <= PC_R
    return {"gs": share(goalside, marked), "gsc": share(goalside & local, marked & local),
            "tight": tight, "duels": wsum, "pc": pc, "cent": cent,
            "pc_near": np.minimum(dpc.min(axis=2), 30.0), "pc_n6": (dpc <= 6.0).sum(axis=2)}


# ---- the block -------------------------------------------------------------------------------

def build_own5(match):
    """(X float32, names, from_ball, subgroup, kind) of the own5 block for one team-view.

    Reads match['ball'], ['players'], ['period'] only.
    """
    feats3._warn_off()
    k = _kinematics(match)
    period, boundary, vis, n = k["period"], k["boundary"], k["vis"], k["n"]
    _vis, gap_back, gap_fwd, _last, _nxt = feats3.clocks(match)
    gap_total = np.where(vis, 0.0, np.minimum(gap_back + gap_fwd, 1e5))
    cols = feats3._Cols()
    visf = vis.astype(np.float64)

    def pair(arr, stem, from_ball=True):
        cols.add_pair(arr[:, 0], arr[:, 1], "own5_" + stem, from_ball, SUB)

    def diff(arr, stem, from_ball=True):
        cols.add_diff(arr[:, 0], arr[:, 1], "own5_" + stem, from_ball, SUB)

    def sym(arr, stem, from_ball=True):
        cols.add(arr, "own5_" + stem, from_ball, SUB, "sym")

    def since(ev):
        return np.stack([feats3._since_period(ev[:, s], period, CAP) for s in (0, 1)], axis=1).astype(np.float64)

    def until(ev):
        return np.stack([feats3._until_period(ev[:, s], period, CAP) for s in (0, 1)], axis=1).astype(np.float64)

    # 1. control runs
    dmin, has, pos, _vel, rel = _nearest(k)
    ctl, excl = _control(k, dmin, has, rel, gap_total)
    runs = [_runs(excl[:, s], boundary) for s in (0, 1)]
    el = np.stack([r[0] for r in runs], axis=1)
    rem = np.stack([r[1] for r in runs], axis=1)
    tot = np.stack([r[2] for r in runs], axis=1)
    conf = excl & (tot >= CONFIRM)
    pair(np.where(has, np.minimum(rel, 30.0), 30.0), "relspeed")
    pair(ctl, "ctl")
    pair(excl, "excl")
    pair(np.log1p(el), "run_el")
    pair(np.log1p(rem), "run_rem")
    pair(np.log1p(tot), "run_tot")
    duel = ctl[:, 0] & ctl[:, 1]
    sym(duel, "duel")
    sym(_wsum(duel, 25, 25, period) / np.maximum(_wsum(visf, 25, 25, period), 1.0), "duel_50")
    exf = excl.astype(np.float64)
    for w in SHARE_W:
        den = np.maximum(_wsum(visf, w, w, period), 1.0)
        sh = np.stack([_wsum(exf[:, s], w, w, period) / den for s in (0, 1)], axis=1)
        pair(sh, "share_%d" % w)
        diff(sh, "share_%d" % w)
    w = 50
    past = np.stack([_wsum(exf[:, s], w, 0, period) for s in (0, 1)], axis=1) / \
        np.maximum(_wsum(visf, w, 0, period), 1.0)[:, None]
    fut = np.stack([_wsum(exf[:, s], 0, w, period) for s in (0, 1)], axis=1) / \
        np.maximum(_wsum(visf, 0, w, period), 1.0)[:, None]
    pair(past, "share_past50")
    pair(fut, "share_fut50")
    pair(fut - past, "share_d50")

    # 2. brackets: last / next confirmed control, turnover candidates
    sc, uc = since(conf), until(conf)
    pair(np.log1p(sc), "since_conf")
    pair(np.log1p(uc), "until_conf")
    diff(np.log1p(sc[:, ::-1]), "last_lead")         # (a - h) of the reversed pair: + = home more recent
    diff(np.log1p(uc[:, ::-1]), "next_lead")
    last_is = (sc < sc[:, ::-1]) & (sc < CAP)
    next_is = (uc < uc[:, ::-1]) & (uc < CAP)
    keep = last_is & next_is
    turn_to = last_is[:, ::-1] & next_is             # the other team before, this team after
    span = sc.min(axis=1) + uc.min(axis=1)
    any_turn = turn_to[:, 0] | turn_to[:, 1]
    pair(last_is, "last_is")
    pair(next_is, "next_is")
    pair(keep, "keep")
    pair(turn_to, "turn_to")
    pair(turn_to & (span <= FAST_TURN)[:, None], "turn_fast_to")
    sym(np.where(keep.any(axis=1) | any_turn, np.log1p(span), np.log1p(2.0 * CAP)), "brk_span")
    sym(np.where(any_turn, np.log1p(span), 0.0), "turn_span")
    sym(np.where(any_turn, sc.min(axis=1) / np.maximum(span, 1.0), 0.5), "turn_pos")

    # 3. impulses
    b32 = k["ball"].astype(np.float32)
    vp, okp, vf, okf = feats3._displacement(b32, IMP_W, period)
    both = okp & okf
    dv = np.where(both, _norm(vf.astype(np.float64) - vp), 0.0)
    dsp = np.where(both, _norm(vf.astype(np.float64)) - _norm(vp.astype(np.float64)), 0.0)
    gate = has & (dmin < IMP_GATE) & ((dmin + IMP_LEAD) < dmin[:, ::-1]) & both[:, None]
    sym(np.minimum(dv, 30.0), "dv")
    pair(np.where(gate, np.minimum(dv, 30.0)[:, None], 0.0), "imp")
    kick = gate & (dsp > IMP_MIN)[:, None]
    recv = gate & (dsp < -IMP_MIN)[:, None]
    pair(np.log1p(since(kick)), "since_kick")
    pair(np.log1p(until(kick)), "until_kick")
    pair(np.log1p(since(recv)), "since_recv")
    pair(np.log1p(until(recv)), "until_recv")
    ev = (gate & (dv > IMP_MIN)[:, None]).astype(np.float64)
    den = np.maximum(_wsum(visf, 50, 50, period), 1.0)
    pair(np.stack([_wsum(ev[:, s], 50, 50, period) / den for s in (0, 1)], axis=1), "imp_rate100")

    # 4. cross-team interaction around the ball
    gap, n4, press, mate = _pressure(k, pos, has)
    pair(gap, "opp_gap")
    pair(n4, "opp_n4")
    pair(press, "press_on")
    pair(mate, "mate_gap")
    tint = _intercept(k)
    pair(tint, "tint")
    diff(tint[:, ::-1], "tint_lead")                 # + = home meets the ball first
    in_flight = vis & (np.where(has, dmin, np.inf).min(axis=1) > R_OUT) & (_norm(k["bv"]) > 4.0)
    sym(in_flight, "in_flight")
    lead = np.where(in_flight[:, None], tint[:, ::-1] - tint, 0.0)
    cols.add(lead[:, 0], "own5_flight_lead_diff", True, SUB, "diff")

    # 5. ball-free proxies
    sign = feats3._attack_sign(k["xy"].astype(np.float32), period).astype(np.float64)
    mk = _marking(k, sign)
    for stem in ("gs", "gsc"):
        g = mk[stem]
        pair(g, stem, False)
        diff(g, stem, False)
        cnt = np.maximum(_wsum(np.ones(n), 50, 50, period), 1.0)
        sm = np.stack([_wsum(g[:, s], 50, 50, period) / cnt for s in (0, 1)], axis=1)
        pair(sm, stem + "_100", False)
        pa = np.stack([_wsum(g[:, s], 50, 0, period) for s in (0, 1)], axis=1) / \
            np.maximum(_wsum(np.ones(n), 50, 0, period), 1.0)[:, None]
        fu = np.stack([_wsum(g[:, s], 0, 50, period) for s in (0, 1)], axis=1) / \
            np.maximum(_wsum(np.ones(n), 0, 50, period), 1.0)[:, None]
        pair(fu - pa, stem + "_d50", False)
    pair(mk["tight"], "tight", False)
    sym(np.log1p(mk["duels"]), "duels", False)
    pc = mk["pc"]
    pc_att = np.where(sign > 0, pc[:, 0, None], PITCH_L - pc[:, 0, None])
    cent_att = np.where(sign > 0, mk["cent"][..., 0], PITCH_L - mk["cent"][..., 0])
    pair(pc_att, "pc_att", False)
    pair(pc_att - cent_att, "pc_ahead", False)
    pcv = (_shift(pc, 12, period) - _shift(pc, -12, period)) * (FPS / 24.0)
    pcv = np.nan_to_num(pcv[:, 0])
    pcv = np.where(np.abs(pcv) > 15.0, 0.0, pcv)
    pair(sign * pcv[:, None], "pc_vatt", False)
    pair(mk["pc_near"], "pc_near", False)
    pair(mk["pc_n6"], "pc_n6", False)
    sym(np.where(vis, np.minimum(_norm(np.nan_to_num(k["ball"]) - pc), 30.0), 30.0), "pc_ball")
    return cols.stack()


def swap_plan(names, kind):
    """(permutation, sign) so that x[:, perm] * sign is the block of the team-swapped match."""
    pos = {nm: i for i, nm in enumerate(names)}
    perm = np.arange(len(names))
    sgn = np.ones(len(names), np.float32)
    for i, (nm, kd) in enumerate(zip(names, kind)):
        if kd in ("h", "a"):
            perm[i] = pos[feats3.partner_name(nm)]
        elif kd == "diff":
            sgn[i] = -1.0
    return perm, sgn


def swap_view(x, names, kind):
    perm, sgn = swap_plan(names, kind)
    return np.asarray(x)[:, perm] * sgn


def build_features3(match, groups=(), context_drop=(), with_own5=True):
    """feats3.build_features with own5 appended as the last column group. The feats3 columns are
    byte-identical to feats3.build_features(match, groups, context_drop)."""
    x, names, ball, sub, kind = feats3.build_features(match, groups, context_drop)
    if not with_own5:
        return x, names, ball, sub, kind
    xo, no, bo, so, ko = build_own5(match)
    return (np.concatenate([x, xo], axis=1), list(names) + list(no), np.concatenate([ball, bo]),
            list(sub) + list(so), list(kind) + list(ko))


# ---- sidecar cache and the 'v3own5' config ---------------------------------------------------

def bdir(deg, seed):
    return os.path.join(CACHE, deg, "s%d" % seed)


def _log(msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _free_gb(path):
    import shutil
    return shutil.disk_usage(os.path.splitdrive(path)[0] + os.sep).free / 1e9


def build_block(deg_name, seed, index, match):
    """Degrade one open match exactly as run_v3cand.build_one does and store its own5 block.
    Prints shapes and timings only."""
    from bench.dfl import load
    from bench import degrade_v5
    from bench.degrade_v4 import DEGRADERS
    degrade_v5.register()
    stem = os.path.join(bdir(deg_name, seed), match)
    if os.path.isfile(stem + ".ometa.json"):
        with open(stem + ".ometa.json") as f:
            if json.load(f).get("rev") == REV:
                return
    if _free_gb(RUN_DIR) < MIN_FREE_GB:
        raise SystemExit("less than %.0f GB free on the cache drive; not building" % MIN_FREE_GB)
    os.makedirs(bdir(deg_name, seed), exist_ok=True)
    t0 = time.time()
    deg = DEGRADERS[deg_name](load(match), 1000 * seed + index)
    x, names, ball, _sub, kind = build_own5(deg)
    if not np.isfinite(x).all():
        raise AssertionError("non-finite own5 features")
    np.save(stem + ".o.npy", x)
    np.save(stem + ".operiod.npy", deg["period"])
    np.save(stem + ".ovis.npy", ~np.isnan(deg["ball"][:, 0]))
    perm, sgn = swap_plan(names, kind)
    meta = {"rev": REV, "names": names, "kind": list(kind), "from_ball": [bool(b) for b in ball],
            "perm": perm.tolist(), "sign": sgn.tolist(), "frames": int(len(x)),
            "seconds": round(time.time() - t0, 1)}
    with open(stem + ".ometa.json", "w") as f:
        json.dump(meta, f)
    _log("own5 built %s %s seed %d (%d frames, %d cols, %.0fs)" % (deg_name, match, seed, len(x),
                                                                 x.shape[1], meta["seconds"]))


def load_block(deg, seed, match):
    stem = os.path.join(bdir(deg, seed), match)
    with open(stem + ".ometa.json") as f:
        meta = json.load(f)
    if meta.get("rev") != REV:
        raise RuntimeError("own5 cache of %s is rev %s, code is %s: rebuild" % (match, meta.get("rev"), REV))
    return np.load(stem + ".o.npy", mmap_mode="r"), meta


def attach(pack, deg, seed, matches):
    """Hang the own5 block on the cached v3 matrices of a run_v3cand pack (attribute .own5)."""
    for m in matches:
        o, meta = load_block(deg, seed, m)
        stem = os.path.join(bdir(deg, seed), m)
        if not (np.array_equal(np.load(stem + ".operiod.npy"), pack[m]["period"])
                and np.array_equal(np.load(stem + ".ovis.npy"), pack[m]["vis"])):
            raise RuntimeError("own5 block of %s is not the same degraded copy as the v3 cache" % m)
        perm, sgn = np.asarray(meta["perm"]), np.asarray(meta["sign"], np.float32)
        pack[m]["x"].own5 = (o, None, None)
        pack[m]["xs"].own5 = (o, perm, sgn)
    return pack


def make_spec(rv, shifts=(25, 50, 125)):
    class Own5Spec(rv.Spec):
        """v3 view plus the own5 columns at the end."""

        def view(self, mm, meta, period, rows=None):
            base = rv.Spec.view(self, mm, meta, period, rows)
            extra = getattr(mm, "own5", None)
            if extra is None:
                raise RuntimeError("no own5 block attached (run `python -m bench.own5 build` first)")
            o, perm, sgn = extra
            o = np.asarray(o if rows is None else o[rows], np.float32)
            if perm is not None:
                o = o[:, perm] * sgn
            return np.concatenate([base, o], axis=1)

    return Own5Spec("v3", shifts=shifts)


def register(rv):
    """Add config 'v3own5' to run_v3cand and make its load_pack attach the block when it exists."""
    if getattr(rv, "_own5_registered", False):
        return
    rv.CONFIGS["v3own5"] = lambda: make_spec(rv)
    plain = rv.load_pack

    def load_pack(deg, seed, tag=""):
        pack = plain(deg, seed, tag)
        have = all(os.path.isfile(os.path.join(bdir(deg, seed), m + ".ometa.json")) for m in rv.ORDER)
        if have and not tag:
            attach(pack, deg, seed, rv.ORDER)
        return pack

    rv.load_pack = load_pack
    rv._own5_registered = True


def cmd_build(degs, seeds):
    from bench import run_hedge, run_v3cand as rv
    run_hedge.set_low_priority()
    for deg in degs:
        for seed in seeds:
            for i, m in enumerate(rv.ORDER):
                build_block(deg, seed, i, m)


def cmd_hedge(degs, seeds, extra):
    """v3 and v3own5 through bench.run_hedge, pooled L1+L2 target, boundary x20, results in RUN_DIR."""
    from bench import run_hedge, run_v3cand as rv
    register(rv)
    run_hedge.RES, run_hedge.LOG = RES, LOG
    for deg in degs:
        for config in ("v3", "v3own5"):
            sys.argv = ["run_hedge", "--deg", deg, "--config", config, "--seeds"] + [str(s) for s in seeds] + \
                       ["--boosts", "20", "--train-defs", "L12"] + list(extra)
            run_hedge.main()


def verdict(records, min_mean=0.02):
    """Per-seed gain of v3own5 over v3 (mean of the L1 and L2 TEST scores, mean over degraders).
    Adopt only if the mean gain is >= min_mean and the gain is positive on every seed."""
    cell = {}
    for r in records:
        if r["train_def"] != "L12" or float(r["boost"]) != 20.0 or r["config"] not in ("v3", "v3own5"):
            continue
        val = float(np.mean([np.mean(list(r["test"][d].values())) for d in ("L1", "L2")]))
        cell[(r["deg"], r["seed"], r["config"])] = val
    pairs = sorted({(d, s) for (d, s, c) in cell if (d, s, "v3") in cell and (d, s, "v3own5") in cell})
    gains = {}
    for d, s in pairs:
        g = cell[(d, s, "v3own5")] - cell[(d, s, "v3")]
        gains.setdefault(s, []).append(g)
        print("%-9s seed %d  v3 %.4f  v3own5 %.4f  gain %+.4f" % (d, s, cell[(d, s, "v3")],
                                                              cell[(d, s, "v3own5")], g))
    if not gains:
        print("no paired results")
        return None
    per_seed = {s: float(np.mean(v)) for s, v in sorted(gains.items())}
    mean = float(np.mean(list(per_seed.values())))
    ok = len(per_seed) >= 3 and mean >= min_mean and all(v > 0 for v in per_seed.values())
    print("per-seed gain %s  mean %+.4f  ->  %s" % (
        {s: round(v, 4) for s, v in per_seed.items()}, mean, "ADOPT" if ok else "REJECT"))
    return ok


# ---- tests (synthetic only) ------------------------------------------------------------------

def _expect(name, cond, detail=""):
    if not cond:
        raise AssertionError("%s (%s)" % (name, detail))
    print("PASS", name)


def _synth(n=3000, seed=0, hide=0.45):
    """Two halves with the sides exchanged, 11 players a side plus an empty slot, a ball that is
    carried, passed and sometimes intercepted, and random ball gaps. Returns (match, true owner)."""
    rng = np.random.default_rng(seed)
    s = 12
    half = n // 2
    period = np.ones(n, np.int8)
    period[half:] = 2
    form = np.array([[4, 34], [18, 10], [17, 26], [17, 42], [18, 58], [38, 14], [36, 34], [38, 54],
                     [58, 12], [62, 34], [58, 56]], np.float64)
    players = np.full((n, 2, s, 2), np.nan)
    for side in (0, 1):
        for j in range(11):
            walk = np.cumsum(rng.normal(0, 0.06, (n, 2)), axis=0)
            walk -= np.linspace(0, 1, n)[:, None] * walk[-1]
            p = form[j] + walk + rng.normal(0, 1.5, 2)
            x_att = p[:, 0] * 0.9 + 4.0
            left = (side == 0) == (np.arange(n) < half)      # home defends x=0 in half 1, x=105 in half 2
            players[:, side, j, 0] = np.clip(np.where(left, x_att, PITCH_L - x_att), 0.5, 104.5)
            players[:, side, j, 1] = np.clip(p[:, 1], 0.5, 67.5)
    players[rng.random((n, 2, s)) < 0.01] = np.nan          # random drop-outs
    players[:, :, 11] = np.nan
    ball = np.zeros((n, 3))
    owner = np.full(n, -1, np.int8)
    t = 0
    team, slot = 0, 6
    while t < n:
        hold = int(rng.integers(20, 90))
        e = min(n, t + hold)
        pos = np.nan_to_num(players[t:e, team, slot], nan=50.0)
        ball[t:e, :2] = pos + 0.4
        owner[t:e] = team
        t = e
        if t >= n:
            break
        nteam = team if rng.random() < 0.65 else 1 - team
        nslot = int(rng.integers(1, 11))
        src = ball[t - 1, :2]
        dst = np.nan_to_num(players[min(n - 1, t + 20), nteam, nslot], nan=50.0) + 0.4
        fl = max(4, int(np.linalg.norm(dst - src) / 0.5))    # 12.5 m/s
        e = min(n, t + fl)
        a = np.arange(1, e - t + 1)[:, None] / fl
        ball[t:e, :2] = src + a * (dst - src)
        t = e
        team, slot = nteam, nslot
    hidden = np.zeros(n, bool)
    while hidden.mean() < hide:
        a = int(rng.integers(0, n))
        hidden[a:a + int(rng.integers(15, 200))] = True
    ball[hidden] = np.nan
    return {"ball": ball.astype(np.float32), "players": players.astype(np.float32), "period": period}, owner


def _mirror(match, fx, fy):
    out = dict(match)
    pl = np.array(match["players"], np.float64)
    bl = np.array(match["ball"], np.float64)
    if fx:
        pl[..., 0] = PITCH_L - pl[..., 0]
        bl[:, 0] = PITCH_L - bl[:, 0]
    if fy:
        pl[..., 1] = PITCH_W - pl[..., 1]
        bl[:, 1] = PITCH_W - bl[:, 1]
    out["players"], out["ball"] = pl, bl
    return out


def _bad_cols(a, b, names, tol=1e-3):
    return [nm for j, nm in enumerate(names) if not np.allclose(a[:, j], b[:, j], rtol=1e-4, atol=tol)]


def _scenario():
    """Home player carries for 2 s, plays a 1 s pass that an away player controls for 2 s."""
    n = 150
    players = np.full((n, 2, 3, 2), np.nan, np.float32)
    players[:, 0, 0] = (3.0, 34.0)
    players[:, 1, 0] = (102.0, 34.0)
    t = np.arange(n, dtype=np.float32)
    players[:, 0, 1, 0] = 30.0 + 0.1 * np.minimum(t, 50)      # carrier, 2.5 m/s then stops
    players[:, 0, 1, 1] = 30.0
    players[:, 1, 1] = (55.0, 30.0)                           # receiver, standing
    players[:, 0, 2] = (20.0, 50.0)
    players[:, 1, 2] = (80.0, 50.0)
    ball = np.zeros((n, 3), np.float32)
    ball[:, 1] = 30.2
    ball[:50, 0] = players[:50, 0, 1, 0] + 0.3
    ball[50:75, 0] = np.linspace(ball[49, 0], 54.7, 26)[1:]   # ~19 m in 1 s
    ball[75:, 0] = 54.7
    return {"ball": ball, "players": players, "period": np.ones(n, np.int8)}


def self_test():
    t0 = time.time()
    match, owner = _synth()
    x, names, fb, sub, kind = build_own5(match)
    ix = {nm: i for i, nm in enumerate(names)}
    n = len(x)
    _expect("shape, float32, finite, one subgroup", x.dtype == np.float32 and np.isfinite(x).all()
            and set(sub) == {SUB} and len(names) == x.shape[1], "%d cols" % x.shape[1])
    hidden = float(np.isnan(match["ball"][:, 0]).mean())
    _expect("synthetic ball is missing on 40-55% of frames", 0.40 <= hidden <= 0.55, "%.2f" % hidden)
    x2 = build_own5(match)[0]
    _expect("deterministic (two calls, same bytes)", np.array_equal(x, x2))

    xs, names_s, _, _, kind_s = build_own5(methods.swap_teams(match))
    _expect("team swap keeps the schema", names_s == names and kind_s == kind)
    bad = _bad_cols(swap_view(x, names, kind), xs, names)
    _expect("team-swap invariance: swapped match == column permutation of the block", not bad, str(bad[:8]))
    _expect("every h column has an a partner and the plan is an involution",
            np.array_equal(swap_plan(names, kind)[0][swap_plan(names, kind)[0]], np.arange(len(names))))
    for fx, fy in ((True, False), (False, True), (True, True)):
        xm = build_own5(_mirror(match, fx, fy))[0]
        bad = _bad_cols(x, xm, names)
        _expect("mirror invariance (flip x=%s, y=%s)" % (fx, fy), not bad, str(bad[:8]))

    rng = np.random.default_rng(1)
    perm = match["players"].copy()
    for side in (0, 1):
        perm[:, side] = perm[:, side][:, rng.permutation(perm.shape[2])]
    extra = np.concatenate([perm, np.full((n, 2, 2, 2), np.nan, np.float32)], axis=2)
    xp = build_own5(dict(match, players=extra))[0]
    bad = _bad_cols(x, xp, names)
    _expect("slot order and empty slots do not matter", not bad, str(bad[:8]))

    scr = dict(match, alive=np.zeros(n, bool), owner=np.ones(n, np.int8))
    _expect("labels are not inputs", np.array_equal(build_own5(scr)[0], x))
    rolled = dict(match, ball=np.roll(match["ball"], 97, axis=0))
    xr = build_own5(rolled)[0]
    free = [j for j in range(len(names)) if not fb[j]]
    _expect("ball-free columns do not move with the ball, the others do",
            len(free) >= 20 and np.array_equal(x[:, free], xr[:, free]) and not np.array_equal(x, xr),
            "%d ball-free" % len(free))
    noball = dict(match, ball=np.full_like(match["ball"], np.nan))
    xn = build_own5(noball)[0]
    _expect("no ball at all: finite, ball-free columns unchanged",
            np.isfinite(xn).all() and np.array_equal(xn[:, free], x[:, free]))

    vis = ~np.isnan(match["ball"][:, 0])
    eh, ea = x[:, ix["own5_excl_h"]] > 0, x[:, ix["own5_excl_a"]] > 0
    carried = vis & (owner >= 0)
    acc = float(np.mean(np.where(owner[carried] == 0, eh[carried], ea[carried])))
    wrong = float(np.mean(np.where(owner[carried] == 0, ea[carried], eh[carried])))
    _expect("exclusive control finds the true carrier on visible carried frames", acc > 0.85 and wrong < 0.03,
            "hit %.3f wrong %.3f" % (acc, wrong))
    flight = vis & (owner < 0)
    _expect("a ball in flight is rarely called controlled", float((eh | ea)[flight].mean()) < 0.25,
            "%.3f" % float((eh | ea)[flight].mean()))
    lead = x[:, ix["own5_last_lead_diff"]] + x[:, ix["own5_next_lead_diff"]]
    # Gaps longer than a carry hide whole possessions, where no bracket can know the carrier;
    # the claim tested is for 1.4 s gaps (a second copy of the game, hidden 35 frames in every 120).
    short, owner_s = _synth(seed=5, hide=0.0)
    cutm = (np.arange(n) % 120) >= 85
    short["ball"][cutm] = np.nan
    xq = build_own5(short)[0]
    lead = xq[:, ix["own5_last_lead_diff"]] + xq[:, ix["own5_next_lead_diff"]]
    gapc = cutm & (owner_s >= 0) & (lead != 0)
    agree = float(np.mean(np.where(owner_s[gapc] == 0, lead[gapc] > 0, lead[gapc] < 0)))
    _expect("ball hidden for 1.4 s: the bracket still points at the true carrier",
            int(gapc.sum()) >= 200 and agree > 0.75, "%d frames, %.3f" % (int(gapc.sum()), agree))

    sc = _scenario()
    y, nm, _, _, _ = build_own5(sc)
    iy = {k_: i for i, k_ in enumerate(nm)}
    _expect("scenario: home controls the carry, away the reception, nobody the pass",
            y[10:45, iy["own5_excl_h"]].all() and y[85:140, iy["own5_excl_a"]].all()
            and not y[58:68, iy["own5_ctl_h"]].any() and not y[58:68, iy["own5_ctl_a"]].any())
    _expect("scenario: the pass is a fast turnover candidate towards away",
            y[55:70, iy["own5_turn_fast_to_a"]].all() and not y[:, iy["own5_turn_to_h"]].any()
            and y[20, iy["own5_keep_h"]] == 1)
    _expect("scenario: release impulse goes to home, reception to away",
            y[40:60, iy["own5_imp_h"]].max() > 5 and y[40:60, iy["own5_imp_a"]].max() == 0
            and y[66:84, iy["own5_imp_a"]].max() > 5 and y[60, iy["own5_since_kick_h"]] < y[60, iy["own5_since_kick_a"]])
    _expect("scenario: away meets the in-flight ball first",
            y[58, iy["own5_in_flight"]] == 1 and y[58, iy["own5_flight_lead_diff"]] < 0)
    cut = dict(sc, ball=sc["ball"].copy())
    cut["ball"][45:80] = np.nan
    yc = build_own5(cut)[0]
    _expect("scenario with the whole pass hidden: turnover bracket survives",
            yc[50:75, iy["own5_turn_to_a"]].all() and np.isfinite(yc).all())
    stat = feats3._synthetic_shape()
    ys, ns_, _, _, _ = build_own5(stat)
    js = {k_: i for i, k_ in enumerate(ns_)}
    _expect("static blocks: ball at a home player's feet is home control for the whole clip",
            ys[:, js["own5_excl_h"]].all() and not ys[:, js["own5_ctl_a"]].any()
            and np.allclose(ys[:, js["own5_share_50_h"]], 1.0))

    # degrader options (synthetic match of the degrader's own test)
    from bench import degrade_v5
    syn = degrade_v5._synth(n=4000, s0=6, seed=3)
    for nm_, kw in (("v4/v5 default", {}), ("v5ball", dict(ball_stress=True)), ("v5stress", degrade_v5.STRESS)):
        deg = degrade_v5.degrade(syn, 11, **kw)
        xd, nd, _, _, kd = build_own5(deg)
        xds = build_own5(methods.swap_teams(deg))[0]
        bad = _bad_cols(swap_view(xd, nd, kd), xds, nd)
        _expect("degrader %s: finite and swap-invariant (%d slots)" % (nm_, deg["players"].shape[2]),
                np.isfinite(xd).all() and nd == names and not bad, str(bad[:8]))

    # plugging into feats3 and into the v3 config
    xa, na, _, sa, _ = build_features3(stat, ("context", "lang_dead"))
    xb, nb, _, _, _ = feats3.build_features(stat, ("context", "lang_dead"))
    _expect("feats3 columns unchanged, own5 appended as the last group",
            na[:len(nb)] == nb and np.array_equal(xa[:, :len(nb)], xb) and sa[len(nb):] == [SUB] * len(ns_)
            and np.array_equal(xa[:, len(nb):], ys))
    import tempfile
    from bench import run_v3cand as rv
    with tempfile.TemporaryDirectory() as tmp:
        m2 = dict(match, players=match["players"][:600], ball=match["ball"][:600], period=match["period"][:600])
        full, fnames, _, fsub, _ = feats3.build_features(m2, rv.GROUPS)
        np.save(os.path.join(tmp, "x.npy"), full)
        np.save(os.path.join(tmp, "o.npy"), build_own5(m2)[0])
        mm = np.load(os.path.join(tmp, "x.npy"), mmap_mode="r")
        o = np.load(os.path.join(tmp, "o.npy"), mmap_mode="r")
        meta = {"names": fnames, "sub": list(fsub)}
        spec = make_spec(rv)
        v3 = rv.Spec("v3").view(mm, meta, m2["period"])
        try:
            spec.view(mm, meta, m2["period"])
            raised = False
        except RuntimeError:
            raised = True
        p_, s_ = swap_plan(names, kind)
        mm.own5 = (o, p_, s_)
        v5 = spec.view(mm, meta, m2["period"])
        rows = np.arange(0, 600, 5)
        _expect("config v3own5 = v3 view + block (swapped by the plan), refuses to run without it",
                raised and np.array_equal(v5[:, :v3.shape[1]], v3)
                and np.array_equal(v5[:, v3.shape[1]:], swap_view(o, names, kind))
                and np.array_equal(spec.view(mm, meta, m2["period"], rows), v5[rows]))
        del mm, o, v3, v5
    recs = [{"deg": "v4", "seed": s_, "config": c, "train_def": "L12", "boost": 20.0,
             "test": {"L1": {"m": 0.60 + g}, "L2": {"m": 0.62 + g}}}
            for s_, gs in enumerate(((0.03, 0.0), (0.02, 0.0), (0.03, 0.0)))
            for c, g in (("v3own5", gs[0]), ("v3", gs[1]))]
    ok = verdict(recs)
    recs[2]["test"]["L1"]["m"] = recs[2]["test"]["L2"]["m"] = 0.5
    _expect("adoption rule: mean >= +0.02 and positive on every seed", ok is True and verdict(recs) is False)
    print("own5 self_test ok: %d columns (%d ball-free) in %.1fs" % (len(names), len(free), time.time() - t0))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="selftest", choices=("selftest", "build", "hedge", "verdict"))
    ap.add_argument("--deg", nargs="+", default=["v4", "v5ball"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a, extra = ap.parse_known_args()
    if a.cmd == "selftest":
        self_test()
    elif a.cmd == "build":
        cmd_build(a.deg, a.seeds)
    elif a.cmd == "hedge":
        cmd_hedge(a.deg, a.seeds, extra)
    else:
        if not os.path.isfile(RES):
            raise SystemExit("no results at " + RES)
        verdict([json.loads(line) for line in open(RES, encoding="utf-8")])


if __name__ == "__main__":
    main()
