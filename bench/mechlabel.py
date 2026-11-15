"""Mechanical team-possession labels from clean tracking.

Literature, fixed before any spell-length comparison (not fit to a match):

- Savitzky-Golay, order 2, 7 frames, on the ball (Vidal-Codina et al. 2022).
- Possession zone r_p. Pre-registered default 1.0 m (Vidal-Codina providers B and C).
  Sweep 0.5 / 1.0 / 1.5 / 2.0. databallpy's 1.5 m is only a reported row.
- Duel radius 1.0 m: both teams inside it, duel prevails, owner does not change
  (Vidal-Codina; the task text).
- A zone run is a real gain only with a touch: entry-vs-exit direction change
  > 10 degrees, or a speed change > 5 m/s (databallpy defaults for Vidal-Codina).
  The 5 m/s test is applied to the raw frame difference and to any 7-frame
  window, because the prescribed smoother wipes a one-frame spike.
- A ball that is essentially stopped inside the zone is control, not a fly-by.
- Team owner is sticky through loose balls (nobody within 3 m), flight and
  deflections. The opponent becomes owner only after T_c seconds of confirmed
  proximity. Pre-registered default T_c = 1.0 s (Link & Hoernig 2017; the rule
  note's start value). Sweep 0, 0.5, 1, 2, 3. Gaps of up to 1 s pause the
  clock and do not count as contact (Link & Hoernig). The flip is written on
  the frame T_c is reached; earlier frames keep the previous owner.
- D is the provider alive flag and nothing else. Dead frames discard the owner.
  The next alive stretch is filled from the team that first completes control,
  so a restart has a possessor.

Clean IDSSE ball tracks have no gaps. ``impute=True`` (degraded tracks only)
holds the last observed ball for distances, and never treats a gap frame as
contact, so a teleport when the ball reappears is not a touch.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter

H, A, D = 0, 1, 2
FPS = 25

SG_WINDOW = 7
SG_POLY = 2
DUEL_R = 1.0
LOOSE_R = 3.0
ANGLE_DEG = 10.0
SPEED_DELTA = 5.0          # m/s
SPEED_WINDOW = 7           # same scale as the Savitzky-Golay window
REST_SPEED = 1.0           # m/s, ball at a foot rather than passing
REST_MIN_S = 0.5
GAP_S = 1.0                # Link & Hoernig: a 1 s loss of contact ends the attempt
MIN_STEP = 1.0 / FPS       # 1 m/s, below which a direction is numerical noise

# Pre-registered. The sweep reports every other pair and does not select them.
DEFAULT_R_P = 1.0
DEFAULT_T_C = 1.0

R_P_SWEEP = (0.5, 1.0, 1.5, 2.0)
T_C_SWEEP = (0.0, 0.5, 1.0, 2.0, 3.0)


def tc_frames(t_c):
    """Confirmed-contact frames required before the owner flips. 0 s -> one frame."""
    if t_c <= 0:
        return 1
    return max(1, int(float(t_c) * FPS + 0.5))


def _stretches(observed):
    idx = np.flatnonzero(observed)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1) + 1
    return np.split(idx, cuts)


def _sg(col):
    n = len(col)
    w = SG_WINDOW if n >= SG_WINDOW else (n if n % 2 else n - 1)
    if w <= SG_POLY:
        return col
    return savgol_filter(col, w, SG_POLY, mode="interp")


def _smooth(xy, observed):
    """Savitzky-Golay inside each contiguous observed stretch. NaN stays NaN."""
    out = np.array(xy, dtype=np.float64, copy=True)
    out[~observed] = np.nan
    for ix in _stretches(observed):
        if len(ix) < 3:
            continue
        for j in range(2):
            out[ix, j] = _sg(out[ix, j])
    return out


def _hold_fill(xy, observed):
    """Hold last observed sample; back-fill the leading gap from the first one."""
    n = len(xy)
    out = np.array(xy, dtype=np.float64, copy=True)
    ids = np.where(observed, np.arange(n), -1)
    last = np.maximum.accumulate(ids)
    ok = last >= 0
    out[ok] = xy[last[ok]]
    if n and not observed[0] and observed.any():
        first = int(np.argmax(observed))
        out[:first] = out[first]
    return out


def _side_min(players_side, xy):
    diff = players_side.astype(np.float32, copy=False) - xy.astype(np.float32)[:, None, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    dist[~np.isfinite(dist)] = np.inf
    return dist.min(axis=1).astype(np.float32)


def _kinematics(xy, observed):
    """Speed (m/s) and a touch-scale speed-change mask. Invalid off observed stretches."""
    n = len(xy)
    speed = np.full(n, np.nan, np.float32)
    hot = np.zeros(n, dtype=bool)
    for ix in _stretches(observed):
        m = len(ix)
        if m == 1:
            speed[ix[0]] = 0.0
            continue
        step = np.linalg.norm(np.diff(xy[ix], axis=0), axis=1) * FPS
        sp = np.empty(m, np.float32)
        sp[0] = step[0]
        sp[1:] = step
        speed[ix] = sp
        if m >= 2:
            hot[ix[1:]] |= np.abs(np.diff(sp)) > SPEED_DELTA
        if m >= SPEED_WINDOW:
            sw = sliding_window_view(sp, SPEED_WINDOW)
            span = sw.max(axis=1) - sw.min(axis=1)
            hot_w = span > SPEED_DELTA
            acc = np.zeros(m + 1, np.int32)
            ii = np.flatnonzero(hot_w)
            acc[ii] += 1
            acc[ii + SPEED_WINDOW] -= 1
            hot[ix] |= np.cumsum(acc[:-1]) > 0
    return speed, hot


def _angle_ok(v_in, v_out):
    n1 = float(np.hypot(v_in[0], v_in[1]))
    n2 = float(np.hypot(v_out[0], v_out[1]))
    if n1 < MIN_STEP or n2 < MIN_STEP:
        return False
    c = (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (n1 * n2)
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return float(np.degrees(np.arccos(c))) > ANGLE_DEG


def prepare(match, impute=False):
    """Per-period arrays shared by every (r_p, T_c) pair. Distances are metres."""
    ball = match["ball"]
    players = match["players"]
    period = np.asarray(match["period"])
    n = len(ball)
    observed = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    if impute:
        alive = np.ones(n, dtype=bool)
    else:
        alive = np.asarray(match["alive"], dtype=bool)
    parts = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        xy_raw = ball[idx, :2]
        obs = observed[idx]
        xy = _smooth(xy_raw, obs)
        if impute:
            pos = _hold_fill(xy, obs)
        else:
            pos = xy
        speed, hot = _kinematics(xy, obs)
        dh = _side_min(players[idx, 0], pos)
        da = _side_min(players[idx, 1], pos)
        parts.append({
            "idx": idx,
            "xy": xy,
            "pos": np.asarray(pos, np.float32),
            "speed": speed,
            "hot": hot,
            "dh": dh,
            "da": da,
            "alive": alive[idx],
            "observed": obs,
        })
    return parts


def _zone(part, r_p):
    """Closest team inside r_p, or -1. A duel or a loose ball clears the zone."""
    dh, da = part["dh"], part["da"]
    near = np.minimum(dh, da)
    alive = part["alive"]
    obs = part["observed"]
    duel = alive & obs & (dh <= DUEL_R) & (da <= DUEL_R)
    loose = near > LOOSE_R
    in_pz = alive & obs & (near <= r_p) & ~duel & ~loose
    closest = np.where(dh <= da, np.int8(0), np.int8(1))
    zone = np.where(in_pz, closest, np.int8(-1)).astype(np.int8)
    return zone


def _confirm(part, zone, info=None):
    """Mark frames that belong to a touch-confirmed (or at-rest) zone run."""
    xy, speed, hot = part["xy"], part["speed"], part["hot"]
    obs = part["observed"]
    n = len(zone)
    control = np.full(n, -1, np.int8)
    if n == 0:
        return control
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = zone[1:] != zone[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:] - 1
    ends[-1] = n - 1
    rest_min = max(1, int(REST_MIN_S * FPS + 0.5))
    for s, e in zip(starts, ends):
        team = int(zone[s])
        if team < 0:
            continue
        length = int(e - s + 1)
        if info is not None:
            info["zone_runs"] += 1
            info["zone_run_frames"] += length
        by_angle = False
        if s > 0 and e + 1 < n and obs[s] and obs[s - 1] and obs[e] and obs[e + 1]:
            by_angle = _angle_ok(xy[s] - xy[s - 1], xy[e + 1] - xy[e])
        elif e > s and obs[s] and obs[s - 1] and obs[e] and obs[e - 1]:
            by_angle = _angle_ok(xy[s] - xy[s - 1], xy[e] - xy[e - 1])
        sp = speed[s:e + 1]
        by_speed = bool(hot[s:e + 1].any())
        by_rest = bool(np.isfinite(sp).all() and length >= rest_min
                       and float(np.median(sp)) < REST_SPEED)
        if not (by_angle or by_speed or by_rest):
            continue
        control[s:e + 1] = team
        if info is not None:
            info["confirmed_runs"] += 1
            info["by_angle"] += int(by_angle)
            info["by_speed"] += int(by_speed)
            info["by_rest"] += int(by_rest)
    return control


def _possess(control, alive, t_c):
    """Sticky owner. Flip after T_c confirmed-contact frames; gaps up to 1 s pause."""
    n = len(control)
    need = tc_frames(t_c)
    gap_max = int(GAP_S * FPS)  # 25 frames: a gap of 1.00 s is still the same attempt
    owners = np.full(n, -1, np.int16)
    owner = -1
    att = -1
    contact = 0
    gap = 0
    start = 0
    for t in range(n):
        if not alive[t]:
            owners[t] = -2
            owner = -1
            att = -1
            contact = 0
            gap = 0
            continue
        ct = int(control[t])
        if ct >= 0:
            if att != ct:
                att = ct
                contact = 1
                gap = 0
                start = t
            else:
                contact += 1
                gap = 0
            if owner != att and contact >= need:
                owner = att
                if start < t:
                    # Establishment after a dead ball: the attempt itself belongs to
                    # the new team. A steal from a known owner is not rewritten; the
                    # frames before this one were stored as that owner and stay.
                    unknown = owners[start:t] < 0
                    if unknown.any() and not (owners[start:t] >= 0).any():
                        owners[start:t] = owner
                owners[t] = owner
            else:
                owners[t] = owner if owner >= 0 else -1
        else:
            if att >= 0:
                gap += 1
                if gap > gap_max:
                    att = -1
                    contact = 0
                    gap = 0
            owners[t] = owner if owner >= 0 else -1
    return owners


def _close_owners(owners, zone):
    """Fill a restart's unknown frames from the team that first completes control.

    Frames already given to a team are left alone, so a steal is not back-dated.
    An alive stretch that never comes under control takes the majority zone team,
    or home if nobody entered the zone. Returns the fallback frame count.
    """
    n = len(owners)
    fallback = 0
    i = 0
    while i < n:
        if owners[i] == -2:
            i += 1
            continue
        j = i + 1
        while j < n and owners[j] != -2:
            j += 1
        seg = owners[i:j]
        known = np.flatnonzero(seg >= 0)
        if len(known) == 0:
            z = zone[i:j]
            h = int(np.sum(z == 0))
            a = int(np.sum(z == 1))
            seg[:] = 0 if h >= a else 1
            fallback += int(j - i)
        else:
            seg[:int(known[0])] = seg[int(known[0])]
            idx = np.where(seg >= 0, np.arange(len(seg)), -1)
            last = np.maximum.accumulate(idx)
            hole = seg < 0
            if hole.any():
                seg[hole] = seg[last[hole]]
        i = j
    return fallback


def _to_codes(owners):
    codes = np.full(len(owners), H, np.int8)
    codes[owners == 1] = A
    codes[owners < 0] = D
    return codes


def label_prepared(parts, r_p, t_c, info=None):
    """Int8 H/A/D codes aligned with the original match, from ``prepare`` output."""
    if not parts:
        return np.zeros(0, np.int8)
    n = int(parts[-1]["idx"][-1]) + 1
    # periods may not cover a dense 0..n-1 if a period id were missing; size from max idx
    n = max(n, max(int(p["idx"][-1]) for p in parts) + 1)
    codes = np.full(n, D, np.int8)
    for part in parts:
        zone = _zone(part, r_p)
        control = _confirm(part, zone, info)
        owners = _possess(control, part["alive"], t_c)
        fb = _close_owners(owners, zone)
        if info is not None:
            info["fallback_frames"] += fb
            info["frames"] += int(len(zone))
        codes[part["idx"]] = _to_codes(owners)
    return codes


def sweep_match(match, radii=R_P_SWEEP, tcs=T_C_SWEEP):
    """One prepare, then every (r_p, T_c). Returns codes, confirm-info per radius.

    Confirm counts do not depend on T_c. Fallback frames do, and are returned
    on each codes' companion dict ``fallback``.
    """
    parts = prepare(match, impute=False)
    n = len(match["ball"])
    codes = {}
    confirm = {}
    fallback = {}
    for r_p in radii:
        info = _blank_info()
        zones, controls = [], []
        for part in parts:
            zone = _zone(part, r_p)
            zones.append(zone)
            controls.append(_confirm(part, zone, info))
            info["frames"] += int(len(zone))
        confirm[float(r_p)] = info
        for t_c in tcs:
            out = np.full(n, D, np.int8)
            fb = 0
            for part, zone, control in zip(parts, zones, controls):
                owners = _possess(control, part["alive"], t_c)
                fb += _close_owners(owners, zone)
                out[part["idx"]] = _to_codes(owners)
            codes[(float(r_p), float(t_c))] = out
            fallback[(float(r_p), float(t_c))] = int(fb)
    return codes, confirm, fallback


def label_match(match, r_p=DEFAULT_R_P, t_c=DEFAULT_T_C, impute=False, return_info=False):
    """Label one match. ``impute`` holds the last ball and ignores the alive flag."""
    parts = prepare(match, impute=impute)
    info = None
    if return_info:
        info = _blank_info()
    codes = label_prepared(parts, r_p, t_c, info)
    if return_info:
        return codes, info
    return codes


def _blank_info():
    return {
        "zone_runs": 0, "zone_run_frames": 0, "confirmed_runs": 0,
        "by_angle": 0, "by_speed": 0, "by_rest": 0,
        "fallback_frames": 0, "frames": 0,
    }


def zone_counts(players, xy, r_p):
    """How many players of each team sit inside r_p. Shape (n, 2), float32."""
    n = len(xy)
    out = np.zeros((n, 2), np.float32)
    xy32 = xy.astype(np.float32, copy=False)
    for side in (0, 1):
        diff = players[:, side].astype(np.float32, copy=False) - xy32[:, None, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        out[:, side] = (np.isfinite(dist) & (dist <= r_p)).sum(axis=1)
    return out


def mechanical_state(match, r_p=DEFAULT_R_P, t_c=DEFAULT_T_C):
    """Owner, seconds since the last owner change, and zone occupancy.

    Computed from positions only. The provider alive flag and owner flag are
    not read. Ball gaps are held last for distances; gap frames are not contact.
    Columns: owner_home, since_change_s, pz_home, pz_away.
    """
    parts = prepare(match, impute=True)
    n = len(match["ball"])
    codes = label_prepared(parts, r_p, t_c, None)
    owner_home = (codes == H).astype(np.float32)
    since = np.zeros(n, np.float32)
    pz = np.zeros((n, 2), np.float32)
    players = match["players"]
    for part in parts:
        idx = part["idx"]
        own = np.where(codes[idx] == A, 1, 0)
        ch = np.ones(len(own), dtype=bool)
        if len(own) > 1:
            ch[1:] = own[1:] != own[:-1]
        ids = np.where(ch, np.arange(len(own)), -1)
        last = np.maximum.accumulate(ids)
        since[idx] = np.minimum((np.arange(len(own)) - last) / float(FPS), 30.0)
        pz[idx] = zone_counts(players[idx], part["pos"], r_p)
    block = np.column_stack([owner_home, since, pz[:, 0], pz[:, 1]]).astype(np.float32)
    names = ["mech_owner_home", "mech_since_change_s", "mech_pz_home", "mech_pz_away"]
    return block, names


def spell_lengths(codes, period, fps=FPS):
    """H/A spell lengths in seconds. D splits a spell. Halves are separate."""
    lengths = []
    period = np.asarray(period)
    for p in np.unique(period):
        c = codes[period == p]
        if len(c) == 0:
            continue
        cuts = np.flatnonzero(c[1:] != c[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(c)]))
        for s, e in zip(starts, ends):
            if c[s] == H or c[s] == A:
                lengths.append((e - s) / float(fps))
    return np.asarray(lengths, np.float64)


def spell_starts(codes, period):
    """Global frame index and team of each H/A spell start."""
    frames, teams = [], []
    period = np.asarray(period)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        c = codes[idx]
        if len(c) == 0:
            continue
        new = np.ones(len(c), dtype=bool)
        if len(c) > 1:
            new[1:] = c[1:] != c[:-1]
        keep = new & ((c == H) | (c == A))
        frames.append(idx[keep])
        teams.append(c[keep])
    if not frames:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)
    return np.concatenate(frames), np.concatenate(teams).astype(np.int8)


def turnover_starts(codes, period):
    """Spell starts whose previous in-half frame is the other team, not a dead ball."""
    frames, teams = [], []
    period = np.asarray(period)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        c = codes[idx]
        if len(c) < 2:
            continue
        flip = np.zeros(len(c), dtype=bool)
        flip[1:] = ((c[1:] == H) & (c[:-1] == A)) | ((c[1:] == A) & (c[:-1] == H))
        frames.append(idx[flip])
        teams.append(c[flip])
    if not frames:
        return np.zeros(0, np.int64), np.zeros(0, np.int8)
    return np.concatenate(frames), np.concatenate(teams).astype(np.int8)


def signed_lags(src_f, src_team, dst_f, dst_team):
    """src_frame - nearest same-team dst_frame, in frames. One value per src spell."""
    out = []
    for team in (H, A):
        s = src_f[src_team == team]
        d = np.sort(dst_f[dst_team == team])
        if len(s) == 0 or len(d) == 0:
            continue
        pos = np.searchsorted(d, s)
        for i, f in enumerate(s):
            j = int(pos[i])
            best = None
            for k in (j - 1, j):
                if 0 <= k < len(d):
                    lag = int(f) - int(d[k])
                    if best is None or abs(lag) < abs(best):
                        best = lag
            if best is not None:
                out.append(best)
    return np.asarray(out, np.float64)


def length_stats(lengths_s):
    if len(lengths_s) == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "lt4": 0.0,
                "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    q = np.percentile(lengths_s, [10, 25, 50, 75, 90])
    return {
        "n": int(len(lengths_s)),
        "mean": float(lengths_s.mean()),
        "median": float(np.median(lengths_s)),
        "lt4": float(np.mean(lengths_s < 4.0)),
        "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p90": float(q[4]),
    }


def frame_agreement(mech, prov):
    alive = prov != D
    out = {
        "agree_all": float(np.mean(mech == prov)) if len(mech) else 0.0,
        "agree_alive": float(np.mean(mech[alive] == prov[alive])) if alive.any() else 0.0,
        "d_share": float(np.mean(mech == D)) if len(mech) else 0.0,
    }
    for code, name in ((H, "H"), (A, "A"), (D, "D")):
        m = prov == code
        out["recall_" + name] = float(np.mean(mech[m] == code)) if m.any() else 0.0
    return out


def _blank(n, period=1):
    return {
        "period": np.full(n, period, np.int8),
        "ball": np.zeros((n, 3), np.float32),
        "alive": np.ones(n, dtype=bool),
        "owner": np.zeros(n, np.int8),
        "players": np.full((n, 2, 4, 2), np.nan, np.float32),
    }


def _expect(name, cond, detail):
    if not cond:
        raise AssertionError("{} ({})".format(name, detail))
    print("PASS", name)


def _selftest():
    # Direct state machine: home control, then a long away control. Flip after 25.
    control = np.array([H] * 40 + [A] * 40, np.int8)
    alive = np.ones(80, dtype=bool)
    owners = _possess(control, alive, 1.0)
    owners = owners.copy()
    _close_owners(owners, control)
    codes = _to_codes(owners)
    _expect("steal flips at T_c, not at first contact",
            np.all(codes[:64] == H) and np.all(codes[64:] == A),
            "A frames {}".format(int((codes == A).sum())))

    # Short contact does not flip.
    control = np.array([H] * 40 + [A] * 8 + [-1] * 20, np.int8)
    alive = np.ones(len(control), dtype=bool)
    owners = _possess(control, alive, 1.0)
    _close_owners(owners, control)
    codes = _to_codes(owners)
    _expect("touch shorter than T_c keeps the owner", np.all(codes == H),
            "A frames {}".format(int((codes == A).sum())))

    # Gap of 10 frames inside an away attempt still completes.
    control = np.array([H] * 30 + [A] * 15 + [-1] * 10 + [A] * 15, np.int8)
    alive = np.ones(len(control), dtype=bool)
    owners = _possess(control, alive, 1.0)
    _close_owners(owners, control)
    codes = _to_codes(owners)
    _expect("gap under 1 s does not reset the clock", codes[63] == H and codes[64] == A,
            "first A at {}".format(int(np.argmax(codes == A)) if (codes == A).any() else -1))

    # Gap of 30 frames resets. Second streak is only 15 frames.
    control = np.array([H] * 30 + [A] * 15 + [-1] * 30 + [A] * 15, np.int8)
    alive = np.ones(len(control), dtype=bool)
    owners = _possess(control, alive, 1.0)
    _close_owners(owners, control)
    codes = _to_codes(owners)
    _expect("gap over 1 s cancels the attempt", np.all(codes == H),
            "A frames {}".format(int((codes == A).sum())))

    # Dead ball discards the owner; the restart team is back-filled.
    control = np.array([H] * 30 + [-1] * 10 + [A] * 40, np.int8)
    alive = np.array([True] * 30 + [False] * 10 + [True] * 40)
    owners = _possess(control, alive, 1.0)
    _close_owners(owners, control)
    codes = _to_codes(owners)
    _expect("dead discards the owner and the restart is the new team",
            np.all(codes[:30] == H) and np.all(codes[30:40] == D) and np.all(codes[40:] == A),
            "D {} A {}".format(int((codes == D).sum()), int((codes == A).sum())))

    # T_c = 0 flips on the first confirmed frame and then sticks.
    control = np.array([H] * 10 + [A] * 5 + [-1] * 10, np.int8)
    alive = np.ones(len(control), dtype=bool)
    owners = _possess(control, alive, 0.0)
    _close_owners(owners, control)
    codes = _to_codes(owners)
    _expect("T_c 0 flips on the confirming frame and sticks",
            np.all(codes[:10] == H) and np.all(codes[10:] == A),
            "A frames {}".format(int((codes == A).sum())))

    # Full pipeline: stopped ball at a home player's feet.
    m = _blank(50)
    m["ball"][:, 0] = 40.0
    m["ball"][:, 1] = 34.0
    m["players"][:, 0, 0] = (40.2, 34.0)
    m["players"][:, 1, 0] = (90.0, 10.0)
    codes = label_match(m, r_p=1.0, t_c=1.0)
    _expect("resting ball is home possession", np.all(codes == H) and np.all((codes == D) == ~m["alive"]),
            "H {}".format(int((codes == H).sum())))

    # Loose ball keeps the owner. D only where alive is false.
    m = _blank(80)
    m["ball"][:40, 0] = 40.0
    m["ball"][:40, 1] = 34.0
    m["ball"][40:, 0] = 10.0
    m["ball"][40:, 1] = 10.0
    m["players"][:, 0, 0] = (40.2, 34.0)
    m["players"][:, 1, 0] = (90.0, 10.0)
    m["alive"][60:70] = False
    codes = label_match(m, r_p=1.0, t_c=1.0)
    _expect("loose ball and only the dead flag become D",
            np.all(codes[:60] == H) and np.all(codes[60:70] == D) and np.all(codes[70:] == H),
            "D {}".format(int((codes == D).sum())))

    # Radius: 1.2 m is outside 1.0 and inside 1.5. The ball stays put so smoothing
# does not delay the moment the opponent steps into the zone.
    m = _blank(80)
    m["ball"][:, 0] = 40.0
    m["ball"][:, 1] = 34.0
    m["players"][:31, 0, 0] = (40.2, 34.0)
    m["players"][31:, 0, 0] = (80.0, 10.0)
    m["players"][:, 1, 0] = (90.0, 60.0)
    m["players"][31:, 1, 0] = (41.2, 34.0)
    c1 = label_match(m, r_p=1.0, t_c=1.0)
    c15 = label_match(m, r_p=1.5, t_c=1.0)
    _expect("r_p 1.0 ignores a player at 1.2 m", np.all(c1 == H),
            "A {}".format(int((c1 == A).sum())))
    _expect("r_p 1.5 accepts a player at 1.2 m after T_c",
            c15[54] == H and c15[55] == A and c15[-1] == A,
            "first A {}".format(int(np.argmax(c15 == A))))

    # Straight constant-speed carry is not a touch, so the owner stays.
    m = _blank(100)
    m["ball"][:30, 0] = 20.0
    m["ball"][:30, 1] = 34.0
    m["players"][:30, 0, 0] = (20.2, 34.0)
    t = np.arange(70)
    m["ball"][30:, 0] = 20.0 + 0.32 * t
    m["ball"][30:, 1] = 34.0
    m["players"][30:, 0, 0] = (20.0, 34.0)
    m["players"][:, 1, 0] = (80.0, 60.0)
    m["players"][50:, 1, 0, 0] = m["ball"][50:, 0] + 0.2
    m["players"][50:, 1, 0, 1] = 34.0
    c = label_match(m, r_p=1.0, t_c=1.0)
    _expect("straight constant-speed run is not a gain", np.all(c == H),
            "A {}".format(int((c == A).sum())))

    # Same geometry with a speed step while the opponent is next to the ball.
    m = _blank(100)
    m["ball"][:30, 0] = 20.0
    m["ball"][:30, 1] = 34.0
    m["players"][:30, 0, 0] = (20.2, 34.0)
    x = np.empty(70, np.float32)
    x[0] = 20.0
    step = np.full(69, 0.32, np.float32)
    step[20:26] = np.linspace(0.32, 0.80, 6)
    step[26:] = 0.80
    x[1:] = 20.0 + np.cumsum(step)
    m["ball"][30:, 0] = x
    m["ball"][30:, 1] = 34.0
    m["players"][30:, 0, 0] = (20.0, 34.0)
    m["players"][:, 1, 0] = (100.0, 60.0)
    m["players"][40:, 1, 0, 0] = m["ball"][40:, 0] + 0.2
    m["players"][40:, 1, 0, 1] = 34.0
    c, info = label_match(m, r_p=1.0, t_c=1.0, return_info=True)
    _expect("speed change near the opponent flips after T_c",
            (c == A).any() and c[-1] == A and info["by_speed"] > 0,
            "A {} speed-confirmed runs {}".format(int((c == A).sum()), info["by_speed"]))

    # Direction change: the opponent is in the zone on the way in and the way out.
    m = _blank(100)
    m["ball"][:30, 0] = 20.0
    m["ball"][:30, 1] = 20.0
    m["players"][:30, 0, 0] = (20.2, 20.0)
    m["players"][:, 1, 0] = (80.0, 50.0)
    for t in range(30, 50):
        m["ball"][t, 0] = 20.0 + 0.40 * (t - 30)
        m["ball"][t, 1] = 20.0
    for t in range(50, 100):
        m["ball"][t, 0] = m["ball"][49, 0]
        m["ball"][t, 1] = 20.0 + 0.40 * (t - 49)
    m["players"][30:, 0, 0] = (20.0, 20.0)
    # Away stands on the corner, so the zone run covers the last +x frames and the +y leg.
    m["players"][40:, 1, 0, 0] = m["ball"][49, 0]
    m["players"][40:, 1, 0, 1] = 20.0
    # Keep them on the ball once it has turned, so the run is long enough for T_c.
    m["players"][50:, 1, 0, 0] = m["ball"][50:, 0] + 0.2
    m["players"][50:, 1, 0, 1] = m["ball"][50:, 1]
    c, info = label_match(m, r_p=1.0, t_c=1.0, return_info=True)
    _expect("direction change near the opponent flips after T_c",
            c[-1] == A and info["by_angle"] > 0,
            "A {} angle-confirmed runs {}".format(int((c == A).sum()), info["by_angle"]))

    # Home owns, then both opponents are inside 1 m with away closer. Duel: no flip.
    m = _blank(80)
    m["ball"][:, 0] = 50.0
    m["ball"][:, 1] = 34.0
    m["players"][:30, 0, 0] = (50.2, 34.0)
    m["players"][:30, 1, 0] = (90.0, 10.0)
    m["players"][30:, 0, 0] = (50.6, 34.0)
    m["players"][30:, 1, 0] = (50.15, 34.0)
    c = label_match(m, r_p=1.0, t_c=0.0)
    _expect("a duel does not change the owner", np.all(c == H),
            "A {}".format(int((c == A).sum())))

    # Halves do not inherit an owner.
    m = _blank(60)
    m["period"][30:] = 2
    m["ball"][:, 0] = 40.0
    m["ball"][:, 1] = 34.0
    m["players"][:30, 0, 0] = (40.2, 34.0)
    m["players"][:30, 1, 0] = (90.0, 10.0)
    m["players"][30:, 0, 0] = (90.0, 10.0)
    m["players"][30:, 1, 0] = (40.2, 34.0)
    c = label_match(m, r_p=1.0, t_c=1.0)
    _expect("a new period starts a new owner", np.all(c[:30] == H) and np.all(c[30:] == A),
            "second-half H {}".format(int((c[30:] == H).sum())))

    # Nobody ever in the zone: fallback home, and it is not dead.
    m = _blank(20)
    m["ball"][:, 0] = 50.0
    m["ball"][:, 1] = 34.0
    c, info = label_match(m, return_info=True)
    _expect("no control falls back without inventing a dead ball",
            np.all(c == H) and info["fallback_frames"] == 20,
            "fallback {}".format(info["fallback_frames"]))

    # A gap in the track is not a touch, even at T_c 0 next to an opponent.
    m = _blank(80)
    m["ball"][:30, 0] = 20.0
    m["ball"][:30, 1] = 34.0
    m["players"][:30, 0, 0] = (20.2, 34.0)
    m["players"][:, 1, 0] = (80.0, 50.0)
    m["ball"][30:50] = np.nan
    t = np.arange(5)
    m["ball"][50:55, 0] = 70.0 + 0.40 * t
    m["ball"][50:55, 1] = 34.0
    m["players"][50:55, 1, 0, 0] = m["ball"][50:55, 0] + 0.2
    m["players"][50:55, 1, 0, 1] = 34.0
    m["ball"][55:, 0] = 90.0
    m["ball"][55:, 1] = 10.0
    c = label_match(m, r_p=1.0, t_c=0.0, impute=True)
    _expect("reappearing ball is not a touch by itself", np.all(c == H),
            "A {}".format(int((c == A).sum())))

    # State features ignore the provider flags.
    m = _blank(40)
    m["ball"][:, 0] = 40.0
    m["ball"][:, 1] = 34.0
    m["players"][:, 0, 0] = (40.2, 34.0)
    m["players"][:, 1, 0] = (90.0, 10.0)
    a, names = mechanical_state(m)
    m2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in m.items()}
    m2["alive"] = np.zeros(40, dtype=bool)
    m2["owner"] = np.ones(40, np.int8)
    b, _ = mechanical_state(m2)
    _expect("mechanical state ignores alive and owner flags",
            a.shape == (40, 4) and names[0] == "mech_owner_home" and np.allclose(a, b),
            "max abs diff {}".format(float(np.max(np.abs(a - b)))))

    c1 = label_match(m, r_p=1.0, t_c=1.0)
    c2 = label_match(m, r_p=1.0, t_c=1.0)
    _expect("labeller is deterministic", np.array_equal(c1, c2), "")
    print("selftest ok")


if __name__ == "__main__":
    _selftest()
