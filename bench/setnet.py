"""SetNet: an end-to-end player-set neural model for per-frame team possession (H/A/D).

Bench protocol (docs/team/BENCH.md): open DFL/IDSSE matches degraded with bench.degrade
(PLAYER_SMOOTH = 12, calibrated on real aggregates), provider labels (bench.run.labels),
leave-one-match-out on TRAIN for epochs and switch costs, TEST scored once with the official
scorer (evaluation/possession_metrics.py through bench.run.score).

Per 5 Hz step the model sees:
  * the set of tracked players of both teams (positions canonicalised so the home side attacks
    +x, the attack direction being inferred per period from player positions with
    bench.feats3._attack_sign; velocities from centred differences of the smoothed tracks;
    a team sign; distance to the nearest team-mate and opponent; offsets to the last and next
    known ball position),
  * one ball token (last / next known position, visible flag, seconds since last seen and until
    next seen, ball velocity when visible).
Missing slots are masked; duplicates stay in (the real tracking has them too).
A set transformer (SAB blocks with a learnable CLS token) embeds each step, a temporal module
(BiGRU, dilated TCN or none) runs over a 60 s window and a linear head gives H/A/D logits.
Team symmetry: H/A swap augmentation in training and averaging of both views at inference.
Nothing here reads the alive or owner flags except as training targets (bench.run.labels).
"""

import hashlib
import os
from types import SimpleNamespace

import numpy as np

H, A, D = 0, 1, 2
SWAP = np.array([A, H, D])
FPS = 25
STEP = 5                 # 25 Hz -> 5 Hz
PL, PW = 105.0, 68.0
PLAYER_SMOOTH = 12       # brief: calibrated against real aggregates (research/runs/decouple/CALIB.txt)
VEL_K = 2                # centred difference over +-2 frames
KMAX = 26                # tracked objects kept per frame (11 + 1 duplicate per team, plus margin)
FP = 16                  # player-token features
FB = 12                  # ball-token features

TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
TEST = ["J03WPY", "J03WQQ", "J03WR9"]
ORDER = TRAIN + TEST


# -- degraded data -----------------------------------------------------------------------------

def degraded(match_id, seed):
    """Degraded copy and provider labels for one bench match. seed rule = 1000*seed + index."""
    from bench import dfl
    from bench import degrade as dg
    from bench.run import labels
    dg.PLAYER_SMOOTH = PLAYER_SMOOTH
    index = ORDER.index(match_id)
    clean = dfl.load(match_id)
    deg = dg.degrade(clean, seed=1000 * int(seed) + index)
    y = labels(clean)
    del clean
    return deg, y


def digest(deg):
    """Short hash of the degraded inputs, to check that two processes saw the same copy."""
    h = hashlib.sha1()
    h.update(np.nan_to_num(np.asarray(deg["ball"], np.float32), nan=-999.0).tobytes())
    h.update(np.nan_to_num(np.asarray(deg["players"], np.float32), nan=-999.0).tobytes())
    return h.hexdigest()[:16]


def central_velocity(arr, period, k, vmax):
    """arr (n, K, 2) with NaN gaps -> (velocity (n, K, 2) m/s, ok (n, K)). Inside periods only."""
    n = arr.shape[0]
    vel = np.zeros(arr.shape, np.float32)
    ok = np.zeros(arr.shape[:2], bool)
    if n <= 2 * k:
        return vel, ok
    step = arr[2 * k:] - arr[:-2 * k]
    same = (period[2 * k:] == period[:-2 * k])[:, None]
    good = np.isfinite(step).all(axis=-1) & same
    v = np.where(good[..., None], step * (FPS / (2.0 * k)), 0.0)
    sp = np.linalg.norm(v, axis=-1)
    good &= sp <= vmax
    v = np.where(good[..., None], v, 0.0)
    vel[k:n - k] = v
    ok[k:n - k] = good
    return vel, ok


def prepare(deg, y):
    """Compact per-frame arrays for the set model. Reads players, ball and period only."""
    from bench.feats3 import _attack_sign, clocks
    players = np.asarray(deg["players"], np.float32)
    n, _, S, _ = players.shape
    period = np.asarray(deg["period"])
    sign = _attack_sign(players, period)          # (n, 2); sign[:, 0] = +1 -> home attacks +x
    flip = sign[:, 0] < 0
    flat = players.reshape(n, 2 * S, 2).copy()
    team_slot = np.repeat(np.array([0, 1], np.int8), S)
    ball = np.asarray(deg["ball"][:, :2], np.float32).copy()
    if flip.any():
        flat[flip, :, 0] = PL - flat[flip, :, 0]
        flat[flip, :, 1] = PW - flat[flip, :, 1]
        ball[flip, 0] = PL - ball[flip, 0]
        ball[flip, 1] = PW - ball[flip, 1]
    vel, vok = central_velocity(flat, period, VEL_K, 12.0)
    bvel, _ = central_velocity(ball[:, None, :], period, VEL_K, 40.0)
    valid = np.isfinite(flat[..., 0])
    count = valid.sum(axis=1)
    order = np.argsort(~valid, axis=1, kind="stable")[:, :KMAX]      # valid slots first
    tok_valid = np.take_along_axis(valid, order, axis=1)
    pos = np.take_along_axis(flat, order[..., None], axis=1)
    vel = np.take_along_axis(vel, order[..., None], axis=1)
    vok = np.take_along_axis(vok, order, axis=1) & tok_valid
    team = np.where(tok_valid, team_slot[order], -1).astype(np.int8)
    pos = np.where(tok_valid[..., None], pos, 0.0).astype(np.float32)
    vel = np.where(tok_valid[..., None], vel, 0.0).astype(np.float32)
    vis, _, _, last, nxt = clocks(deg)
    return {
        "pos": pos, "vel": vel, "vok": vok, "team": team,
        "ball": np.nan_to_num(ball, nan=0.0).astype(np.float32),
        "bvel": bvel[:, 0].astype(np.float32),
        "vis": np.asarray(vis, bool), "last": np.asarray(last, np.int32),
        "nxt": np.asarray(nxt, np.int32), "period": period.astype(np.int8),
        "y": np.asarray(y, np.int8),
        "n_over_k": np.int64((count > KMAX).sum()), "max_count": np.int64(count.max()),
        "flip_share": np.float64(flip.mean()),
    }


def prep_path(cache, seed, match):
    return os.path.join(cache, "prep", "s%d_%s.npz" % (seed, match))


def load_prep(cache, seed, match, log=None):
    """Prepared arrays for (seed, match); built and cached on first use."""
    path = prep_path(cache, seed, match)
    if not os.path.isfile(path):
        deg, y = degraded(match, seed)
        dig = digest(deg)
        p = prepare(deg, y)
        p["digest"] = np.array(dig)
        del deg
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **p)
        os.replace(tmp, path)
        if log:
            log("prep s%d %s: %d frames, max objects %d, frames over K %d, digest %s" % (
                seed, match, len(y), int(p["max_count"]), int(p["n_over_k"]), dig))
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def periods_of(period):
    """[(start, stop)] half-open per period, in frame order."""
    out = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


# -- decoding and scoring ------------------------------------------------------------------------

_VIT = None


def _compile_viterbi():
    from numba import njit

    @njit(cache=False)
    def _vit(e, st, sd):
        n = e.shape[0]
        back = np.empty((n, 3), np.int8)
        s0 = e[0, 0]
        s1 = e[0, 1]
        s2 = e[0, 2]
        for t in range(1, n):
            a0 = s0
            a1 = s1
            a2 = s2
            # to H
            src = 0
            best = a0
            if a1 - st > best:
                src = 1
                best = a1 - st
            if a2 - sd > best:
                src = 2
                best = a2 - sd
            back[t, 0] = src
            s0 = best + e[t, 0]
            # to A
            src = 0
            best = a0 - st
            if a1 > best:
                src = 1
                best = a1
            if a2 - sd > best:
                src = 2
                best = a2 - sd
            back[t, 1] = src
            s1 = best + e[t, 1]
            # to D
            src = 0
            best = a0 - sd
            if a1 - sd > best:
                src = 1
                best = a1 - sd
            if a2 > best:
                src = 2
                best = a2
            back[t, 2] = src
            s2 = best + e[t, 2]
        last = 0
        bests = s0
        if s1 > bests:
            last = 1
            bests = s1
        if s2 > bests:
            last = 2
        path = np.empty(n, np.int8)
        path[n - 1] = last
        for t in range(n - 1, 0, -1):
            last = back[t, last]
            path[t - 1] = last
        return path

    return _vit


def viterbi(e, switch_team, switch_dead):
    """Same 3-state decode as bench.methods.viterbi (ties keep the lower state), numba-compiled."""
    global _VIT
    e = np.ascontiguousarray(e, dtype=np.float64)
    if len(e) == 0:
        return np.zeros(0, np.int8)
    if _VIT is None:
        _VIT = _compile_viterbi()
    return _VIT(e, float(switch_team), float(switch_dead))


def decode_periods(e, period, switch_team, switch_dead):
    out = np.full(len(e), D, np.int8)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = viterbi(e[idx], switch_team, switch_dead)
    return out


def check_viterbi():
    """Parity with bench.methods.viterbi on random emissions (with planted ties)."""
    from bench import methods
    rng = np.random.default_rng(0)
    e = rng.normal(size=(600, 3))
    e[10] = 0.0
    e[11] = 1.0
    e[200:230] = 0.5
    for st, sd in ((0.0, 0.0), (3.0, 3.0), (5.0, 40.0), (80.0, 10.0), (1.5, 0.7)):
        if not np.array_equal(viterbi(e, st, sd), methods.viterbi(e, st, sd)):
            raise AssertionError("viterbi mismatch at %s %s" % (st, sd))
    return True


def score(y, pred, period):
    """Official scorer via bench.run.score: (spellF1, macroF1, n_pred_spells, n_gt_spells)."""
    from bench.run import score as bench_score
    s, m, npred, ngt = bench_score(y, pred, period)
    return float(s), float(m), int(npred), int(ngt)


def tune_costs(ems, ys, periods, grid_t, grid_d):
    """Switch costs maximising mean spellF1 over the given (out-of-fold) matches.
    Ties keep the earlier grid point. Returns (mean spellF1, team cost, dead cost)."""
    best = None
    for st in grid_t:
        for sd in grid_d:
            s = [score(y, decode_periods(e, p, st, sd), p)[0] for e, y, p in zip(ems, ys, periods)]
            m = float(np.mean(s))
            if best is None or m > best[0]:
                best = (m, float(st), float(sd))
    return best


def emissions_from_proba(p, prior):
    return (np.log(np.asarray(p, np.float64) + 1e-6) - np.log(np.asarray(prior, np.float64))).astype(np.float32)


# -- torch model ---------------------------------------------------------------------------------

_TP = None


def torch_parts():
    """Torch classes, defined lazily so the CPU-only tools never import torch."""
    global _TP
    if _TP is not None:
        return _TP
    import torch
    from torch import nn
    import torch.nn.functional as Fn

    class SAB(nn.Module):
        """Pre-norm set-attention block. Plain matmul attention (deterministic), key mask."""

        def __init__(self, d, heads, drop):
            super().__init__()
            self.h = heads
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, keymask):
            n, L, d = x.shape
            q, k, v = self.qkv(self.ln1(x)).view(n, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
            att = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / np.sqrt(d // self.h))
            att = att.float().masked_fill(~keymask[:, None, None, :], -1e9)
            att = torch.softmax(att, dim=-1).to(v.dtype)
            o = torch.matmul(att, v).transpose(1, 2).reshape(n, L, d)
            x = x + self.drop(self.proj(o))
            x = x + self.drop(self.ff(self.ln2(x)))
            return x

    class TCN(nn.Module):
        def __init__(self, d, dilations=(1, 2, 4, 8, 16, 32, 64), drop=0.1):
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, padding=r, dilation=r) for r in dilations])
            self.norms = nn.ModuleList([nn.GroupNorm(1, d) for _ in dilations])
            self.drop = nn.Dropout(drop)

        def forward(self, z):            # (B, T, d)
            h = z.transpose(1, 2)
            for conv, norm in zip(self.convs, self.norms):
                h = h + self.drop(Fn.gelu(norm(conv(h))))
            return h.transpose(1, 2)

    class SetNet(nn.Module):
        def __init__(self, d=64, heads=4, n_sab=2, temporal="gru", hidden=64, rnn_layers=1,
                     drop=0.1):
            super().__init__()
            self.d = d
            self.temporal = temporal
            self.pin = nn.Sequential(nn.Linear(FP, d), nn.GELU(), nn.Linear(d, d))
            self.bin = nn.Sequential(nn.Linear(FB, d), nn.GELU(), nn.Linear(d, d))
            self.team_vec = nn.Parameter(torch.randn(d) * 0.2)
            self.ball_type = nn.Parameter(torch.randn(d) * 0.2)
            self.cls = nn.Parameter(torch.randn(d) * 0.2)
            self.sabs = nn.ModuleList([SAB(d, heads, drop) for _ in range(n_sab)])
            self.lnf = nn.LayerNorm(d)
            if temporal == "gru":
                self.rnn = nn.GRU(d, hidden, num_layers=rnn_layers, batch_first=True,
                                  bidirectional=True, dropout=drop if rnn_layers > 1 else 0.0)
                out = 2 * hidden
            elif temporal == "tcn":
                self.tcn = TCN(d, drop=drop)
                out = d
            elif temporal == "none":
                out = d
            else:
                raise ValueError(temporal)
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(out, 3))

        def forward(self, ptok, pmask, btok, bmask, ts):
            B, T, K, _ = ptok.shape
            hp = self.pin(ptok) + ts[..., None] * self.team_vec
            hb = self.bin(btok) + self.ball_type
            cls = self.cls.view(1, 1, 1, -1).expand(B, T, 1, self.d)
            x = torch.cat([cls, hb[:, :, None, :], hp], dim=2).reshape(B * T, K + 2, self.d)
            km = torch.cat([torch.ones(B, T, 1, dtype=torch.bool, device=ptok.device),
                            bmask[:, :, None], pmask], dim=2).reshape(B * T, K + 2)
            for sab in self.sabs:
                x = sab(x, km)
            z = self.lnf(x[:, 0]).reshape(B, T, self.d)
            if self.temporal == "gru":
                with torch.autocast(device_type=z.device.type, enabled=False):
                    z, _ = self.rnn(z.float())
            elif self.temporal == "tcn":
                z = self.tcn(z)
            return self.head(z)

    _TP = SimpleNamespace(torch=torch, nn=nn, F=Fn, SAB=SAB, TCN=TCN, SetNet=SetNet)
    return _TP


class Bank:
    """Prepared matches of one degradation seed, concatenated on the device."""

    def __init__(self, preps, names, device):
        tp = torch_parts()
        torch = tp.torch
        self.device = device
        self.names = list(names)
        self.offset = {}
        self.periods = {}
        self.length = {}
        off = 0
        lasts, nxts = [], []
        for name, p in zip(names, preps):
            n = len(p["y"])
            self.offset[name] = off
            self.length[name] = n
            self.periods[name] = [(off + a, off + b) for a, b in periods_of(p["period"])]
            lasts.append(np.where(p["last"] >= 0, p["last"].astype(np.int64) + off, -1))
            nxts.append(np.where(p["nxt"] >= 0, p["nxt"].astype(np.int64) + off, -1))
            off += n
        self.n = off

        def cat(key, dtype=None):
            a = np.concatenate([p[key] for p in preps])
            if dtype is not None:
                a = a.astype(dtype)
            return torch.from_numpy(a).to(device)

        self.pos = cat("pos")
        self.vel = cat("vel")
        self.vok = cat("vok")
        self.team = cat("team", np.int64)
        self.ball = cat("ball")
        self.bvel = cat("bvel")
        self.vis = cat("vis")
        self.y = cat("y", np.int64)
        self.last = torch.from_numpy(np.concatenate(lasts)).to(device)
        self.nxt = torch.from_numpy(np.concatenate(nxts)).to(device)
        self.swap_lut = torch.tensor(SWAP, dtype=torch.int64, device=device)


def make_inputs(bank, starts, pstarts, pends, T, use_ball=True, swap=None, mirror=None,
                mask_a=None, mask_b=None):
    """Model inputs for windows of T steps at 5 Hz starting at global frames `starts` (B,).

    swap / mirror: (B,) bool tensors (team exchange = 180 degree rotation + team sign flip +
    label swap; mirror = y reflection). mask_a / mask_b: (B,) frame span of extra ball hiding
    (a == b: none). Returns a dict of device tensors.
    """
    torch = torch_parts().torch
    dev = bank.device
    B = starts.shape[0]
    ar = torch.arange(T, device=dev, dtype=torch.int64) * STEP
    F = starts[:, None] + ar[None, :]
    stepv = F < pends[:, None]
    F = torch.minimum(F, pends[:, None] - 1)
    pos = bank.pos[F]                       # (B, T, K, 2)
    vel = bank.vel[F]
    vok = bank.vok[F]
    team = bank.team[F]                     # -1 pad
    valid = team >= 0
    ts = torch.where(team == 0, 1.0, -1.0) * valid
    y = bank.y[F]

    if use_ball:
        vis = bank.vis[F]
        last = bank.last[F]
        nxt = bank.nxt[F]
        if mask_a is not None:
            a = mask_a[:, None]
            b = mask_b[:, None]
            lb = torch.where(a - 1 >= pstarts[:, None], bank.last[(a - 1).clamp(min=0)],
                             torch.full_like(a, -1))
            nb = torch.where(b < pends[:, None], bank.nxt[b.clamp(max=bank.n - 1)],
                             torch.full_like(b, -1))
            last = torch.where((last >= a) & (last < b), lb.expand_as(last), last)
            nxt = torch.where((nxt >= a) & (nxt < b), nb.expand_as(nxt), nxt)
            vis = vis & ~((F >= a) & (F < b))
        kl = last >= 0
        kn = nxt >= 0
        bl = bank.ball[last.clamp(min=0)]   # (B, T, 2)
        bn = bank.ball[nxt.clamp(min=0)]
        age_l = torch.where(kl, (F - last).float() / FPS, torch.full_like(F, 600).float())
        age_n = torch.where(kn, (nxt - F).float() / FPS, torch.full_like(F, 600).float())
        bv = torch.where(vis[..., None], bank.bvel[F], torch.zeros_like(bank.bvel[F]))

    # geometric augmentation: swap = 180 degree rotation, mirror = y reflection
    if swap is not None or mirror is not None:
        sw = swap if swap is not None else torch.zeros(B, dtype=torch.bool, device=dev)
        mi = mirror if mirror is not None else torch.zeros(B, dtype=torch.bool, device=dev)
        fx = sw
        fy = sw ^ mi

        def tpos(xy, nd):
            shape = (B,) + (1,) * (nd - 1)
            x = torch.where(fx.view(shape), PL - xy[..., 0], xy[..., 0])
            yy = torch.where(fy.view(shape), PW - xy[..., 1], xy[..., 1])
            return torch.stack([x, yy], dim=-1)

        def tvel(v, nd):
            shape = (B,) + (1,) * (nd - 1)
            x = torch.where(fx.view(shape), -v[..., 0], v[..., 0])
            yy = torch.where(fy.view(shape), -v[..., 1], v[..., 1])
            return torch.stack([x, yy], dim=-1)

        pos = tpos(pos, 3)
        vel = tvel(vel, 3)
        ts = torch.where(sw.view(B, 1, 1), -ts, ts)
        y = torch.where(sw.view(B, 1), bank.swap_lut[y], y)
        if use_ball:
            bl = tpos(bl, 2)
            bn = tpos(bn, 2)
            bv = tvel(bv, 2)

    validf = valid.float()
    feats = [
        pos[..., 0:1] / 52.5 - 1.0, pos[..., 1:2] / 34.0 - 1.0,
        vel / 5.0, torch.linalg.norm(vel, dim=-1, keepdim=True) / 5.0, vok.float()[..., None],
    ]
    if use_ball:
        rl = pos - bl[:, :, None, :]
        rn = pos - bn[:, :, None, :]
        klf = kl.float()[:, :, None, None]
        knf = kn.float()[:, :, None, None]
        feats += [
            (rl / 20.0).clamp(-4, 4) * klf,
            (torch.linalg.norm(rl, dim=-1, keepdim=True) / 20.0).clamp(0, 4) * klf,
            (rn / 20.0).clamp(-4, 4) * knf,
            (torch.linalg.norm(rn, dim=-1, keepdim=True) / 20.0).clamp(0, 4) * knf,
            klf.expand(B, T, pos.shape[2], 1), knf.expand(B, T, pos.shape[2], 1),
        ]
    else:
        feats.append(torch.zeros(B, T, pos.shape[2], 8, device=dev))
    # nearest team-mate / opponent (duplicates show up as a very close team-mate)
    dist = torch.cdist(pos.reshape(B * T, -1, 2), pos.reshape(B * T, -1, 2)).reshape(
        B, T, pos.shape[2], pos.shape[2])
    same = (team[..., :, None] == team[..., None, :])
    vj = valid[..., None, :]
    eye = torch.eye(pos.shape[2], dtype=torch.bool, device=dev)
    big = torch.full_like(dist, 300.0)
    d_team = torch.where(same & vj & ~eye, dist, big).amin(dim=-1)
    d_opp = torch.where(~same & vj, dist, big).amin(dim=-1)
    feats += [(d_team / 10.0).clamp(0, 3)[..., None], (d_opp / 10.0).clamp(0, 3)[..., None]]
    ptok = torch.cat(feats, dim=-1) * validf[..., None]

    if use_ball:
        btok = torch.cat([
            (bl[..., 0:1] / 52.5 - 1.0) * kl.float()[..., None],
            (bl[..., 1:2] / 34.0 - 1.0) * kl.float()[..., None],
            (bn[..., 0:1] / 52.5 - 1.0) * kn.float()[..., None],
            (bn[..., 1:2] / 34.0 - 1.0) * kn.float()[..., None],
            kl.float()[..., None], kn.float()[..., None], vis.float()[..., None],
            torch.log1p(age_l)[..., None] / 3.0, torch.log1p(age_n)[..., None] / 3.0,
            bv / 10.0, torch.linalg.norm(bv, dim=-1, keepdim=True) / 10.0,
        ], dim=-1)
        bmask = torch.ones(B, T, dtype=torch.bool, device=dev)
    else:
        btok = torch.zeros(B, T, FB, device=dev)
        bmask = torch.zeros(B, T, dtype=torch.bool, device=dev)
    return {"ptok": ptok, "pmask": valid, "btok": btok, "bmask": bmask, "ts": ts.float(),
            "y": y, "stepv": stepv}


# -- training ------------------------------------------------------------------------------------

DEFAULT_CFG = dict(
    d=64, heads=4, n_sab=2, temporal="gru", hidden=64, rnn_layers=1, drop=0.1,
    T=300, batch=16, lr=2e-3, wd=0.01, gamma=0.95, warmup=1, label_smooth=0.05,
    use_ball=True, p_swap=0.5, p_mirror=0.5, p_mask=0.5, boundary_w=1.0,
    epochs=60, ckpt_every=6, ema=0.98, hop=150, tta="swap_mirror", infer_batch=64,
)


def sample_windows(bank, matches, T, rng):
    """One epoch: about one window per T*STEP frames of each period, random starts (any phase)."""
    span = T * STEP
    out = []
    for m in matches:
        for a, b in bank.periods[m]:
            length = b - a
            k = max(1, int(round(length / float(span))))
            hi = max(a, b - span)
            starts = rng.integers(a, hi + 1, size=k)
            out += [(int(s), a, b) for s in starts]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def boundary_weight(y, stepv, w):
    """Weight w on steps within +-5 steps (1 s) of a label change inside a window."""
    torch = torch_parts().torch
    ch = torch.zeros_like(y, dtype=torch.float32)
    ch[:, 1:] = (y[:, 1:] != y[:, :-1]).float()
    k = torch.ones(1, 1, 11, device=y.device)
    dil = torch.nn.functional.conv1d(ch[:, None], k, padding=5)[:, 0]
    return torch.where(dil > 0, torch.full_like(ch, float(w)), torch.ones_like(ch)) * stepv.float()


class EMA:
    def __init__(self, model, decay):
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        for mod in self.model.modules():
            if hasattr(mod, "flatten_parameters"):
                mod.flatten_parameters()

    def update(self, model):
        torch = torch_parts().torch
        with torch.no_grad():
            for pe, pm in zip(self.model.parameters(), model.parameters()):
                pe.mul_(self.decay).add_(pm.detach(), alpha=1.0 - self.decay)
            for be, bm in zip(self.model.buffers(), model.buffers()):
                be.copy_(bm)


def train(bank, train_matches, cfg, seed, epochs, on_ckpt=None, log=None):
    """Train one SetNet on `train_matches` of the bank. on_ckpt(model, epoch) is called with the
    EMA weights at every cfg['ckpt_every'] epochs (and at the last epoch). Returns the EMA model."""
    tp = torch_parts()
    torch = tp.torch
    dev = bank.device
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(10007 * int(seed) + 17)
    model = tp.SetNet(d=cfg["d"], heads=cfg["heads"], n_sab=cfg["n_sab"], temporal=cfg["temporal"],
                      hidden=cfg["hidden"], rnn_layers=cfg["rnn_layers"], drop=cfg["drop"]).to(dev)
    ema = EMA(model, cfg["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    T = cfg["T"]
    span = T * STEP
    step_i = 0
    use_amp = dev.type == "cuda"
    for epoch in range(1, epochs + 1):
        model.train()
        wins = sample_windows(bank, train_matches, T, rng)
        nb = (len(wins) + cfg["batch"] - 1) // cfg["batch"]
        tot, cnt = 0.0, 0
        for bi in range(nb):
            chunk = wins[bi * cfg["batch"]:(bi + 1) * cfg["batch"]]
            Bn = len(chunk)
            if epoch <= cfg["warmup"]:
                lr = cfg["lr"] * (bi + 1) / nb
            else:
                lr = cfg["lr"] * cfg["gamma"] ** (epoch - 1 - cfg["warmup"])
            for g in opt.param_groups:
                g["lr"] = lr
            st = torch.tensor([c[0] for c in chunk], dtype=torch.int64, device=dev)
            ps = torch.tensor([c[1] for c in chunk], dtype=torch.int64, device=dev)
            pe = torch.tensor([c[2] for c in chunk], dtype=torch.int64, device=dev)
            sw = torch.from_numpy(rng.random(Bn) < cfg["p_swap"]).to(dev)
            mi = torch.from_numpy(rng.random(Bn) < cfg["p_mirror"]).to(dev)
            ma = mb = None
            if cfg["use_ball"] and cfg["p_mask"] > 0:
                on = rng.random(Bn) < cfg["p_mask"]
                frac = rng.uniform(0.1, 1.0, size=Bn)
                L = np.where(on, np.round(frac * span), 0).astype(np.int64)
                off = (rng.random(Bn) * (span - L + 1)).astype(np.int64)
                a = np.array([c[0] for c in chunk], np.int64) + off
                ma = torch.from_numpy(a).to(dev)
                mb = torch.from_numpy(a + L).to(dev)
            with torch.no_grad():
                inp = make_inputs(bank, st, ps, pe, T, cfg["use_ball"], sw, mi, ma, mb)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(inp["ptok"], inp["pmask"], inp["btok"], inp["bmask"], inp["ts"])
            logits = logits.float()
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 3), inp["y"].reshape(-1), reduction="none",
                label_smoothing=cfg["label_smooth"]).reshape(Bn, T)
            if cfg["boundary_w"] != 1.0:
                w = boundary_weight(inp["y"], inp["stepv"], cfg["boundary_w"])
            else:
                w = inp["stepv"].float()
            loss = (nll * w).sum() / w.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step_i += 1
            tot += float(loss.detach()) * Bn
            cnt += Bn
        if log and (epoch % cfg["ckpt_every"] == 0 or epoch == 1):
            log("  epoch %d loss %.4f lr %.2e" % (epoch, tot / max(cnt, 1), lr))
        if on_ckpt is not None and (epoch % cfg["ckpt_every"] == 0 or epoch == epochs):
            on_ckpt(ema.model, epoch)
    return ema.model


def predict_match(model, bank, match, cfg, use_ball=None, views=None, drop_players=0.0, seed=0):
    """Per-frame probabilities (n, 3) for one match: 5 phase-shifted 5 Hz passes cover every
    frame; overlapping windows are averaged with centre-heavy weights; team-swap (and mirror)
    views are averaged in probability space after mapping back to the original labels."""
    tp = torch_parts()
    torch = tp.torch
    dev = bank.device
    use_ball = cfg["use_ball"] if use_ball is None else use_ball
    if views is None:
        views = [(False, False), (True, False)]
        if cfg.get("tta", "swap") == "swap_mirror":
            views += [(False, True), (True, True)]
    T, hop = cfg["T"], cfg["hop"]
    off = bank.offset[match]
    n = bank.length[match]
    w_win = torch.minimum(torch.arange(1, T + 1, device=dev), torch.arange(T, 0, -1, device=dev)).float()
    w_win = (w_win / w_win.max()).clamp_min(0.02)
    model.eval()
    prob = np.zeros((n, 3), np.float64)
    saved_vok = None
    if drop_players > 0:
        # robustness probe: hide a random share of player tokens for whole 2 s blocks
        g = np.random.default_rng(seed)
        saved_vok = bank.team[off:off + n].clone()
        blocks = (n + 49) // 50
        hide = g.random((blocks, bank.team.shape[1])) < drop_players
        hide = np.repeat(hide, 50, axis=0)[:n]
        t = bank.team[off:off + n]
        t[torch.from_numpy(hide).to(dev)] = -1
    try:
        for (sw, mi) in views:
            acc = torch.zeros(n, 3, device=dev, dtype=torch.float64)
            for (a, b) in bank.periods[match]:
                for phase in range(STEP):
                    s0 = a + phase
                    if s0 >= b:
                        continue
                    Tp = (b - s0 + STEP - 1) // STEP
                    Tw = min(T, Tp)
                    # regular windows every `hop` steps (overlap-add by chunks), plus one window
                    # aligned to the period end when the grid does not reach it
                    J = (Tp - Tw) // hop + 1
                    starts = [j * hop for j in range(J)]
                    tail = (Tp - Tw) % hop != 0
                    if tail:
                        starts.append(Tp - Tw)
                    ww = w_win[:Tw] if Tw == T else torch.ones(Tw, device=dev)
                    lps = []
                    for i in range(0, len(starts), cfg["infer_batch"]):
                        sub = starts[i:i + cfg["infer_batch"]]
                        Bn = len(sub)
                        st = torch.tensor([s0 + STEP * s for s in sub], dtype=torch.int64, device=dev)
                        ps = torch.full((Bn,), a, dtype=torch.int64, device=dev)
                        pe = torch.full((Bn,), b, dtype=torch.int64, device=dev)
                        swt = torch.full((Bn,), bool(sw), dtype=torch.bool, device=dev)
                        mit = torch.full((Bn,), bool(mi), dtype=torch.bool, device=dev)
                        with torch.no_grad():
                            inp = make_inputs(bank, st, ps, pe, Tw, use_ball, swt, mit)
                            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                enabled=dev.type == "cuda"):
                                logits = model(inp["ptok"], inp["pmask"], inp["btok"], inp["bmask"], inp["ts"])
                            lps.append(torch.log_softmax(logits.float(), dim=-1))   # (Bn, Tw, 3)
                    lp = torch.cat(lps, dim=0) * ww[None, :, None]
                    lp_acc = torch.zeros(Tp, 3, device=dev)
                    w_acc = torch.zeros(Tp, device=dev)
                    if Tw == T and T % hop == 0:
                        R = T // hop
                        nch = J - 1 + R
                        X = lp[:J].reshape(J, R, hop, 3)
                        Wt = ww.reshape(R, hop)
                        ca = torch.zeros(nch, hop, 3, device=dev)
                        cw = torch.zeros(nch, hop, device=dev)
                        for r in range(R):
                            ca[r:r + J] += X[:, r]
                            cw[r:r + J] += Wt[r]
                        lp_acc[:nch * hop] += ca.reshape(-1, 3)
                        w_acc[:nch * hop] += cw.reshape(-1)
                    else:
                        for j in range(J):
                            s = starts[j]
                            lp_acc[s:s + Tw] += lp[j]
                            w_acc[s:s + Tw] += ww
                    if tail:
                        s = starts[-1]
                        lp_acc[s:s + Tw] += lp[-1]
                        w_acc[s:s + Tw] += ww
                    lp_mean = lp_acc / w_acc[:, None]
                    p = torch.softmax(lp_mean, dim=-1)                      # renormalised
                    if sw:
                        p = p[:, torch.tensor(SWAP, device=dev)]
                    frames = torch.arange(s0, b, STEP, device=dev) - off
                    acc[frames] += p.double()
            prob += acc.cpu().numpy()
    finally:
        if saved_vok is not None:
            bank.team[off:off + n] = saved_vok
    prob /= len(views)
    prob /= prob.sum(axis=1, keepdims=True)
    return prob


def self_test():
    """Synthetic checks: viterbi parity, label independence of the inputs, swap consistency."""
    check_viterbi()
    from bench.feats3 import _synthetic_gap
    m = _synthetic_gap()
    n = len(m["ball"])
    m["players"] = np.concatenate([m["players"], np.full((n, 2, 1, 2), np.nan, np.float32)], axis=2)
    y = np.zeros(n, np.int8)
    p1 = prepare(m, y)
    m2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in m.items()}
    m2["alive"] = ~m2["alive"]
    m2["owner"] = 1 - m2["owner"]
    p2 = prepare(m2, y)
    for k in ("pos", "vel", "vok", "team", "ball", "bvel", "vis", "last", "nxt"):
        if not np.array_equal(p1[k], p2[k]):
            raise AssertionError("inputs changed with labels: %s" % k)
    # canonical frame: home attacks +x in both periods
    tp = torch_parts()
    torch = tp.torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = Bank([p1], ["syn"], dev)
    st = torch.tensor([0], device=dev)
    ps = torch.tensor([0], device=dev)
    pe = torch.tensor([30], device=dev)
    a = make_inputs(bank, st, ps, pe, 4, True)
    b = make_inputs(bank, st, ps, pe, 4, True, swap=torch.tensor([True], device=dev))
    # swapping twice is identity on the label and the team sign
    if not torch.equal(a["ts"], -b["ts"]):
        raise AssertionError("swap did not flip the team sign")
    if not torch.allclose(a["ptok"][..., 0] * a["pmask"], -b["ptok"][..., 0] * b["pmask"], atol=1e-5):
        raise AssertionError("swap did not rotate x")
    net = tp.SetNet(d=16, heads=2, n_sab=1, hidden=8).to(dev)
    out = net(a["ptok"], a["pmask"], a["btok"], a["bmask"], a["ts"])
    if out.shape != (1, 4, 3) or not torch.isfinite(out).all():
        raise AssertionError("forward shape")
    # permutation invariance over player tokens (eval mode)
    net.eval()
    perm = torch.randperm(a["ptok"].shape[2], device=dev)
    o1 = net(a["ptok"], a["pmask"], a["btok"], a["bmask"], a["ts"])
    o2 = net(a["ptok"][:, :, perm], a["pmask"][:, :, perm], a["btok"], a["bmask"], a["ts"][:, :, perm])
    if not torch.allclose(o1, o2, atol=1e-4):
        raise AssertionError("not permutation invariant")
    print("setnet self_test ok")
    return True
