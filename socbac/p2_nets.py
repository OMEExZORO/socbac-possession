"""Networks of pipeline_final2 (open IDSSE data only for training).

Faithful copies of the round-2/3 research code (research/runs/round2/code/r2.py, tn.py and
research/runs/round3/code/r3x.py), without their hard-coded paths, logs and globals:
  * SetNet2: setnet big with ball-regression, nearest-player pointer and L0 heads (r2.net_class)
  * train_rec: r2.train with auxiliary losses (bf16 autocast on the GPU, as in the research run)
  * predict_aux: r3x.predict_aux with a precision switch; amp=False is the float32 path used for
    every shipped inference (CPU float32)
  * estimates: r3x.estimates (reconstruction anchored to the visible ball on both sides of a gap)
  * touchnet: tn.features / sym_vectors / view / make_net / predict, r3x.train (distillation term)
  * fill_inputs: r3x.tn_stage feature block of touchnet-R (reconstructed ball + 5 channels, 63 inputs)
Prints nothing about the data.
"""
import numpy as np

from socbac import setnet_pool as sp

PTR_R = 2.0
TAU = 2.0
NF0 = 58                                   # touchnet inputs with the observed (or clean) ball
TN_CFG = dict(hidden=128, layers=2, drop=0.1, T=1500, batch=16, lr=2e-3, wd=0.01, gamma=0.96, epochs=80,
              ckpt_every=8, ema=0.98, p_mask=0.5, label_smooth=0.05, aux=0.3)
REC_AUX = dict(ball=3.0, ptr=1.0, l0=0.3)


def rec_cfg(epochs=45):
    cfg = dict(sp.DEFAULT_CFG)
    cfg.update({"d": 96, "n_sab": 3})
    cfg.update(infer_batch=32, epochs=epochs)
    return cfg


# ------------------------------------------------------------------ bank
def make_bank(preps, names, dev):
    """sp.Bank plus the privileged training arrays (zeros / False where absent, e.g. real match)."""
    import torch
    bank = sp.Bank(preps, names, dev)

    def get(p, k, shape_fn, dtype):
        return p[k] if k in p else np.zeros(shape_fn(p), dtype)

    n_of = lambda p: (len(p["y"]),)
    bank.cball = torch.from_numpy(np.concatenate([get(p, "cball", lambda q: (len(q["y"]), 2), np.float32)
                                                  for p in preps])).to(dev)
    bank.cok = torch.from_numpy(np.concatenate([get(p, "cok", n_of, bool) for p in preps])).to(dev)
    bank.y0 = torch.from_numpy(np.concatenate([get(p, "y0", n_of, np.int8) for p in preps]).astype(np.int64)).to(dev)
    return bank


# ------------------------------------------------------------------ reconstructor (setnet + aux heads)
_NET = None


def net_class():
    global _NET
    if _NET is not None:
        return _NET
    tp = sp.torch_parts()
    torch, nn = tp.torch, tp.nn

    class SetNet2(tp.SetNet):
        def __init__(self, **kw):
            super().__init__(**kw)
            out = self.head[1].in_features
            d = self.d
            self.ball_head = nn.Linear(out, 2)
            self.head0 = nn.Linear(out, 3)
            self.ptr_q = nn.Linear(out, d)
            self.ptr_k = nn.Linear(d, d)
            self.ptr_none = nn.Linear(out, 1)

        def forward(self, ptok, pmask, btok, bmask, ts, aux=False):
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
            with torch.autocast(device_type=z.device.type, enabled=False):
                z, _ = self.rnn(z.float())
            logits = self.head(z)
            if not aux:
                return logits
            keys = self.ptr_k(self.lnf(x[:, 2:])).reshape(B, T, K, self.d).float()
            q = self.ptr_q(z).float()
            s = (keys * q[:, :, None, :]).sum(-1) * (1.0 / np.sqrt(self.d))
            s = s.masked_fill(~pmask, -1e9)
            ptr = torch.cat([s, self.ptr_none(z).float()], dim=-1)
            return logits, self.ball_head(z).float(), ptr, self.head0(z).float()

    _NET = SetNet2
    return _NET


def new_rec(cfg, dev):
    return net_class()(d=cfg["d"], heads=cfg["heads"], n_sab=cfg["n_sab"], temporal="gru",
                       hidden=cfg["hidden"], rnn_layers=cfg["rnn_layers"], drop=cfg["drop"]).to(dev)


def train_rec(bank, train_matches, cfg, aux, seed, epochs, logf=None):
    """r2.train (aux heads), returns the EMA model."""
    tp = sp.torch_parts()
    torch = tp.torch
    Fn = torch.nn.functional
    dev = bank.device
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(10007 * int(seed) + 17)
    model = new_rec(cfg, dev)
    ema = sp.EMA(model, cfg["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    T = cfg["T"]
    span = T * sp.STEP
    ar = torch.arange(T, device=dev, dtype=torch.int64) * sp.STEP
    use_amp = dev.type == "cuda"
    for epoch in range(1, epochs + 1):
        model.train()
        wins = sp.sample_windows(bank, train_matches, T, rng)
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
            on = rng.random(Bn) < cfg["p_mask"]
            frac = rng.uniform(0.1, 1.0, size=Bn)
            L = np.where(on, np.round(frac * span), 0).astype(np.int64)
            off = (rng.random(Bn) * (span - L + 1)).astype(np.int64)
            a = np.array([c[0] for c in chunk], np.int64) + off
            ma = torch.from_numpy(a).to(dev)
            mb = torch.from_numpy(a + L).to(dev)
            with torch.no_grad():
                inp = sp.make_inputs(bank, st, ps, pe, T, True, sw, mi, ma, mb)
                F = torch.minimum(st[:, None] + ar[None, :], pe[:, None] - 1)
                cb = bank.cball[F]
                ok = bank.cok[F] & inp["stepv"]
                pos = bank.pos[F]
                valid = bank.team[F] >= 0
                dist = torch.linalg.norm(pos - cb[:, :, None, :], dim=-1).masked_fill(~valid, 1e6)
                dmin, imin = dist.min(dim=-1)
                K = pos.shape[2]
                ptr_t = torch.where(dmin < PTR_R, imin, torch.full_like(imin, K))
                fx, fy = sw, sw ^ mi
                bx = torch.where(fx[:, None], sp.PL - cb[..., 0], cb[..., 0]) / 52.5 - 1.0
                by = torch.where(fy[:, None], sp.PW - cb[..., 1], cb[..., 1]) / 34.0 - 1.0
                ball_t = torch.stack([bx, by], dim=-1)
                y0 = bank.y0[F]
                y0 = torch.where(sw[:, None], bank.swap_lut[y0], y0)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(inp["ptok"], inp["pmask"], inp["btok"], inp["bmask"], inp["ts"], aux=True)
            logits = out[0].float()

            def _ce(lg, yy):
                return Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), yy.reshape(-1), reduction="none",
                                        label_smoothing=cfg["label_smooth"]).reshape(Bn, T)

            w = inp["stepv"].float()
            nll = 0.5 * (_ce(logits, inp["y"]) + _ce(logits, inp["y2"]))
            loss = (nll * w).sum() / w.sum().clamp_min(1.0)
            okf = ok.float()
            den = okf.sum().clamp_min(1.0)
            lb = (Fn.smooth_l1_loss(out[1], ball_t, beta=0.05, reduction="none").sum(-1) * okf).sum() / den
            lp = (Fn.cross_entropy(out[2].reshape(Bn * T, -1), ptr_t.reshape(-1), reduction="none"
                                   ).reshape(Bn, T) * okf).sum() / den
            l0 = (_ce(out[3], y0) * w).sum() / w.sum().clamp_min(1.0)
            loss = loss + aux["ball"] * lb + aux["ptr"] * lp + aux["l0"] * l0
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot += float(loss.detach()) * Bn
            cnt += Bn
        if logf and (epoch % cfg["ckpt_every"] == 0 or epoch == 1):
            logf("  rec epoch %d loss %.4f lr %.2e" % (epoch, tot / max(cnt, 1), lr))
    return ema.model


def predict_aux(model, bank, name, cfg, amp=True):
    """(n, 7): ball head xy [m], pointer-mixture xy [m], P(nearest home / away / none).
    amp=True: bfloat16 autocast on CUDA (research path); amp=False: float32 everywhere."""
    import torch
    dev = bank.device
    T, hop, S, ib = cfg["T"], cfg["hop"], sp.STEP, cfg["infer_batch"]
    off, n = bank.offset[name], bank.length[name]
    out = torch.zeros(n, 7, device=dev)
    model.eval()
    ww = torch.minimum(torch.arange(1, T + 1, device=dev), torch.arange(T, 0, -1, device=dev)).float()
    ww = (ww / ww.max()).clamp_min(0.02)
    use_amp = bool(amp) and dev.type == "cuda"
    for a, b in bank.periods[name]:
        for ph in range(S):
            s0 = a + ph
            if s0 >= b:
                continue
            Tp = (b - s0 + S - 1) // S
            Tw = min(T, Tp)
            starts = list(range(0, Tp - Tw + 1, hop))
            if starts[-1] != Tp - Tw:
                starts.append(Tp - Tw)
            w = (ww[:Tw] if Tw == T else torch.ones(Tw, device=dev))[:, None]
            acc = torch.zeros(Tp, 7, device=dev)
            wac = torch.zeros(Tp, 1, device=dev)
            ar = torch.arange(Tw, device=dev)[None] * S
            for i in range(0, len(starts), ib):
                sub = starts[i:i + ib]
                st = torch.tensor([s0 + S * s for s in sub], dtype=torch.int64, device=dev)
                ps, pe = torch.full_like(st, a), torch.full_like(st, b)
                with torch.no_grad():
                    inp = sp.make_inputs(bank, st, ps, pe, Tw, True)
                    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                        _, ball, ptr, _ = model(inp["ptok"], inp["pmask"], inp["btok"], inp["bmask"], inp["ts"], aux=True)
                    F = torch.minimum(st[:, None] + ar, pe[:, None] - 1)
                    pos, team = bank.pos[F], bank.team[F]
                    K = pos.shape[2]
                    p = torch.softmax(ptr.float(), -1)
                    pn = p[..., K]
                    pp = (p[..., :K, None] * pos).sum(-2) / (1 - pn).clamp_min(1e-3)[..., None]
                    o = torch.stack([(ball[..., 0].float() + 1) * 52.5, (ball[..., 1].float() + 1) * 34.0,
                                     pp[..., 0], pp[..., 1], (p[..., :K] * (team == 0)).sum(-1),
                                     (p[..., :K] * (team == 1)).sum(-1), pn], -1)
                for j, s in enumerate(sub):
                    acc[s:s + Tw] += o[j] * w
                    wac[s:s + Tw] += w
            out[torch.arange(s0, b, S, device=dev) - off] = acc / wac
    return out


def estimates(bank, R, mode="bh"):
    import torch
    N = bank.n
    est = {"bh": R[:, 0:2], "pp": R[:, 2:4], "avg": 0.5 * (R[:, 0:2] + R[:, 2:4])}[mode]
    k = torch.ones(1, 1, 13, device=R.device) / 13.0
    est = torch.nn.functional.conv1d(est.t()[:, None, :], k, padding=6)[:, 0, :].t().contiguous()
    idx = torch.arange(N, device=R.device)
    kl, kn = bank.last >= 0, bank.nxt >= 0
    li, ni = bank.last.clamp(min=0), bank.nxt.clamp(min=0)
    al = torch.where(kl, (idx - li).float() / 25.0, torch.full((N,), 600.0, device=R.device))
    an = torch.where(kn, (ni - idx).float() / 25.0, torch.full((N,), 600.0, device=R.device))
    rl = (bank.ball[li] - est[li]) * kl[:, None]
    rn = (bank.ball[ni] - est[ni]) * kn[:, None]
    wl = torch.where(kl & kn, an / (al + an).clamp_min(1e-3), kl.float())
    wn = torch.where(kl & kn, al / (al + an).clamp_min(1e-3), kn.float())
    decay = torch.exp(-torch.minimum(al, an) / TAU)
    out = est + (rl * wl[:, None] + rn * wn[:, None]) * decay[:, None]
    lin = bank.ball[li] * wl[:, None] + bank.ball[ni] * wn[:, None]
    return out, lin, al, an


# ------------------------------------------------------------------ touchnet
def tn_features(bank, nf=NF0):
    """tn.features: (N, nf) float32 plus the auxiliary nearest-team target (N,) in {0, 1, 2}."""
    import torch
    N = bank.n
    X = torch.zeros(N, nf, device=bank.device)
    aux = torch.full((N,), 2, dtype=torch.int64, device=bank.device)
    bv = bank.bvel
    vis = bank.vis
    acc = torch.zeros_like(bv)
    acc[2:-2] = (bv[4:] - bv[:-4]) * (25.0 / 4.0)
    aok = torch.zeros_like(vis)
    aok[2:-2] = vis[4:] & vis[:-4] & vis[2:-2]
    acc = acc * aok[:, None]
    for a in range(0, N, 20000):
        b = min(N, a + 20000)
        pos, vel, team = bank.pos[a:b], bank.vel[a:b], bank.team[a:b]
        ball, v = bank.ball[a:b], bv[a:b]
        vf = vis[a:b].float()[:, None]
        d = pos - ball[:, None, :]
        dist = torch.linalg.norm(d, dim=-1)
        edge = torch.minimum(torch.minimum(ball[:, 0], 105.0 - ball[:, 0]), torch.minimum(ball[:, 1], 68.0 - ball[:, 1]))
        x = [ball[:, 0:1] / 52.5 - 1.0, ball[:, 1:2] / 34.0 - 1.0, v / 10.0,
             torch.linalg.norm(v, dim=-1, keepdim=True) / 10.0, (acc[a:b] / 20.0).clamp(-3, 3),
             (torch.linalg.norm(acc[a:b], dim=-1, keepdim=True) / 20.0).clamp(0, 3),
             (edge[:, None] / 10.0).clamp(-1, 3), torch.ones_like(vf)]
        ctx = []
        for s in (0, 1):
            m = team == s
            ds = dist.masked_fill(~m, 1e3)
            dk, ik = torch.topk(ds, 3, dim=1, largest=False)
            ok = (dk < 1e2).float()[..., None]
            g = lambda t: torch.gather(t, 1, ik[..., None].expand(-1, -1, 2))
            pv = g(vel)
            blk = torch.cat([(g(d) / 5.0).clamp(-4, 4), (dk[..., None] / 5.0).clamp(0, 4), (pv - v[:, None, :]) / 5.0,
                             torch.linalg.norm(pv, dim=-1, keepdim=True) / 5.0, torch.ones_like(ok)], dim=-1) * ok
            x.append(blk.reshape(b - a, 21))
            mf = m.float()[..., None]
            cnt = mf.sum(1).clamp_min(1.0)
            cen = (pos * mf).sum(1) / cnt
            ctx.append(torch.cat([cen[:, 0:1] / 52.5 - 1.0, cen[:, 1:2] / 34.0 - 1.0,
                                  ((vel[..., 0:1] * mf).sum(1) / cnt) / 5.0], dim=-1))
        xb = torch.cat(x, dim=-1) * vf
        X[a:b] = torch.cat([xb] + ctx, dim=-1)
        dc = torch.linalg.norm(pos - bank.cball[a:b][:, None, :], dim=-1).masked_fill(team < 0, 1e6)
        dmin, imin = dc.min(dim=1)
        tt = torch.gather(team, 1, imin[:, None])[:, 0]
        aux[a:b] = torch.where((dmin < PTR_R) & bank.cok[a:b], tt, torch.full_like(tt, 2))
    return X, aux


def sym_vectors(dev, extended=False):
    import torch
    sx, sy = np.ones(NF0, np.float32), np.ones(NF0, np.float32)
    sx[[0, 2, 5]] = -1
    sy[[1, 3, 6]] = -1
    for base in (10, 31):
        for k in range(3):
            o = base + 7 * k
            sx[[o, o + 3]] = -1
            sy[[o + 1, o + 4]] = -1
    for base in (52, 55):
        sx[[base, base + 2]] = -1
        sy[base + 1] = -1
    perm = np.concatenate([np.arange(10), np.arange(31, 52), np.arange(10, 31), np.arange(55, 58), np.arange(52, 55)])
    t = lambda a: torch.from_numpy(a).to(dev)
    sx, sy, perm = t(sx), t(sy), t(perm)
    if extended:                     # + P(home), P(away), P(none), time to visible ball on each side
        one = torch.ones(5, device=dev)
        sx, sy = torch.cat([sx, one]), torch.cat([sy, one])
        perm = torch.cat([perm, torch.tensor([59, 58, 60, 61, 62], device=dev)])
    return sx, sy, perm


def view(x, sw, mi, sym):
    import torch
    sx, sy, perm = sym
    if isinstance(sw, bool):
        if sw:
            x = x[..., perm] * sx * sy
        if mi:
            x = x * sy
        return x
    xs = x[..., perm] * sx * sy
    x = torch.where(sw[:, None, None], xs, x)
    return torch.where(mi[:, None, None], x * sy, x)


def make_tn(cfg, dev, nf):
    import torch
    from torch import nn

    class TouchNet(nn.Module):
        def __init__(self):
            super().__init__()
            h = cfg["hidden"]
            self.inp = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Linear(h, h))
            self.rnn = nn.GRU(h, h, num_layers=cfg["layers"], batch_first=True, bidirectional=True, dropout=cfg["drop"])
            self.drop = nn.Dropout(cfg["drop"])
            self.head = nn.Linear(2 * h, 3)
            self.head_aux = nn.Linear(2 * h, 3)
            self.head0 = nn.Linear(2 * h, 3)

        def forward(self, x, aux=False):
            z, _ = self.rnn(self.inp(x))
            z = self.drop(z)
            if aux:
                return self.head(z), self.head_aux(z), self.head0(z)
            return self.head(z)

    return TouchNet().to(dev)


def train_tn(bank, X, AUX, sym, matches, cfg, seed, epochs, on_ckpt=None, logf=None, TL=None):
    """r3x.train: touchnet with optional distillation towards teacher log-probabilities TL."""
    import torch
    Fn = torch.nn.functional
    dev = bank.device
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(10007 * int(seed) + 23)
    model = make_tn(cfg, dev, X.shape[1])
    ema = sp.EMA(model, cfg["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    T = cfg["T"]
    kd = cfg.get("kd", 0.0) if TL is not None else 0.0
    ar = torch.arange(T, device=dev)
    lut = bank.swap_lut
    for epoch in range(1, epochs + 1):
        model.train()
        wins = []
        for m in matches:
            for a, b in bank.periods[m]:
                k = max(1, int(round((b - a) / float(T))))
                wins += [(int(s), b) for s in rng.integers(a, max(a, b - T) + 1, size=k)]
        wins = [wins[i] for i in rng.permutation(len(wins))]
        nb = (len(wins) + cfg["batch"] - 1) // cfg["batch"]
        tot, cnt = 0.0, 0
        for bi in range(nb):
            chunk = wins[bi * cfg["batch"]:(bi + 1) * cfg["batch"]]
            Bn = len(chunk)
            lr = cfg["lr"] * ((bi + 1) / nb if epoch == 1 else cfg["gamma"] ** (epoch - 2))
            for g in opt.param_groups:
                g["lr"] = lr
            st = torch.tensor([c[0] for c in chunk], device=dev)
            pe = torch.tensor([c[1] for c in chunk], device=dev)
            F = st[:, None] + ar[None, :]
            w = (F < pe[:, None]).float()
            F = torch.minimum(F, pe[:, None] - 1)
            sw = torch.from_numpy(rng.random(Bn) < 0.5).to(dev)
            mi = torch.from_numpy(rng.random(Bn) < 0.5).to(dev)
            x = X[F]
            if cfg["p_mask"] > 0:
                on = rng.random(Bn) < cfg["p_mask"]
                Lh = np.where(on, np.round(rng.uniform(0.1, 1.0, size=Bn) * T), 0).astype(np.int64)
                off = (rng.random(Bn) * (T - Lh + 1)).astype(np.int64)
                ma = torch.from_numpy(off).to(dev)[:, None]
                mb = torch.from_numpy(off + Lh).to(dev)[:, None]
                hide = (ar[None, :] >= ma) & (ar[None, :] < mb)
                x = torch.cat([x[..., :52] * (~hide).float()[..., None], x[..., 52:]], dim=-1)
            x = view(x, sw, mi, sym)
            y1, y2, y0, ya = [torch.where(sw[:, None], lut[t], t) for t in (bank.y[F], bank.y2[F], bank.y0[F], AUX[F])]
            out = model(x, aux=True)
            den = w.sum().clamp_min(1.0)

            def ce(lg, yy):
                return (Fn.cross_entropy(lg.float().reshape(-1, 3), yy.reshape(-1), reduction="none",
                                         label_smoothing=cfg["label_smooth"]).reshape(Bn, T) * w).sum() / den

            loss = 0.5 * (ce(out[0], y1) + ce(out[0], y2))
            if kd > 0:
                tl = TL[F]
                tl = torch.where(sw[:, None, None], tl[..., lut], tl)
                pt = torch.softmax(tl, -1)
                kl = (pt * (torch.log(pt.clamp_min(1e-8)) - torch.log_softmax(out[0].float(), -1))).sum(-1)
                loss = (1 - kd) * loss + kd * (kl * w).sum() / den
            if cfg["aux"] > 0:
                loss = loss + cfg["aux"] * (ce(out[1], ya) + ce(out[2], y0))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tot += float(loss.detach()) * Bn
            cnt += Bn
        if logf and (epoch % cfg["ckpt_every"] == 0 or epoch == 1):
            logf("  tn epoch %d loss %.4f lr %.2e" % (epoch, tot / max(cnt, 1), lr))
        if on_ckpt is not None and (epoch % cfg["ckpt_every"] == 0 or epoch == epochs):
            on_ckpt(ema.model, epoch)
    return ema.model


def predict_tn(model, bank, X, sym, match):
    """tn.predict: 4 views (swap x mirror), 6000-frame windows with hop 3000, float32."""
    import torch
    dev = bank.device
    off, n = bank.offset[match], bank.length[match]
    prob = torch.zeros(n, 3, device=dev, dtype=torch.float64)
    model.eval()
    sl = torch.tensor(sp.SWAP, device=dev)
    with torch.no_grad():
        for a, b in bank.periods[match]:
            for sw in (False, True):
                for mi in (False, True):
                    x = view(X[a:b][None], sw, mi, sym)
                    nper, W, hop = b - a, 6000, 3000
                    acc = torch.zeros(nper, 3, device=dev)
                    wac = torch.zeros(nper, 1, device=dev)
                    tri = torch.minimum(torch.arange(1, W + 1, device=dev), torch.arange(W, 0, -1, device=dev)).float()[:, None]
                    starts = list(range(0, max(1, nper - W + 1), hop))
                    if starts[-1] + W < nper:
                        starts.append(nper - W)
                    for s in starts:
                        e = min(nper, s + W)
                        lp = torch.log_softmax(model(x[:, s:e].contiguous())[0].float(), dim=-1)
                        acc[s:e] += lp * tri[:e - s]
                        wac[s:e] += tri[:e - s]
                    p = torch.softmax(acc / wac, dim=-1)
                    if sw:
                        p = p[:, sl]
                    prob[a - off:b - off] += p.double() / 4.0
    p = prob.cpu().numpy()
    return p / p.sum(axis=1, keepdims=True)


def clean_ball_inputs(bank):
    """Teacher inputs: the clean, never-hidden ball (training only). Restores the bank afterwards."""
    import torch
    keep = (bank.ball, bank.bvel, bank.vis)
    cb = bank.cball
    v = torch.zeros_like(cb)
    v[2:-2] = (cb[4:] - cb[:-4]) * (25.0 / 4.0)
    ok = bank.cok.clone()
    ok[2:-2] &= bank.cok[4:] & bank.cok[:-4] & (torch.linalg.norm(v[2:-2], dim=-1) <= 40.0)
    bank.ball, bank.bvel, bank.vis = cb * bank.cok[:, None], v * ok[:, None], bank.cok.clone()
    XT, AUX = tn_features(bank)
    bank.ball, bank.bvel, bank.vis = keep
    return XT, AUX


def fill_inputs(bank, R):
    """Touchnet-R inputs (63): observed ball where visible, anchored reconstruction where hidden,
    true visible flag in column 9, P(home/away/none nearest), log time to the visible ball on each side.
    Modifies bank.ball / bvel / vis in place (as r3x.tn_stage)."""
    import torch
    est, _, al, an = estimates(bank, R, "bh")
    vis0 = bank.vis.clone()
    ball = torch.where(vis0[:, None], bank.ball, est)
    bv = torch.zeros_like(ball)
    bv[2:-2] = (ball[4:] - ball[:-4]) * (25.0 / 4.0)
    bv = bv * (torch.linalg.norm(bv, dim=-1, keepdim=True) <= 40.0)
    bank.bvel = torch.where(vis0[:, None], bank.bvel, bv)
    bank.ball = ball
    bank.vis = torch.ones_like(vis0)
    X, AUX = tn_features(bank)
    X[:, 9] = vis0.float()
    X = torch.cat([X, R[:, 4:7], (torch.log1p(al) / 3.0)[:, None], (torch.log1p(an) / 3.0)[:, None]], dim=-1)
    return X, AUX
