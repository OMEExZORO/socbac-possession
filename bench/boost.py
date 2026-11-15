"""Model-class experiments on top of v1 + context + lang_dead.

Team-symmetric training matches bench.methods.LGBMViterbi: every match is
presented twice, with the teams swapped. Switch costs are chosen on TRAIN
only. Nothing here reads the competition match.
"""

import os
import gc

import numpy as np

from bench import methods
from bench.feats3 import decode_periods_fast, viterbi_fast

H, A, D = methods.H, methods.A, methods.D
SWAP = methods.SWAP
GRID_TEAM = (5.0, 10.0, 20.0, 40.0, 80.0)
GRID_DEAD = (10.0, 20.0, 40.0, 80.0, 160.0)
GRID_TEAM_FINE = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 40.0)
GRID_DEAD_FINE = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 160.0)
ADOPT = 0.03

_VITERBI = None


def _compile_viterbi():
    """Numba port of feats3.viterbi_fast. Ties keep the lower state index."""
    from numba import njit

    @njit(cache=True)
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
            e0 = e[t, 0]
            e1 = e[t, 1]
            e2 = e[t, 2]
            b0 = a0
            b1 = a1 - st
            b2 = a2 - sd
            src = 0
            best = b0
            if b1 > best:
                src = 1
                best = b1
            if b2 > best:
                src = 2
                best = b2
            back[t, 0] = src
            s0 = best + e0
            b0 = a0 - st
            b1 = a1
            b2 = a2 - sd
            src = 0
            best = b0
            if b1 > best:
                src = 1
                best = b1
            if b2 > best:
                src = 2
                best = b2
            back[t, 1] = src
            s1 = best + e1
            b0 = a0 - sd
            b1 = a1 - sd
            b2 = a2
            src = 0
            best = b0
            if b1 > best:
                src = 1
                best = b1
            if b2 > best:
                src = 2
                best = b2
            back[t, 2] = src
            s2 = best + e2
        if s1 > s0:
            last = 1
            bests = s1
        else:
            last = 0
            bests = s0
        if s2 > bests:
            last = 2
        path = np.empty(n, np.int8)
        path[n - 1] = last
        for t in range(n - 1, 0, -1):
            last = back[t, last]
            path[t - 1] = last
        return path

    return _vit


def viterbi_rows(e, switch_team, switch_dead):
    """One period. Prefers the numba loop when it matches viterbi_fast."""
    global _VITERBI
    e = np.asarray(e, dtype=np.float64)
    if len(e) == 0:
        return np.zeros(0, np.int8)
    if _VITERBI is None and _VITERBI != "off":
        try:
            _VITERBI = _compile_viterbi()
        except Exception:
            _VITERBI = "off"
    if _VITERBI not in (None, "off"):
        return _VITERBI(e, float(switch_team), float(switch_dead))
    return viterbi_fast(e, switch_team, switch_dead)


def decode_periods(e, period, switch_team, switch_dead):
    e = np.asarray(e, dtype=np.float64)
    period = np.asarray(period)
    out = np.full(len(e), D, np.int8)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        out[idx] = viterbi_rows(e[idx], switch_team, switch_dead)
    return out


def atomic_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npy"
    np.save(tmp, np.ascontiguousarray(arr))
    os.replace(tmp, path)


def take(mm, cols, rows=None):
    """Copy selected columns. rows=None reads the whole matrix in blocks."""
    cols = np.asarray(cols, dtype=np.int32)
    if rows is None:
        n = int(mm.shape[0])
        out = np.empty((n, len(cols)), np.float32)
        for i in range(0, n, 20000):
            out[i:i + 20000] = np.asarray(mm[i:i + 20000][:, cols], dtype=np.float32)
        return out
    rows = np.asarray(rows, dtype=np.int32)
    out = np.empty((len(rows), len(cols)), np.float32)
    for i in range(0, len(rows), 20000):
        r = rows[i:i + 20000]
        out[i:i + 20000] = np.asarray(mm[r][:, cols], dtype=np.float32)
    return out


def class_vector(y, scheme):
    """Per-class multiplier. Both team views are included by the caller, so H and A match."""
    if scheme in ("none", "boundary"):
        return None
    if scheme == "balanced":
        counts = np.bincount(y, minlength=3).astype(np.float64)
        return (counts.sum() / (3.0 * np.maximum(counts, 1.0))).astype(np.float32)
    table = {
        "alive2": (2.0, 2.0, 1.0),
        "alive4": (4.0, 4.0, 1.0),
        "dead2": (1.0, 1.0, 2.0),
    }
    if scheme not in table:
        raise ValueError(scheme)
    return np.asarray(table[scheme], dtype=np.float32)


def boundary_weights(y, period, radius=12, boost=4.0):
    """Upweight frames within `radius` of a label change, inside each period."""
    y = np.asarray(y)
    period = np.asarray(period)
    w = np.ones(len(y), np.float32)
    kernel = np.ones(2 * int(radius) + 1, np.float32)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        yy = y[idx]
        ch = np.zeros(len(idx), np.float32)
        if len(yy) > 1:
            ch[1:] = (yy[1:] != yy[:-1]).astype(np.float32)
        dil = np.convolve(ch, kernel, mode="same")
        w[idx] = np.where(dil > 0.0, np.float32(boost), np.float32(1.0))
    return w


def _stride_rows(n, stride):
    return np.arange(0, int(n), int(stride), dtype=np.int32)


def collect_train(cache, matches, cols, stride, scheme):
    """Stack both team views at the given stride. Returns x, y, w (w may be None)."""
    cols = np.asarray(cols, dtype=np.int32)
    stride = int(stride)
    plan = []
    y_parts = []
    for m in matches:
        y = np.load(os.path.join(cache, m + ".y.npy"))
        period = np.load(os.path.join(cache, m + ".period.npy"))
        rows = _stride_rows(len(y), stride)
        plan.append((m, y, period, rows))
        y_parts.append(y[rows])
        y_parts.append(SWAP[y[rows]])
    ycat = np.concatenate(y_parts) if y_parts else np.zeros(0, np.int32)
    cw = class_vector(ycat, scheme)
    n = int(sum(len(rows) * 2 for _, _, _, rows in plan))
    x = np.empty((n, len(cols)), np.float32)
    y_out = np.empty(n, np.int32)
    w_out = None if scheme == "none" else np.empty(n, np.float32)
    i = 0
    for m, y, period, rows in plan:
        mm = np.load(os.path.join(cache, m + ".x.npy"), mmap_mode="r")
        ms = np.load(os.path.join(cache, m + ".xs.npy"), mmap_mode="r")
        block = take(mm, cols, rows)
        k = len(rows)
        x[i:i + k] = block
        y_out[i:i + k] = y[rows]
        block_s = take(ms, cols, rows)
        x[i + k:i + 2 * k] = block_s
        y_out[i + k:i + 2 * k] = SWAP[y[rows]]
        if w_out is not None:
            if scheme == "boundary":
                wf = boundary_weights(y, period)[rows]
            else:
                wf = cw[y[rows]]
            w_out[i:i + k] = wf
            w_out[i + k:i + 2 * k] = wf
        i += 2 * k
        del mm, ms, block, block_s
    if i != n:
        raise RuntimeError("row count mismatch")
    return x, y_out, w_out


def predict_matrix(model, kind, x):
    out = np.empty((len(x), 3), np.float64)
    for i in range(0, len(x), 20000):
        block = x[i:i + 20000]
        if kind == "lgbm":
            pred = model.predict(block)
        elif kind == "xgb":
            import xgboost as xgb
            pred = model.predict(xgb.DMatrix(block))
            if pred.ndim == 1:
                pred = pred.reshape(-1, 3)
        elif kind == "cat":
            pred = model.predict_proba(block)
        elif kind == "et":
            raw = model.predict_proba(block)
            pred = np.zeros((len(block), 3), np.float64)
            for j, c in enumerate(model.classes_):
                pred[:, int(c)] = raw[:, j]
        else:
            raise ValueError(kind)
        out[i:i + len(block)] = pred
    out = np.clip(out, 1e-8, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def predict_match(model, kind, cache, match, cols):
    """Symmetric average of the two team views, in the original label order."""
    cols = np.asarray(cols, dtype=np.int32)
    mx = np.load(os.path.join(cache, match + ".x.npy"), mmap_mode="r")
    p = predict_matrix(model, kind, take(mx, cols))
    del mx
    ms = np.load(os.path.join(cache, match + ".xs.npy"), mmap_mode="r")
    ps = predict_matrix(model, kind, take(ms, cols))
    del ms
    return 0.5 * (p + ps[:, SWAP])


def _prior_of(y):
    prior = np.bincount(np.asarray(y), minlength=3).astype(np.float64)
    prior /= max(prior.sum(), 1.0)
    return prior


def _importance(kind, model, n_features):
    gain = np.zeros(n_features, np.float64)
    if kind == "lgbm":
        gain[:] = np.asarray(model.feature_importance(importance_type="gain"), dtype=np.float64)
    elif kind == "xgb":
        score = model.get_score(importance_type="gain")
        for key, val in score.items():
            if key.startswith("f") and key[1:].isdigit():
                j = int(key[1:])
                if 0 <= j < n_features:
                    gain[j] = float(val)
    elif kind == "cat":
        gain[:] = np.asarray(model.get_feature_importance(), dtype=np.float64)
    elif kind == "et":
        gain[:] = np.asarray(model.feature_importances_, dtype=np.float64)
    return gain


def fit_model(kind, x, y, w, params, rounds):
    """Fit one booster. Returns model, prior, gain."""
    y = np.asarray(y)
    prior = _prior_of(y)
    n_features = int(x.shape[1])
    if kind == "lgbm":
        import lightgbm as lgb
        kw = {}
        if w is not None:
            kw["weight"] = np.asarray(w, dtype=np.float32)
        dset = lgb.Dataset(x, label=y, free_raw_data=True, **kw)
        model = lgb.train(dict(params), dset, num_boost_round=int(rounds))
        del dset
    elif kind == "xgb":
        import xgboost as xgb
        kw = {"label": y}
        if w is not None:
            kw["weight"] = np.asarray(w, dtype=np.float32)
        dm = xgb.DMatrix(x, **kw)
        model = xgb.train(dict(params), dm, num_boost_round=int(rounds))
        del dm
    elif kind == "cat":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**dict(params))
        model.set_params(iterations=int(rounds))
        model.fit(x, y, sample_weight=None if w is None else np.asarray(w, dtype=np.float32))
    elif kind == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        from threadpoolctl import threadpool_limits
        kw = dict(params)
        ms = min(int(kw.pop("max_samples")), len(y))
        kw["max_samples"] = ms
        model = ExtraTreesClassifier(**kw)
        with threadpool_limits(limits=1):
            model.fit(x, y, sample_weight=None if w is None else np.asarray(w, dtype=np.float32))
    else:
        raise ValueError(kind)
    gain = _importance(kind, model, n_features)
    return model, prior, gain


def lgbm_params(num_leaves=63, min_child_samples=200, feature_fraction=0.8, learning_rate=0.05):
    params = dict(methods.LGBMViterbi.PARAMS)
    params.update(
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        feature_fraction=float(feature_fraction),
        learning_rate=float(learning_rate),
        num_threads=1,
        deterministic=True,
        verbose=-1,
        seed=0,
    )
    return params


def xgb_params():
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": 1,
        "seed": 0,
        "verbosity": 0,
    }


def cat_params():
    return dict(
        loss_function="MultiClass",
        depth=6,
        learning_rate=0.05,
        thread_count=1,
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        border_count=128,
        used_ram_limit="1200mb",
        rsm=0.8,
    )


def et_params():
    return dict(
        n_estimators=20,
        max_depth=8,
        min_samples_leaf=40,
        max_features=16,
        max_samples=12000,
        bootstrap=True,
        n_jobs=1,
        random_state=0,
    )


def emissions(proba, prior):
    p = np.asarray(proba, dtype=np.float64)
    pr = np.asarray(prior, dtype=np.float64)
    return np.log(p + 1e-6) - np.log(pr)


def frame_metrics(y, pred, period):
    from bench.run import score
    spell, macro, npred, ngt = score(y, pred, period)
    alive_t = np.asarray(y) != D
    alive_p = np.asarray(pred) != D
    if not np.any(alive_t) or not np.any(alive_p):
        share = 1.0
    else:
        share = float(abs(np.mean(pred[alive_p] == H) - np.mean(y[alive_t] == H)))
    return {
        "spell": float(spell),
        "macro": float(macro),
        "n_pred": int(npred),
        "n_gt": int(ngt),
        "share_err": share,
    }


def tune_costs(ems, ys, periods, grid_t, grid_d):
    """Pick switch costs by mean spellF1. Ties keep the earlier (smaller) costs."""
    best = None
    per = None
    for st in grid_t:
        for sd in grid_d:
            spells = []
            rows = []
            for e, y, period in zip(ems, ys, periods):
                pred = decode_periods(e, period, st, sd)
                rec = frame_metrics(y, pred, period)
                spells.append(rec["spell"])
                rows.append(rec)
            mean = float(np.mean(spells))
            if best is None or mean > best[0]:
                best = (mean, float(st), float(sd))
                per = rows
    return best[0], best[1], best[2], per


def apply_temperature(proba, temperature):
    logp = np.log(np.asarray(proba, dtype=np.float64) + 1e-6) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    e /= e.sum(axis=1, keepdims=True)
    return e


def fit_isotonic(probas, labels):
    """Three independent isotonic maps, one per class. Fit on the stacked rows."""
    from sklearn.isotonic import IsotonicRegression
    models = []
    x = np.concatenate([np.asarray(p, dtype=np.float64) for p in probas], axis=0)
    y = np.concatenate([np.asarray(lab) for lab in labels], axis=0)
    # Isotonic on every frame is expensive and over-confident. A stride keeps the fit small.
    step = max(1, len(y) // 80000)
    x = x[::step]
    y = y[::step]
    for c in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1.0 - 1e-4)
        iso.fit(x[:, c], (y == c).astype(np.float64))
        models.append(iso)
    return models


def apply_isotonic(models, proba):
    p = np.asarray(proba, dtype=np.float64)
    out = np.stack([m.predict(p[:, c]) for c, m in enumerate(models)], axis=1)
    out = np.clip(out, 1e-6, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def geo_mean(probas):
    acc = None
    for p in probas:
        lp = np.log(np.asarray(p, dtype=np.float64) + 1e-6)
        acc = lp if acc is None else acc + lp
    acc /= float(len(probas))
    acc -= acc.max(axis=1, keepdims=True)
    e = np.exp(acc)
    e /= e.sum(axis=1, keepdims=True)
    return e


def hold_fill(n, frames, proba):
    """Hold each stride probability forward until the next sample. `frames` is increasing."""
    frames = np.asarray(frames, dtype=np.int32)
    proba = np.asarray(proba, dtype=np.float64)
    if len(frames) == 0:
        raise RuntimeError("no frames to hold")
    out = np.empty((int(n), proba.shape[1]), np.float64)
    pos = np.searchsorted(frames, np.arange(n), side="right") - 1
    pos = np.clip(pos, 0, len(frames) - 1)
    out[:] = proba[pos]
    return out


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def period_strides(period, stride):
    period = np.asarray(period)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        sub = idx[::int(stride)]
        if len(sub):
            yield sub


def feature_norm(cache, matches, cols, stride):
    """Mean and std of the selected columns over both team views. TRAIN matches only."""
    cols = np.asarray(cols, dtype=np.int32)
    parts = []
    for m in matches:
        for key in ("x", "xs"):
            mm = np.load(os.path.join(cache, m + "." + key + ".npy"), mmap_mode="r")
            rows = _stride_rows(mm.shape[0], stride)
            parts.append(take(mm, cols, rows))
            del mm
    f = np.concatenate(parts, axis=0)
    mu = f.mean(axis=0).astype(np.float32)
    sd = np.maximum(f.std(axis=0), 1e-3).astype(np.float32)
    return mu, sd


def _seq_weight(y):
    """Boundary weight on an already-strided sequence. Radius 5 steps is 1 s at 5 Hz."""
    w = np.ones(len(y), np.float32)
    if len(y) < 2:
        return w
    ch = np.zeros(len(y), np.float32)
    ch[1:] = (y[1:] != y[:-1]).astype(np.float32)
    dil = np.convolve(ch, np.ones(11, np.float32), mode="same")
    return np.where(dil > 0.0, np.float32(4.0), np.float32(1.0))


def sequences_for_match(cache, match, cols, proba, mu, sd, stride):
    """Per-period 5 Hz sequences for both team views. proba is the original-view posterior."""
    cols = np.asarray(cols, dtype=np.int32)
    y = np.load(os.path.join(cache, match + ".y.npy"))
    period = np.load(os.path.join(cache, match + ".period.npy"))
    proba = np.asarray(proba, dtype=np.float32)
    views = []
    for key, p_use, y_use in (("x", proba, y), ("xs", proba[:, SWAP], SWAP[y])):
        mm = np.load(os.path.join(cache, match + "." + key + ".npy"), mmap_mode="r")
        seqs = []
        for sub in period_strides(period, stride):
            feat = take(mm, cols, sub)
            feat = (feat - mu) / sd
            pv = np.asarray(p_use[sub], dtype=np.float32)
            dp = np.zeros_like(pv)
            if len(pv) > 1:
                dp[1:] = pv[1:] - pv[:-1]
            xb = np.concatenate([pv, dp, feat], axis=1).astype(np.float32, copy=False)
            yy = np.asarray(y_use[sub], dtype=np.int64)
            seqs.append({
                "x": xb,
                "logp": np.log(np.clip(pv, 1e-6, 1.0)).astype(np.float32),
                "y": yy,
                "w": _seq_weight(yy),
                "frames": np.asarray(sub, dtype=np.int32),
            })
        del mm
        views.append(seqs)
    return views, y, period


class ResidualTCN:
    """Built lazily so importing this module does not import torch."""

    def __new__(cls, in_dim, hidden=32, dilations=(1, 2, 4, 8, 16, 32, 64, 128), dropout=0.1):
        import torch
        from torch import nn

        class _Block(nn.Module):
            def __init__(self, channels, dilation, drop):
                super().__init__()
                self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
                self.norm = nn.GroupNorm(4, channels)
                self.drop = nn.Dropout(drop)
                self.act = nn.GELU()

            def forward(self, h):
                y = self.act(self.norm(self.conv(h)))
                return h + self.drop(y)

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Conv1d(in_dim, hidden, 1)
                self.blocks = nn.ModuleList([_Block(hidden, d, dropout) for d in dilations])
                self.head = nn.Conv1d(hidden, 3, 1)
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)

            def forward(self, x, log_p):
                # x, log_p: (B, T, C). Zero head => output equals log_p at init.
                h = self.in_proj(x.transpose(1, 2))
                for block in self.blocks:
                    h = block(h)
                delta = self.head(h).transpose(1, 2)
                return log_p + delta

        return _Net()


def _tcn_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _iter_chunks(seqs, chunk, step):
    chunks = []
    for seq in seqs:
        t = len(seq["y"])
        if t == 0:
            continue
        if t <= chunk:
            spans = [(0, t)]
        else:
            starts = list(range(0, t - chunk + 1, step))
            if starts[-1] != t - chunk:
                starts.append(t - chunk)
            spans = [(s, chunk) for s in starts]
        for start, length in spans:
            chunks.append((seq, start, length))
    return chunks


def _predict_views(model, views, device):
    """Average the two team views in the original label space. Returns list of (frames, proba) per period."""
    import torch
    model.eval()
    out = []
    view0, view1 = views
    with torch.no_grad():
        for s0, s1 in zip(view0, view1):
            def _fwd(seq):
                x = torch.from_numpy(seq["x"][None]).to(device)
                logp = torch.from_numpy(seq["logp"][None]).to(device)
                logits = model(x, logp)[0].detach().cpu().numpy()
                return softmax_np(logits)

            p0 = _fwd(s0)
            p1 = _fwd(s1)[:, SWAP]
            out.append((s0["frames"], 0.5 * (p0 + p1)))
    return out


def train_tcn_fold(train_views, held_views, held_y, held_period, prior, switch_team, switch_dead,
                   ball_idx, seed, epochs=4, chunk=160, step=80, batch=16):
    """Train on TRAIN views. Keep the epoch with the best held-out spellF1 at frozen costs.

    The last layer starts at zero, so the network begins as the GBDT posterior and only
    learns a residual. Ball-linked feature channels are zeroed on random spans.
    """
    import torch
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(2)
    device = _tcn_device()
    in_dim = train_views[0][0]["x"].shape[1]
    model = ResidualTCN(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(1000 + int(seed))
    flat = [seq for view in train_views for seq in view]
    best_spell = -1.0
    best_state = None
    best_epoch = 1
    history = []
    for epoch in range(epochs):
        model.train()
        chunks = _iter_chunks(flat, chunk, step)
        order = rng.permutation(len(chunks))
        total = 0.0
        seen = 0
        for b0 in range(0, len(order), batch):
            pick = order[b0:b0 + batch]
            x = np.zeros((len(pick), chunk, in_dim), np.float32)
            logp = np.zeros((len(pick), chunk, 3), np.float32)
            y = np.zeros((len(pick), chunk), np.int64)
            w = np.zeros((len(pick), chunk), np.float32)
            mask = np.zeros((len(pick), chunk), np.float32)
            for i, j in enumerate(pick):
                seq, start, length = chunks[int(j)]
                xb = seq["x"][start:start + length].copy()
                if len(ball_idx) and rng.random() < 0.3 and length > 4:
                    a = int(rng.integers(0, max(1, length // 2)))
                    b = int(rng.integers(a + 1, length))
                    xb[a:b, ball_idx] = 0.0
                x[i, :length] = xb
                logp[i, :length] = seq["logp"][start:start + length]
                y[i, :length] = seq["y"][start:start + length]
                w[i, :length] = seq["w"][start:start + length]
                mask[i, :length] = 1.0
            xt = torch.from_numpy(x).to(device)
            lt = torch.from_numpy(logp).to(device)
            yt = torch.from_numpy(y).to(device)
            wt = torch.from_numpy(w * mask).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xt, lt)
            log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            nll = -log_prob.gather(-1, yt[:, :, None]).squeeze(-1)
            denom = wt.sum().clamp_min(1.0)
            loss = (nll * wt).sum() / denom
            delta = logits - lt
            loss = loss + 0.01 * (delta.pow(2).mean())
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite TCN loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(pick)
            seen += len(pick)
        loss_mean = total / max(seen, 1)
        if held_views is None:
            # Final model: no held-out match. Keep the last epoch.
            best_epoch = epoch + 1
            best_spell = -1.0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append({"epoch": epoch + 1, "loss": loss_mean})
        else:
            spell = _held_spell(
                model, held_views, held_y, held_period, prior, switch_team, switch_dead, device)
            history.append({"epoch": epoch + 1, "loss": loss_mean, "spell": spell})
            if spell > best_spell:
                best_spell = spell
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("TCN produced no epoch")
    # Snapshots are on CPU. Load them there, then return the net to the training device.
    model.cpu()
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch, best_spell, history, device


def _held_spell(model, views, y, period, prior, switch_team, switch_dead, device):
    parts = _predict_views(model, views, device)
    frames = np.concatenate([f for f, _ in parts])
    proba = np.concatenate([p for _, p in parts])
    order = np.argsort(frames, kind="mergesort")
    full = hold_fill(len(y), frames[order], proba[order])
    pred = decode_periods(emissions(full, prior), period, switch_team, switch_dead)
    return frame_metrics(y, pred, period)["spell"]


def tcn_full_proba(model, views, n, device):
    parts = _predict_views(model, views, device)
    frames = np.concatenate([f for f, _ in parts])
    proba = np.concatenate([p for _, p in parts])
    order = np.argsort(frames, kind="mergesort")
    return hold_fill(n, frames[order], proba[order])


def ball_anchor(match):
    """Last visible ball in this period, else the pitch centre. Does not look at labels."""
    from bench.feats3 import clocks
    ball = np.asarray(match["ball"])
    _, _, _, last, _ = clocks(match)
    n = len(ball)
    xy = np.empty((n, 2), np.float32)
    known = np.zeros(n, np.float32)
    ok = last >= 0
    if np.any(ok):
        xy[ok] = np.asarray(ball[last[ok], :2], dtype=np.float32)
        known[ok] = 1.0
    miss = ~ok
    xy[miss, 0] = 52.5
    xy[miss, 1] = 34.0
    bad = ~np.isfinite(xy).all(axis=1)
    xy[bad, 0] = 52.5
    xy[bad, 1] = 34.0
    known[bad] = 0.0
    return xy, known


def player_velocity(players, period):
    vel = np.zeros_like(players, dtype=np.float32)
    prev = players[:-1]
    nxt = players[1:]
    both = np.isfinite(prev).all(axis=-1) & np.isfinite(nxt).all(axis=-1)
    delta = np.zeros(nxt.shape, np.float32)
    delta[both] = (nxt[both] - prev[both]) * np.float32(25.0)
    vel[1:] = delta
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        if len(idx):
            vel[idx[0]] = 0.0
    return np.clip(vel, -15.0, 15.0)


def build_tokens(match):
    """(n, slots, 6) and a pad mask. Channels: dx, dy, vx, vy, team, ball-known.

    The set model never reads alive or owner. Team +1 is the side stored first.
    """
    players = np.asarray(match["players"], dtype=np.float32)
    n, sides, slots, _ = players.shape
    anchor, known = ball_anchor(match)
    vel = player_velocity(players, match["period"])
    finite = np.isfinite(players).all(axis=-1)
    rel = np.nan_to_num(players - anchor[:, None, None, :], nan=0.0)
    rel = np.clip(rel, -80.0, 80.0)
    vel = np.nan_to_num(vel, nan=0.0)
    width = sides * slots
    tok = np.zeros((n, width, 6), np.float32)
    pad = np.ones((n, width), dtype=bool)
    for side in range(sides):
        sl = slice(side * slots, (side + 1) * slots)
        tok[:, sl, 0:2] = rel[:, side]
        tok[:, sl, 2:4] = vel[:, side]
        tok[:, sl, 4] = np.float32(1.0 if side == 0 else -1.0)
        tok[:, sl, 5] = known[:, None]
        pad[:, sl] = ~finite[:, side]
    tok[pad] = 0.0
    return tok, pad


class _SetNetHolder:
    pass


def SetEncoder(d=32, heads=4, layers=1, dropout=0.1):
    import torch
    from torch import nn
    try:
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
    except TypeError:
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 2, dropout=dropout, batch_first=True,
        )
    enc = nn.TransformerEncoder(layer, num_layers=layers)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(6, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.enc = enc
            self.head = nn.Linear(d, 3)

        def forward(self, x, pad):
            h = self.proj(x)
            b = h.shape[0]
            cls = self.cls.expand(b, -1, -1)
            h = torch.cat([cls, h], dim=1)
            pad_cls = torch.zeros(b, 1, dtype=torch.bool, device=h.device)
            pad_all = torch.cat([pad_cls, pad], dim=1)
            h = self.enc(h, src_key_padding_mask=pad_all)
            return self.head(h[:, 0])

    return _Net()


def train_set_epoch(model, opt, tokens, pads, labels, device, batch, seed, epoch, mask_prob=0.3):
    """One epoch. tokens/pads/labels are lists of arrays already strided."""
    import torch
    xs = np.concatenate(tokens, axis=0)
    ps = np.concatenate(pads, axis=0)
    ys = np.concatenate(labels, axis=0)
    rng = np.random.default_rng(2000 + int(seed) * 17 + int(epoch))
    order = rng.permutation(len(ys))
    model.train()
    counts = np.bincount(ys, minlength=3).astype(np.float64)
    cw = counts.sum() / (3.0 * np.maximum(counts, 1.0))
    weight = torch.tensor(cw, dtype=torch.float32, device=device)
    total = 0.0
    seen = 0
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        xb = xs[idx].copy()
        pb = ps[idx].copy()
        if rng.random() < mask_prob:
            # Drop the ball anchor on a random subset of the batch: dx, dy, known.
            rows = rng.random(len(idx)) < 0.5
            xb[rows, :, 0:2] = 0.0
            xb[rows, :, 5] = 0.0
        xt = torch.from_numpy(xb).to(device)
        pt = torch.from_numpy(pb).to(device)
        yt = torch.from_numpy(ys[idx].astype(np.int64)).to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xt, pt)
        loss = torch.nn.functional.cross_entropy(logits, yt, weight=weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.detach().cpu()) * len(idx)
        seen += len(idx)
    return total / max(seen, 1)


def predict_set(model, tokens, pads, device, batch=64):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            xt = torch.from_numpy(tokens[i:i + batch]).to(device)
            pt = torch.from_numpy(pads[i:i + batch]).to(device)
            logits = model(xt, pt).detach().cpu().numpy()
            outs.append(softmax_np(logits))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 3), np.float64)


def execute_build(job):
    """Degrade one match and write both team views of the full feature matrix."""
    import time
    from bench.dfl import load
    from bench.degrade import degrade
    from bench.methods import swap_teams
    from bench.run import labels as labeller
    from bench.feats3 import ALL_GROUPS, build_features, clocks

    t0 = time.time()
    clean = load(job["match"])
    deg = degrade(clean, seed=1000 * int(job["seed"]) + int(job["index"]))
    y = labeller(clean)
    del clean
    x, names, ball, sub, kind = build_features(deg, ALL_GROUPS)
    if not np.isfinite(x).all():
        raise RuntimeError("non-finite features for %s" % job["match"])
    atomic_npy(job["x_path"], x)
    del x
    gc.collect()
    xs, names2, _, _, _ = build_features(swap_teams(deg), ALL_GROUPS)
    if names2 != names:
        raise RuntimeError("swap changed feature names for %s" % job["match"])
    if not np.isfinite(xs).all():
        raise RuntimeError("non-finite swapped features for %s" % job["match"])
    atomic_npy(job["xs_path"], xs)
    del xs
    gc.collect()
    vis, gap_back, _, _, _ = clocks(deg)
    atomic_npy(job["y_path"], np.asarray(y))
    atomic_npy(job["period_path"], np.asarray(deg["period"]))
    atomic_npy(job["gap_path"], np.asarray(gap_back))
    atomic_npy(job["vis_path"], np.asarray(vis))
    return {
        "match": job["match"],
        "names": list(names),
        "ball": [int(bool(v)) for v in ball],
        "sub": list(sub),
        "kind": list(kind),
        "frames": int(len(y)),
        "feats": int(len(names)),
        "seconds": round(time.time() - t0, 1),
    }


def _collect_and_fit(kind, cache, train_matches, cols, stride, scheme, params, rounds):
    """Build the training matrix, hand it to the booster, then drop it before the long loop.

    The numpy block is the large allocation. Histogram boosters keep a bin copy, so the
    raw block can go away once that copy exists.
    """
    x, y, w = collect_train(cache, train_matches, cols, stride, scheme)
    rows, feats = int(x.shape[0]), int(x.shape[1])
    prior = _prior_of(y)
    if kind == "lgbm":
        import lightgbm as lgb
        kw = {}
        if w is not None:
            kw["weight"] = np.asarray(w, dtype=np.float32)
        dset = lgb.Dataset(x, label=y, params=dict(params), free_raw_data=True, **kw)
        dset.construct()
        del x, y, w
        gc.collect()
        model = lgb.train(dict(params), dset, num_boost_round=int(rounds))
        del dset
    elif kind == "xgb":
        import xgboost as xgb
        kw = {"label": y}
        if w is not None:
            kw["weight"] = np.asarray(w, dtype=np.float32)
        dm = xgb.DMatrix(x, **kw)
        del x, y, w
        gc.collect()
        model = xgb.train(dict(params), dm, num_boost_round=int(rounds))
        del dm
    else:
        model, prior, gain = fit_model(kind, x, y, w, params, rounds)
        del x, y, w
        gc.collect()
        return model, prior, gain, rows, feats
    gain = _importance(kind, model, feats)
    gc.collect()
    return model, prior, gain, rows, feats


def _fit_and_predict(kind, cache, train_matches, predict_matches, cols, stride, scheme, params, rounds):
    """Fit, drop the training matrix, then predict. Peak RAM is one matrix, not two."""
    model, prior, gain, rows, feats = _collect_and_fit(
        kind, cache, train_matches, cols, stride, scheme, params, rounds)
    written = []
    for m in predict_matches:
        proba = predict_match(model, kind, cache, m, cols)
        written.append((m, np.asarray(proba, dtype=np.float32)))
        del proba
        gc.collect()
    del model
    gc.collect()
    return {
        "prior": [float(v) for v in prior],
        "gain": [float(v) for v in gain],
        "rows": rows,
        "feats": feats,
        "probas": written,
    }


def gain_keep(gain, mass=0.995, min_keep=30):
    """Indices of features that carry the leading `mass` of total gain, at least min_keep."""
    gain = np.asarray(gain, dtype=np.float64)
    order = np.argsort(-gain, kind="mergesort")
    total = float(gain.sum())
    if total <= 0:
        return order
    csum = np.cumsum(gain[order])
    n_keep = int(np.searchsorted(csum, mass * total, side="left") + 1)
    n_keep = min(len(gain), max(int(min_keep), n_keep))
    return order[:n_keep]


def self_test():
    """Synthetic checks only. Prints a pass line, not feature values."""
    import torch
    from bench.feats3 import build_features, column_index, _synthetic_shape

    # Viterbi parity, including planted ties.
    rng = np.random.default_rng(0)
    e = rng.normal(size=(180, 3))
    e[10] = 0.0
    e[11] = 1.0
    for st, sd in ((0.0, 0.0), (3.0, 3.0), (5.0, 40.0), (80.0, 10.0)):
        got = viterbi_rows(e, st, sd)
        exp = viterbi_fast(e, st, sd)
        if not np.array_equal(got, exp):
            raise AssertionError("numba viterbi mismatch at %s %s" % (st, sd))
    period = np.array([1, 1, 1, 2, 2, 2] * 30, np.int8)
    full = decode_periods(e, period, 5.0, 40.0)
    exp = decode_periods_fast(e, period, 5.0, 40.0)
    if not np.array_equal(full, exp):
        raise AssertionError("decode_periods mismatch")

    # Column schema of v2.
    shape = _synthetic_shape()
    x, names, ball, sub, kind = build_features(shape, ("context", "lang_dead"))
    cols, used = column_index(names, sub, np.asarray(ball), ("context", "lang_dead"))
    v1_cols, v1_names = column_index(names, sub, np.asarray(ball), ())
    if len(v1_cols) != 49:
        raise AssertionError("v1 width %d" % len(v1_cols))
    if len(cols) != 351 or len(used) != 351:
        raise AssertionError("v2 width %d" % len(cols))
    if not np.isfinite(x).all():
        raise AssertionError("non-finite synthetic v2 features")

    # Boundary weights ignore team identity and mark a change.
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2], np.int8)
    per = np.ones(len(y), np.int8)
    w = boundary_weights(y, per, radius=1, boost=4.0)
    if not np.array_equal(w, boundary_weights(SWAP[y], per, radius=1, boost=4.0)):
        raise AssertionError("boundary weights moved when teams swapped")
    if w[3] != 4.0 or w[0] != 1.0:
        raise AssertionError("boundary weight placement")

    # Temperature 1 and the geometric mean of a model with itself.
    p = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]], np.float64)
    if not np.allclose(apply_temperature(p, 1.0), p / p.sum(axis=1, keepdims=True), atol=1e-6):
        # apply_temperature renorms from log(p+eps), so it is close, not identical.
        q = apply_temperature(p, 1.0)
        if np.max(np.abs(q - p)) > 1e-3:
            raise AssertionError("temperature 1 drifted")
    if np.max(np.abs(geo_mean([p, p]) - p / p.sum(1, keepdims=True))) > 1e-3:
        raise AssertionError("geo mean of copies")
    sharp = apply_temperature(p, 0.5)
    if sharp[0, 2] <= p[0, 2]:
        raise AssertionError("temperature below 1 should sharpen the mode")

    # Hold-forward at stride 5.
    frames = np.array([0, 5], np.int32)
    pr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    held = hold_fill(10, frames, pr)
    if not np.allclose(held[:5, 0], 1.0) or not np.allclose(held[5:, 1], 1.0):
        raise AssertionError("hold-fill")

    # Zero-init residual TCN reproduces log p.
    torch.manual_seed(0)
    net = ResidualTCN(16, hidden=32, dropout=0.0)
    net.eval()
    xb = torch.randn(2, 12, 16)
    lp = torch.randn(2, 12, 3)
    if not torch.allclose(net(xb, lp), lp):
        raise AssertionError("TCN residual is not zero at init")

    # Set tokens ignore labels, and swapping teams swaps the side channels.
    scrambled = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in shape.items()}
    scrambled["owner"] = np.ones(len(shape["ball"]), np.int8)
    scrambled["alive"] = np.zeros(len(shape["ball"]), dtype=bool)
    tok, pad = build_tokens(shape)
    tok2, pad2 = build_tokens(scrambled)
    if not np.array_equal(tok, tok2) or not np.array_equal(pad, pad2):
        raise AssertionError("set tokens changed when labels changed")
    if not np.isfinite(tok).all():
        raise AssertionError("non-finite set tokens")
    swapped = build_tokens(methods.swap_teams(shape))[0]
    slots = shape["players"].shape[2]
    if not np.allclose(swapped[:, :slots, 0:2], tok[:, slots:2 * slots, 0:2]):
        raise AssertionError("swap_teams did not exchange set-token positions")

    # Tiny deterministic boosters. CatBoost is included when the import works.
    rng = np.random.default_rng(1)
    xx = rng.normal(size=(240, 6)).astype(np.float32)
    yy = np.array([0, 1, 2] * 80, dtype=np.int32)
    m1, _, _ = fit_model("lgbm", xx, yy, None, lgbm_params(num_leaves=7, min_child_samples=5), 8)
    m2, _, _ = fit_model("lgbm", xx, yy, None, lgbm_params(num_leaves=7, min_child_samples=5), 8)
    a = predict_matrix(m1, "lgbm", xx[:20])
    b = predict_matrix(m2, "lgbm", xx[:20])
    if not np.allclose(a, b):
        raise AssertionError("LightGBM is not deterministic")
    if np.max(np.abs(a.sum(1) - 1)) > 1e-5:
        raise AssertionError("probabilities do not sum to 1")
    mx, _, _ = fit_model("xgb", xx, yy, None, xgb_params(), 6)
    px = predict_matrix(mx, "xgb", xx[:15])
    if px.shape != (15, 3):
        raise AssertionError("xgb predict shape")
    et_kw = et_params()
    et_kw["n_estimators"] = 3
    et_kw["max_depth"] = 3
    et_kw["max_samples"] = 40
    et_kw["min_samples_leaf"] = 2
    et_kw["max_features"] = 3
    me, _, _ = fit_model("et", xx, yy, None, et_kw, 1)
    pe = predict_matrix(me, "et", xx[:15])
    if pe.shape != (15, 3):
        raise AssertionError("extratrees predict shape")
    try:
        mc, _, _ = fit_model("cat", xx, yy, None, cat_params(), 4)
        pc = predict_matrix(mc, "cat", xx[:12])
        if pc.shape != (12, 3):
            raise AssertionError("catboost predict shape")
    except Exception as exc:
        raise AssertionError("catboost tiny fit failed: %s" % exc)

    # Set encoder runs on a fully padded row and a normal row.
    torch.manual_seed(0)
    device = torch.device("cpu")
    sen = SetEncoder(d=32, heads=4, layers=1, dropout=0.0).to(device)
    tokens = torch.randn(2, 5, 6)
    padm = torch.zeros(2, 5, dtype=torch.bool)
    padm[1] = True
    logits = sen(tokens, padm)
    if logits.shape != (2, 3) or not torch.isfinite(logits).all():
        raise AssertionError("set encoder forward")
    print("boost self_test ok")
    return True
