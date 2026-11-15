"""Print visibility, gap quantiles and the alive<->dead gap-edge probe for v4 and v2s (aggregates only)."""
import numpy as np
from bench import dfl
from bench.degrade_v4 import DEGRADERS, gap_lengths, edge_probe
from bench.run import TRAIN, TEST
ORD = TRAIN + TEST
for name in ("v4", "v2s"):
    vis, gaps, hit, ch, ntr, dead_hid, alive_hid = [], [], [], [], 0, [], []
    for i, m in enumerate(ORD):
        c = dfl.load(m); d = DEGRADERS[name](c, 1000 * 0 + i)
        bx = d["ball"][:, 0]
        vis.append(1 - np.isnan(bx).mean()); gaps.append(gap_lengths(bx, d["period"]))
        a, b, n = edge_probe(c["alive"], bx); hit.append(a * n); ch.append(b); ntr += n
        hid = np.isnan(bx); dead_hid.append(hid[~c["alive"]].mean()); alive_hid.append(hid[c["alive"]].mean())
    g = np.concatenate(gaps)
    print(name, "visible %.3f" % np.mean(vis), "gaps n=%d median %.0f mean %.0f p90 %.0f" % (len(g), np.median(g), g.mean(), np.percentile(g, 90)),
          "| alive<->dead transitions with gap edge within +-2: %.1f%% (chance %.1f%%, n=%d)" % (100 * sum(hit) / ntr, 100 * np.mean(ch), ntr),
          "| hidden share dead %.2f alive %.2f" % (np.mean(dead_hid), np.mean(alive_hid)), flush=True)
