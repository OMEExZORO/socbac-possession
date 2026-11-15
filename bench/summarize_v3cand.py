"""Summarise results.jsonl: mean +- sd over (match, seed) runs, per-seed means, gains."""
import json, collections
import numpy as np
RES = r"D:\Barca_Hackathon\research\runs\v3cand\results.jsonl"
R = collections.defaultdict(dict)
for line in open(RES, encoding="utf-8"):
    r = json.loads(line); R[(r["deg"], r["config"])][r["seed"]] = r
def seedmean(r): return float(np.mean([x["spell"] for x in r["rows"]]))
print("%-10s %-10s %-6s %-14s %-7s %-12s %s" % ("deg", "config", "seeds", "spellF1+-sd", "macro", "poss pred/gt", "per-seed"))
for (deg, cfg), d in sorted(R.items()):
    rows = [x for r in d.values() for x in r["rows"]]
    s = np.array([x["spell"] for x in rows])
    print("%-10s %-10s %-6s %.3f+-%.3f   %.3f   %.0f/%.0f      %s  costs %s" % (
        deg, cfg, ",".join(map(str, sorted(d))), s.mean(), s.std(), np.mean([x["macro"] for x in rows]),
        np.mean([x["npred"] for x in rows]), np.mean([x["ngt"] for x in rows]),
        " ".join("s%d=%.3f" % (k, seedmean(d[k])) for k in sorted(d)), [d[k]["costs"] for k in sorted(d)]))
print("\nPer-seed gains (spellF1)")
for deg in sorted({k[0] for k in R}):
    for a, b in (("v1", "baseline"), ("v2", "v1"), ("v3", "v2")):
        da, db = R.get((deg, a), {}), R.get((deg, b), {})
        ks = sorted(set(da) & set(db))
        if ks: print(deg, "%s over %s:" % (a, b), " ".join("s%d %+.3f" % (k, seedmean(da[k]) - seedmean(db[k])) for k in ks))
