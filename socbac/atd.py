"""Read the provided tactical ATD into the same array layout the bench uses.

Streams the raw EPTS text exactly as starter/baseline.py does (numpy only). Nothing here prints
or returns data to a caller other than the pipeline; per-frame arrays are cached under
data/derived/ (gitignored) only.

Layout returned (n = 199319 video frames, video_frame = frame_id - 1):
    ball     (n, 3)  metres, z NaN (the ATD has no ball height); NaN where the ball is missing
    players  (n, 2, S, 2)  metres, team 0 = home, 1 = away, S slots per team, NaN if not tracked
    period   (n,)    1 / 2 on the graded halves, 0 elsewhere (games.half_video_blocks)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402  starter/baseline.py, MIT

CACHE = os.path.join(ROOT, "data", "derived", "atd_arrays.npz")


def _cache_path(raw_path):
    st = os.stat(raw_path)
    return CACHE.replace(".npz", "_{}_{}.npz".format(st.st_size, int(st.st_mtime)))


def load(game=GAME, use_cache=True, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = _cache_path(raw_path)
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    team = team_of_each_track(meta_path)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    slot_of = np.zeros(len(team), int)
    counts = [0, 0]
    for i, t in enumerate(team):
        slot_of[i] = counts[t]
        counts[t] += 1
    s = max(counts)

    ball = np.full((n, 3), np.nan, np.float32)
    players = np.full((n, 2, s, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            t = int(parts[0]) - 1
            b = parts[2].split(",")
            if b[0][:1] != "N":
                ball[t, 0] = float(b[0]) * length
                ball[t, 1] = float(b[1]) * width
            for k, group in enumerate(parts[1].split(";")):
                g = group.split(",")
                if g[0][:1] == "N":
                    continue
                players[t, team[k], slot_of[k]] = (float(g[0]) * length, float(g[1]) * width)

    period = np.zeros(n, np.int8)
    for half, first, last in half_video_blocks(game):
        period[first:last + 1] = half

    out = {"ball": ball, "players": players, "period": period}
    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **out)
    return out
