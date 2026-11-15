"""Open DFL/Sportec (IDSSE) matches -> compact per-frame arrays.

Source: Bassek et al., "An integrated dataset of spatiotemporal and event data in elite soccer"
(IDSSE), CC-BY 4.0, mirrored at https://huggingface.co/datasets/pysport/idsse-data.

The raw position files carry, for every frame, a ball-alive flag and the team in possession.
Those are exactly the two things the challenge ground truth is derived from (dead -> D, otherwise
the owning team), so they serve as reference labels with no relabelling.

    python -m bench.dfl fetch      # download the 7 matches (~2.6 GB raw XML)
    python -m bench.dfl convert    # parse to bench/cache/npz/<match>.npz (~75 s per match)
"""

import argparse
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RAW = os.path.join(CACHE, "raw")
NPZ = os.path.join(CACHE, "npz")
# pinned to a commit (see bench/idsse_manifest.txt for hashes and DOIs); "main" can change under us
HF_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
HF = "https://huggingface.co/datasets/pysport/idsse-data/resolve/" + HF_REVISION + "/"

# match id -> competition id in the file names
MATCHES = {
    "J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
    "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002",
}
PITCH = (105.0, 68.0)
MAX_PLAYERS = 20  # per team, squad members who appear; padded with NaN


def file_names(match):
    comp = MATCHES[match]
    return ("DFL_02_01_matchinformation_DFL-COM-{}_DFL-MAT-{}.xml".format(comp, match),
            "DFL_04_03_positions_raw_observed_DFL-COM-{}_DFL-MAT-{}.xml".format(comp, match))


def fetch():
    os.makedirs(RAW, exist_ok=True)
    for match in MATCHES:
        for name in file_names(match):
            path = os.path.join(RAW, name)
            if os.path.isfile(path) and os.path.getsize(path) > 1000:
                continue
            print("downloading", name)
            urllib.request.urlretrieve(HF + name, path)


def convert(match):
    """Parse one match and write bench/cache/npz/<match>.npz."""
    from kloppy import sportec
    from kloppy.domain import BallState

    meta, raw = (os.path.join(RAW, n) for n in file_names(match))
    ds = sportec.load_tracking(raw_data=raw, meta_data=meta, only_alive=False)
    home_team, away_team = ds.metadata.teams
    length, width = PITCH

    slots = {}
    for side, team in ((0, home_team), (1, away_team)):
        for i, player in enumerate(team.players[:MAX_PLAYERS]):
            slots[player.player_id] = (side, i)

    n = len(ds.frames)
    period = np.zeros(n, np.int8)
    ball = np.full((n, 3), np.nan, np.float32)
    alive = np.zeros(n, bool)
    owner = np.full(n, -1, np.int8)  # 0 home, 1 away, -1 unknown
    players = np.full((n, 2, MAX_PLAYERS, 2), np.nan, np.float32)

    for t, frame in enumerate(ds.frames):
        period[t] = frame.period.id
        alive[t] = frame.ball_state == BallState.ALIVE
        if frame.ball_owning_team is not None:
            owner[t] = 0 if frame.ball_owning_team.team_id == home_team.team_id else 1
        b = frame.ball_coordinates
        if b is not None and b.x is not None:
            ball[t] = (b.x * length, b.y * width, b.z if b.z is not None else 0.0)
        for player, data in frame.players_data.items():
            slot = slots.get(player.player_id)
            if slot is None or data.coordinates is None:
                continue
            players[t, slot[0], slot[1]] = (data.coordinates.x * length,
                                            data.coordinates.y * width)

    os.makedirs(NPZ, exist_ok=True)
    out = os.path.join(NPZ, match + ".npz")
    np.savez_compressed(out, period=period, ball=ball, alive=alive, owner=owner, players=players)
    print("{}: {} frames, alive {:.1%}, owner known {:.1%}".format(
        match, n, alive.mean(), (owner >= 0).mean()))
    return out


def load(match):
    with np.load(os.path.join(NPZ, match + ".npz")) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["fetch", "convert"])
    parser.add_argument("--match", nargs="*", default=list(MATCHES))
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch()
    else:
        for m in args.match:
            if os.path.isfile(os.path.join(NPZ, m + ".npz")):
                print(m, "already converted")
                continue
            convert(m)
    sys.exit(0)
