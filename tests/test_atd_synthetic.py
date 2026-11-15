"""socbac.atd on a tiny synthetic EPTS file (no competition data involved)."""

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import asset_paths, GAMES  # noqa: E402
from socbac import atd  # noqa: E402


def test_parse_synthetic():
    with tempfile.TemporaryDirectory() as tmp:
        paths = asset_paths(data_dir=tmp)
        os.makedirs(os.path.dirname(paths["atd_raw"]))
        with open(paths["atd_meta"], "w") as fh:
            fh.write("<Root><Teams><Team id='T1'><Name>Home</Name></Team><Team id='T2'><Name>Away</Name></Team></Teams>"
                     "<Players><Player id='P1' teamId='T1'/><Player id='P2' teamId='T2'/><Player id='P3' teamId='T1'/></Players></Root>")
        with open(paths["atd_raw"], "w") as fh:
            fh.write("1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,0.0:0.5,0.25,3.0\n")
            fh.write("11186:NaN,NaN,NaN;0.9,0.1,2.0;NaN,NaN,NaN:NaN,NaN,NaN\n")
        out = atd.load(use_cache=False, data_dir=tmp)

    length, width = GAMES["BAR-ATH"]["pitch"]
    n = 199319
    assert out["ball"].shape == (n, 3) and out["players"].shape == (n, 2, 2, 2)
    assert np.allclose(out["ball"][0, :2], (0.5 * length, 0.25 * width))
    assert np.isnan(out["ball"][11185, 0])
    assert np.allclose(out["players"][0, 0, 0], (0.5 * length, 0.5 * width))   # P1 home slot 0
    assert np.allclose(out["players"][0, 0, 1], (0.1 * length, 0.2 * width))   # P3 home slot 1
    assert np.isnan(out["players"][0, 1, 0, 0])                                 # P2 untracked
    assert np.allclose(out["players"][11185, 1, 0], (0.9 * length, 0.1 * width))
    assert out["period"][0] == 0 and out["period"][11185] == 1 and out["period"][110808] == 2
    assert out["period"][95000] == 0


if __name__ == "__main__":
    test_parse_synthetic()
    print("ok")
