"""Install already-unzipped dataset folders into the layout games.asset_paths() expects.

    python -m socbac.place_folders SEARCH_DIR [--videos]

Like place_data.py but for folders that were extracted from the Drive zip parts already
(SEARCH_DIR/**/<asset file name>). Files are copied, never moved. Videos are skipped unless
--videos is given. Prints file names, sizes and ok/missing only, never contents (AGENTS.md s.1).
"""

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import SOURCES, asset_paths  # noqa: E402


def main(search_dir, videos):
    want = {}
    for source in SOURCES:
        for path in asset_paths(source=source).values():
            if path.lower().endswith(".mp4") and not videos:
                continue
            want[os.path.basename(path)] = path
    found = {}
    for dirpath, _dirs, files in os.walk(search_dir):
        if os.path.abspath(dirpath).startswith(os.path.join(ROOT, "")):
            continue
        for f in files:
            if f in want and f not in found:
                found[f] = os.path.join(dirpath, f)
    for name, dst in sorted(want.items()):
        src = found.get(name)
        if src is None:
            print("missing  ", name)
            continue
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print("already  ", name, os.path.getsize(dst))
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        print("copied   ", name, os.path.getsize(dst))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], "--videos" in sys.argv)
