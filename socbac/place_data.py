"""Install the downloaded dataset into the layout games.asset_paths() expects.

    python -m socbac.place_data [DOWNLOAD_DIR]        # default: ~/Downloads

Looks in DOWNLOAD_DIR for:
  - the Drive zip parts ("More Than A Hack 2027*.zip"), extracting each known asset by file name;
  - the two videos (BAR-ATH_tactical*.mp4, BAR-ATH_broadcast*.mp4; Drive appends "-001" etc.).

Files are copied, never moved, so the downloads stay where they are. A target that already exists
with the same size is skipped. Output is file names, sizes and ok/missing only, never contents
(AGENTS.md section 1).
"""

import glob
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import SOURCES, asset_paths  # noqa: E402


def targets():
    """file name -> destination path, for every asset of every source."""
    out = {}
    for source in SOURCES:
        for path in asset_paths(source=source).values():
            out[os.path.basename(path)] = path
    return out


def copy_stream(src_fh, dst, size):
    tmp = dst + ".part"
    with open(tmp, "wb") as out:
        shutil.copyfileobj(src_fh, out, length=16 * 1024 * 1024)
    if os.path.getsize(tmp) != size:
        os.remove(tmp)
        raise IOError("size mismatch writing " + os.path.basename(dst))
    os.replace(tmp, dst)


def main(download_dir):
    want = targets()
    done = {}

    for z in sorted(glob.glob(os.path.join(download_dir, "More Than A Hack 2027*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                name = os.path.basename(info.filename)
                if name not in want:
                    continue
                dst = want[name]
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                    done[name] = "already there"
                    continue
                with zf.open(info) as fh:
                    copy_stream(fh, dst, info.file_size)
                done[name] = "extracted"

    for source in SOURCES:
        dst = asset_paths(source=source)["video"]
        name = os.path.basename(dst)
        stem = name[:-len(".mp4")]
        cands = sorted(glob.glob(os.path.join(download_dir, stem + "*.mp4")))
        if not cands:
            continue
        src = cands[0]
        size = os.path.getsize(src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            done[name] = "already there"
            continue
        with open(src, "rb") as fh:
            copy_stream(fh, dst, size)
        done[name] = "copied"

    missing = 0
    for name, dst in sorted(want.items()):
        if os.path.exists(dst):
            print("ok       {:>14,}  {}  ({})".format(os.path.getsize(dst), name, done.get(name, "present")))
        else:
            missing += 1
            print("MISSING  {:>14}  {}".format("", name))
    print("{} of {} assets in place".format(len(want) - missing, len(want)))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads")))
