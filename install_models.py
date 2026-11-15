"""Copy the shipped model files into the folder the pipeline reads, and check every SHA-256.

    python install_models.py

Source: models/final3_ship (tracked).  Target: data/derived/pipeline_final3/ship (git-ignored).
The pipeline checks the same manifest again before it uses any file.
"""
import hashlib, json, shutil, sys
from pathlib import Path

SRC = Path(__file__).parent / "models" / "final3_ship"
DST = Path(__file__).parent / "data" / "derived" / "pipeline_final3" / "ship"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


man = json.loads((SRC / "manifest.json").read_text())
DST.mkdir(parents=True, exist_ok=True)
for name, meta in man.items():
    s = SRC / name
    if sha(s) != meta["sha256"]:
        sys.exit(f"REFUSED: {name} does not match the manifest (line endings changed? see .gitattributes)")
    shutil.copyfile(s, DST / name)
shutil.copyfile(SRC / "manifest.json", DST / "manifest.json")
print(f"installed {len(man)} files + manifest into {DST}")
