# SOCBAC: frame-level possession for More Than a Hack 2027 (Phase 1)

Team **SOCBAC**. This repository regenerates our submission `SOCBAC.csv` exactly: 199,319
rows, one label per frame of the tactical video (`H`, `A`, `D`; `X` outside the graded halves).
The method, data and limitations are described in `SOCBAC.pdf` (technical document).

**Reference output:** `SOCBAC.csv`, SHA-256 `ae09ef16c9d5296a1b3a29abfd82672ac607ab117fc300a21d32a82ccb95975d`. It was reproduced
byte-for-byte on two different machines (Windows 11; one with an RTX 3050 GPU running the CPU path,
one CPU-only), from a fresh clone and a fresh environment.

## Method in one paragraph
Three models, all trained **only on 7 open DFL/IDSSE matches** (CC BY 4.0). The open matches are
degraded to resemble the provided automatic tracking.
- **Gradient boosting (LightGBM)**: per-frame features from ball and player proximity, ball-free
  team shape, dead-ball activity and team-shape look-ahead. It was trained on copies of the open matches degraded with broad random settings; the tracking smoothing, for example, is drawn for each copy (ball 5–40 frames, players 3–25).
- **Ball reconstructor**: a set transformer over the players that estimates where the ball is
  while the tracking has lost it.
- **Touch network**: reads the players and the reconstructed ball. It was trained with
  distillation from a teacher that saw the clean ball.

The two possession models' per-frame probabilities of H / A / D are averaged and decoded per half
by a three-state Viterbi path with team-switch cost 30 (dead-ball cost 20). Possession labels count a change only after controlled possession:
merges of opposing runs shorter than 1 s and 2 s, pooled.

A **tempo wrapper** around all three models measures one label-free number on the provided match,
its playing pace. If the match is slower than the training matches, it resamples the frames to the
training pace before prediction and maps the result back.

No label, video frame or ground-truth value of the provided match is used to train the models or to choose their settings. A few label-free summaries of the provided tracking, and early checks against the public 5-minute window, influenced the design; the technical document lists them (section 9).

## Reproduce

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/). No GPU is needed.

```bash
git clone https://github.com/OMEExZORO/socbac-possession socbac && cd socbac
uv venv --python 3.11 .venv
# 1) all packages except torch (PyPI)            2) torch, CPU build (PyTorch index)
uv pip sync    --python .venv/Scripts/python.exe requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -r requirements-torch.txt
.venv/Scripts/python.exe install_models.py      # copies models/final3_ship into data/derived/pipeline_final3/ship and checks SHA-256
# (macOS / Linux: use .venv/bin/python)
```

**The provided match data** (not included; from the organisers): place it as

```
data/tactical/BAR-ATH_tactical{.mp4,_atd.xml,_atd.txt,_homography.csv,_smart_tagging.xml}
```

or set `MTAH_DATA_DIR` to the folder that contains `tactical/`.

**The submission, in one command** (from the shipped weights; SHA-256 checked before use):

```bash
.venv/Scripts/python.exe -m socbac.pipeline_final3_c30 --out evaluation/results/SOCBAC.csv --device cpu
.venv/Scripts/python.exe evaluation/validate_submission.py evaluation/results/SOCBAC.csv
```

Runtime: about 3 to 8 minutes on a laptop CPU. The command prints the team-switch cost (30), the tempo decision and the
SHA-256 of the CSV, which must be `ae09ef16c9d5296a1b3a29abfd82672ac607ab117fc300a21d32a82ccb95975d`. `--tempo-off` reproduces the unwrapped pipeline with the same decoder (`d3294602bfa6510b6b06be484c4724c1829fd506f50f28e8b913c31351d81164`).

**To retrain or rerun the bench** (open data: 7 IDSSE/DFL matches, about 2.6 GB, pinned to a fixed
Hugging Face commit):

```bash
.venv/Scripts/python.exe -m bench.dfl fetch
.venv/Scripts/python.exe -m bench.dfl convert
.venv/Scripts/python.exe -m socbac.pipeline_final3 --variant pcf --bench          # open-data bench, TRAIN 4 -> TEST 3 (GPU recommended)
.venv/Scripts/python.exe -m socbac.pipeline_final3 --variant pcf --train-final    # retrain the shipped weights on all 7 matches
```

### What the command does
1. Checks the shipped files (`data/derived/pipeline_final3/ship`, filled by `install_models.py` from `models/final3_ship`) against their SHA-256 manifest. It refuses on any
   mismatch.
2. Reads the provided tracking and keeps the graded halves. The second half starts at the
   organisers' published kick-off anchor; frames before it are `D`.
3. **Tempo rule.** It computes the pace: the 75th percentile, over graded frames, of the per-frame
   median player speed. Then `k = 2.23 / pace`, where 2.23 m/s is the mean pace of the 4 open
   training matches, measured on their degraded copies.
   - If `k` > 1.03 (the match is more than 3% slower), the frames are resampled by `k` inside each
     half. Otherwise nothing changes.
   - Faster matches are never resampled.
   - On BAR-ATH the factor is 1.12, so the wrapper acts.
4. Runs the GBM, the ball reconstructor and the touch network on CPU in float32, maps their
   outputs back to the original frames, averages them, decodes them, and writes and validates the
   CSV.

### Why the weights are shipped
- The neural weights were trained with this repository's code on the 7 open matches only. They
  contain nothing from the provided match.
- Neural training is bit-reproducible only on the same GPU and software stack. Retraining elsewhere
  gives weights that differ in the last bits, and a CSV that differs in a few frames.
- Shipping the weights and predicting on CPU makes the CSV identical on any machine.
- `--train-final` retrains them; expect a slightly different CSV.

## Repository map
| Path | What |
|---|---|
| `socbac/pipeline_final3_c30.py` (entry point), `socbac/pipeline_final3_tempo.py`, `socbac/pipeline_final2_tempo.py`, `socbac/tempo.py` | The one-command submission (tempo wrapper around the final pipeline) |
| `socbac/pipeline_final3.py`, `socbac/p2_nets.py`, `socbac/pc_degrade.py`, `socbac/pipeline_final2.py` | Final pipeline: bench, training on the open matches (`pc_degrade`: degradation of the open matches), inference; `pipeline_final2` supplies the network loaders |
| `socbac/pipeline_final.py`, `socbac/setnet_pool.py`, `socbac/atd.py`, `socbac/pipeline_v2.py` | GBM features, set model helpers, reading the provided tracking, graded-half layout |
| `bench/` | Open-data loading (`dfl.py`), degradation, labels, features: the bench used for every design decision |
| `models/final3_ship`, `install_models.py` | Shipped weights, decoder settings, SHA-256 manifest, and the step that installs them |
| `evaluation/`, `starter/`, `games.py`, `*.md` (upper case) | The organisers' files, unmodified |
| `AGENTS.md` | Rules for AI coding tools (no match data to any AI service) |

## Data and licences
- IDSSE/DFL open data: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset
  of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195.
  doi:10.1038/s41597-025-04505-y. CC BY 4.0. Mirror used: Hugging Face `pysport/idsse-data`.
- The provided match data and ground truth belong to the organisers and are not redistributed here.
- Code: see `LICENSE` (the organisers' files keep their own licence).
