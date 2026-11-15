# Instructions for AI coding assistants

These rules apply to every AI tool used on this repo: Claude Code, Codex, Cursor, Copilot, Gemini,
ChatGPT and any others. Humans should also read [docs/team/RULES_DIGEST.md](docs/team/RULES_DIGEST.md).

This repo is our team's entry to **More Than a Hack 2027**, Phase 1: label every frame of
BAR-ATH as `H`/`A`/`D`/`X`. Its upstream is `metrica-sports/more-than-a-hack-2027`.

## 1. Data firewall (non-negotiable)

The match data is licensed under terms that forbid giving any cloud-hosted LLM the videos,
datasets or **extracts** of them (T&C §8). You are probably a cloud-hosted LLM. Breaking this can
disqualify the team.

**Never:**
- Open, read, `cat`/`head`/`tail`/`type`/`Get-Content`, grep or print the contents of anything
  in `data/`, or of any file named `*_atd.txt`, `*_atd.xml`, `*_homography.csv`,
  `*_smart_tagging.xml` or `*.mp4`, wherever it is (`data/`, Downloads, `MTAH_DATA_DIR`, ...).
- Write or run code whose output shows raw rows, values, samples or per-frame records of the data.
  That rules out `df.head()`, `print(arr[:10])`, `describe()` on raw columns, and dumping parsed
  XML.
- View, render or describe video frames, images or plots made from the data.
- Paste data into prompts, issues, commit messages or docs.
- Work around a permission denial by copying, renaming, or reading the file through
  `python -c`. A denial means stop and ask the user.

**You may:**
- Write code that loads and processes the data **locally**. Get paths from `games.py`
  (`asset_paths()` / `require()`) and never hard-code them.
- Run that code and read back:
  - scores from `evaluation/run_evaluation.py`
  - the output of `evaluation/validate_submission.py`
  - label and possession counts of **our own** predictions
  - shapes, row counts, timings, and pass/fail assertions
  - tracebacks, as long as they contain no data values. Catch those errors and re-raise them
    without the values.
- Use facts already published in the repo docs (`DATA.md`, `CHALLENGE.md`, `EVALUATION.md`, ...).
- Read the public files `evaluation/gt/BAR-ATH_possession_gt_public*` and
  `evaluation/results/baseline_*.csv`.

**Writing scripts:**
- Print aggregate summary lines only.
- Write per-frame intermediate output (features, caches, extracted frames) **only under
  `data/derived/`**. That folder is gitignored and blocked for agents.
- If you are unsure whether an output counts as an extract, don't print it. Ask the user.

## 2. Competition rules your code must respect

- **Labels come from algorithms only.** No hand labels or hand corrections on BAR-ATH, and no
  constants or special cases tied to specific frames of this match.
- **The public 5-minute GT (frames 11185–18684) is a pipeline check only.** Never use it to train,
  tune hyperparameters or choose between methods. Don't write search loops that score against it.
- **No human-produced data about this match** from any source (event data, corrected tracking,
  annotations). Data about other matches is fine, but it must be documented.
- **Reproducible to the byte.** Seed all randomness, keep the processing order deterministic, and
  pin dependencies in `requirements-lock.txt`. A single command must regenerate the submission.
- **Never commit the data or anything that could rebuild it.** That covers `data/`, frame-level
  exports, cached features, submissions other than the shipped `baseline_*.csv`, and the
  Drive or submission-form links. Run `git status` before every commit.

## 3. Repo conventions

- Upstream files stay unedited so organiser updates merge cleanly
  (`git fetch upstream && git merge upstream/main`). Our code goes in new files and folders.
- Python 3.11 in `.venv`, managed by uv (`docs/team/SETUP.md`). On Windows, run
  `.venv\Scripts\python.exe`.
- Ranking metric: `spellF1`. The noise band on the full match is ±0.04, and the floor is 0.2433.
