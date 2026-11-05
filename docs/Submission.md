# Submission checklist (Assessment v2.0 §10)

## 1. Pull request (do this first)

**Branch:** `dev-test01` → **`main`** on your repo.

Open: **https://github.com/kashifmayar/measure-it/compare/main...dev-test01**

**Pull request:** https://github.com/kashifmayar/measure-it/pull/1 (`dev-test01` → `main`, 9 commits)

## 2. Recalibrate (last step before you submit)

Only after PR + validation docs are ready:

```powershell
.\.venv\Scripts\python.exe calibration\calibrate.py --square-size 20.0
.\.venv\Scripts\python.exe main.py --image dataset\raw\pen_0042.HEIC --marker-size-mm 198.0 --checkpoint models\runs\unet_pen\best.pt --allow-poor-calibration --skip-undistort --output-dir measurement\output
```

Drop `--skip-undistort` / `--allow-poor-calibration` when reprojection ≤ 2 px.

## 3. Ruler table (≥10)

1. Open `validation/ruler_refs.csv` — **fill** `reference_length_mm` and `reference_width_mm` (system columns are hints only).
2. Run:

```powershell
.\.venv\Scripts\python.exe scripts\validate_ruler.py --csv validation\ruler_refs.csv --marker-size-mm 198.0 --checkpoint models\runs\unet_pen\best.pt --allow-poor-calibration --skip-undistort
```

3. Paste output into `docs/Objectives.md` → Measurement validation.

## 4. Loom (≤5 min)

Follow **`docs/Loom_Recording.md`** (full script + checklist); **`docs/Loom_Guide.md`** (K1–K4 rules).

## 5. Email

Reply to the assessment message with **PR link + Loom link** within 24 h of your start time.

## 6. Model weights

If `best.pt` is not in Git: run `scripts/hash_checkpoint.ps1` and put the SHA-256 in README; upload weights to a release or drive link if required.
