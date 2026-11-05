# Loom recording — full script and checklist (≤5 min)

**Assessment:** xis.ai AI / Computer Vision Technical Assessment v2.0 §7  
**Branch / PR:** `dev-test01` → https://github.com/kashifmayar/measure-it/pull/1  
**Rule:** Rehearse once; on camera speak naturally — **do not read this word-for-word**.

Also see: [`marker_measure_which_side.png`](marker_measure_which_side.png) (which mm to measure on the print).

---

## One-page checklist (print or second monitor)

| Step | Done |
|------|------|
| `pytest tests\ -q` passes | ☐ |
| `models\runs\unet_pen\best.pt` exists | ☐ |
| ArUco printed; **outer black square** measured in mm (not inner pattern) | ☐ |
| Pen + marker flat on desk; ruler/calliper ready | ☐ |
| Loom: screen + webcam, **max 5:00** | ☐ |
| Email: **PR #1 + Loom URL** within 24 h | ☐ |

**Your marker (fit-to-page print):** outer black square **≈198 mm** per side → use `--marker-size-mm 198.0` (remeasure and use your average).

---

## Timeline (assessor rubric)

| Time | What you show |
|------|----------------|
| **0:00–1:00** | New photo → `main.py` → mm on screen → ruler/calliper on **same pen** |
| **1:00–2:30** | Code path: load → undistort/skip → ArUco → rectified plane → U-Net → mm |
| **2:30–5:00** | Answer **K1, K2, K3, K4** out loud |

---

## Before Record — terminal prep

```powershell
cd "E:\Project\measure it"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pytest tests\ -q
Test-Path models\runs\unet_pen\best.pt
```

Copy a fresh iPhone photo to e.g. `measurement\images\loom_demo.HEIC` (or use fallback `dataset\raw\pen_0042.HEIC`).

**Optional B-roll:** `.\.venv\Scripts\python.exe demo\serve.py` → http://127.0.0.1:8765/index.html

---

## PART A — Live demo (~1 minute)

### Say (intro, ~15 s)

> “This is **Measure It**: one iPhone photo of a pen and a printed ArUco marker on the same flat surface. The CLI reports **length** and **max width** in millimetres using a U-Net I trained on 87 CVAT masks — PyTorch, not YOLO.”

> “Scale comes from the **outer black square** of the marker. I measured mine at about **198 millimetres** per side because I printed fit-to-page — **not** the small inner pattern.”

### Do on camera

1. Lay **pen + ArUco** flat; all **four** marker corners visible.
2. Take **one new photo** (rear camera).
3. Copy to PC as `measurement\images\loom_demo.HEIC`.

### Run (replace marker mm if you remeasured)

```powershell
cd "E:\Project\measure it"
.\.venv\Scripts\python.exe main.py `
  --image measurement\images\loom_demo.HEIC `
  --marker-size-mm 198.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\output
```

**While calibration reprojection is ~17 px:** keep `--allow-poor-calibration` and `--skip-undistort` on **raw HEIC**.  
**After good calibration (≤2 px):** remove both flags.

### Show on screen

- Terminal: `OK: length=… mm, max_width=… mm (unet_segmentation)`
- Open `measurement\output\loom_demo_annotated.jpg` (or `*_rectified.jpg`)
- **Same pen** with ruler/calliper — read length aloud (close is fine; width may differ from “max width from mask”)

**Known-good fallback** (if live photo fails — say so honestly):

```powershell
.\.venv\Scripts\python.exe main.py `
  --image dataset\raw\pen_0042.HEIC `
  --marker-size-mm 198.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\output
```

Example on this repo (ruler ~135 mm length, ~14 mm body width):

- System: **138.37 mm** length, **18.47 mm** max width  
- JSON: `measurement/output/pen_0042_result.json`

### Say (wrap demo, ~10 s)

> “If marker or pen fails checks, the JSON gives a **rejection reason**. Here we got **success** with marker quality and pen confidence in the JSON.”

---

## PART B — Code walk (~1.5 minutes)

### Say

> “I'll trace **one image** through the repo on branch **dev-test01**.”

**Scroll in order — name each file aloud:**

| Order | File | One line to say |
|-------|------|-----------------|
| 1 | `main.py` | Parses CLI, builds `MeasurementConfig`, calls `measure_pen_image`, writes JSON and overlays. |
| 2 | `measurement/measure_pen.py` | Loads HEIC, optional `_undistort_bgr` or `--skip-undistort`, then marker + rectification + pen detect. |
| 3 | `measurement/aruco_detector.py` | `DICT_4X4_50`, ID 0; corners are the **outer black square**. |
| 4 | `measurement/perspective.py` | `rectify_plane()` — homography; marker side = `--marker-size-mm` on the metric plane. |
| 5 | `measurement/pen_detector.py` | U-Net mask on rectified image; `length_mm = length_px / pixels_per_mm`. |
| 6 | `models/unet.py` + `models/train.py` | U-Net trained on committed splits; metrics in `validation/test_metrics.json`. |
| 7 | `dataset/splits/test.txt` | Test held out before training. |

### Say (dataset)

> “**87** labelled images, splits **61 / 17 / 9**, masks from CVAT; splits committed before training.”

---

## PART C — Four questions (~2.5 minutes)

Answer **from memory**. Use **your** numbers below.

### K1 — Pixels to millimetres + valid scene

**Question:** Which file/line turns pixels into mm? What must be true about the scene?

**Answer guide:**

1. **Metric plane:** `measurement/perspective.py` — `rectify_plane()` maps ArUco corners so one marker side = `--marker-size-mm` with `pixels_per_mm` (default 8). See `_destination_corners` (~lines 36–41).
2. **Pen size:** `measurement/pen_detector.py` — `length_mm = length_px / ppm`, `max_width_mm = width_px / ppm` (~lines 157–158) on the **rectified** image.
3. **Scene:** Pen and marker on the **same flat plane**; all **four** corners visible; marker size = **measured outer black square** (see `docs/marker_measure_which_side.png`); not heavily tilted. Coplanarity: `docs/Plan.md` §6.

**Story you can tell:** “I first used **34 mm** on the inner pattern and got ~**23 mm** pen length; with **198 mm** outer square I got ~**138 mm**, matching my ruler ~**135 mm**.”

---

### K2 — Undistortion + your numbers

**Question:** Where is undistortion applied, and how much did skipping it change error?

**Answer guide:**

- **Where:** `measurement/measure_pen.py` — `_undistort_bgr()` using `calibration/camera_params.npz`. Dataset prep: `calibration/undistort.py` / `scripts/prepare_dataset.py`.
- **Your numbers:** `validation/raw_vs_undistorted_ruler_stems.json` — on 10 stems **7/10** succeed with `--skip-undistort`, **1/10** with full undistort while reprojection ~**17 px**.
- **Example:** `pen_0038` raw **22.57 mm** vs undistorted **20.3 mm** (~**2.3 mm** on length).
- **Honest line:** Raw HEIC + skip until recalibration; documented in README.

---

### K3 — One thing you would reverse

Pick **one** (your choice):

| Choice | Cost |
|--------|------|
| Kept **poor calibration** too long | Reshoot 25+ checkerboard-only images, re-run `calibrate.py` — **hours**, but trustworthy undistort |
| **CPU-only** U-Net (test IoU ~**0.60**, target 0.80) | Colab GPU + more epochs — **time**, higher segmentation quality |
| Printed ArUco **fit-to-page** (198 mm marker) | Reprint at 100% scale (~40–50 mm outer) — easier desk photos |

---

### K4 — One thing you did not understand

Pick **one** (honest):

| Topic | What you did |
|-------|----------------|
| **34 mm vs 198 mm** marker | Inner pattern ≠ OpenCV outer corners; fixed `--marker-size-mm` and added `marker_measure_which_side.png` |
| **`dataset/images` vs `dataset/raw`** | Training JPGs often **no ArUco**; measure on **raw HEIC** |
| **Double undistort** | Undistorted JPG + cal again broke detection → `--skip-undistort` |
| **MO-2 below target** | Reported real `test_metrics.json`; planned GPU retrain |

### Closing (~10 s)

> “Pull request **#1** is open on my repo. Weights are local **`best.pt`** with SHA-256 in the README.”

---

## After recording

1. Watch once — under **5:00**? All **four** answers audible?
2. Email assessors: **PR #1** + **Loom link**.
3. Optional: merge PR to `main` after review.

---

## Submission pointers

| Item | Doc / command |
|------|----------------|
| Full checklist | `docs/Submission.md` |
| Ruler table (≥10) | `validation/ruler_refs.csv` + `scripts/validate_ruler.py --marker-size-mm 198.0` |
| Tests | `pytest tests\ -q` |

**Do not claim perfect mm** until calibration reprojection ≤ 2 px and ruler CSV is filled.
