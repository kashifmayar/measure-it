# Loom demonstration outline (≤5 min)

**Assessment:** xis.ai AI / Computer Vision Technical Assessment v2.0 §7  
**Branch / PR:** `dev-test01` → https://github.com/kashifmayar/measure-it/pull/1  

**ArUco scale:** measure **outer black square** — [`marker_measure_which_side.png`](marker_measure_which_side.png). This project’s fit-to-page print: **~198 mm** → `--marker-size-mm 198.0`.

---

## Pre-flight checklist

| Step | Requirement |
|------|-------------|
| Tests | `pytest tests\ -q` passes |
| Weights | `models\runs\unet_pen\best.pt` present locally |
| Marker | Outer black square measured in mm |
| Scene | Pen + marker coplanar; ruler/calliper for same pen |
| Length | Recording ≤ **5:00** |
| Submit | Email **PR #1** + **Loom URL** (see `docs/Submission.md`) |

---

## Timeline

| Time | Content |
|------|---------|
| **0:00–1:00** | New photo → `main.py` → mm output → ruler/calliper on same pen |
| **1:00–2:30** | Code path: load → undistort or skip → ArUco → rectified plane → U-Net → mm |
| **2:30–5:00** | **K1–K4** (assessment §7.2) |

---

## Pre-flight commands

```powershell
cd "<repo-root>"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pytest tests\ -q
Test-Path models\runs\unet_pen\best.pt
```

Optional: `.\.venv\Scripts\python.exe demo\serve.py` → http://127.0.0.1:8765/index.html

---

## PART A — Live demo (~1 min)

**Intro**

> Measure It: one iPhone photo, pen + ArUco on a flat surface. CLI reports length and max width (mm) via U-Net on 87 CVAT masks (PyTorch). Scale from **outer black square** of marker (~**198 mm** on this print).

**Steps**

1. Pen + ArUco flat; four marker corners visible.
2. New rear-camera photo → e.g. `measurement\images\loom_demo.HEIC`.

```powershell
.\.venv\Scripts\python.exe main.py `
  --image measurement\images\loom_demo.HEIC `
  --marker-size-mm 198.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\output
```

While reprojection ~**17 px**: keep `--allow-poor-calibration` and `--skip-undistort` on raw HEIC. After recalibration (≤2 px): omit both.

**Show:** terminal OK line; `*_annotated.jpg`; ruler on same pen.

**Fallback** (`dataset\raw\pen_0042.HEIC`) if live capture fails — state rejection reason if used:

```powershell
.\.venv\Scripts\python.exe main.py `
  --image dataset\raw\pen_0042.HEIC `
  --marker-size-mm 198.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\output
```

Evidence: `measurement/output/pen_0042_result.json` — **138.37 mm** length (ruler ~135 mm), **18.47 mm** max width.

---

## PART B — Code walk (~1.5 min)

| File | Role |
|------|------|
| `main.py` | CLI → `measure_pen_image` → JSON and overlays |
| `measurement/measure_pen.py` | Load HEIC; `_undistort_bgr` or `--skip-undistort` |
| `measurement/aruco_detector.py` | `DICT_4X4_50`, ID 0; outer black square corners |
| `measurement/perspective.py` | `rectify_plane()`; `--marker-size-mm` on metric plane |
| `measurement/pen_detector.py` | U-Net mask; `length_mm = length_px / pixels_per_mm` |
| `models/unet.py`, `models/train.py` | Training; test metrics in `validation/test_metrics.json` |
| `dataset/splits/test.txt` | Held-out test (61/17/9 split) |

---

## PART C — K1–K4 (technical content)

### K1 — Pixels to millimetres + scene

- **Plane:** `measurement/perspective.py` — `rectify_plane()`, `_destination_corners` (~36–41).
- **Pen:** `measurement/pen_detector.py` — `length_mm` / `max_width_mm` from rectified px ÷ `pixels_per_mm` (~157–158).
- **Scene:** Coplanar pen + marker; four corners visible; measured **outer black** side; see `docs/Plan.md` §6.

### K2 — Undistortion

- **Code:** `measurement/measure_pen.py` — `_undistort_bgr()`; dataset via `calibration/undistort.py`.
- **Evidence:** `validation/raw_vs_undistorted_ruler_stems.json` — **7/10** raw + skip vs **1/10** undistort at ~**17 px** reprojection; example `pen_0038` ~**2.3 mm** length delta.

### K3 — One reversal + cost

Examples: recalibrate checkerboard set; GPU retrain for IoU; reprint ArUco at 100% scale.

### K4 — One gap + response

Examples: inner vs outer marker mm; `dataset/raw` vs `dataset/images`; skip-undistort until cal fixed; honest MO-2 metrics.

**Close:** PR #1; `best.pt` local with SHA-256 in README.

---

## Related submission steps

| Item | Location |
|------|----------|
| Checklist | `docs/Submission.md` |
| Ruler validation | `validation/ruler_refs.csv` + `validate_ruler.py --marker-size-mm 198.0` |
| K1–K4 rules | `docs/Loom_Guide.md` |
