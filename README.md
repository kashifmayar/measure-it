# Measure It

Computer vision pipeline for the **xis.ai AI / Computer Vision Technical Assessment (v2.0)**. Photograph **pens** with an **iPhone 12** rear camera, calibrate lens distortion, then measure **length** and **maximum width (mm)** using a printed **ArUco** marker (`DICT_4X4_50`, ID 0) on the same flat plane.

**Host:** Windows, Python 3.11, CPU inference (Intel i5-8365U, 16 GB RAM). PyTorch U-Net segmentation is **planned** (Kaggle GPU for training).

---

## Implementation status

| Component | Status |
|-----------|--------|
| Pattern generation (`generate_patterns.py`) | **Implemented**, tested |
| Camera calibration (`calibrate.py`, HEIC, ≥20 images) | **Implemented**, **run locally** (see calibration report) |
| Undistortion (`undistort.py`) | **Implemented**, tested |
| Image inventory (`scripts/inventory_images.py`) | **Implemented**, tested |
| ArUco + homography + baseline pen CV (`measurement/`, `main.py`) | **Implemented**, **needs valid calibration + measured marker mm** |
| Trained U-Net segmentation | **Not started** |
| Ruler validation / raw vs undistorted study | **Not started** |

---

## Repository layout

```
measure-it/
├── main.py                      # pen measurement CLI
├── requirements.txt
├── calibration/
│   ├── generate_patterns.py
│   ├── calibrate.py
│   ├── undistort.py
│   ├── camera_params.py
│   ├── image_io.py              # HEIC + portrait normalize
│   ├── images/                  # your photos (gitignored)
│   ├── patterns/                # generated PNGs
│   └── output/                  # inventory CSV/JSON
├── measurement/
│   ├── aruco_detector.py
│   ├── perspective.py
│   ├── pen_detector.py          # classical baseline (not U-Net)
│   ├── measure_pen.py
│   ├── config.py
│   ├── images/                  # pen+ArUco captures (gitignored)
│   └── output/                  # annotated / JSON results
├── scripts/
│   └── inventory_images.py
├── tests/
└── docs/
```

---

## Install (PowerShell)

```powershell
cd "E:\Project\measure it"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Patterns and physical sizes

- **Checkerboard:** 9×7 **inner corners** (10×8 squares). Measured square size for this project: **20 mm** → `calibrate.py --square-size 20.0`.
- **ArUco:** generated at **600×600 px** only; **measure the printed black square** with a ruler and pass `--marker-size-mm` to `main.py`. The code does **not** assume 50 mm or any default marker size.

Generate patterns:

```powershell
.\.venv\Scripts\python.exe calibration\generate_patterns.py --square-size 20.0
```

---

## Image inventory

Scans `calibration/images` and `measurement/images` (does not move files):

```powershell
.\.venv\Scripts\python.exe scripts\inventory_images.py
```

Outputs: `calibration/output/image_inventory.csv` and `.json`.

---

## Camera calibration

1. Place checkerboard photos in `calibration/images/` (HEIC supported via `pillow-heif`).
2. Do **not** mix pen+ArUco photos in this folder for calibration (move them to `measurement/images/`).
3. Run:

```powershell
.\.venv\Scripts\python.exe calibration\calibrate.py --square-size 20.0 --blur-threshold 30.0
```

Writes `calibration/camera_params.npz` and `calibration/calibration_report.json` only when ≥20 detections succeed.

**Current run (local):** 23/28 detections, resolution 3024×4032, mean reprojection error **~16.9 px** — **poor**; remeasure printed squares, confirm 9×7 inner corners, and recalibrate before trusting millimetres. Measurement CLI rejects when error &gt; 2 px unless `--allow-poor-calibration` (development only).

---

## Undistortion

```powershell
.\.venv\Scripts\python.exe calibration\undistort.py `
  --input calibration\images\IMG_2178.HEIC `
  --output calibration\output\IMG_2178_undistorted.jpg `
  --save-comparison calibration\output\IMG_2178_compare.jpg
```

Portrait orientation is normalized to match calibration (landscape phone shots rotated to height ≥ width).

---

## Pen measurement (baseline CV)

1. Copy pen+ArUco HEIC/JPEG files to `measurement/images/` (hardlink OK to save space).
2. Measure printed ArUco **black square** side in mm.
3. Run:

```powershell
.\.venv\Scripts\python.exe main.py `
  --image measurement\images\IMG_2205.HEIC `
  --marker-size-mm <YOUR_MEASURED_MM> `
  --pixels-per-mm 8 `
  --output-dir measurement\output
```

Outputs per stem: `*_annotated.jpg`, `*_rectified.jpg`, `*_mask.png`, `*_result.json`.

**Geometric limits:** pen and marker on one flat plane; all four marker corners visible; tilted pens and bad calibration invalidate mm accuracy. Segmentation is **classical_cv_baseline**, not the final trained model.

Development override (not for submission claims):

```powershell
.\.venv\Scripts\python.exe main.py --image ... --marker-size-mm <mm> --allow-poor-calibration
```

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q
```

Synthetic ArUco/board tests only; real-world mm numbers must come from your runs.

---

## Geometry (short)

| Step | What it does |
|------|----------------|
| **Undistortion** | Removes lens distortion using `camera_matrix` / `dist_coeffs`. |
| **Perspective rectification** | Homography from ArUco corners → metric plane (mm via `pixels_per_mm`). |
| **mm conversion** | Done **in the rectified plane**, not with a single global px/mm on a distorted image. |

---

## Next steps (assessment)

1. Fix calibration quality (reprojection &lt; ~0.5–1 px).
2. Move pen shots out of `calibration/images/` into `measurement/images/`.
3. Collect ≥70 pen images, masks, 70/20/10 split, U-Net training on Kaggle GPU.
4. Replace baseline pen detector with trained model in a later `main.py` / `inference/` path.

See `docs/Plan.md`, `docs/Objectives.md`, `docs/Definition_of_Done.md`.
