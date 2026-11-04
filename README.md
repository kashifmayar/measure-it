# Measure It

Computer vision pipeline for the **xis.ai AI / Computer Vision Technical Assessment (v2.0)**. Photograph **pens** with an **iPhone 12** rear camera, calibrate lens distortion, then measure **length** and **maximum width (mm)** using a printed **ArUco** marker (`DICT_4X4_50`, ID 0) on the same flat plane.

**Host:** Windows, Python 3.11, CPU inference (Intel i5-8365U, 16 GB RAM). U-Net trains on CPU locally or on a Kaggle GPU with the same commands.

---

## Implementation status

| Component | Status |
|-----------|--------|
| Pattern generation (`generate_patterns.py`) | **Implemented**, tested |
| Camera calibration (`calibrate.py`, HEIC, ≥20 images) | **Implemented**, **run locally** (see calibration report) |
| Undistortion (`undistort.py`) | **Implemented**, tested |
| Image inventory (`scripts/inventory_images.py`) | **Implemented**, tested |
| ArUco + homography + baseline pen CV (`measurement/`, `main.py`) | **Implemented**, **needs valid calibration + measured marker mm** |
| Dataset splits + CVAT masks (87 images) | **Done** — `scripts/prepare_dataset.py validate` |
| U-Net train / eval (`models/train.py`, `evaluate.py`) | **Implemented** — run locally; metrics in `models/runs/unet_pen/` |
| Measurement with U-Net (`main.py --checkpoint`) | **Implemented** — falls back to baseline CV unless `--no-fallback-baseline` |
| Ruler validation / raw vs undistorted | **`validate_ruler.py`**, **`compare_raw_undistorted.py`** — need your mm CSV + marker size |

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
│   ├── pen_detector.py          # U-Net + classical baseline
│   ├── measure_pen.py
│   ├── config.py
│   ├── images/                  # pen+ArUco captures (gitignored)
│   └── output/                  # annotated / JSON results
├── models/
│   ├── unet.py
│   ├── train.py                 # loss curves -> models/runs/unet_pen/training_curves.png
│   ├── evaluate.py
│   └── inference.py
├── scripts/
│   ├── inventory_images.py
│   ├── prepare_dataset.py
│   ├── cvat_xml_to_masks.py
│   └── validate_ruler.py
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
- **ArUco:** generated at **600×600 px** only; **measured black square for this project: 34 mm** → pass `--marker-size-mm 34.0` to `main.py` and validation scripts (never assume a default in code).

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

**Current run (local):** 23/28 detections, resolution 3024×4032, mean reprojection error **~16.9 px** — **poor**; remeasure printed squares, confirm 9×7 inner corners, and recalibrate before trusting millimetres. Measurement CLI rejects when error &gt; 2 px unless `--allow-poor-calibration`. Until recalibration, measure from **`dataset/raw/*.HEIC`** with **`--skip-undistort`** (training JPGs in `dataset/images/` are already undistorted and often omit the marker).

---

## Undistortion

```powershell
.\.venv\Scripts\python.exe calibration\undistort.py `
  --input calibration\images\IMG_2178.HEIC `
  --output calibration\output\IMG_2178_undistorted.jpg `
  --save-comparison calibration\output\IMG_2178_compare.jpg
```

Portrait orientation is normalized to match calibration (landscape phone shots rotated to height ≥ width).

**Rejection thresholds** (see `measurement/config.py`): min ArUco sharpness 30, min marker side 20 px, min pen confidence 0.35, min pen aspect 2.0, calibration reprojection must be ≤ 2 px unless `--allow-poor-calibration`.

Batch rejection stats on a split:

```powershell
.\.venv\Scripts\python.exe scripts\rejection_report.py --split test --marker-size-mm 34.0 --checkpoint models\runs\unet_pen\best.pt --allow-poor-calibration
```

---

## U-Net training and evaluation

Requires `dataset/images/`, `dataset/masks/`, and `dataset/splits/*.txt` (see `dataset/README.md`).

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py validate
.\.venv\Scripts\python.exe models\train.py --epochs 20 --image-size 384 --batch-size 2
.\.venv\Scripts\python.exe models\evaluate.py --checkpoint models\runs\unet_pen\best.pt --split test
```

Artifacts: `models/runs/unet_pen/best.pt`, `config.json`, `history.json`, `training_curves.png`, `test_metrics.json`.

Use a GPU when available: `models\train.py --device cuda --batch-size 4`.

---

## Pen measurement (U-Net or baseline CV)

1. Copy pen+ArUco HEIC/JPEG files to `measurement/images/` (hardlink OK to save space).
2. Measure printed ArUco **black square** side in mm.
3. Run:

```powershell
.\.venv\Scripts\python.exe main.py `
  --image measurement\images\IMG_2205.HEIC `
  --marker-size-mm 34.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --pixels-per-mm 8 `
  --output-dir measurement\output
```

Outputs per stem: `*_annotated.jpg`, `*_rectified.jpg`, `*_mask.png`, `*_result.json`.

**Geometric limits:** pen and marker on one flat plane; all four marker corners visible; tilted pens and bad calibration invalidate mm accuracy. Omit `--checkpoint` to use **classical_cv_baseline** only.

### Ruler / calliper validation

CSV columns: `image,reference_length_mm,reference_width_mm` (stem matches `dataset/images/pen_XXXX.jpg`).

```powershell
.\.venv\Scripts\python.exe scripts\validate_ruler.py `
  --csv validation\ruler_refs.csv `
  --marker-size-mm 34.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration
```

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

## Dataset (segmentation training)

See **`dataset/README.md`**. Prepare with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py organize --include-measurement
.\.venv\Scripts\python.exe scripts\prepare_dataset.py undistort
.\.venv\Scripts\python.exe scripts\prepare_dataset.py split
.\.venv\Scripts\python.exe scripts\prepare_dataset.py validate
```

Commit `dataset/splits/` and `dataset/meta/` **before** training; add masks in `dataset/masks/` after annotation.

## Next steps (assessment)

1. Fix calibration quality (reprojection &lt; ~0.5–1 px).
2. Improve calibration (reprojection &lt; ~1 px) before submission-grade mm claims.
3. Fill `validation/ruler_refs.csv` with ≥10 manual measurements and run `validate_ruler.py`.
4. Re-train on GPU if CPU epochs are too slow; keep `dataset/splits/` fixed.

See `docs/Plan.md`, `docs/Objectives.md`, `docs/Definition_of_Done.md`.
