# Measure It

Computer vision pipeline for the **xis.ai AI / Computer Vision Technical Assessment (v2.0)**. The goal is to photograph physical **pens** with an **iPhone rear main camera**, segment each pen in the image, and report real-world dimensions in millimetres using a printed **ArUco** marker as a metric reference—after **OpenCV camera calibration** and lens undistortion.

---

## Problem

Industrial-style measurement from a single phone photo requires several steps to stay consistent: remove lens distortion, find the object boundary reliably, and convert pixels to millimetres using a reference of known size in the **same plane** as the object. A strong segmentation mask with wrong scale—or correct scale on a distorted image—still yields a wrong measurement.

This project implements that workflow for pens (3–5 different pens, ruler/calliper ground truth), with evidence at each stage.

---

## Key features

| Feature | Status |
|--------|--------|
| Printable checkerboard (9×7 inner corners) and ArUco (`DICT_4X4_50`, ID 0) PNG generation | **Implemented** |
| Checkerboard-based camera calibration (≥20 successful detections) | **Implemented** (script); **not run** on project hardware yet |
| Save `camera_matrix`, `dist_coeffs`, reprojection error report | **Implemented** (script outputs) |
| Single-image undistortion CLI | **Implemented** |
| Pen image dataset (≥70), segmentation masks, 70/20/10 split file lists | **Planned** |
| U-Net (PyTorch) training and evaluation (no Ultralytics YOLO / Roboflow models) | **Planned** |
| One-command image → mm inference with overlay and rejection | **Planned** |
| Ruler/calliper validation, raw vs undistorted comparison | **Planned** |

---

## Architecture (target)

```
Phone photo
    → undistort (camera_params.npz)
    → segmentation (trained U-Net)
    → ArUco detection + known marker size (mm)
    → pixel → mm on measurement plane
    → length & max width (mm) + confidence / reject if untrusted
```

Calibration is **offline** (checkerboard photos). Measurement photos use the **same locked** camera settings and include the pen and ArUco on a **flat, coplanar** surface when possible.

---

## Technology stack

| Layer | Choice |
|-------|--------|
| Language | Python 3 |
| Calibration / geometry | OpenCV (`opencv-contrib-python` for ArUco) |
| Segmentation (planned) | PyTorch, U-Net-style encoder–decoder |
| Metric reference | Printed ArUco marker (`DICT_4X4_50`) |
| Version control | Git, branch `dev-test01` |

---

## Repository layout (current)

```
measure-it/
├── calibration/
│   ├── generate_patterns.py   # checkerboard + ArUco PNGs
│   ├── calibrate.py           # OpenCV calibration
│   ├── undistort.py           # lens correction
│   ├── images/                # calibration photos (create locally; not in repo yet)
│   └── patterns/              # generated PNGs (create by running generate_patterns.py)
├── docs/
│   ├── Plan.md
│   ├── Objectives.md
│   └── Definition_of_Done.md
├── .gitignore
└── README.md
```

**Planned** (assessment layout): `dataset/`, `models/`, `inference/`, `measurement/`, `requirements.txt`.

---

## Hardware and capture (document in Objectives when known)

- **Camera:** iPhone rear main camera—lock focus, zoom, and resolution before calibration; keep locked for all later images.
- **Materials:** 3–5 pens, ruler/calliper, printed checkerboard, printed ArUco marker, rigid flat surface.
- **Host machine:** Record CPU, GPU, RAM, and OS in `docs/Objectives.md` when measurements are run.

---

## Installation (Windows PowerShell)

From the repository root:

```powershell
cd "E:\Project\measure it"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install opencv-contrib-python numpy
pip install Pillow
```

`Pillow` is optional but recommended for checkerboard PNG DPI metadata in `generate_patterns.py`. PyTorch and other training dependencies will be added in `requirements.txt` when the dataset and model stages are implemented.

---

## Generate calibration patterns

Default inner corners: **9×7** (10×8 squares). Target square size **25 mm** at **300 DPI** (verify with a ruler after printing).

```powershell
.\.venv\Scripts\python.exe calibration\generate_patterns.py
```

Custom output directory or parameters:

```powershell
.\.venv\Scripts\python.exe calibration\generate_patterns.py --out-dir calibration\patterns
.\.venv\Scripts\python.exe calibration\generate_patterns.py --cols 9 --rows 7 --square-size 25.0
.\.venv\Scripts\python.exe calibration\generate_patterns.py --aruco-id 0 --aruco-size-px 600
```

Outputs (default `--out-dir calibration\patterns`):

- `checkerboard_9x7.png`
- `aruco_4x4_50_id0.png`

Print at **100% / actual size**. Measure several checkerboard squares and record the **mean** side length in mm. **Physically measure** the printed ArUco marker edge length in mm for the measurement pipeline (the generator only sets pixel size for the marker image).

---

## Collect calibration images

1. Tape the printed checkerboard flat to a rigid board.
2. Capture **≥20** photos with the **same** phone settings you will use for pen images.
3. Vary distance, angle, and position; keep the full board in frame, in focus, and evenly lit.
4. Save images under:

```text
calibration/images/
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`.

---

## Run camera calibration

`--square-size` is **required** and must be the **measured** square side in mm (not the print target alone).

```powershell
.\.venv\Scripts\python.exe calibration\calibrate.py --square-size 24.7
```

Optional arguments:

```powershell
.\.venv\Scripts\python.exe calibration\calibrate.py `
    --images-dir calibration\images `
    --cols 9 --rows 7 `
    --square-size 24.7 `
    --out-params calibration\camera_params.npz `
    --out-report calibration\calibration_report.json `
    --show
```

On success the script writes:

- `calibration/camera_params.npz` — `camera_matrix`, `dist_coeffs`, `image_size`
- `calibration/calibration_report.json` — image counts, RMS/mean reprojection error, matrices

Calibration has **not** been executed in this repository yet; no parameter files are committed until you run this step locally.

---

## Undistort an image

Requires `calibration/camera_params.npz` from `calibrate.py`.

```powershell
.\.venv\Scripts\python.exe calibration\undistort.py --input path\to\photo.jpg
```

Writes `photo_undistorted.jpg` next to the input unless `--output` is set.

```powershell
.\.venv\Scripts\python.exe calibration\undistort.py `
    --input path\to\photo.jpg `
    --output path\to\photo_corrected.jpg `
    --params calibration\camera_params.npz `
    --crop `
    --save-comparison path\to\comparison.jpg `
    --show
```

You are “calibration-ready” when straight edges in the scene stay straight after undistortion.

---

## Tests

No automated test suite is in the repository yet. Verification is currently **manual**: pattern generation, successful calibration with ≥20 detections, and visual undistortion check.

---

## Training and inference (not yet available)

The following will be documented here once implemented:

- Dataset layout under `dataset/` and fixed split files (`train` / `val` / `test`, 70% / 20% / 10%)
- U-Net training command and config under `models/`
- Single CLI under `inference/` — undistort, segment, detect ArUco, output overlay with length and max width (mm), confidence, and explicit handling when the pen or reference is not found
- Evaluation metrics on the **test** split only (mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall)

---

## Limitations

- **Planar metric assumption:** ArUco and pen should lie on the same plane; perspective and out-of-plane tilt bias length and width.
- **Printer scale:** Neither checkerboard nor ArUco physical size should be assumed from file metadata alone—always measure after printing.
- **Single camera profile:** Changing focus, zoom, or resolution after calibration invalidates saved parameters.
- **Assessment floor:** Deliverables 1–4 (calibration, dataset, trained model, one-command mm output) must work before later accuracy and failure-mode claims.

---

## Development status

| Stage | Status |
|-------|--------|
| Documentation (`docs/`, README) | In progress |
| Calibration scripts | Implemented, not yet run on device |
| Dataset, training, inference, measurement | Not started |
| Loom recording and PR submission | Not started |

See `docs/Plan.md`, `docs/Objectives.md`, and `docs/Definition_of_Done.md` for intent, measurable targets, and acceptance checklist.

---

## Future work

- Add pinned `requirements.txt` and scaffold `dataset/`, `models/`, `inference/`, `measurement/`
- Collect ≥70 pen images and masks; commit split lists **before** training
- Train and evaluate segmentation; publish weights or release link with SHA-256 in this README
- Implement measurement CLI, rejection rules, validation tables, and raw vs undistorted error study
- Close the Plan **decision record** (approach taken, rejected, and cost)

---

## Assessment documents

- [Plan](docs/Plan.md) — order of work, time budget, technical approach
- [Objectives](docs/Objectives.md) — measurable targets (MO-1, MO-2, and assessment alignment)
- [Definition of Done](docs/Definition_of_Done.md) — acceptance checklist with evidence

---

## License / submission

Public GitHub repository on branch **`dev-test01`**, pull request into **`main`**, per assessment instructions. Model weights in-repo or linked with hash in this README when available.
