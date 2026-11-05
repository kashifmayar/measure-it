# Plan — Measure It

**Assessment:** xis.ai AI / Computer Vision Technical Assessment v2.0  
**Branch:** `dev-test01`  
**Object:** Physical pens (3–5 types), iPhone rear main camera  
**Status:** Pre-implementation plan (committed before pipeline code in git history)

---

## 1. Overview and problem statement

Build a reproducible pipeline that:

1. Calibrates the phone camera and undistorts images used for measurement.
2. Collects and labels a private pen dataset with segmentation masks.
3. Trains a segmentation model (U-Net in PyTorch—not Ultralytics YOLO or Roboflow).
4. Runs **one command** on a new photo: undistort → segment pen → detect ArUco reference → report **length** and **maximum width** in millimetres with confidence, or refuse when untrusted.

Success is judged on **explainable, checkable measurements**, not only model scores.

---

## 2. Scope and limitations

**In scope**

- Checkerboard calibration (≥20 successful detections), saved intrinsics and distortion coefficients.
- ≥70 original pen images, undistorted for labelling/training, 70/20/10 split committed before training.
- ArUco (`DICT_4X4_50`, generated ID 0 by default) as flat reference; physical marker size measured after printing.
- Test-set evaluation metrics and ≥10 ruler/calliper comparisons.
- Raw vs undistorted measurement comparison on the same instances.
- One chosen failure condition (e.g. camera tilt) and measurement rejection rules.

**Out of scope / explicit skips if time runs out**

- Multi-view or depth-based 3D reconstruction.
- Sub-millimetre metrology without documenting plane and tilt limits.
- Using YOLO or Roboflow **as the segmentation model** (assessment rule).

If work stops early, unfinished items will be listed in `Definition_of_Done.md` with reasons (assessment §2).

---

## 3. Hardware and software requirements

| Item | Requirement |
|------|-------------|
| Camera | iPhone rear main; fixed focus, zoom, resolution after calibration lock |
| Calibration target | Printed 9×7 inner-corner checkerboard; measured square size (mm) |
| Reference | Printed ArUco 4×4_50; measured marker size (mm) |
| Ground truth | Ruler or calliper |
| Software | Python 3, OpenCV (contrib for ArUco), NumPy; PyTorch for training (later) |
| OS | Windows (development); document exact machine in Objectives when running metrics |

---

## 4. Proposed system architecture

```text
[Calibration phase]
  checkerboard photos → calibrate.py → camera_params.npz + report

[Dataset phase]
  pen photos → undistort → manual masks → split lists (70/20/10)

[Training phase]
  train U-Net on train; tune on val only; select checkpoint without test leakage

[Inference phase]
  new image → undistort → segment → ArUco → mm geometry → overlay + JSON/stdout

[Validation phase]
  calliper table, raw vs undistorted, failure sweep, rejection statistics
```

---

## 5. Technical approach

| Stage | Approach |
|-------|----------|
| Distortion | OpenCV `calibrateCamera` + `undistort` / optimal new camera matrix |
| Segmentation | U-Net (encoder–decoder), pretrained backbone optional; trained on our masks |
| Scale | ArUco side length in mm ÷ detected side in px → mm/px on reference plane |
| Pen dimensions | From largest mask: length along principal axis; max width perpendicular (definitions fixed in Objectives) |
| Rejection | Thresholds on mask confidence/area, ArUco detection, optional plane/tilt heuristics |

**Why not YOLO/Roboflow for segmentation:** Assessment forbids those as the segmentation model; U-Net gives full-pixel masks needed for width/length from the silhouette and matches the “build and verify” goal.

---

## 6. Concepts (assessment clarity)

### Camera calibration vs lens distortion correction

- **Calibration** estimates intrinsics (`camera_matrix`) and **distortion coefficients** from many checkerboard views. Output: parameters saved to `camera_params.npz`.
- **Distortion correction (undistortion)** applies those parameters to each new image so straight lines in the scene map to straight lines in the image. Implemented in `undistort.py`; it does **not** by itself yield millimetres.

### Perspective correction vs pixel-to-millimetre conversion

- **Perspective correction** (homography / “rectifying” the plane) warps the image so a reference plane appears fronto-parallel. Useful when the phone is oblique; optional beyond basic undistortion depending on time.
- **Pixel-to-mm conversion** uses a **known physical length** on the **same plane** as the pen (ArUco edge) to compute scale (mm/px) at that plane. It is **not** the same as undistortion.

### Segmentation vs object measurement

- **Segmentation** assigns pen vs background pixels (mask quality: IoU, mAP, etc.).
- **Measurement** derives scalar lengths from the mask (and scale from ArUco). A perfect mask with wrong ArUco size or non-coplanar layout still fails validation.

### Coplanarity, perspective, and tilt

For **planar** metric measurement, the ArUco marker and pen must lie on the **same plane** (e.g. one desk surface). If the phone is tilted or the pen/marker pop out of that plane, the mm/px ratio varies across the image: lengths shrink or grow depending on depth and angle. The pipeline will document this limit; rejection rules will flag low-confidence geometry (e.g. extreme ArUco pose or partial marker).

---

## 7. Development phases, milestones, and git commits

Work in order; **one commit theme per step** (calibration → data → training → measurement).

| Phase | Milestone | Target commit theme |
|-------|-----------|---------------------|
| 0 | Plan, Objectives, DoD, README | Docs only (first commit) |
| 1 | `requirements.txt`, folder scaffold | Environment and layout |
| 2 | Calibration scripts (already written) | `calibration/*.py` |
| 3 | ≥20 cal images, run calibrate, save npz/report | Calibration results + evidence |
| 4 | ≥70 pen images | Dataset images |
| 5 | Masks + `train/val/test` lists | Split **before** training |
| 6 | Training code + config | `models/` |
| 7 | Weights + test metrics | Evaluation outputs in Objectives |
| 8 | `inference/` one-command CLI | End-to-end mm |
| 9 | Validation + failure + rejection | `measurement/` + doc updates |

---

## 8. Eight-hour time allocation (working time)

| Block | Hours | Activity |
|-------|-------|----------|
| A | 0.5 | Docs, repo setup, print patterns |
| B | 1.0 | Calibration capture + `calibrate.py` |
| C | 0.5 | Undistort proof; lock camera settings |
| D | 2.0 | Pen capture + mask annotation |
| E | 0.25 | Commit splits; start training (wall clock may overlap) |
| F | 2.0 | Train U-Net; val tuning; test eval once |
| G | 1.25 | Inference CLI + ArUco scale + overlays |
| H | 0.5 | ≥10 calliper rows; raw vs undistorted |
| I | 0.25 | One failure condition + rejection demo |

Training/install wall clock does not count toward the 8 hours (assessment §2).

---

## 9. Dataset collection strategy

- Same phone settings as calibration.
- Pens on flat background with ArUco visible in measurement photos (for inference stage).
- Variety: pen type, colour, orientation, distance, mild lighting changes.
- Store undistorted images for labelling consistency (or document undistort-on-load in training code).
- **≥70** images minimum (assessment v2.0); aim higher if annotation time allows.
- No downloaded pen images; all captured locally.

---

## 10. Camera calibration strategy

- Generate patterns: `generate_patterns.py` (9×7 inner corners, 25 mm target squares at 300 DPI).
- Print at 100%; measure mean square size; pass to `--square-size`.
- Capture ≥20 images → `calibration/images/`.
- Run `calibrate.py`; require ≥20 successful corner detections (`MIN_IMAGES = 20` in script).
- Accept if mean reprojection error is reasonable (script warns above 0.5 px and 1.0 px).
- Persist `camera_params.npz` and `calibration_report.json`; record counts and errors in Objectives/DoD.

---

## 11. Segmentation model strategy

- Architecture: U-Net (PyTorch), input size fixed in config, binary pen mask.
- Augmentations: flips, small rotation, brightness (document in config).
- Train on train split only; early stopping / checkpoint on validation loss or IoU.
- **Never** use test images for tuning or checkpoint selection.
- Evaluate on test: mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall (implementation in `models/` or `measurement/`).

---

## 12. Measurement methodology

1. Undistort input image with saved intrinsics.
2. Run segmentation → binary mask; compute confidence from model output.
3. Detect ArUco; estimate marker corner geometry; use **measured** marker side length (mm).
4. Compute mm/px from marker (planar assumption).
5. From pen mask: length (mm) and max width (mm) per agreed geometry definitions.
6. Emit overlay image and numeric output; if pen or ArUco missing, or below thresholds, **reject** with explicit reason.

Compare ≥10 instances to calliper/ruler; compute MAE and mean percentage error; repeat on **raw** vs **undistorted** inputs for the same instances.

---

## 13. Testing and validation strategy

- **Calibration:** reprojection error, visual undistortion on straight edges.
- **Segmentation:** held-out test metrics only.
- **Measurement:** per-instance table, mean/median/worst error; raw vs undistorted side-by-side.
- **Failure condition:** one of tilt, distance, lighting, or edge-of-frame—plot or table error vs severity.
- **Rejection:** document threshold values and fraction of images refused.

---

## 14. Risks and mitigation

| Risk | Mitigation |
|------|------------|
| Printer scales board/marker wrong | Measure squares and ArUco; never trust file defaults alone |
| Too few good cal images | Shoot extra; vary angles; exclude blur |
| Annotation slow | Start with fewer pens; simplify background |
| Training over 8 h wall clock | Start early; smaller U-Net; fewer epochs with clear config |
| Test leakage | Commit split files before training; code reads lists only |
| Coplanarity violated | Shoot face-on; rejection; note in Loom K1 |
| Git rejection (one big commit) | Commit per phase as in §7 |

---

## 15. Decision record (fill when work completes)

| Decision | Chosen approach | Rejected alternative | Cost of rejection |
|----------|-----------------|----------------------|-------------------|
| Segmentation | U-Net + PyTorch (`models/unet.py`, BCE+Dice) | Ultralytics YOLO / Roboflow seg | Faster off-the-shelf but assessment-invalid |
| Metric reference | ArUco on same plane as pen | Ruler only in image | Ruler detection harder; ArUco gives stable scale |
| Scale from | Measured **outer black square** of ArUco (~**198 mm** fit-to-page print; see `docs/marker_measure_which_side.png`) | Inner pattern only (~34 mm) or assumed 50 mm | Wrong side measured → ~6× length error until fixed |
| Annotation | CVAT polygons → raster masks (`cvat_xml_to_masks.py`) | Roboflow auto-label | Manual QC time; trustworthy masks |
| Training images | Undistorted JPG in `dataset/images/` | Train on distorted raw | Consistent with labels; measurement may use raw HEIC + `--skip-undistort` until cal is fixed |
| Inference fallback | U-Net then classical CV (`--no-fallback-baseline` to disable) | U-Net only | Baseline rescues edge cases when seg fails |
| Poor calibration | CLI rejects if reprojection > 2 px; `--allow-poor-calibration` for dev | Silent bad mm | Forces honest recalibration for MO-1 |

---

## 16. Plan changes

| Date | Change | Reason |
|------|--------|--------|
| _(none yet)_ | | |

If this plan diverges from what was done, the table above will record **why** (preferred over rewriting history to match outcomes).
