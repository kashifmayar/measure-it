# Objectives — Measure It

Measurable objectives for the xis.ai assessment v2.0. Targets are **chosen before** full implementation; **results** are filled in only after running the pipeline on this machine. Until then, verification columns remain **pending**.

**Hardware / environment (fill when first metrics are run)**

| Field | Value |
|-------|--------|
| CPU | Intel Core i5-8365U |
| GPU | Intel UHD Graphics 620 (CPU inference) |
| RAM | 16 GB |
| OS | Windows (local dev) |
| Camera | **iPhone 12**, rear main camera (fixed settings after calibration lock) |
| Checkerboard square (measured) | **20 mm** (use `calibrate.py --square-size 20.0`) |
| ArUco marker edge (physical) | **Outer black square** measured after print (this project **~198 mm** fit-to-page; use `--marker-size-mm` to match — see `docs/marker_measure_which_side.png`) |
| Image resolution | 3024×4032 px (portrait-normalized HEIC from iPhone 12) |

---

## Required measurable objectives (MO-1 and MO-2)

### MO-1 — Measurement accuracy (required)

| Field | Entry |
|-------|--------|
| **What is measured** | Absolute error of pen **length** and **maximum width** (mm) vs ruler/calliper on the same instance |
| **How** | Script or table in `measurement/` comparing system output to hand-recorded ground truth for each photo; report MAE (mm) and mean percentage error across ≥10 instances |
| **Target** | MAE ≤ **2.0 mm** and mean absolute percentage error ≤ **5%** on **undistorted**, face-on, coplanar ArUco+pen photos (justified: useful for rough inventory sizing of pens, stricter than casual ±1 cm) |
| **Conditions** | Same iPhone settings as calibration; indoor diffuse light; pen and ArUco flat on one surface; inference CLI version committed on `dev-test01` |
| **Not included** | Heavy motion blur; marker under 50% visible; pen occluded over 30%; photos where rejection logic fires |

**Verification:** `validation/ruler_refs.csv` (20 instances, reference columns filled) + `scripts/validate_ruler.py` at `--marker-size-mm 198.0 --skip-undistort --allow-poor-calibration`. **Status: verified 2026-11-05 — target NOT MET** (see results log).

---

### MO-2 — Segmentation quality on held-out test set

| Field | Entry |
|-------|--------|
| **What is measured** | Mask **IoU** (mean over test images) and **mAP@0.5** for pen class |
| **How** | Evaluation script reads `dataset/splits/test.txt` only; compares predicted masks to ground truth; raw console log pasted into this section |
| **Target** | Mean mask IoU ≥ **0.80** and mAP@0.5 ≥ **0.75** on test split |
| **Conditions** | Fixed split committed before training; single trained checkpoint; pinned seed and versions in `requirements.txt` |
| **Not included** | Train/val images; any checkpoint chosen using test metrics |

**Verification (when complete):** Raw metric output + commit hash of eval script and split file. **Status: not verified.**

---

## Assessment alignment — objectives and verification

Each row maps to assessment deliverables. **Verification** describes how completion is checked; none are marked complete until evidence exists.

| ID | Objective | Target / criterion | How completion is verified | Status |
|----|-----------|-------------------|------------------------------|--------|
| O-CAL | Calibrate with ≥20 successful checkerboard detections | ≥20 images used; params saved | `calibration_report.json` shows `images_used` ≥ 20; `camera_params.npz` loads in `undistort.py`; mean reprojection error recorded here | Pending |
| O-CAL-OUT | Record intrinsics and distortion | Matrix + coeffs in npz and report | Keys `camera_matrix`, `dist_coeffs` in npz; values pasted or linked from JSON report | Pending |
| O-DATA | Labelled pen dataset | ≥70 original images | Count files referenced by split lists + README dataset summary | Pending |
| O-DATA-UND | Undistorted images for training | All training paths undistorted or undistort in loader | Document in README; spot-check hashes or naming convention | Pending |
| O-SPLIT | Fixed 70/20/10 split | Committed lists before training | Git commit timestamp of `dataset/splits/*.txt` before first training commit; sums to 100% | Pending |
| O-SEG | Train segmentation without YOLO/Roboflow | U-Net (or other allowed arch) trained on our data | Training script in repo; Plan decision record explains choice | Pending |
| O-MET | Test metrics | IoU, mAP@0.5, mAP@0.5:0.95, precision, recall | Raw eval log on **test** split only, pasted below | Pending |
| O-INF | One command → mm | Overlay + length, width, confidence | README command runs on new image; stdout/artifact saved | Pending |
| O-INF-ERR | Failure cases | Object not found; reference not found | Demo logs or screenshots for both cases | Pending |
| O-VAL | Calliper validation | ≥10 instances | Table with ruler/calliper vs system; MAE and mean % error | Pending |
| O-UND | Undistortion matters | Same instances raw vs undistorted | Side-by-side error columns + short explanation with **our** numbers | Pending |
| O-MO2-ALT | Speed or accuracy target (assessment §3 item 7) | Covered by MO-2 (or add MO-3 inference latency if preferred) | See MO-2 or optional MO-3 below | Pending |
| O-FAIL | One worsening condition | e.g. camera tilt series | Table/plot of error vs tilt angle or distance | Pending |
| O-REJ | Reject untrusted measurements | Documented thresholds + refusal rate | Log counts: accepted vs rejected; example rejected images | Pending |

---

## Optional MO-3 — Inference latency (if MO-2 alone is insufficient for item 7)

| Field | Entry |
|-------|--------|
| **What is measured** | End-to-end CLI time for one 12 MP image (undistort + segment + ArUco + measure) |
| **How** | PowerShell loop, 5 runs, discard first warm-up; report median and (max − min) |
| **Target** | Median ≤ **2.0 s** on documented CPU/GPU |
| **Conditions** | Same test image; batch size 1; no debug visualization |

**Verification:** Paste timing table. **Status: not verified.**

---

## Results log (paste raw outputs here after runs)

### Calibration

```text
2026-11-04 run: 28 discovered, 23 used, 5 failed detection (incl. pen shots in cal folder)
mean_reprojection_error_px ≈ 16.86 (POOR — recalibrate before MO-1)
See calibration/calibration_report.json
```

### Segmentation (test split)

```text
2026-11-05 eval (12-epoch CPU train, image_size=256, best val IoU=0.631):
  mean_iou: 0.596
  precision: 0.756
  recall: 0.674
  map_at_0.5: 0.778
  map_0.5_0.95: 0.344
  split: test, num_images: 9
  Source: models/runs/unet_pen/test_metrics.json
MO-2 target (IoU ≥ 0.80, mAP@0.5 ≥ 0.75): NOT MET — retrain on GPU, more epochs, then re-eval once.
```

### Measurement validation (≥10 instances)

```text
Command:
  scripts/validate_ruler.py --csv validation/ruler_refs.csv --marker-size-mm 198.0
    --checkpoint models/runs/unet_pen/best.pt --allow-poor-calibration --skip-undistort

Summary (validation/ruler_comparison.json, 2026-11-05):
  count: 20, successful: 17 (rejected: pen_0036, pen_0051, pen_0076)
  length MAE: 20.93 mm, length mean % error: 15.12%
  width MAE: 6.19 mm, width mean % error: 51.29%
  MO-1 target (MAE ≤ 2 mm, mean |%| ≤ 5%): NOT MET on full set

Best length cases (|error| ≤ ~4 mm): pen_0038, pen_0045, pen_0062, pen_0070, pen_0074, pen_0077, pen_0078, pen_0086
Partial-seg / fallback failures inflate MAE (e.g. pen_0037, pen_0050, pen_0066, pen_0072, pen_0087).

Per-instance table: validation/ruler_refs.csv (reference_* manual; system_* from script above).

Demo single image: measurement/output/pen_0042_result.json — 138.37 mm vs ruler ~135 mm.

Note: Conditions used raw HEIC + skip undistort (cal ~17 px); not yet undistorted face-on set.
```

### Raw vs undistorted comparison

```text
Ruler stems (10): validation/raw_vs_undistorted_ruler_stems.json
  raw_success: 7/10, undist_success: 1/10 (bad calibration ~17 px — undistort breaks ArUco on most frames)
  Use --skip-undistort on raw HEIC until recalibration; then re-run scripts/raw_vs_undist_ruler_stems.py
Single instance: validation/raw_vs_undistorted_pen_0042.json
```

---

## Rules for reported numbers (assessment §6)

- Every reported number comes from **this repository’s scripts** on **this machine** (raw paste).
- Errors: show **every instance**, then mean, median, and worst case.
- Timings: **5 runs**, report median and spread.
- Do not report metrics before the corresponding code and data exist on `dev-test01`.
