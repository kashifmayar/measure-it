# Manual vs automated work (why some rubric items score low)

## MO-1 ruler proof (≥10, MAE) — **manual only**

| Who | What |
|-----|------|
| **You** | Open each photo in `dataset/raw/` (match `validation/ruler_refs.csv` stem). Measure the **same physical pen** with ruler/calliper. Type mm into `reference_length_mm` and `reference_width_mm`. |
| **Script** | `validate_ruler.py` computes system mm, MAE, and % error after references exist. |

**Why score was 0:** Empty reference columns → no ground truth → MAE cannot exist. **AI cannot invent ruler mm.**

---

## Raw vs undistorted — **half manual**

| Who | What |
|-----|------|
| **You** | Recalibrate (`calibrate.py`) so undistort path works. |
| **Script** | `raw_vs_undist_ruler_stems.py` builds comparison JSON for CSV stems. |

**Why score was low:** Bad calibration (~17 px) → undistort breaks ArUco; only `--skip-undistort` works on many frames.

---

## Failure condition — **you choose, script helps**

| Who | What |
|-----|------|
| **You** | Pick one condition (distance / tilt / lighting). Optionally shoot 3–5 photos at different severity. |
| **Script** | `failure_condition_distance.py` (synthetic distance proxy) or measure each photo with `main.py`. |

**Why score was low:** No committed table tying **error vs severity** with your numbers.

---

## Rejection rules — **mostly done; run report**

| Who | What |
|-----|------|
| **You** | Read `validation/rejection_report_raw.json` and paste summary into Objectives. |
| **Code** | `measurement/config.py`, `measure_pen.py` (ArUco, confidence, cal gate). |

**Why score was low:** Test-split report on `dataset/images` shows 0/9 (no ArUco in training crops) — misleading unless you explain and show **raw** report.

---

## MO-2 — **automated; target partly missed**

| Who | What |
|-----|------|
| **Script** | `models/evaluate.py` — metrics in `validation/test_metrics.json`. |
| **Optional you** | GPU retrain for IoU ≥ 0.80 (mAP@0.5 already ~0.78). |

**Why not full marks:** IoU **0.596** vs target **0.80** (honest miss); mAP@0.5 **0.778** ≈ pass.
