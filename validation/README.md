# Validation artifacts

Committed JSON files are **reproducible** from scripts in `scripts/` (marker **34 mm**).

| File | How to regenerate |
|------|-------------------|
| `test_metrics.json` | `models/evaluate.py --split test` |
| `test_split_measurement_report.json` | `scripts/run_test_eval_and_reports.py --marker-size-mm 34.0` |
| `raw_vs_undistorted_pen_0042.json` | `scripts/compare_raw_undistorted.py --image dataset/raw/pen_0042.HEIC ...` |
| `ruler_refs.csv` | Copy from `ruler_refs.csv.example`; fill ≥10 manual mm measurements |

MO-1 ruler table: run `scripts/validate_ruler.py` after `ruler_refs.csv` is complete.
