# Definition of Done — Measure It

Acceptance checklist for assessment v2.0. Boxes are ticked only when **evidence** (number, file path, commit link, or raw log) is recorded next to the item. An tick without evidence counts as **not done**.

**Evidence column:** fill with path, metric, or `commit:<hash>`.

---

## Project setup and Git workflow

- [ ] Public GitHub repo created under my account — **Evidence:** _repo URL_
- [ ] Active branch is `dev-test01` — **Evidence:** _branch name / remote_
- [ ] Plan, Objectives, and Definition of Done committed **before** pipeline code — **Evidence:** _first commit hash / message_
- [ ] Multiple incremental commits (calibration → data → training → measurement), not one final blob — **Evidence:** _git log summary_
- [ ] Commit messages describe what changed and why — **Evidence:** _example messages_
- [ ] `requirements.txt` with pinned versions — **Evidence:** _file path_
- [ ] `.gitignore` excludes `.venv/`, large weights (or weights released separately) — **Evidence:** _file path_

---

## Camera calibration

- [ ] Printed checkerboard (9×7 inner corners) and ArUco pattern generated — **Evidence:** _paths under `calibration/patterns/`_
- [ ] Printed square size measured with ruler; value used in `calibrate.py --square-size` — **Evidence:** _measured mm_
- [ ] ≥20 calibration images captured in `calibration/images/` — **Evidence:** _count_
- [ ] `calibrate.py` completed with ≥20 successful detections — **Evidence:** `calibration_report.json` → `images_used`
- [ ] `camera_params.npz` saved and loads in `undistort.py` — **Evidence:** _path_
- [ ] Camera matrix and distortion coefficients recorded — **Evidence:** _report JSON or Objectives paste_
- [ ] Mean (and RMS) reprojection error reported — **Evidence:** _px values from report_
- [ ] At least one undistorted image checked visually (straight edges straight) — **Evidence:** _image path or comparison PNG_

---

## Dataset collection

- [ ] Camera focus, zoom, resolution locked and documented — **Evidence:** _README / Objectives_
- [ ] ≥70 original pen photos captured by me — **Evidence:** _count_
- [ ] Images undistorted for labelling/training (or documented equivalent) — **Evidence:** _convention_
- [ ] 3–5 different pens represented — **Evidence:** _brief list_
- [ ] No third-party/downloaded pen images — **Evidence:** _provenance note_

---

## Dataset annotation and splitting

- [ ] Segmentation masks created for all training/val/test images — **Evidence:** _label directory_
- [ ] Labelling tool named in README — **Evidence:** _tool name_
- [ ] Split files `train` / `val` / `test` at 70% / 20% / 10% — **Evidence:** _paths + counts_
- [ ] Split committed **before** training started — **Evidence:** _commit link before training commit_
- [ ] Test filenames never used for training, tuning, or checkpoint pick — **Evidence:** _procedure note + code review_

---

## Model training

- [ ] Segmentation model is **not** Ultralytics YOLO or Roboflow — **Evidence:** _architecture name in code_
- [ ] Model trained (or fine-tuned) on our masks — **Evidence:** _train script + config_
- [ ] Random seed and config pinned — **Evidence:** _config file_
- [ ] Trained weights in repo or release with SHA-256 in README — **Evidence:** _hash_

---

## Model evaluation

- [ ] Evaluation run on **test** split only — **Evidence:** _eval script invocation_
- [ ] Mask IoU reported — **Evidence:** _value_
- [ ] mAP@0.5 reported — **Evidence:** _value_
- [ ] mAP@0.5:0.95 reported — **Evidence:** _value_
- [ ] Precision and recall reported — **Evidence:** _values_
- [ ] Raw metric log pasted in `Objectives.md` — **Evidence:** _section link_

---

## Measurement pipeline

- [ ] Single command: new image → undistort → segment → reference → mm — **Evidence:** _README command_
- [ ] Outputs mask overlay with length (mm) and max width (mm) and confidence — **Evidence:** _sample output path_
- [ ] ArUco physical size measured and configured — **Evidence:** _mm value + config_
- [ ] Handles **object not found** cleanly — **Evidence:** _log or screenshot_
- [ ] Handles **reference not found** cleanly — **Evidence:** _log or screenshot_

---

## Accuracy validation

- [ ] ≥10 instances measured with ruler/calliper and system — **Evidence:** _table path_
- [ ] Per-instance table committed — **Evidence:** _CSV/JSON_
- [ ] MAE (mm) and mean percentage error computed — **Evidence:** _values_
- [ ] Same instances measured on **raw** and **undistorted** images — **Evidence:** _table columns_
- [ ] Explanation of difference uses **our** numbers — **Evidence:** _Objectives or README paragraph_

---

## Failure-condition testing

- [ ] One condition chosen (tilt, distance, lighting, or edge-of-frame) — **Evidence:** _which one_
- [ ] Error measured as condition worsens — **Evidence:** _table or plot path_
- [ ] Short rationale for choice — **Evidence:** _Plan or here_

---

## Measurement rejection

- [ ] Rejection rules implemented (e.g. low mask confidence, partial ArUco, tilt) — **Evidence:** _code path_
- [ ] Threshold values documented — **Evidence:** _config/constants_
- [ ] Count or rate of refused images reported — **Evidence:** _numbers_

---

## Documentation and final demonstration

- [ ] `README.md` — setup, run each stage, dataset summary — **Evidence:** _reviewed_
- [ ] `docs/Plan.md` — decision record closed (taken / rejected / cost) — **Evidence:** _§15 filled_
- [ ] `docs/Objectives.md` — MO-1 and MO-2 verified with raw outputs — **Evidence:** _results log_
- [ ] MO-1 and MO-2 targets stated with justification — **Evidence:** _Objectives tables_
- [ ] Loom ≤5 min with live demo + four knowledge questions — **Evidence:** _URL_
- [ ] PR from `dev-test01` → `main` on my repo — **Evidence:** https://github.com/kashifmayar/measure-it/pull/1
- [ ] Submission reply with PR + Loom links within 24 h — **Evidence:** _sent timestamp_

---

## Work not finished (assessment honesty)

List anything not completed and why:

```text
- Calibration: ~16.9 px reprojection — recalibrate (last step before submit).
- MO-2: test IoU 0.535 (target 0.80) — document honestly; optional GPU retrain.
- MO-1: validation/ruler_refs.csv has 10+ system rows; user must fill reference_* mm then run validate_ruler.py.
- PR dev-test01→main: open via GitHub compare (gh not logged in on dev machine).
- Loom + submission email: user (see docs/Loom_Guide.md).
- Weights: models/runs/unet_pen/best.pt local; add SHA-256 to README (scripts/hash_checkpoint.ps1).
```

---

## Sign-off

- [ ] I did not tick boxes without evidence — **Evidence:** _self-review date_
- [ ] All numbers in docs are reproducible from this repo — **Evidence:** _review date_

**Last updated:** 2026-11-05 — pipeline largely implemented; evidence columns still need your runs and commits.
