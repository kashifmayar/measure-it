# Loom recording guide (Assessment v2.0 §7)

**Max length:** 5 minutes. **Tool:** Loom (or equivalent). Submit link with PR within 24 h of start.

## Timeline (from PDF)

| Time | Content |
|------|---------|
| ~1 min | New photo of pen + ArUco → run `main.py` → ruler/calliper same object on camera |
| ~1.5 min | Walk through **your code** for that image: undistort (or skip), segment, ArUco, mm |
| ~2.5 min | Answer **K1–K4** out loud (no reading notes) |

## Demo commands (after recalibration, adjust flags)

**Before recalibration (current project):**

```powershell
cd "E:\Project\measure it"
.\.venv\Scripts\python.exe main.py `
  --image dataset\raw\pen_0042.HEIC `
  --marker-size-mm 34.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\output
```

**After good calibration (target):** drop `--skip-undistort` and `--allow-poor-calibration` when reprojection ≤ 2 px.

Pre-flight: `pytest tests\ -q`, `models\evaluate.py`, checkpoint exists at `models\runs\unet_pen\best.pt`.

### Browser dashboard (optional, good for Loom B-roll)

Shows metrics, pipeline, sample JSON, training curves:

```powershell
.\.venv\Scripts\python.exe demo\serve.py
```

Opens http://127.0.0.1:8765/index.html — record this alongside the terminal `main.py` run.

---

## The four questions (answer on camera)

From **AI / Computer Vision Technical Assessment v2.0**, §7.2:

| ID | Question |
|----|----------|
| **K1** | Which **file and line** turns pixels into millimetres? What has to be true about the **scene** for that conversion to be valid? |
| **K2** | Where in your code is **undistortion** applied, and **how much did skipping it change** your measurement error? Use **your own numbers**. |
| **K3** | Name **one thing in your own work you would reverse**, and what changing it would **cost**. |
| **K4** | Name **one thing you did not understand**, and **what you did about it**. |

### Suggested talking points (verify against your repo before recording)

**K1 — mm conversion**

- **File:** `measurement/perspective.py` — homography from ArUco corners to a metric plane; scale from `--marker-size-mm` (34 mm measured) and `pixels_per_mm`.
- **Geometry:** `measurement/pen_detector.py` — length/width from mask in rectified plane using `pixels_per_mm`.
- **Scene must be true:** pen and ArUco on the **same flat plane**; marker fully visible; coplanarity (see `docs/Plan.md` §6).

**K2 — undistortion**

- **Applied in:** `measurement/measure_pen.py` → `_undistort_bgr()` before ArUco; also `calibration/undistort.py` for dataset prep.
- **Your numbers:** paste from `validation/raw_vs_undistorted_batch.json` or `compare_raw_undistorted.py` — e.g. `pen_0042` success with skip vs reject with undistort until recalibrated.

**K3 — would reverse**

- Example: train only on CPU with 15 epochs → would use GPU + more epochs; cost: time / MO-2 still below 0.80 IoU target.
- Or: poor calibration run kept early → would reshoot checkerboard-only set; cost: reshoot time.

**K4 — did not understand**

- Honest example: double-undistort on `dataset/images/` broke ArUco → learned to use raw HEIC + `--skip-undistort` until cal fixed; documented in README.

**Rule:** “I do not know, here is what I tried” is OK. Confident wrong answers fail review.

---

## PR (step 1 — you do once)

1. https://github.com/kashifmayar/measure-it/compare/main...dev-test01  
2. Create PR, title e.g. *Measure It assessment pipeline*  
3. Paste PR URL in submission email with Loom link.

**Weights:** Add SHA-256 of `models/runs/unet_pen/best.pt` to README after training (see `scripts/hash_checkpoint.ps1` if added).
