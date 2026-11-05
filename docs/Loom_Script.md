# Loom script — Measure It (≤5 minutes)

Read **`docs/Loom_Guide.md`** for rules. **Do not read this script on camera** for K1–K4 — rehearse once, then speak naturally.

**PR for submission:** https://github.com/kashifmayar/measure-it/pull/1

---

## Before you hit Record

1. **Phone charged** — you will photograph pen + ArUco on desk.
2. **Printed ArUco** (ID 0) + one pen, flat on same surface.
3. **Ruler/calliper** visible for the live measure beat.
4. **PC ready:**
   ```powershell
   cd "E:\Project\measure it"
   .\.venv\Scripts\Activate.ps1
   Test-Path models\runs\unet_pen\best.pt
   ```
   Copy new photo to e.g. `measurement\images\loom_demo.HEIC` (or use USB transfer).
5. **Optional second window:** `demo\serve.py` dashboard OR VS Code on `main.py` / `measure_pen.py`.
6. **Loom:** screen + webcam (or phone for desk shot); **max 5:00**.

---

## Timeline (what assessors expect)

| When | What |
|------|------|
| 0:00–1:00 | New photo → run CLI → show mm output → ruler same pen |
| 1:00–2:30 | Your code path: file → mm (scroll slowly) |
| 2:30–5:00 | K1, K2, K3, K4 out loud |

---

## PART A — Live demo (~1 minute)

**Say (example):**

> “This is Measure It — pen length and max width in millimetres from one iPhone photo, using a printed ArUco marker at 34 millimetres and a U-Net trained on my own dataset.”

**Do on camera:**

1. Lay **pen + ArUco** flat; take **one new photo** with iPhone (rear camera).
2. AirDrop / copy to PC as `measurement\images\loom_demo.HEIC`.

**Run in terminal (adjust path):**

```powershell
cd "E:\Project\measure it"
.\.venv\Scripts\python.exe main.py `
  --image measurement\images\loom_demo.HEIC `
  --marker-size-mm 34.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\output
```

If you **recalibrated** (reprojection ≤ 2 px), **remove** `--allow-poor-calibration` and `--skip-undistort`.

**Show on screen:**

- Terminal: `OK: length=… mm, max_width=… mm (unet_segmentation)`
- Open `measurement\output\loom_demo_rectified.jpg` or overlay image if saved
- **Hold ruler/calliper** to the **same pen** and read length (does not have to match perfectly if cal still poor — be honest)

**Say:**

> “The pipeline rejected or accepted with a reason if the marker or pen is missing; here we got a successful measurement with confidence in the JSON.”

---

## PART B — Code walk (~1.5 minutes)

**Say:**

> “I'll follow one image through the repo on branch dev-test01.”

**Scroll in this order (name files aloud):**

1. **`main.py`** — builds config, calls `measure_pen_image`, writes JSON and images.
2. **`measurement/measure_pen.py`**
   - Load image (`read_image_bgr`)
   - **`_undistort_bgr`** around line 58 — or **`skip_undistort`** if flag set
   - **`select_marker`** in `aruco_detector.py` — DICT_4X4_50, ID 0
   - **`rectify_plane`** in `perspective.py` — homography; marker defines metric plane
3. **`measurement/pen_detector.py`**
   - **`detect_pen_unet`** → mask on rectified image
   - **`length_mm = length_px / ppm`** around lines 157–158 — mm from rectified pixels
4. **`models/train.py`** + **`models/unet.py`** — one sentence: trained on 87 masks, not YOLO.
5. **`dataset/splits/test.txt`** — test held out; metrics in `validation/test_metrics.json`.

**Say:**

> “Dataset is 87 images with CVAT masks; splits were committed before training.”

---

## PART C — Four questions (~2.5 minutes)

Answer **without reading**. Use **your** repo facts below.

### K1 — Pixels to millimetres + valid scene

**Question:** Which file and line turns pixels into millimetres? What must be true about the scene?

**Your answer (guide):**

- **Scale on the plane:** `measurement/perspective.py` — `rectify_plane()` builds a homography so the ArUco square is **34 mm** with **`pixels_per_mm`** (default 8). See `_destination_corners` and warp (~lines 36–81).
- **Pen dimensions:** `measurement/pen_detector.py` — **`length_mm = length_px / ppm`** and **`max_width_mm = width_px / ppm`** (~lines 157–158), where `ppm` is `config.pixels_per_mm` on the **rectified** image.
- **Scene must be true:** pen and ArUco on the **same flat plane**; all **four marker corners** visible; not heavily tilted; marker size **measured** (34 mm), not guessed. See `docs/Plan.md` §6 coplanarity.

---

### K2 — Undistortion + your numbers

**Question:** Where is undistortion applied, and how much did skipping it change error?

**Your answer (guide):**

- **Where:** `measurement/measure_pen.py` — **`_undistort_bgr()`** (~line 58) using `calibration/camera_params.npz`. Training images undistorted in `scripts/prepare_dataset.py` / `calibration/undistort.py`.
- **Your numbers (example from `validation/raw_vs_undistorted_ruler_stems.json`):**
  - On 10 validation stems: **7/10 succeed with `--skip-undistort`**, only **1/10 with full undistort** while calibration reprojection is ~**17 px**.
  - Example **pen_0038:** raw length **22.57 mm** vs undistorted **20.3 mm** (~**2.3 mm** difference on length).
- **Honest line:** I use skip-undistort on raw HEIC until recalibration; I documented that in the README.

---

### K3 — One thing you would reverse

**Question:** One thing you would reverse and the cost?

**Pick one (honest):**

- **Poor calibration kept too long** → would reshoot 25+ checkerboard-only photos and re-run `calibrate.py`. **Cost:** a few hours capture + trust in mm.
- **CPU-only U-Net training** → would train on Colab GPU longer. **Cost:** time; **benefit:** higher val IoU (test IoU was ~0.60, target 0.80).
- **Large 1 GB Colab zip** → would use `pack_dataset_for_colab_small.ps1` first. **Cost:** slightly lower-res training data.

---

### K4 — One thing you did not understand

**Question:** One thing you did not understand and what you did?

**Pick one (honest):**

- **Why `dataset/images` JPG failed ArUco in measurement** — training crops are undistorted and often **no marker in frame**; measurement needs **`dataset/raw` HEIC**. I added README notes and `--skip-undistort`.
- **Why double undistort broke detection** — undistorted JPGs fed through cal again; fixed by skip flag.
- **MO-2 IoU below target** — I ran more epochs and Colab GPU; reported actual test metrics instead of inflating numbers.

**Closing line:**

> “Pull request is number one on my repo; weights are local with SHA-256 in the README.”

---

## After recording

1. Watch once — under **5:00**? Can you hear all **four** answers?
2. Email assessors: **PR #1** + **Loom URL**.
3. Optional: merge PR to `main`.

---

## If live demo fails on the new photo

**Say on camera:**

> “Marker not detected — rejection reason is in the JSON; I'll show a known-good run on pen_0042.”

```powershell
.\.venv\Scripts\python.exe main.py --image dataset\raw\pen_0042.HEIC --marker-size-mm 34.0 --checkpoint models\runs\unet_pen\best.pt --allow-poor-calibration --skip-undistort --output-dir measurement\output
```

Assessors accept honest failure + explanation; **do not** fake success.
