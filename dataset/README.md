# Pen segmentation dataset

## Layout

```text
dataset/
  raw/              # Original iPhone HEIC/JPEG (do not edit pixels)
  images/           # Undistorted RGB — IDs like pen_0001.jpg
  masks/            # Final binary PNG masks (pen=255, bg=0) — same ID as images/
  masks_draft/      # Auto GrabCut drafts — review before using as ground truth
  splits/           # train.txt, val.txt, test.txt (IDs: pen_0001, one per line)
  meta/
    filename_map.json   # pen_0001 <-> original IMG_xxxx
    dataset_summary.json
    split_summary.json
  README.md
```

## Friendly file names

iPhone names (`IMG_2244.HEIC`) are hard to track. Rename once to **`pen_0001` … `pen_0087`**:

```powershell
.\.venv\Scripts\python.exe scripts\rename_dataset.py
# Preview only:
.\.venv\Scripts\python.exe scripts\rename_dataset.py --dry-run
```

Mapping is saved in **`meta/filename_map.json`** (original stem preserved).

## Assessment targets

| Item | Target | Current status |
|------|--------|----------------|
| Pen images | **≥ 70** | See `meta/dataset_summary.json` after running prepare scripts |
| Masks | **1 per training image** | You must annotate (see below) |
| Split | **70% / 20% / 10%** | `splits/*.txt` (fixed seed **42**) |

## Prepare images (PowerShell)

From repo root:

```powershell
# 1. Move loose HEIC from dataset/ root into raw/ (skips .MOV)
.\.venv\Scripts\python.exe scripts\prepare_dataset.py organize

# 2. Undistort raw -> images/ (requires calibration/camera_params.npz)
.\.venv\Scripts\python.exe scripts\prepare_dataset.py undistort

# 3. Create train/val/test lists from images/ (before training)
.\.venv\Scripts\python.exe scripts\prepare_dataset.py split

# 4. Check readiness (pairs, counts)
.\.venv\Scripts\python.exe scripts\prepare_dataset.py validate
```

Optional: include `measurement/images/` in organize:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py organize --include-measurement
```

## Annotation (required before training)

Train on **undistorted** files in `dataset/images/`, not `raw/`.

### Automatic draft masks (fast bootstrap — not final labels)

GrabCut guesses foreground; **you must review** every mask before training.

```powershell
# Try 5 images first
.\.venv\Scripts\python.exe scripts\generate_draft_masks.py --limit 5
# All images (slow on full 12 MP)
.\.venv\Scripts\python.exe scripts\generate_draft_masks.py
```

Outputs: `dataset/masks_draft/pen_XXXX.png`. Fix in Label Studio, then save finals to `dataset/masks/`.

**Easier manual options (assessment-allowed):**

| Tool | Automatic help |
|------|----------------|
| **[CVAT](https://app.cvat.ai/)** | Polygons, mask brush, optional AI interactors — see **CVAT workflow** below |
| **[Label Studio](https://labelstud.io/)** | Brush/polygon; optional ML backend (e.g. SAM) if you install models |
| **Roboflow Annotate** | Allowed as **labelling tool only** (not as your trained segmentation model) |

There is no fully reliable fully-automatic pen mask for all colours/backgrounds without review.

### CVAT workflow (recommended for this project)

Images to upload: **`dataset/images/pen_*.jpg`** (87 undistorted files).  
Goal: one binary mask per image in **`dataset/masks/pen_XXXX.png`**.

1. **Account:** [https://app.cvat.ai/](https://app.cvat.ai/) (free tier is enough to start).
2. **Project:** `Measure It` → add label **`pen`** (single class).
3. **Create task:**
   - **Select files** from `E:\Project\measure it\dataset\images\` (all `pen_*.jpg`).
   - If upload is slow, zip the folder first and upload the zip.
   - **Sort:** keep default order; filenames must stay `pen_0001.jpg`, etc.
4. **Open the job** → choose **Standard** or **Attribute annotation** workspace.
5. **Draw masks:**
   - Tool: **Polygon** (`N`) or **Mask** / brush (segmentation mode).
   - Outline **pen only** — exclude ArUco marker and table when possible.
   - **Save** (`Ctrl+S`) often; **Submit** when the frame is done, then next (`F`) / sidebar.
6. **Shortcuts:** `N` polygon, `F` next frame, `Ctrl+S` save, scroll wheel zoom.
7. **Export** (task or project menu → **Export task dataset**):
   - Format: **`Segmentation mask 1.1`** (PNG masks; preferred), or **`COCO 1.0`** (needs conversion).
   - Download ZIP and extract.
8. **Import into this repo:**

   ```powershell
   .\.venv\Scripts\python.exe scripts\prepare_dataset.py import-masks --from-dir path\to\extracted\masks
   ```

   If mask filenames differ, rename so each mask matches the image stem (`pen_0042.png` ↔ `pen_0042.jpg`).

9. **Validate:**

   ```powershell
   .\.venv\Scripts\python.exe scripts\prepare_dataset.py validate
   ```

**Local CVAT (optional):** [CVAT Docker](https://opencv.github.io/cvat/docs/administration/basics/installation/) if you prefer not to upload to the cloud.

**Time:** ~2–4 hours for 87 images after ~10 practice frames (similar to Label Studio).

### Label Studio (manual or semi-auto)

1. Install [Label Studio](https://labelstud.io/) locally or use Docker.
2. Create project → **Semantic segmentation** with masks.
3. Import images from `dataset/images/*.jpg`.
4. Draw **polygon/brushes** around the **pen only** (exclude ArUco unless you want two classes; binary pen is enough).
5. Export **PNG masks** (or COCO and convert).
6. Place masks in `dataset/masks/` with **identical basename**:

   ```text
   images/IMG_2207.jpg  <->  masks/IMG_2207.png
   ```

7. Run validate:

   ```powershell
   .\.venv\Scripts\python.exe scripts\prepare_dataset.py validate
   ```

### Import helper (Label Studio PNG export folder)

If exports live in a single folder as `{basename}-mask.png`:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py import-masks --from-dir path\to\export
```

## Training readiness checklist

- [ ] ≥ 70 images in `dataset/images/`
- [ ] Every train/val/test list entry has `masks/{stem}.png`
- [ ] `validate` reports **ready_for_training: true**
- [ ] Commit **`dataset/splits/*.txt`** before any training commit
- [ ] Do **not** use test filenames for tuning or checkpoint selection

## Notes

- **`.MOV` files** are Live Photo sidecars; they are ignored.
- Masks produced by the classical `main.py` baseline are **not** ground truth for training unless manually corrected.
- Fix camera calibration (low reprojection error) before trusting geometry; undistortion still uses current `camera_params.npz`.
