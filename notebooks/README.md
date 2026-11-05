# Colab GPU training

Open **`MeasureIt_Colab_GPU.ipynb`** in [Google Colab](https://colab.research.google.com/):

1. **File → Upload notebook** (or upload from GitHub: `notebooks/MeasureIt_Colab_GPU.ipynb` on `dev-test01`).
2. **Runtime → Change runtime type → T4 GPU**.
3. **Avoid Colab “Choose files” upload** — it is slow for hundreds of MB.

   Zip on your PC (full or smaller):

```powershell
.\scripts\pack_dataset_for_colab.ps1
# or faster upload:
.\scripts\pack_dataset_for_colab_small.ps1
```

   Upload the zip to **Google Drive** (web or Drive for desktop). In the notebook set `DRIVE_ZIP_PATH` and `USE_DRIVE = True`.

4. Zip layout (same for both scripts):
   - `dataset/images/` (all `pen_*.jpg`)
   - `dataset/masks/` (all `pen_*.png`)
   - Layout inside zip: `images/...` and `masks/...` so extract lands under `dataset/`.
5. Run all cells **except** the Colab `files.upload()` cell if you used Drive; download **`best.pt`** and copy to:

   `E:\Project\measure it\models\runs\unet_pen\best.pt`

6. Re-run `models\evaluate.py` locally or paste Colab `test_metrics_colab.json` into `docs\Objectives.md`.

**Not in Colab:** calibration, `main.py` measurement on HEIC (do that on Windows with downloaded weights).
