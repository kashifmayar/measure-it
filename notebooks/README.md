# Colab GPU training

Open **`MeasureIt_Colab_GPU.ipynb`** in [Google Colab](https://colab.research.google.com/):

1. **File → Upload notebook** (or upload from GitHub: `notebooks/MeasureIt_Colab_GPU.ipynb` on `dev-test01`).
2. **Runtime → Change runtime type → T4 GPU**.
3. Zip on your PC:
   - `dataset/images/` (all `pen_*.jpg`)
   - `dataset/masks/` (all `pen_*.png`)
   - Layout inside zip: `images/...` and `masks/...` so extract lands under `dataset/`.
4. Run all cells; download **`best.pt`** and copy to:

   `E:\Project\measure it\models\runs\unet_pen\best.pt`

5. Re-run `models\evaluate.py` locally or paste Colab `test_metrics_colab.json` into `docs\Objectives.md`.

**Not in Colab:** calibration, `main.py` measurement on HEIC (do that on Windows with downloaded weights).
