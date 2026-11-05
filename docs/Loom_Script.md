# Loom script (short index)

**Full step-by-step script, checklist, and K1–K4 talking points:** [`Loom_Recording.md`](Loom_Recording.md)

**Assessment rules only:** [`Loom_Guide.md`](Loom_Guide.md)

**PR:** https://github.com/kashifmayar/measure-it/pull/1

**Demo command (your measured outer black square ≈198 mm):**

```powershell
cd "E:\Project\measure it"
.\.venv\Scripts\python.exe main.py `
  --image dataset\raw\pen_0042.HEIC `
  --marker-size-mm 198.0 `
  --checkpoint models\runs\unet_pen\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\output
```

**Marker diagram:** [`marker_measure_which_side.png`](marker_measure_which_side.png)
