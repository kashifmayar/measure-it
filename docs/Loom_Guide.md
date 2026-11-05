# Loom guide — assessment rules (v2.0 §7)

**Follow this recording:** [`Loom_Recording.md`](Loom_Recording.md) (full script + checklist).

**Max length:** 5 minutes. Submit **PR + Loom URL** within 24 h (see `docs/Submission.md`).

---

## Timeline

| Time | Content |
|------|---------|
| ~1 min | New pen + ArUco photo → `main.py` → show mm → ruler/calliper same pen |
| ~1.5 min | Code walk: undistort or skip → segment → ArUco → mm |
| ~2.5 min | **K1–K4** out loud (no reading notes) |

---

## Pre-flight

```powershell
cd "E:\Project\measure it"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pytest tests\ -q
Test-Path models\runs\unet_pen\best.pt
```

Use **`--marker-size-mm`** = **outer black square** you measured (see [`marker_measure_which_side.png`](marker_measure_which_side.png)). Example for fit-to-page print: **198.0**.

With poor calibration (~17 px reprojection): raw HEIC, `--skip-undistort`, `--allow-poor-calibration`.

Optional dashboard: `.\.venv\Scripts\python.exe demo\serve.py` → http://127.0.0.1:8765/index.html

---

## K1–K4 (must answer on camera)

| ID | Question |
|----|----------|
| **K1** | Which **file/line** turns pixels into **mm**? What must be true about the **scene**? |
| **K2** | Where is **undistortion** applied, and **how much did skipping it change** error? **Your numbers.** |
| **K3** | One thing you would **reverse**, and the **cost**. |
| **K4** | One thing you **did not understand**, and **what you did**. |

Suggested answers tied to this repo: **`Loom_Recording.md` → PART C**.

**Rule:** “I don't know, here's what I tried” beats a confident wrong answer.

---

## PR

https://github.com/kashifmayar/measure-it/compare/main...dev-test01  
Open PR **#1**; link in submission email with Loom.

Weights: `models/runs/unet_pen/best.pt` (local); SHA-256 in README via `scripts/hash_checkpoint.ps1`.
