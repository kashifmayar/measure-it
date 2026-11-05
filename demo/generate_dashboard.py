"""
Build demo/index.html from validation JSON and sample measurement output.

  .\\.venv\\Scripts\\python.exe demo\\generate_dashboard.py
"""

from __future__ import annotations

import json
import html
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
OUT = DEMO / "index.html"


def _load(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _esc(obj: object) -> str:
    return html.escape(json.dumps(obj, indent=2) if obj is not None else "(missing)")


def main() -> None:
    test_metrics = _load(ROOT / "validation" / "test_metrics.json")
    pen_result = _load(ROOT / "measurement" / "output" / "pen_0042_result.json")
    rejection = _load(ROOT / "validation" / "rejection_report_test.json")
    cal_report = _load(ROOT / "calibration" / "calibration_report.json")

    cal_err = None
    if isinstance(cal_report, dict):
        cal_err = cal_report.get("calibration_summary", {}).get("mean_reprojection_error_px")

    curves_src = ROOT / "models" / "runs" / "unet_pen" / "training_curves.png"
    curves_rel = ""
    if curves_src.is_file():
        shutil.copy2(curves_src, DEMO / "training_curves.png")
        curves_rel = "training_curves.png"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Measure It — Demo Dashboard</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3b82f6;
      --ok: #22c55e;
      --warn: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px 32px 48px;
      line-height: 1.5;
    }}
    h1 {{ font-size: 1.75rem; margin: 0 0 8px; }}
    .sub {{ color: var(--muted); margin-bottom: 28px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card);
      border-radius: 12px;
      padding: 18px 20px;
      border: 1px solid #2a3548;
    }}
    .card h2 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted); margin: 0 0 12px; }}
    .metric {{ font-size: 1.8rem; font-weight: 600; }}
    .metric small {{ font-size: 0.9rem; color: var(--muted); font-weight: 400; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-ok {{ background: #14532d; color: var(--ok); }}
    .badge-warn {{ background: #422006; color: var(--warn); }}
    pre {{
      background: #0d1117;
      padding: 14px;
      border-radius: 8px;
      overflow: auto;
      font-size: 0.78rem;
      max-height: 220px;
    }}
    .flow {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 12px 0;
    }}
    .step {{
      background: #243044;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.85rem;
    }}
    .arrow {{ color: var(--muted); }}
    img {{ max-width: 100%; border-radius: 8px; margin-top: 8px; }}
    code {{ background: #243044; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
    .cmd {{
      background: #0d1117;
      padding: 14px;
      border-radius: 8px;
      font-family: Consolas, monospace;
      font-size: 0.8rem;
      white-space: pre-wrap;
      word-break: break-all;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <h1>Measure It</h1>
  <p class="sub">xis.ai CV Assessment v2.0 — live demo dashboard for Loom (regenerate before recording)</p>

  <div class="grid">
    <div class="card">
      <h2>Calibration</h2>
      <div class="metric">{html.escape(str(cal_err) if cal_err is not None else "—")} <small>px mean reproj.</small></div>
      <p style="margin:12px 0 0;font-size:0.85rem">
        {"<span class='badge badge-warn'>Recalibrate before final submit</span>" if cal_err and float(cal_err) > 2 else "<span class='badge badge-ok'>OK for CLI</span>"}
      </p>
    </div>
    <div class="card">
      <h2>Test segmentation (U-Net)</h2>
      <div class="metric">{test_metrics.get("mean_iou", "—") if test_metrics else "—"} <small>mean IoU</small></div>
      <p style="font-size:0.85rem;color:var(--muted);margin:8px 0 0">
        mAP@0.5: {test_metrics.get("map_at_0.5", "—") if test_metrics else "—"}
      </p>
    </div>
    <div class="card">
      <h2>Demo measurement</h2>
      <div class="metric">{pen_result.get("pen_length_mm", "—") if pen_result else "—"} <small>mm length</small></div>
      <p style="font-size:0.85rem;margin:8px 0 0">
        width {pen_result.get("pen_max_width_mm", "—") if pen_result else "—"} mm ·
        {pen_result.get("segmentation_method", "") if pen_result else ""}
      </p>
    </div>
    <div class="card">
      <h2>Reference</h2>
      <p style="margin:0;font-size:0.95rem">ArUco black square: <strong>34 mm</strong></p>
      <p style="margin:8px 0 0;font-size:0.85rem;color:var(--muted)">
        PR: <a href="https://github.com/kashifmayar/measure-it/pull/1" target="_blank">#1</a>
      </p>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px">
    <h2>Pipeline (one command)</h2>
    <div class="flow">
      <span class="step">Input image</span><span class="arrow">→</span>
      <span class="step">Undistort</span><span class="arrow">→</span>
      <span class="step">ArUco + homography</span><span class="arrow">→</span>
      <span class="step">U-Net mask</span><span class="arrow">→</span>
      <span class="step">Length & width (mm)</span>
    </div>
    <div class="cmd">.\\.venv\\Scripts\\python.exe main.py `
  --image dataset\\raw\\pen_0042.HEIC `
  --marker-size-mm 34.0 `
  --checkpoint models\\runs\\unet_pen\\best.pt `
  --allow-poor-calibration `
  --skip-undistort `
  --output-dir measurement\\output</div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Sample result JSON</h2>
      <pre>{_esc(pen_result)}</pre>
    </div>
    <div class="card">
      <h2>Test metrics JSON</h2>
      <pre>{_esc(test_metrics)}</pre>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h2>Test-split rejection (dataset/images — no ArUco in crop)</h2>
    <pre>{_esc(rejection.get("summary") if isinstance(rejection, dict) else rejection)}</pre>
  </div>

  {"<div class='card' style='margin-top:16px'><h2>Training curves</h2><img src='" + html.escape(curves_rel) + "' alt='training curves'/></div>" if curves_rel else ""}

  <p class="sub" style="margin-top:32px">
    Loom K1–K4: see <code>docs/Loom_Guide.md</code> · Regenerate: <code>python demo/generate_dashboard.py</code>
  </p>
</body>
</html>
"""
    OUT.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
