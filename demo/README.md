# Demo dashboard (Loom)

Local browser UI summarizing calibration, test metrics, and a sample measurement.

```powershell
cd "E:\Project\measure it"
.\.venv\Scripts\python.exe demo\serve.py
```

- Regenerates `index.html` from `validation/*.json` and `measurement/output/pen_0042_result.json`
- Opens http://127.0.0.1:8765/index.html

For the recording: show this dashboard **and** a live `main.py` run on a new photo (assessment requires both).
