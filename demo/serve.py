"""
Regenerate dashboard and serve on http://127.0.0.1:8765 for Loom recording.

  .\\.venv\\Scripts\\python.exe demo\\serve.py
"""

from __future__ import annotations

import http.server
import os
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
PORT = 8765


def main() -> None:
    subprocess.run([sys.executable, str(DEMO / "generate_dashboard.py")], check=True, cwd=ROOT)

    os.chdir(DEMO)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(format % args)

    with socketserver.TCPServer(("127.0.0.1", PORT), Quiet) as httpd:
        url = f"http://127.0.0.1:{PORT}/index.html"
        print(f"Serving demo dashboard at {url}")
        print("Press Ctrl+C to stop.")
        webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
