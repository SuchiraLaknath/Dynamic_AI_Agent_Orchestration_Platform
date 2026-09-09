"""Renders docs/codebase-guide.html to docs/codebase-guide.pdf.

The guide is hand-authored HTML, so this only has to print it. The Chrome
machinery lives in `pdf_renderer.py`, shared with `render-report-pdf.py`.

Run:  .venv/bin/python docs/render-guide-pdf.py
"""

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS))

from pdf_renderer import render_to_pdf  # noqa: E402

SOURCE = DOCS / "codebase-guide.html"
OUTPUT = DOCS / "codebase-guide.pdf"


if __name__ == "__main__":
    render_to_pdf(SOURCE, OUTPUT)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size // 1024} KB)")
