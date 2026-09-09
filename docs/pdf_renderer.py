"""Renders a local HTML file to PDF with headless Chrome.

Shared by `render-guide-pdf.py` and `render-report-pdf.py`, which is the whole
reason it is a module rather than a function copied twice.

It drives Chrome over the DevTools Protocol rather than the simpler
`--print-to-pdf` flag for one reason: the flag's only choices are no footer at
all, or Chrome's default footer, which stamps the local `file:///...` path onto
every page. The protocol lets us supply our own footer, so pages carry a plain
page number and nothing else.

Chrome is used because this machine has no LaTeX, weasyprint, wkhtmltopdf or
typst, so the installed pandoc has no PDF engine to reach.
"""

import asyncio
import base64
import json
import subprocess
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEBUG_PORT = 9333

PAGE_NUMBER_FOOTER = (
    "<div style='width:100%;font-family:Georgia,serif;font-size:8pt;"
    "color:#8a94a0;padding:0 16mm;text-align:center;'>"
    "<span class='pageNumber'></span></div>"
)

A4 = {"paperWidth": 8.27, "paperHeight": 11.69}
MARGINS = {"marginTop": 0.7, "marginBottom": 0.7, "marginLeft": 0.65, "marginRight": 0.65}


async def _wait_for_devtools(timeout_seconds: float = 30.0) -> str:
    """Poll Chrome's debug endpoint until it answers, returning its socket URL."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/version") as reply:
                return json.load(reply)["webSocketDebuggerUrl"]
        except OSError:
            await asyncio.sleep(0.5)
    raise RuntimeError("Chrome never exposed its DevTools endpoint")


async def render(source: Path, output: Path, settle_seconds: float = 3.0) -> None:
    """Open `source` in headless Chrome and write it to `output` as a PDF."""
    chrome = subprocess.Popen(
        [CHROME, "--headless", "--disable-gpu", f"--remote-debugging-port={DEBUG_PORT}",
         "--no-first-run", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        socket_url = await _wait_for_devtools()
        async with websockets.connect(socket_url, max_size=None) as connection:
            counter = {"id": 0}

            async def call(method: str, params: dict | None = None, session: str | None = None):
                counter["id"] += 1
                message = {"id": counter["id"], "method": method, "params": params or {}}
                if session:
                    message["sessionId"] = session
                await connection.send(json.dumps(message))
                while True:
                    reply = json.loads(await connection.recv())
                    if reply.get("id") == counter["id"]:
                        return reply.get("result", {})

            target = await call("Target.createTarget", {"url": "about:blank"})
            attached = await call(
                "Target.attachToTarget", {"targetId": target["targetId"], "flatten": True}
            )
            session = attached["sessionId"]

            await call("Page.enable", session=session)
            await call("Page.navigate", {"url": source.as_uri()}, session=session)
            # Fonts and inline SVG need a moment to lay out before printing.
            await asyncio.sleep(settle_seconds)

            result = await call("Page.printToPDF", {
                "printBackground": True,
                **A4,
                **MARGINS,
                "displayHeaderFooter": True,
                "headerTemplate": "<div></div>",
                "footerTemplate": PAGE_NUMBER_FOOTER,
            }, session=session)

            output.write_bytes(base64.b64decode(result["data"]))
    finally:
        chrome.terminate()


def render_to_pdf(source: Path, output: Path) -> Path:
    """Synchronous wrapper, for scripts that have nothing else to await."""
    asyncio.run(render(source, output))
    return output
