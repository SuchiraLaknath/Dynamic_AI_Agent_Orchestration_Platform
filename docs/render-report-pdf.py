"""Builds docs/project-report.pdf from README.md and docs/architecture.md.

One submission document containing everything in both sources. It is *generated*
from them rather than written separately, so the report cannot drift out of step
with the documentation it summarises -- edit the markdown, re-run this, and the
PDF follows.

Three things need handling that a plain markdown-to-PDF conversion gets wrong:

- **Mermaid.** Both documents embed the compiled LangGraph as a ```mermaid fence,
  which GitHub renders natively but a PDF cannot. Each fence is replaced with
  `graph.png`, produced by `render-graph.py` from the same compiled object.
- **Images.** `architecture.svg` is referenced by relative path from two
  different directories. Every image is inlined as a data URI so the HTML is
  self-contained and Chrome does not have to resolve anything.
- **Cross-document links.** README links to architecture.md, which is now Part 2
  of this same file, so those links are rewritten to name the part instead.

Requires pandoc for the markdown conversion; Chrome does the printing, via
`pdf_renderer.py`.

Run:  .venv/bin/python docs/render-report-pdf.py
"""

import base64
import re
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS))

from pdf_renderer import render_to_pdf  # noqa: E402

README = ROOT / "README.md"
ARCHITECTURE = DOCS / "architecture.md"
GRAPH_PNG = DOCS / "graph.png"

BUILT_HTML = DOCS / "project-report.html"
OUTPUT_PDF = DOCS / "project-report.pdf"


class PandocMissingError(RuntimeError):
    """Raised when pandoc is not on PATH, since the report cannot be built without it."""


def markdown_to_html(path: Path) -> str:
    """Convert one markdown file to an HTML fragment."""
    try:
        result = subprocess.run(
            ["pandoc", str(path), "-f", "gfm", "-t", "html5", "--syntax-highlighting=none"],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError as error:
        raise PandocMissingError(
            "pandoc is required to build the report. Install it with `brew install pandoc`."
        ) from error
    return result.stdout


def data_uri(path: Path) -> str:
    """Inline a file so the built HTML depends on nothing outside itself."""
    media = {".svg": "image/svg+xml", ".png": "image/png"}[path.suffix]
    return f"data:{media};base64,{base64.b64encode(path.read_bytes()).decode()}"


def replace_mermaid_with_image(html: str, image: str) -> str:
    """Swap each ```mermaid fence for the rendered graph.

    The fence is the authoritative artifact on GitHub; in print it is unreadable
    source, so the picture generated from the same compiled graph stands in.
    """
    figure = (
        f'<figure class="graph"><img src="{image}" alt="The compiled LangGraph">'
        "<figcaption>The compiled graph: four nodes, whatever the plan asks for. The "
        "dotted edges out of <code>dispatcher</code> are the conditional ones."
        "</figcaption></figure>"
    )
    return re.sub(
        r'<pre class="mermaid"><code>.*?</code></pre>', figure, html, flags=re.DOTALL
    )


def inline_images(html: str) -> str:
    """Turn every relative <img src> into a data URI."""

    def swap(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith("data:"):
            return match.group(0)
        candidate = (ROOT / src) if (ROOT / src).exists() else (DOCS / Path(src).name)
        if not candidate.exists():
            return match.group(0)
        return match.group(0).replace(src, data_uri(candidate))

    return re.sub(r'<img[^>]*src="([^"]+)"[^>]*>', swap, html)


def rewrite_cross_document_links(html: str) -> str:
    """README's links to architecture.md now point inside this same document.

    `re.DOTALL` matters here: pandoc line-wraps its output, so a link's href and
    text routinely straddle a newline and a non-DOTALL pattern silently skips it.
    """

    def swap(match: re.Match) -> str:
        # A link like "architecture.md §5" becomes "Part 2 §5"; anything else
        # just becomes "Part 2", since that is where the reader should look.
        text = " ".join(match.group(1).split())
        section = re.search(r"§\s*(\d+)", text)
        return f"<b>Part 2 §{section.group(1)}</b>" if section else "<b>Part 2</b>"

    # `<a\s+href` rather than `<a href`: pandoc wraps long lines wherever it
    # likes, including between the tag and its first attribute.
    return re.sub(
        r'<a\s+href="(?:docs/)?architecture\.md[^"]*">(.*?)</a>',
        swap,
        html,
        flags=re.DOTALL,
    )


def prepare(path: Path, graph_image: str) -> str:
    """Convert one source document and apply every print-time fix."""
    html = markdown_to_html(path)
    html = replace_mermaid_with_image(html, graph_image)
    html = inline_images(html)
    html = rewrite_cross_document_links(html)
    # The first <h1> becomes the part title supplied by the template instead.
    return re.sub(r"<h1[^>]*>.*?</h1>", "", html, count=1, flags=re.DOTALL)


def build_html() -> Path:
    """Assemble the two documents into one self-contained HTML file."""
    graph_image = data_uri(GRAPH_PNG)
    readme_html = prepare(README, graph_image)
    architecture_html = prepare(ARCHITECTURE, graph_image)

    BUILT_HTML.write_text(
        TEMPLATE.replace("{{README}}", readme_html).replace(
            "{{ARCHITECTURE}}", architecture_html
        )
    )
    return BUILT_HTML


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dynamic AI Agent Orchestration Platform — Project Report</title>
<style>
  @page { size: A4; }
  :root {
    --ink:#1b2733; --navy:#14395e; --accent:#1f6feb; --muted:#66727f;
    --line:#d8dee6; --wash:#f4f6f9; --code-bg:#f6f8fa;
  }
  * { box-sizing: border-box; }
  body { font-family: Georgia, Charter, "Times New Roman", serif; font-size:10.4pt;
         line-height:1.58; color:var(--ink); margin:0; }
  h1,h2,h3,h4 { font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;
                color:var(--navy); line-height:1.25; page-break-after:avoid; }
  h1 { font-size:20pt; margin:0 0 .5em; }
  h2 { font-size:14.5pt; margin:1.7em 0 .5em; padding-bottom:.3em;
       border-bottom:2px solid var(--navy); }
  h3 { font-size:11.8pt; margin:1.5em 0 .45em; }
  h4 { font-size:10.4pt; margin:1.2em 0 .4em; color:#26445f; }
  p { margin:0 0 .75em; }
  ul,ol { margin:0 0 .85em; padding-left:1.35em; }
  li { margin-bottom:.32em; }
  blockquote { margin:0 0 1em; padding:10px 14px; background:var(--wash);
               border-left:3px solid var(--accent); }
  blockquote p:last-child { margin-bottom:0; }
  code { font-family:Menlo,Consolas,monospace; font-size:8.7pt; background:var(--code-bg);
         border:1px solid #e3e8ee; border-radius:3px; padding:.5px 4px; color:#0f3b6b; }
  pre { font-family:Menlo,Consolas,monospace; font-size:8.2pt; line-height:1.5;
        background:var(--code-bg); border:1px solid #e0e6ed; border-left:3px solid var(--accent);
        border-radius:4px; padding:9px 11px; white-space:pre-wrap; word-break:break-word;
        margin:0 0 .9em; page-break-inside:avoid; }
  pre code { background:none; border:none; padding:0; font-size:inherit; color:inherit; }
  table { border-collapse:collapse; width:100%; font-size:9.2pt; margin:0 0 1em;
          page-break-inside:avoid; }
  th,td { border:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }
  th { background:#eaeff5; font-family:"Helvetica Neue",Arial,sans-serif; font-size:8.8pt;
       color:var(--navy); }
  td code { font-size:8.1pt; }
  img { max-width:100%; height:auto; }
  figure { margin:0 0 1.2em; text-align:center; page-break-inside:avoid; }
  figure.graph img { max-height:250px; }
  figcaption { font-size:8.9pt; color:var(--muted); font-style:italic; margin-top:.5em; }
  hr { border:none; border-top:1px solid var(--line); margin:1.6em 0; }
  .page-break { page-break-before:always; }

  .cover { height:243mm; display:flex; flex-direction:column; justify-content:center; }
  .cover .eyebrow { font-family:"Helvetica Neue",Arial,sans-serif; font-size:9pt;
    letter-spacing:2.4px; text-transform:uppercase; color:var(--accent); margin-bottom:14px; }
  .cover h1 { font-size:30pt; line-height:1.14; }
  .cover .sub { font-size:13pt; color:var(--muted); font-style:italic; margin:6px 0 30px; }
  .cover .rule { height:3px; width:78px; background:var(--accent); margin-bottom:28px; }
  .cover .meta { font-size:9.7pt; color:var(--muted); }
  .cover .meta b { color:var(--ink); }
  .note { background:var(--wash); border-left:3px solid var(--muted); padding:11px 14px;
          font-size:9.8pt; margin-top:30px; }
  .part-title { font-family:"Helvetica Neue",Arial,sans-serif; font-size:9pt;
    letter-spacing:2.4px; text-transform:uppercase; color:var(--accent); margin-bottom:8px; }
  .toc { font-family:"Helvetica Neue",Arial,sans-serif; font-size:9.9pt; }
  .toc .part { margin-top:16px; font-weight:700; color:var(--navy); font-size:10.6pt; }
  .toc .row { padding:2.5px 0 2.5px 17px; border-bottom:1px dotted #dfe5ec; color:#3d4c5c; }
</style>
</head>
<body>

<section class="cover">
  <div class="eyebrow">Project Report</div>
  <h1>Dynamic AI Agent Orchestration Platform</h1>
  <div class="sub">Setup, demonstration, architecture and design rationale</div>
  <div class="rule"></div>
  <div class="meta">
    <p><b>Part 1</b> &nbsp; Project overview &mdash; setup, running it, demonstration
       scenarios, API, assumptions and limitations</p>
    <p><b>Part 2</b> &nbsp; Architecture &mdash; agent design, routing, LangGraph and
       LangChain usage, MCP integration, memory and persistence</p>
  </div>
  <div class="note">
    <p style="margin:0">Part 1 covers what the platform does and how to run it. Part 2
    explains why it is built the way it is, and defends the decisions behind it. A
    companion document, <code>docs/codebase-guide.pdf</code>, walks through the code
    itself for a reader coming to it for the first time.</p>
  </div>
</section>

<section class="page-break">
<h1>Contents</h1>
<div class="toc">
  <div class="part">Part 1 &nbsp;&middot;&nbsp; Project overview</div>
  <div class="row">Quick start &mdash; Docker, local, and running the tests</div>
  <div class="row">Demonstration scenarios</div>
  <div class="row">API</div>
  <div class="row">Adding a capability</div>
  <div class="row">Technology choices</div>
  <div class="row">How it works, briefly &mdash; including the compiled graph</div>
  <div class="row">Assumptions</div>
  <div class="row">Known limitations</div>
  <div class="row">What I would improve for production</div>
  <div class="row">Repository layout</div>

  <div class="part">Part 2 &nbsp;&middot;&nbsp; Architecture and design rationale</div>
  <div class="row">1. The idea the design is organised around</div>
  <div class="row">2. Agent architecture and routing strategy</div>
  <div class="row">3. LangGraph and LangChain &mdash; what each is used for</div>
  <div class="row">4. MCP integration</div>
  <div class="row">5. Memory design and persistence</div>
  <div class="row">6. Cost tracking</div>
  <div class="row">7. The event contract</div>
  <div class="row">8. Notable decisions</div>
  <div class="row">9. What I would change for production</div>
</div>
</section>

<section class="page-break">
<div class="part-title">Part 1</div>
<h1>Project overview</h1>
{{README}}
</section>

<section class="page-break">
<div class="part-title">Part 2</div>
<h1>Architecture and design rationale</h1>
{{ARCHITECTURE}}
</section>

</body>
</html>
"""


if __name__ == "__main__":
    for required in (README, ARCHITECTURE, GRAPH_PNG):
        if not required.exists():
            raise SystemExit(f"Missing {required}. Run docs/render-graph.py first.")

    html = build_html()
    print(f"Built {html.relative_to(ROOT)} ({html.stat().st_size // 1024} KB)")
    render_to_pdf(html, OUTPUT_PDF)
    print(f"Wrote {OUTPUT_PDF.relative_to(ROOT)} ({OUTPUT_PDF.stat().st_size // 1024} KB)")
