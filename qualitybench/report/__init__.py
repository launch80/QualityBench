"""The self-contained HTML report: one file, charts drawn by embedded JS, nothing fetched at view time."""
from __future__ import annotations

import html
import json
from pathlib import Path

_DIR = Path(__file__).parent


def render(doc: dict) -> str:
    tpl = (_DIR / "template.html").read_text()
    # </script> inside a string must not end the script block
    data = json.dumps(doc, separators=(",", ":")).replace("</", "<\\/")
    script = (_DIR / "report.js").read_text().replace("/*__DATA__*/null", data)
    return tpl.replace("__TITLE__", html.escape(doc.get("title", "QualityBench"))).replace("/*__SCRIPT__*/", script)


def write(doc: dict, results: Path, out: Path | None = None) -> Path:
    out = out or (results.with_suffix(".html") if results.suffix == ".json" else results.with_name(results.name + ".html"))
    out.write_text(render(doc))
    return out
