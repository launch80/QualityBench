"""The self-contained HTML report: one file, charts drawn by embedded JS, nothing fetched at view time."""
from __future__ import annotations

import html
import json
from pathlib import Path

from ..context import sidecar_for
from ..suite import SUITE

_DIR = Path(__file__).parent


def render(doc: dict, tokens: dict | None = None) -> str:
    """The page for a results document; `tokens` (results.report_tokens) is local data the document has no room for."""
    tpl = (_DIR / "template.html").read_text()
    # </script> inside a string must not end the script block
    data = lambda v: json.dumps(v, separators=(",", ":")).replace("</", "<\\/")
    about = json.dumps({t.id: {"label": t.label, "about": t.about} for t in SUITE}, separators=(",", ":")).replace("</", "<\\/")
    script = (_DIR / "report.js").read_text().replace("/*__TOKENS__*/null", data(tokens)).replace("/*__ABOUT__*/{}", about).replace("/*__DATA__*/null", data(doc))
    return tpl.replace("__TITLE__", html.escape(doc.get("title", "QualityBench"))).replace("/*__SCRIPT__*/", script)


def write(doc: dict, results: Path, out: Path | None = None, tokens: dict | None = None) -> Path:
    """Write the page; `tokens` also goes to the results file's sidecar (tokens.json), for `qualitybench report`."""
    out = out or (results.with_suffix(".html") if results.suffix == ".json" else results.with_name(results.name + ".html"))
    if tokens is not None:
        p = sidecar_for(results) / "tokens.json"
        if any([tokens.get("ref"), tokens.get("on"), *(tokens.get("arms") or {}).values()]):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(tokens))
        else:  # no task counted: no table, and none left from an earlier write
            p.unlink(missing_ok=True)
    out.write_text(render(doc, tokens))
    return out
