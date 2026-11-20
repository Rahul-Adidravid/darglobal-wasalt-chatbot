"""Shared helpers for the scrapers."""
from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

try:
    import lxml  # noqa: F401
    PARSER = "lxml"
except ImportError:  # fall back to Python's built-in parser
    PARSER = "html.parser"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)

NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)

# Phone numbers / emails are never stored (Wasalt llms.txt asks not to extract contact details).
PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def next_data(html: str) -> dict | None:
    m = NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, PARSER)
    for t in soup(["script", "style", "noscript", "svg", "iframe"]):
        t.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for t in soup.find_all(BLOCK_TAGS):
        t.insert_before("\n")
        t.insert_after("\n")
    text = soup.get_text("")
    return clean_text(text)


BLOCK_TAGS = ["p", "div", "section", "article", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3",
              "h4", "h5", "h6", "blockquote", "header", "footer", "main", "figure", "dt", "dd"]


def clean_text(text: str) -> str:
    text = EMAIL_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    # collapse consecutive duplicate lines
    out = []
    for line in lines:
        if not out or out[-1] != line:
            out.append(line)
    return "\n".join(out)


def strip_phones(text: str) -> str:
    return PHONE_RE.sub("[contact via website]", text)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")
