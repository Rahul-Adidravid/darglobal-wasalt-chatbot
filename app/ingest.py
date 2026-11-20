"""Turn the scraped JSONL files into retrieval chunks + dense embeddings.

Run:  python -m app.ingest
Writes data/index/chunks.json and data/index/embeddings.npy
"""
from __future__ import annotations

import json
import re

import numpy as np

from .config import EMBED_CACHE, EMBED_MODEL, INDEX_DIR, RAW_DIR

CHUNK_CHARS = 1100
OVERLAP = 150


def read_jsonl(name: str) -> list[dict]:
    p = RAW_DIR / name
    if not p.exists():
        print(f"  (missing {name}, skipped)")
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def split(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Paragraph-aware splitter."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n" + p if overlap else p
        else:
            cur = p
        while len(cur) > size:  # very long paragraph
            chunks.append(cur[:size])
            cur = cur[size - overlap:]
    if cur.strip():
        chunks.append(cur)
    return chunks


def normalize_listing(r: dict) -> dict:
    """Rent listings carry their price in rent_frequency: {"yearly": {"amount": .., "freq": ..}, ...}."""
    rf = r.get("rent_frequency")
    if isinstance(rf, dict) and rf:
        opts = [v for v in rf.values() if isinstance(v, dict)]
        best = next((v for v in opts if v.get("default_freq")), opts[0] if opts else {})
        if best.get("amount") is not None and r.get("price") in (None, 0):
            r["price"] = float(best["amount"])
        r["rent_frequency"] = best.get("freq", "")
    elif not isinstance(rf, str):
        r["rent_frequency"] = ""
    return r


def fmt_price(v, purpose, freq, cur):
    if v is None:
        return "Price on request"
    s = f"{v:,.0f} {cur}"
    if purpose == "rent":
        per = {"yearly": "year", "monthly": "month", "weekly": "week", "daily": "day"}.get(str(freq).lower(), str(freq).lower())
        s += f" per {per}" if per else " (rent)"
    return s


def listing_text(r: dict) -> str:
    attrs = "; ".join(f"{k}: {v}" for k, v in r.get("attributes", {}).items())
    parts = [
        f"{r['title']} — {r['sub_type'] or r['main_type']} for {r['purpose']} in "
        f"{r['district']}, {r['region']}, {r['city']}.",
        f"Price: {fmt_price(r.get('price'), r['purpose'], r.get('rent_frequency'), r.get('currency', 'SAR'))}.",
    ]
    if attrs:
        parts.append(f"Details: {attrs}.")
    if r.get("furnishing"):
        parts.append(f"Furnishing: {r['furnishing']}.")
    if r.get("amenities"):
        parts.append("Amenities: " + ", ".join(r["amenities"][:15]) + ".")
    if r.get("agency"):
        parts.append(f"Listed by: {r['agency']}.")
    if r.get("published"):
        parts.append(f"Published: {r['published']}.")
    return " ".join(parts)


def build_chunks() -> list[dict]:
    chunks: list[dict] = []

    def add(source, kind, title, url, text, meta=None):
        chunks.append({"id": len(chunks), "source": source, "kind": kind, "title": title,
                       "url": url, "text": text, "meta": meta or {}})

    # DarGlobal pages
    seen_pages = set()
    for r in read_jsonl("darglobal_pages.jsonl"):
        sig = r["text"][:500]
        if sig in seen_pages:  # some URLs redirect to the same page
            continue
        seen_pages.add(sig)
        head = r["title"] + (f"\n{r['description']}" if r.get("description") else "")
        for i, c in enumerate(split(r["text"])):
            add("DarGlobal", r.get("section", "page"), r["title"], r["url"],
                (head + "\n" + c) if i == 0 else f"{r['title']}\n{c}")

    # Wasalt category guides + FAQs
    for r in read_jsonl("wasalt_guides.jsonl"):
        summary = r["title"]
        if r.get("total_listings"):
            summary += f" — Wasalt currently shows {r['total_listings']:,} listings in this category."
        if r.get("description"):
            summary += f"\n{r['description']}"
        if r.get("faq"):
            for c in split(r["faq"]):
                add("Wasalt", "faq", r["title"] + " — FAQ", r["url"], f"{summary}\n{c}")
        for c in split(r.get("text", "")):
            add("Wasalt", "guide", r["title"], r["url"], f"{r['title']}\n{c}")
        if not r.get("faq") and not r.get("text"):
            add("Wasalt", "guide", r["title"], r["url"], summary)

    # Wasalt blog
    for r in read_jsonl("wasalt_blog.jsonl"):
        for c in split(r["text"]):
            add("Wasalt", "blog", r["title"], r["url"], f"{r['title']} ({r.get('date','')})\n{c}")

    # Wasalt listings (one chunk per listing, structured fields kept for filtering)
    for r in read_jsonl("wasalt_listings.jsonl"):
        r = normalize_listing(r)
        meta = {k: r.get(k) for k in ("purpose", "sub_type", "main_type", "city", "region", "district", "price")}
        beds = r.get("attributes", {}).get("Bedrooms")
        meta["bedrooms"] = int(re.sub(r"\D", "", beds) or 0) if beds else None
        add("Wasalt", "listing", r["title"], r["url"], listing_text(r), meta)

    return chunks


def main():
    from fastembed import TextEmbedding

    chunks = build_chunks()
    by = {}
    for c in chunks:
        by[(c["source"], c["kind"])] = by.get((c["source"], c["kind"]), 0) + 1
    print(f"{len(chunks)} chunks:", by)
    # Reuse vectors from a previous build for unchanged texts (embedding is the slow part).
    cache: dict[str, np.ndarray] = {}
    old_c, old_v = INDEX_DIR / "chunks.json", INDEX_DIR / "embeddings.npy"
    if old_c.exists() and old_v.exists():
        prev = json.loads(old_c.read_text(encoding="utf-8"))
        pv = np.load(old_v)
        for c, v in zip(prev, pv):
            t = re.sub(r"per (year|month|week|day)ly\b", r"per \1", c["text"]).replace("per daily", "per day")
            cache[t] = v
    todo = [c["text"] for c in chunks if c["text"] not in cache]
    print(f"reusing {len(chunks) - len(todo)} cached vectors, embedding {len(todo)} new chunks")
    if todo:
        model = TextEmbedding(EMBED_MODEL, cache_dir=EMBED_CACHE)
        new = np.array(list(model.embed(todo, batch_size=64)), dtype=np.float32)
        new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-9
        cache.update(zip(todo, new))
    vecs = np.stack([cache[c["text"]] for c in chunks]).astype(np.float32)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", vecs)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print("index written:", vecs.shape)


if __name__ == "__main__":
    main()
