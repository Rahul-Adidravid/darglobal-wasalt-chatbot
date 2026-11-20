"""Scrape public pages of DarGlobal (darglobal.co.uk).

The site sits behind Imperva/Incapsula, which rejects plain HTTP clients, so a real
headless Chromium (Playwright) renders each page. Images/fonts/media are blocked to keep
load on the site low, and concurrency is limited.

Output: data/raw/darglobal_pages.jsonl  ({url, title, description, section, text})
"""
from __future__ import annotations

import asyncio
import random
import re
import sys
from collections import Counter

import httpx
from playwright.async_api import async_playwright

from common import RAW_DIR, UA, clean_text, html_to_text, strip_phones, write_jsonl

BASE = "https://darglobal.co.uk"
CONCURRENCY = 3
MAX_BLOG = 120

SKIP = re.compile(
    r"/(thank-you|success|pay-online|bookYourUnit|internship-application|get-in-touch|"
    r"landingpage|landing-page|new-home-page|win-a-trip|privacy-policy|terms-|broker-contest)",
    re.I,
)


def sitemap_urls() -> list[str]:
    xml = httpx.get(f"{BASE}/sitemap.xml", headers={"User-Agent": UA}, timeout=30).text
    urls = list(dict.fromkeys(re.findall(r"<loc>([^<]+)</loc>", xml)))
    urls = [u for u in urls if not SKIP.search(u)]
    blog = [u for u in urls if "/blog/" in u][:MAX_BLOG]
    rest = [u for u in urls if "/blog/" not in u]
    return rest + blog


def section_of(url: str) -> str:
    path = url.replace(BASE, "").strip("/")
    first = path.split("/")[0] if path else "home"
    if first in {"blog", "press", "insights", "partners", "projects", "one-of-one"}:
        return first
    return "project-or-page"


JS_EXTRACT = """
() => {
  for (const sel of ['header','footer','nav','form','[class*="cookie" i]','[id*="cookie" i]',
                     '[class*="Footer"]','[class*="Header"]','[class*="modal" i]']) {
    document.querySelectorAll(sel).forEach(e => e.remove());
  }
  const main = document.querySelector('main') || document.body;
  const d = document.querySelector('meta[name="description"]');
  return {html: main.innerHTML, title: document.title, description: d ? d.content : ''};
}
"""


async def fetch(ctx, url: str, sem: asyncio.Semaphore, out: list, fails: list):
    async with sem:
        page = await ctx.new_page()
        try:
            for attempt in range(3):
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(4500)
                html = await page.content()
                if "Incapsula" in html[:3000] and len(html) < 5000:
                    await page.wait_for_timeout(8000 * (attempt + 1))
                    continue
                # scroll once to trigger lazy sections
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1500)
                data = await page.evaluate(JS_EXTRACT)
                text = strip_phones(html_to_text(data["html"]))
                out.append(
                    {
                        "source": "DarGlobal",
                        "url": url,
                        "section": section_of(url),
                        "title": data["title"].strip(),
                        "description": data["description"].strip(),
                        "text": text,
                    }
                )
                print(f"[{len(out)}] ok {url} ({len(text)} chars)", flush=True)
                break
            else:
                fails.append(url)
                print(f"BLOCKED {url}", flush=True)
        except Exception as e:  # noqa: BLE001
            fails.append(url)
            print(f"FAIL {url}: {e}", flush=True)
        finally:
            await page.close()
            await asyncio.sleep(random.uniform(1.0, 2.5))


def drop_boilerplate(rows: list[dict]) -> None:
    """Remove lines that appear on a large share of pages (menus, CTAs, footers)."""
    counts = Counter()
    for r in rows:
        counts.update(set(r["text"].splitlines()))
    limit = max(5, int(len(rows) * 0.25))
    common = {line for line, c in counts.items() if c >= limit}
    for r in rows:
        r["text"] = clean_text("\n".join(l for l in r["text"].splitlines() if l not in common))


async def main(limit: int | None = None):
    urls = sitemap_urls()
    if limit:
        urls = urls[:limit]
    print(f"{len(urls)} DarGlobal URLs to fetch")
    out, fails = [], []
    sem = asyncio.Semaphore(CONCURRENCY)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(user_agent=UA, locale="en-GB")

        async def block(route):
            if route.request.resource_type in {"image", "media", "font"}:
                await route.abort()
            else:
                await route.continue_()

        await ctx.route("**/*", block)
        # warm-up so the protection cookie is set before parallel requests
        await fetch(ctx, urls[0], asyncio.Semaphore(1), out, fails)
        await asyncio.gather(*(fetch(ctx, u, sem, out, fails) for u in urls[1:]))
        await browser.close()

    drop_boilerplate(out)
    out = [r for r in out if len(r["text"]) > 150]
    write_jsonl(RAW_DIR / "darglobal_pages.jsonl", out)
    print(f"failed: {len(fails)}", fails[:20])


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
