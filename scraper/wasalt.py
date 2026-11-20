"""Scrape public content from Wasalt (wasalt.sa), following https://wasalt.sa/llms.txt:

* only public listing / blog content (no /search, no user data),
* moderate crawl rate (one request every ~1.5 s, sequential),
* no phone numbers, WhatsApp links or personal agent names are stored,
* everything is attributed "Source: Wasalt.sa".

Sources
1. City / district category pages (e.g. /en/apartments-for-sale-in-riyadh). Each page embeds
   (in __NEXT_DATA__) 32 listings, an editorial guide, an FAQ and the total listing count.
2. English blog posts via the public WordPress REST API of blog.wasalt.sa.

Output: data/raw/wasalt_listings.jsonl, data/raw/wasalt_guides.jsonl, data/raw/wasalt_blog.jsonl
"""
from __future__ import annotations

import gzip
import random
import re
import sys
import time

import httpx

from common import RAW_DIR, UA, html_to_text, next_data, strip_phones, write_jsonl

DELAY = 1.5
CATEGORY_SITEMAP = "https://cdn.wasalt.sa/sitemap/category_sitemap_en_sa.xml.gz"
BLOG_API = "https://blog.wasalt.sa/en/wp-json/wp/v2/posts"

CITIES = [
    "saudi-arabia", "riyadh", "jeddah", "makkah", "madinah", "dammam", "al-khobar", "taif",
    "tabuk", "abha", "hail", "buraidah", "khamis-mushait", "jazan", "najran", "al-ahsa",
    "north-riyadh", "south-riyadh", "east-riyadh", "west-riyadh", "central-riyadh",
]
TYPES = r"(properties|apartments|villas|lands|floors|buildings|commercial-properties|offices|shops|showrooms|3-bedroom-apartments|2-bedroom-apartments)"
DISTRICT_TYPES = r"(properties|apartments|villas)"
MAX_DISTRICT_PAGES = 120

client = httpx.Client(headers={"User-Agent": UA, "Accept-Language": "en", "Accept-Encoding": "gzip"}, timeout=40, follow_redirects=True)


def get(url: str, **kw) -> httpx.Response | None:
    for attempt in range(3):
        try:
            r = client.get(url, **kw)
            time.sleep(DELAY + random.uniform(0, 0.5))
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            return None
        except httpx.HTTPError as e:
            print("  http error", e)
            time.sleep(5)
    return None


def category_urls() -> list[str]:
    raw = client.get(CATEGORY_SITEMAP).content
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    urls = re.findall(r"<loc>([^<]+)</loc>", raw.decode("utf-8"))
    city_re = re.compile(rf"/en/{TYPES}-for-(sale|rent)-in-({'|'.join(CITIES)})$")
    city_pages = [u for u in urls if city_re.search(u)]
    # popular districts inside the big cities
    dist_re = re.compile(rf"/en/{DISTRICT_TYPES}-for-(sale|rent)-in-((north|south|east|west|central)-riyadh|jeddah|dammam)-[a-z-]+$")
    district_pages = [u for u in urls if dist_re.search(u)]
    random.Random(7).shuffle(district_pages)
    return list(dict.fromkeys(city_pages + district_pages[:MAX_DISTRICT_PAGES]))


def attr_map(attrs: list | None) -> dict:
    out = {}
    for a in attrs or []:
        name = a.get("name") or a.get("key")
        val = a.get("value")
        if name and val not in (None, "", "-"):
            unit = a.get("unit") or ""
            out[name] = f"{val} {unit}".strip()
    return out


def listing_record(p: dict) -> dict | None:
    pi = p.get("propertyInfo") or {}
    slug = pi.get("slug")
    if not slug:
        return None
    purpose = pi.get("propertyFor") or ""
    price = pi.get("salePrice") if purpose == "sale" else pi.get("expectedRent")
    owner = p.get("propertyOwner") or {}
    agency = owner.get("enCmpName") or owner.get("companyName") or owner.get("brand") or ""
    if owner.get("userCapacity") != "company":
        agency = ""  # never store private individuals' names
    rec = {
        "source": "Wasalt",
        "id": p.get("id"),
        "url": f"https://wasalt.sa/en/property/{purpose}/{slug}",
        "title": pi.get("title") or pi.get("propertyName") or "",
        "purpose": purpose,  # sale | rent
        "main_type": pi.get("propertyMainType") or "",
        "sub_type": pi.get("propertySubType") or "",
        "city": pi.get("city") or "",
        "region": pi.get("territory") or "",
        "district": pi.get("district") or pi.get("zone") or "",
        "address": pi.get("address") or "",
        "price": float(price) if price not in (None, "") else None,
        "currency": pi.get("currencyType") or "SAR",
        "rent_frequency": pi.get("rentFreq") or "",
        "furnishing": pi.get("furnishingType") or "",
        "attributes": attr_map(p.get("attributes")),
        "amenities": [a.get("name") for a in (p.get("featureAmenities") or []) if isinstance(a, dict) and a.get("name")],
        "agency": agency,
        "published": (p.get("publishedAt") or "")[:10],
    }
    return rec


def scrape_categories(limit: int | None = None):
    urls = category_urls()
    if limit:
        urls = urls[:limit]
    print(f"{len(urls)} category pages")
    listings: dict[int, dict] = {}
    guides = []
    for i, url in enumerate(urls, 1):
        r = get(url)
        if not r:
            print(f"[{i}] skip {url}")
            continue
        data = next_data(r.text)
        if not data:
            continue
        pp = data.get("props", {}).get("pageProps", {})
        sr = pp.get("searchResult") or {}
        n_new = 0
        items = []
        for p in sr.get("properties") or []:
            items.extend(p if isinstance(p, list) else [p])  # some slots are grouped (projects/units)
        for p in items:
            if not isinstance(p, dict):
                continue
            rec = listing_record(p)
            if rec and rec["id"] not in listings:
                listings[rec["id"]] = rec
                n_new += 1
        meta = pp.get("metaData") or {}
        title = meta.get("title_tag") or meta.get("h1tag") or url.rsplit("/", 1)[-1]
        faq_lines = []
        for q in (pp.get("FAQSchema") or {}).get("mainEntity", []) or []:
            a = (q.get("acceptedAnswer") or {}).get("text", "")
            faq_lines.append(f"Q: {q.get('name','')}\nA: {html_to_text(a)}")
        guide_text = html_to_text(pp.get("para") if isinstance(pp.get("para"), str) else "")
        guides.append(
            {
                "source": "Wasalt",
                "url": url,
                "title": title,
                "total_listings": sr.get("count"),
                "description": meta.get("meta_description") or "",
                "text": strip_phones(guide_text),
                "faq": strip_phones("\n\n".join(faq_lines)),
            }
        )
        print(f"[{i}/{len(urls)}] {title}: +{n_new} listings (total {len(listings)}), count={sr.get('count')}", flush=True)
    write_jsonl(RAW_DIR / "wasalt_listings.jsonl", list(listings.values()))
    write_jsonl(RAW_DIR / "wasalt_guides.jsonl", guides)


def scrape_blog(max_posts: int = 200):
    posts = []
    page = 1
    while len(posts) < max_posts:
        r = get(BLOG_API, params={"per_page": 50, "page": page, "_fields": "link,title,date,content,excerpt"})
        if not r:
            break
        batch = r.json()
        if not batch:
            break
        for p in batch:
            if "/en/" not in p.get("link", ""):
                continue
            text = strip_phones(html_to_text(p["content"]["rendered"]))
            if len(text) < 300:
                continue
            posts.append(
                {
                    "source": "Wasalt",
                    "url": p["link"],
                    "title": html_to_text(p["title"]["rendered"]),
                    "date": p.get("date", "")[:10],
                    "text": text,
                }
            )
        print(f"blog page {page}: {len(posts)} posts", flush=True)
        page += 1
    write_jsonl(RAW_DIR / "wasalt_blog.jsonl", posts[:max_posts])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if what in ("all", "categories"):
        scrape_categories(lim)
    if what in ("all", "blog"):
        scrape_blog()
