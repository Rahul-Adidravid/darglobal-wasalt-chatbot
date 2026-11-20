---
title: DarGlobal & Wasalt Property Chatbot
emoji: 🏙️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Property Guide: DarGlobal & Wasalt AI chatbot

A retrieval-augmented (RAG) chatbot that answers questions about **DarGlobal** (international luxury real-estate developer) and **Wasalt** (Saudi property portal). It uses data scraped from their public websites and a **free LLM via OpenRouter**. It is containerised with Docker and deployed on Hugging Face Spaces.

## Architecture

```
 scraper/ (run once, offline)          data/raw/*.jsonl              Docker image build
 ┌───────────────────────────┐        ┌────────────────────┐        ┌──────────────────────────┐
 │ darglobal.py (Playwright) │──────▶ │ darglobal_pages    │        │ python -m app.ingest     │
 │ wasalt.py   (httpx)       │──────▶ │ wasalt_listings    │──────▶ │  chunk → embed (ONNX     │
 └───────────────────────────┘        │ wasalt_guides      │        │  bge-small) → index      │
                                      │ wasalt_blog        │        └────────────┬─────────────┘
                                      └────────────────────┘                     ▼
  Browser ──▶ FastAPI /api/chat ──▶ query parser (city / type / rent|sale / beds / price)
                                   ──▶ hybrid retrieval: BM25 + dense vectors, fused with RRF
                                   ──▶ OpenRouter free model (streamed, auto-fallback across models)
                                   ──▶ SSE stream back to the UI, with clickable source citations
```

* **Scraping**
  * **Wasalt:** plain HTTP. City and district category pages embed structured JSON (`__NEXT_DATA__`): 32 listings per page, an editorial guide, FAQs and the total listing count. Blog posts come from the public WordPress REST API.
  * **DarGlobal:** project, partner, insight, press and blog pages, rendered with headless Chromium.
  * **Compliance:** follows robots.txt and https://wasalt.sa/llms.txt. Sequential requests with about 1.5 s delay, no `/search` pages, no phone numbers, WhatsApp links or individual agents' names, and every answer is attributed "Source: Wasalt.sa".
* **Retrieval**
  * Each listing becomes one chunk, and its structured fields are kept so price, bedroom, city and type filters are exact rather than "semantic".
  * Pages and blogs are split into paragraph-aware chunks of about 1,100 characters.
  * Embeddings come from `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX, no PyTorch), so the image is small. Keyword search uses `rank-bm25`.
* **LLM**
  * Any `:free` model on OpenRouter. The default is `nvidia/nemotron-3-ultra-550b-a55b:free` (reasoning disabled for speed), with `nemotron-3.5-super-vl` and `gemma-4-31b-it` as automatic fallbacks when one is rate-limited. You can override this with `OPENROUTER_MODELS`.
  * The prompt forces answers to come only from the retrieved context, with `[n]` citations.

## Run locally

```bash
cp .env.example .env            # add your OPENROUTER_API_KEY
docker compose up --build       # → http://localhost:7860
```

Without Docker:

```bash
pip install -r requirements.txt
python -m app.ingest
OPENROUTER_API_KEY=sk-or-... uvicorn app.main:app --port 7860
```

Re-scrape the data (optional):

```bash
pip install -r requirements-scraper.txt && python -m playwright install chromium
cd scraper && python wasalt.py all && python darglobal.py
```

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Chat UI |
| `POST /api/chat` | `{"messages":[{"role":"user","content":"..."}]}` → server-sent events (`sources`, tokens, `done`) |
| `GET /api/search?q=` | Raw retrieval results (debugging, no LLM call) |
| `GET /health` | Index stats and model configuration |

## Limitations

* The data is a snapshot taken when it was scraped. Listing prices and availability change, so the bot always points users to the live site.
* Wasalt has about 60k listings. We index a representative sample (a few thousand from city and district pages) and use the category totals for "how many" questions.
* OpenRouter free models are rate-limited (roughly 20 requests per minute, and a daily cap that depends on account credits). The app falls back across several free models and rate-limits each IP.
