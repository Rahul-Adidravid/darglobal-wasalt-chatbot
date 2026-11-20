import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.getenv("FASTEMBED_CACHE_PATH", str(ROOT / "models"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Primary free model first; the others are tried automatically if it is rate-limited / down.
MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_MODELS",
        "nvidia/nemotron-3-ultra-550b-a55b:free,nvidia/nemotron-3.5-super-vl:free,google/gemma-4-31b-it:free",
    ).split(",")
    if m.strip()
]
APP_URL = os.getenv("APP_URL", "https://huggingface.co/spaces")
