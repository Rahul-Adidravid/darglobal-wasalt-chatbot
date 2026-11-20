"""Deploy to a Hugging Face Docker Space.  Usage:  python deploy_hf.py   (reads HF_TOKEN + OPENROUTER_API_KEY from .env)"""
import os
from pathlib import Path

from huggingface_hub import HfApi

for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

SPACE = os.getenv("HF_SPACE", "darglobal-wasalt-chatbot")
api = HfApi(token=os.environ["HF_TOKEN"])
user = api.whoami()["name"]
repo = f"{user}/{SPACE}"
api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
api.add_space_secret(repo, "OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"])
api.add_space_variable(repo, "APP_URL", f"https://huggingface.co/spaces/{repo}")
api.upload_folder(
    repo_id=repo, repo_type="space", folder_path=".",
    allow_patterns=["app/**", "data/index/**", "scraper/*.py", "Dockerfile", "requirements*.txt",
                    "README.md", "docker-compose.yml", ".env.example", ".dockerignore", ".gitignore"],
    ignore_patterns=["**/__pycache__/**", "*.pyc", ".env", "data/raw/**", "*.log"],
    commit_message="Deploy DarGlobal & Wasalt RAG chatbot",
)
print("Space:", f"https://huggingface.co/spaces/{repo}")
print("App  :", f"https://{repo.replace('/', '-').replace('_', '-').lower()}.hf.space")
