"""Find a catalogue model on disk.

The organizer's store is laid out as  <APCTT_MODEL_STORE>/<org>/<name>/<revision>/  (see
`python tools/get_models.py --snippet <id>`), NOT <store>/<catalog_id>.
"""

from __future__ import annotations

from pathlib import Path

REPOS = {                                           # catalogue id -> Hugging Face repo (contracts/model-catalog.json)
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "deberta-v3-large-zeroshot-v2.0-c": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c",
    "qwen3-4b-instruct-2507-nf4": "Qwen/Qwen3-4B-Instruct-2507",
    "ms-marco-minilm-l6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",
}


def resolve(store: str | None, catalog_id: str) -> str:
    if not store:
        raise FileNotFoundError("APCTT_MODEL_STORE is not set")
    root = Path(store)
    candidates: list[Path] = []
    repo = REPOS.get(catalog_id)
    if repo and (root / repo).is_dir():
        base = root / repo
        candidates += sorted(p for p in base.iterdir() if p.is_dir())      # <repo>/<revision>/
        candidates.append(base)
    candidates.append(root / catalog_id)                                   # fallback: <store>/<catalog_id>
    for path in candidates:
        if (path / "config.json").exists():
            return str(path)
    raise FileNotFoundError(f"model {catalog_id!r} not found under {store}")
