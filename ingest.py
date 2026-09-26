"""
Ingest documents (PDF, Markdown) into a local Chroma vector store,
using Ollama for embeddings.

Usage:
    uv ingest.py /path/to/docs

Re-run any time you add new documents — already-ingested files
(tracked by path + modification time) are skipped automatically.
"""

import sys
import os
import json
import hashlib
import requests
import chromadb
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # overlap between chunks


def embed(text: str) -> list[float]:
    r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def json_to_text(data, prefix: str = "") -> str:
    """Flatten arbitrary JSON into readable 'key: value' lines."""
    lines = []
    if isinstance(data, dict):
        for key, value in data.items():
            path_key = f"{prefix}.{key}" if prefix else str(key)
            lines.append(json_to_text(value, path_key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            lines.append(json_to_text(item, f"{prefix}[{i}]"))
    else:
        lines.append(f"{prefix}: {data}")
    return "\n".join(lines)


def read_json(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json_to_text(data)


def load_text(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in (".md", ".markdown", ".txt"):
        return read_markdown(path)
    if ext == ".json":
        return read_json(path)
    return None  # unsupported type, skip


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def file_id(path: str) -> str:
    """Stable id based on path + mtime, so edited files get re-ingested."""
    mtime = os.path.getmtime(path)
    return hashlib.md5(f"{path}:{mtime}".encode()).hexdigest()


def main(folder: str):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    existing_ids = set(collection.get()["ids"])

    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            text = load_text(path)
            if text is None:
                continue

            base_id = file_id(path)
            # skip if this exact version of the file was already ingested
            if any(cid.startswith(base_id) for cid in existing_ids):
                print(f"skip (unchanged): {path}")
                continue

            chunks = chunk_text(text)
            print(f"ingesting: {path} ({len(chunks)} chunks)")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{base_id}_{i}"
                vector = embed(chunk)
                collection.add(
                    ids=[chunk_id],
                    embeddings=[vector],
                    documents=[chunk],
                    metadatas=[{"source": path, "chunk": i}],
                )

    print("done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py /path/to/docs_folder")
        sys.exit(1)
    main(sys.argv[1])