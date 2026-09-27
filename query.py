"""
Ask questions against your local knowledge base.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    # or: export OPENAI_API_KEY=sk-...
    uv run query.py "What does the doc say about X?"

Set exactly one of OPENAI_API_KEY or ANTHROPIC_API_KEY.
"""

import sys
import os
import requests
import chromadb
import anthropic
from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
TOP_K = 5
LLM_MODEL = "claude-sonnet-4-6"  # swap for the model you want to use
OPENAI_MODEL = "gpt-5.6-sol"  # swap for the model you want to use


def embed(text: str) -> list[float]:
    r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    q_vector = embed(question)
    results = collection.query(query_embeddings=[q_vector], n_results=k)

    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": doc, "source": meta["source"]})
    return hits


def _key_set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def answer_backend() -> str:
    """Return 'openai' or 'anthropic' from which API key is set."""
    openai_set = _key_set("OPENAI_API_KEY")
    anthropic_set = _key_set("ANTHROPIC_API_KEY")
    if openai_set and anthropic_set:
        sys.exit("Set only one of OPENAI_API_KEY or ANTHROPIC_API_KEY.")
    if openai_set:
        return "openai"
    if anthropic_set:
        return "anthropic"
    sys.exit("Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    return f"""Answer the question using only the context below.
If the context doesn't contain the answer, say so — don't make things up.
Cite which source(s) you used.

Context:
{context}

Question: {question}"""


def ask_anthropic(prompt: str) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def ask_openai(prompt: str) -> str:
    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_completion_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def ask_llm(question: str, chunks: list[dict], backend: str) -> str:
    prompt = build_prompt(question, chunks)
    if backend == "openai":
        return ask_openai(prompt)
    return ask_anthropic(prompt)


def main(question: str):
    backend = answer_backend()
    chunks = retrieve(question)
    if not chunks:
        print("No relevant documents found. Have you run ingest.py yet?")
        return

    answer = ask_llm(question, chunks, backend)
    print("\n--- Answer ---\n")
    print(answer)
    print("\n--- Sources used ---")
    for c in chunks:
        print(f"- {c['source']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
