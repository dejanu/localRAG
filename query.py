"""
Ask questions against your local knowledge base.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run query.py "What does the doc say about X?"
"""

import sys
import os
import requests
import chromadb
import anthropic

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
TOP_K = 5
LLM_MODEL = "claude-sonnet-4-6"  # swap for the model you want to use


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


def ask_llm(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""Answer the question using only the context below.
If the context doesn't contain the answer, say so — don't make things up.
Cite which source(s) you used.

Context:
{context}

Question: {question}"""

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main(question: str):
    chunks = retrieve(question)
    if not chunks:
        print("No relevant documents found. Have you run ingest.py yet?")
        return

    answer = ask_llm(question, chunks)
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