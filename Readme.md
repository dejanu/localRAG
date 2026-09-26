## Local RAG pipeline

For Dynamic and Time-Sensitive Information: Daily financial reports, stock market trends, or finance industry news updates 

The info gathering flow is currently a manual batch loading proccess by adding data in `docs` folder

⚠️ A RAG system makes the most sense for information that is highly specific, constantly changing, proprietary, or requires strict verification. 

### Components

* Embeddings model: [nomic-embed-text embedding model from Ollama](https://ollama.com/library/nomic-embed-text-v2-moe)
* Vector DB: [ChromaDB](https://docs.trychroma.com/docs/overview/introduction)
* LLM (remote hosted Anthropic via API calls)


### Project setup for Python deps

```bash
# init project and add dependencies
uv init --name localRAG --python 3.14
uv add chromadb requests anthropic pypdf

# use project as is: install dependencies and update pyproject.toml
uv sync

# cleanup uv project
rm -f pyproject.toml .python-version main.py .gitignore
rm -rf .venv
```

### Steps

* Start the embedding model:

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull nomic-embed-text
```

* Chunk and embed data from docs to vector DB:

    - Supports `json, pdf, md, markdown, txt`
    - re-run as new files are added to docs
    - vector data persists on disk in `./chroma_db`

```bash
# check chromadb
uv run check_db.py

# ingest data
uv run ingest.py docs/
```

* Start quering the knowledge base:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run query.py "What does the knowledge base say about X?"
```

### System diagram

```mermaid
flowchart TB
  docs["docs/ — json, pdf, md, markdown, txt"] --> ingest["ingest.py"]
  ingest --> chunk["Chunk text: 1000 characters, 150 overlap\nSkip files already stored for the same path and mtime"]
  chunk --> embedIn["Ollama nomic-embed-text\nlocalhost:11434"]
  embedIn --> chroma[("ChromaDB ./chroma_db\ncollection knowledge_base")]

  question["query.py question"] --> embedQ["Embed the question\nwith the same Ollama model"]
  embedQ --> search["Similarity search, top 5 chunks"]
  chroma --> search
  search --> prompt["Prompt: question plus retrieved chunks and source paths"]
  question --> prompt
  prompt --> llm["Anthropic claude-sonnet-4-6"]
  llm --> answer["Printed answer and sources"]
```

