# check the vector db for data

import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection("knowledge_base")
except chromadb.errors.NotFoundError:
    print("Collection knowledge_base does not exist. Run ingest.py first.")
    raise SystemExit(1)

print(collection.count())
