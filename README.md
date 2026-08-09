# vectordb

A small, local, embedded vector database. No server, no external service —
just a Python package backed by [hnswlib](https://github.com/nmslib/hnswlib)
(HNSW approximate nearest-neighbor search) and SQLite (IDs + metadata).

Good fit for: RAG/semantic search, recommendations, and dedup, at the
100K–10M vector scale, on a single machine.

## Install

```bash
pip install -r requirements.txt
```

Then drop the `vectordb/` folder into your project (or `pip install -e .`
if you turn it into a proper package later).

## Quickstart

```python
from vectordb import VectorDB
import numpy as np

# Create (or reopen) a DB. `dim` must match your embedding model's output size.
db = VectorDB("./mydb", dim=384, space="cosine")

# Add vectors (upsert by id)
db.add("doc1", embedding_vector, metadata={"title": "Intro to HNSW", "category": "ml"})

# Batch add
db.add(
    ids=["doc2", "doc3"],
    vectors=np.stack([vec2, vec3]),
    metadatas=[{"category": "ml"}, {"category": "finance"}],
)

# Search
results = db.search(query_vector, k=5)
# -> [{"id": "doc1", "score": 0.12, "metadata": {...}}, ...]

# Search with metadata filter (dict = exact-match AND, or pass a callable)
results = db.search(query_vector, k=5, filter={"category": "ml"})
results = db.search(query_vector, k=5, filter=lambda m: m.get("year", 0) > 2020)

# Persist to disk (also called implicitly on write, but call explicitly
# after a batch of writes to flush the HNSW index file)
db.save()

# Delete / update
db.delete("doc2")
db.update_metadata("doc1", {"category": "ml", "reviewed": True})

# Reclaim space after many deletes
db.compact()
```

Reopening later infers `dim`/`space` automatically:

```python
db = VectorDB("./mydb")  # no need to pass dim again
```

## Text embeddings (skip the vector math)

If you'd rather pass raw text than pre-computed vectors, use `TextEmbedder`
(backed by [fastembed](https://github.com/qdrant/fastembed), which runs on
ONNX Runtime — no PyTorch required, fast on CPU):

```python
from vectordb import VectorDB
from vectordb.embeddings import TextEmbedder

embedder = TextEmbedder()  # defaults to BAAI/bge-small-en-v1.5 (384-dim)
db = VectorDB("./mydb", embedder=embedder)  # dim inferred from the embedder

db.add_text("doc1", "The cat sat on the mat", metadata={"category": "animals"})
db.add_text(["doc2", "doc3"], ["Dogs are loyal", "Stocks fell today"])

results = db.search_text("pets and animals", k=3)
```

Note: the first time you construct `TextEmbedder(...)`, it downloads model
weights from the internet (cached afterward in `~/.cache/fastembed`).

## Running as an HTTP service

Useful when multiple processes/services need to share one DB instance
without each opening the SQLite/HNSW files directly.

```bash
# vector-only mode
DB_PATH=./mydb DIM=384 uvicorn vectordb.api:app --host 0.0.0.0 --port 8000

# with text embedding support (adds /add_text and /search_text)
DB_PATH=./mydb EMBED_MODEL=BAAI/bge-small-en-v1.5 uvicorn vectordb.api:app --port 8000
```

Endpoints: `POST /add`, `POST /search`, `POST /add_text`, `POST /search_text`,
`GET /items/{id}`, `PATCH /items/{id}/metadata`, `POST /delete`,
`POST /compact`, `GET /count`, `GET /health`.

```bash
curl -X POST localhost:8000/add -H 'Content-Type: application/json' -d '{
  "items": [{"id": "doc1", "vector": [0.1, 0.2, ...], "metadata": {"category": "ml"}}]
}'

curl -X POST localhost:8000/search -H 'Content-Type: application/json' -d '{
  "vector": [0.1, 0.2, ...], "k": 5, "filter": {"category": "ml"}
}'
```

This server design is **single-writer-process**: one `VectorDB` instance
with an in-process lock handles concurrent requests safely within itself,
but don't point multiple server *processes* at the same DB directory for
concurrent writes — route all writes through one process, or you risk
corrupting the index file.

## ⚠️ Persistence gotcha

Writes (`add`, `delete`, `update_metadata`) update the in-memory HNSW index
immediately, but the index file on disk is only updated when you call
`db.save()`. Metadata (SQLite) is committed on every write, but the vector
index is not. If you write, exit the process, and reopen the DB *without*
having called `save()`, you'll get an empty/stale index with metadata that
references vectors it doesn't have — searches will look empty or throw
errors like `Label not found` on delete.

Call `db.save()` after a batch of writes, or after every write if losing
recent writes on a crash is unacceptable. The `VectorDB.api` HTTP server
already does this for you on every write endpoint.

## Use cases

**Semantic search / RAG** — embed your documents/chunks, `add()` them with
source metadata (title, url, chunk index), and `search()` with the query
embedding. Use the `filter` argument to scope search to a subset (e.g. one
tenant's documents).

**Recommendations** — treat an item's own embedding as the query
(`db.search(item_vector, k=10)`) to get "more like this." Exclude the item
itself in the caller by filtering on `id != item_id`, or just drop the first
result (it will be the item itself with score 0).

**Dedup / near-duplicate detection** — before inserting a new vector, run
`db.search(new_vector, k=1)`; if the returned score is below a threshold
(e.g. cosine distance < 0.02), treat it as a duplicate.

## Design notes

- **Index**: HNSW via hnswlib — logarithmic-ish query time, good recall/speed
  tradeoff, handles 100K–10M+ vectors on one machine. Auto-grows capacity.
- **Metadata & IDs**: SQLite. hnswlib only knows integer labels, so this
  package maintains the string-id ↔ integer-label mapping and JSON metadata.
- **Filtering**: applied post-search (over-fetch then filter), since hnswlib
  has no native filtering. Very selective filters may need a larger `k` or
  the `ef` parameter to get enough candidates — see `search(..., ef=...)`.
- **Deletes**: soft-deleted (hidden from search immediately), space is
  reclaimed by calling `compact()`, which rebuilds the index.
- **Concurrency**: a single `RLock` guards writes/reads on one `VectorDB`
  instance. Fine for a single process; for multi-process access, put it
  behind a small API server (a `search`/`add` HTTP wrapper is ~30 lines with
  FastAPI) or add file locking.
- **Distance metric**: `"cosine"` (default), `"l2"`, or `"ip"` (inner
  product) — pick based on how your embedding model was trained/normalized.

## Tuning

- `ef_construction` / `M` (build time): higher = better recall, slower
  builds, more memory. Defaults (`ef_construction=200`, `M=16`) are solid
  up to a few million vectors.
- `ef` (query time, via `index.set_ef()` or `search(..., ef=...)`): higher =
  better recall, slower queries. Defaults to `max(50, ef_construction)`.

## Limitations to know about

- Single-machine, single-process by default (no built-in replication/sharding).
- Filtering is post-hoc, not index-native — extremely selective filters on a
  huge DB can require large over-fetch. For heavy filtered search at scale,
  consider partitioning into separate `VectorDB` instances per filter value
  (e.g. one per tenant) instead of one giant index.
- No built-in embedding generation — bring your own vectors (OpenAI,
  Sentence-Transformers, Cohere, etc).
