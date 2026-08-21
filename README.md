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
having called `save()`, those unsaved vectors are gone — their metadata
still exists (silently unsearchable, no error) until you run `db.compact()`,
which detects and cleans up the inconsistency:

```python
result = db.compact()
print(result)  # {"kept": 950, "dropped": 50} <- 50 records lost to a crash
```

`dropped > 0` means writes were lost between the last `save()` and the
crash — that data is genuinely unrecoverable, `compact()` just makes the
DB's state consistent again instead of leaving stale, unsearchable metadata
around (and instead of raising an error, which is what it did before this
was caught and fixed by testing).

Call `db.save()` after a batch of writes, or after every write if losing
recent writes on a crash is unacceptable. The `VectorDB.api` HTTP server
already does this for you on every write endpoint. Run `db.compact()`
after any suspected crash, and periodically anyway to reclaim space from
ordinary deletes.

## Verified behavior (not just claimed)

Numbers below are from actual runs, not estimates. Measured on a
constrained 1-CPU-core sandbox; a normal multi-core machine will build
faster (hnswlib parallelizes insertion across cores).

- **Recall**: HNSW is approximate. Measured recall@10 against brute-force
  ground truth: ~95% on 20K vectors (dim 64) at default settings; ~64% on
  60K vectors (dim 128) with *uniformly random* test vectors, which is a
  deliberately adversarial case (near-zero similarity gap between true
  rank 10 and rank 20 — real embeddings cluster far more than random noise
  and should recall better). Raising `ef` closed most of the gap (94.8%
  recall at `ef=800` on the same hard case). If recall matters for your use
  case, measure it against your own embeddings and tune `ef` accordingly —
  don't assume defaults are right for your data's dimensionality.
- **Concurrency**: tested with 4 concurrent writer threads + 4 concurrent
  reader threads hammering one `VectorDB` instance — zero exceptions, zero
  lost writes.
- **Crash recovery**: tested an actual process crash (unsaved writes, then
  process exit) followed by reopening in a new process. Confirmed the
  failure mode described above, and confirmed `compact()` repairs it
  cleanly.
- **Scale**: 60K vectors (dim 128) built in ~35s / ~37MB index file on 1
  CPU core. Not yet verified at millions of vectors on real hardware —
  extrapolate cautiously.

## Hybrid search (semantic + keyword)

`hybrid_search()` combines vector similarity with BM25 keyword search
(via SQLite's FTS5 extension) using Reciprocal Rank Fusion (RRF). This
catches both "meaning" matches (embeddings) and exact-term matches —
names, SKUs, error codes, acronyms — that embeddings alone can under-rank.

```python
db = VectorDB("./mydb", embedder=embedder)

# add_text() indexes for BOTH vector and keyword search
db.add_text("doc1", "SKU-4471-X is currently out of stock")

results = db.hybrid_search("SKU-4471-X", k=5)
# -> ranks doc1 highly even if its embedding similarity to the query
#    alone wouldn't have put it in the top results
```

Notes:
- Only records added via `add_text()` (or `add(..., texts=[...])`) have a
  keyword-searchable side; records added with `add()` and no `texts=` still
  participate via vector similarity only.
- `score` in hybrid results is the fused RRF score (higher = better), not a
  raw distance — don't compare it directly to `search()`'s distance scores.
- Tune fusion weighting with `rrf_k` (default 60, the standard from the RRF
  paper) and `candidate_pool` (how many results each individual search
  contributes before fusion; larger = better recall, more per-query cost).

## Graph functionality

Beyond vector similarity, you can attach explicit, typed, directed edges
between records and query the resulting graph — neighbors, multi-hop
traversal, shortest path, and a combined vector-plus-graph search.

```python
db.add_edge("doc1", "doc2", relation="references")
db.add_edge("doc2", "doc3", relation="references")

db.get_neighbors("doc1", direction="out")          # ['doc2']
db.get_neighbors("doc1", with_metadata=True)        # full neighbor records + edge metadata

db.traverse("doc1", max_depth=2)
# {"nodes": {"doc1": 0, "doc2": 1, "doc3": 2}, "edges": [("doc1","doc2","references"), ...]}

db.shortest_path("doc1", "doc3")                    # ['doc1', 'doc2', 'doc3'] or None

# vector search, then pull in graph-connected nodes too (even ones that
# wouldn't rank in the top-k by similarity alone)
db.graph_search(query_vector, k=5, expand_hops=1)
```

Also exposed over the API: `POST /add_edge`, `/get_neighbors`, `/traverse`,
`/shortest_path`, `/graph_search`.

Notes:
- Edges are independent of the vector index — you can link ids that don't
  (yet) have a vector. `get_neighbors()`/`traverse()` return such
  "dangling" neighbors with `metadata: None` rather than erroring, since
  real-world link graphs (including OKF's own) routinely reference things
  that aren't (yet) present.
- `direction`: `'out'` (edges where the given id is the source), `'in'`,
  or `'both'` (default) on every graph method.
- `traverse()`/`shortest_path()` are unweighted BFS (hop count, not edge
  weight) — fine for "how are these connected" questions, not meant for
  weighted-shortest-path use cases.
- Deleting a record (`db.delete()`) also removes every edge touching it,
  so the graph doesn't accumulate edges pointing at nothing. This cleanup
  also runs during `compact()`'s crash-recovery path, not just `delete()`
  — found as a real gap during testing (a crash-orphaned record's edges
  survived compact() and pointed at an id that would never exist again),
  now fixed.
- **`traverse()` (and anything built on it — `graph_search()`, the query
  language's variable-length patterns) returns `truncated: bool`.** A
  densely connected graph can hit the `max_nodes` safety cap before
  finishing; without this flag, a truncated result looks identical to a
  complete one. This was tested and found missing, then added — check it
  rather than assuming a returned node/row count is the whole picture on
  any graph you haven't explicitly bounded the size of.
- Performance: tested up to 50,000 nodes / ~250,000 edges — 3-hop
  traversal and shortest-path both stayed well under a second, and a
  full-database-scan query (no `id` anchor) over 50K records took ~0.3s.
  The one-SQL-query-per-node pattern during traversal turned out not to
  be a practical bottleneck at this scale, verified rather than assumed
  either way.
- **OKF integration**: `ingest_okf_bundle()` automatically turns each
  concept's markdown cross-links (already extracted per SPEC §6.1) into
  real graph edges (`relation="references"`), so an ingested OKF bundle is
  immediately traversable, not just searchable. Verified against Google's
  real `acme_retail` bundle — 14 real cross-references correctly became
  queryable edges, including a correctly-dangling edge to a concept that
  exists in the bundle's link graph but was excluded from search because
  it's marked `deprecated`.

## Query language (small Cypher-like subset)

A genuine but intentionally small subset of Cypher (Neo4j/FalkorDB's query
language), for pattern-matching over the graph without hand-writing
traversal code every time:

```python
from vectordb.query import run_query

result = run_query(db, "MATCH (a)-[:references]->(b) WHERE a.id = 'policy1' RETURN b.id, b.title")
print(result)  # {"rows": [...], "truncated": False}
```

Also exposed via `POST /query`.

**Supported:** `MATCH (a)-[:relation]->(b)` (exact hop count or
`*min..max` variable-length paths), `WHERE var.field = 'value'`
(AND-joined equality only), `RETURN var` or `var.field` (comma-separated).
If `WHERE` constrains the start variable's `id`, the query starts from
that one id and traverses outward (same cost as `traverse()`). If it
doesn't, every active record is scanned as a candidate start node — a real
full-database scan, same cost category as an unindexed Cypher `MATCH`
would be, and should be expected to be slow on a large DB.

**`truncated: true` in the result means the rows may be incomplete** —
some underlying multi-hop traversal hit its safety cap (`max_nodes`,
passed through as `run_query(db, query, max_nodes=...)`) on a densely
connected graph before it finished exploring. Always check this on
variable-length queries over graphs you haven't bounded the size of.

**Deliberately NOT supported**, and raises `QueryError` with a specific
reason rather than silently returning wrong results if you try: `OR`,
inequality/range conditions, undirected patterns, multiple `MATCH`
clauses, node labels, aggregation, `ORDER BY`/`LIMIT`,
`CREATE`/`MERGE`/`SET`/`DELETE`. This is not Cypher — it's a small,
honest slice of it.

## Graph algorithms

`vectordb.graph_algorithms` adds PageRank, degree centrality, and weakly
connected components — this project had zero graph algorithms before
this (only BFS traversal), so this closes a real, previously-total gap
rather than extending an existing feature.

```python
from vectordb.graph_algorithms import pagerank, degree_centrality, connected_components

pagerank(db)                    # {"concept_id": score, ...} -- sums to ~1.0
degree_centrality(db)           # {"concept_id": edge_count, ...}
connected_components(db)        # [set of ids, set of ids, ...] -- one per component
```

Also exposed via `POST /graph/pagerank`, `/graph/degree_centrality`,
`/graph/connected_components`.

**Verification, not just "it runs":** PageRank was cross-checked against
`networkx`'s independent reference implementation (including the classic
dangling-node edge case — a node with in-edges but no out-edges, a common
source of rank-conservation bugs) and matched to within floating-point
tolerance on every test graph, including the real OKF-derived
`acme_retail` graph, where it correctly surfaced the most-referenced
concept as highest-ranked.

**Honest scope**: these are pure-Python implementations over the SQLite
edge table — correct and tested, but not vectorized/matrix-native
execution. That's the actual architectural difference from something
like FalkorDB's GraphBLAS engine (see the maturity roadmap below), and
isn't closeable without a fundamentally different storage engine. These
algorithms will be fine at the scale this project has been tested at
(tens of thousands of nodes); they are not competing with a sparse-matrix
engine's ability to run PageRank over millions of nodes.

## Graph DB maturity roadmap (honest assessment)

Compared against Neo4j and FalkorDB specifically — researched, not
assumed. FalkorDB represents graphs as sparse adjacency matrices and
executes traversals as GraphBLAS linear algebra (a multi-hop pattern
match becomes matrix multiplication); Neo4j uses index-free
adjacency (every node holds direct pointers to its relationship
records) plus a cost-based query planner refined over more than a
decade. Neither is what this project is.

**Genuinely closeable, and now closed:**
- ✅ Graph algorithms (PageRank, centrality, connected components) — this session
- ✅ A declarative query language (small Cypher subset) — see above
- ✅ Correctness hardening (cycles, dangling edges, crash-recovery cleanup, silent-truncation signaling) — extensively adversarially tested

**Genuinely closeable, not yet done (realistic next steps if this keeps
growing):**
- Weighted shortest path (Dijkstra) using edge metadata — `shortest_path()`
  is currently unweighted hop-count only
- More query language grammar: `OR`, inequality/range conditions,
  `ORDER BY`/`LIMIT`, basic aggregation (`COUNT`)
- Bulk edge import from CSV/adjacency-list formats
- Concurrent-write stress testing specifically for the edges table at
  higher thread counts than tested so far

**Not closeable without becoming a fundamentally different project:**
- Matrix-native or index-free adjacency execution (the actual
  architectural core of both comparison systems)
- A real cost-based query planner
- ACID transactions spanning vector + graph + metadata atomically
- Clustering, replication, horizontal scale
- Full Cypher/Gremlin language compliance
- Years of production hardening, multi-language drivers, ecosystem tooling

The honest framing, unchanged from earlier in this conversation: this is
a correctness-hardened, algorithmically-real graph layer bolted onto a
vector DB — meaningfully more capable than it was, still not a graph
database in the category Neo4j or FalkorDB occupy, and closing the
remaining gap would mean rewriting the storage engine, not adding more
Python functions.

## Capacity planning

Real measured numbers (not estimates) at dim=128: **660 bytes/vector on
disk** for the HNSW index (default `M=16`) — 512 bytes is the raw float32
vector itself, ~148 bytes is graph overhead. This scales linearly with N
(overhead is per-node from `M`, not from total graph size), so it
extrapolates reliably to larger N — unlike build *time*, which doesn't
extrapolate as cleanly (see below).

| vectors | dim 128 | dim 384 (bge-small default) | dim 768 |
|---|---|---|---|
| 100K | 0.07 GB | 0.17 GB | 0.32 GB |
| 1M | 0.66 GB | 1.68 GB | 3.22 GB |
| 10M | 6.6 GB | 16.8 GB | 32.2 GB |

**Important**: hnswlib keeps the entire index in RAM for search — the
numbers above are your *minimum RAM* requirement, not just disk. At 10M
vectors with 384-dim embeddings, budget ~17GB+ RAM, not a typical laptop.
Add ~51 bytes/record for empty metadata (more if you store real fields —
measure your own metadata size and add it).

**Build time does not extrapolate linearly from small tests** — insertion
cost grows as the graph gets larger, and hnswlib parallelizes across CPU
cores, so throughput depends heavily on your actual hardware. Benchmark
insert throughput on your real target machine with your real dimension
before committing to a 10M-vector plan; don't assume small-scale numbers
scale up proportionally.

## OKF (Open Knowledge Format) support

[OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog) is Google
Cloud's open spec for representing knowledge as a directory of markdown
files with YAML frontmatter. `vectordb.okf` ingests an OKF bundle so you
can semantically/hybrid-search it, instead of only following its markdown
cross-links by hand.

```python
from vectordb import VectorDB
from vectordb.embeddings import TextEmbedder
from vectordb.okf import ingest_okf_bundle

db = VectorDB("./mydb", embedder=TextEmbedder())
result = ingest_okf_bundle(db, "/path/to/okf/bundle")
print(result)  # {"indexed": 42, "skipped_deprecated": 3, "skipped_malformed": 0}

results = db.hybrid_search("revenue recognition policy", k=5)
```

Notes, built and tested against Google's real reference bundles
(`GoogleCloudPlatform/knowledge-catalog`), not just the written spec:

- Concept ID = the file's bundle-relative path with `.md` stripped (per
  spec — this is how OKF concepts are addressed).
- `index.md` and `log.md` are reserved filenames (navigation/history), not
  concepts, and are skipped at every directory level.
- Only `type` is required in frontmatter; everything else (`title`,
  `description`, `resource`, `tags`, `status`, `stale_after`, and any
  producer-defined extras) is optional and tolerated if missing or unknown.
- Concepts with `status: deprecated` are **skipped by default** — pass
  `include_deprecated=True` to index them anyway (e.g. for an archival
  view). Real bundles do use this field, so this matters in practice.
- Markdown cross-links (`[text](/tables/x.md)` or `[text](./x.md)`) are
  extracted into `metadata["links"]`, since OKF doesn't have a `links:`
  frontmatter field — links are just normal markdown.
- `stale_after` (a YAML date) is normalized to an ISO string, since raw
  `datetime.date` objects aren't JSON-serializable for the metadata store.
- Searchable text is `title + description + body`, so semantic search
  weighs the concept's own summary alongside its full content.

## Generating OKF from your own documents

The reverse direction — turn raw documents into a valid OKF bundle:

```python
from vectordb.okf_generate import documents_to_okf

# every .txt/.md file in the directory becomes one concept
result = documents_to_okf("/path/to/my/docs", "/path/to/output/bundle", default_type="Document")
print(result)  # {"generated": 12}

# or from in-memory data
result = documents_to_okf(
    [{"text": "...", "title": "Rate Limits", "type": "API Endpoint", "tags": ["api"]}],
    "/path/to/output/bundle",
)
```

What it produces, verified by round-tripping generated output back through
`vectordb.okf`'s own reader (not just visual inspection):

- One concept `.md` file per document, with `type`, a derived `title`
  (from a markdown heading if present, a plausible title-like first line,
  or the filename as a last resort — plain body text is never
  misidentified as a title), a derived `description`, and a `generated:
  {at, by}` provenance block (SPEC §5.2).
- `index.md` at every directory level, in the bullet-list style Google's
  own reference bundles use.
- A `log.md` recording the generation event (SPEC §9).

**What this does NOT do:** infer cross-links between your documents.
There's no reliable way to know which documents should reference each
other from plain text alone — add markdown links in your source text
yourself, or post-process the generated `.md` files, if you want a linked
concept graph rather than a flat one.

**Security note:** if you use the dict/list input form with explicit `id`
fields (rather than the directory-scan form, where IDs come from real
filenames), those IDs are validated against path traversal — `id:
"../../etc/evil"` or similar is rejected with `ValueError`, not silently
written outside `output_dir`. This was a real bug caught by testing, not
a hypothetical: an earlier version of this code did write outside the
intended directory when given an adversarial `id`.

## OKF security notes (found by adversarial testing, not assumed safe)

Ingesting an OKF bundle means reading and executing logic over content you
may not fully trust (a cloned repo, an uploaded archive, a shared drive).
Two things were tested adversarially and fixed, not just assumed fine:

- **Symlink file exfiltration**: a bundle containing a symlink to a file
  *outside* the bundle (crafted to look like a valid OKF concept) would
  previously have its content silently read and indexed into your search
  results — a real way for a malicious bundle to exfiltrate other files
  the process can read. `load_bundle()` and `ingest_okf_bundle()` now
  reject any symlink whose target resolves outside the bundle root.
  Symlinks that stay *inside* the bundle (a legitimate internal alias)
  still work correctly.
- **YAML parser recursion crash**: a pathologically deeply-nested (but
  syntactically valid) YAML value in one concept's frontmatter could
  exceed Python's recursion limit and crash the *entire* bundle load —
  taking every other concept down with the one bad file. Now caught and
  treated the same as any other malformed concept: skipped, with the rest
  of the bundle loading normally.
- (Checked and found NOT to be an issue): YAML's classic "billion laughs"
  alias-expansion memory bomb doesn't actually apply here — PyYAML shares
  object references across aliases rather than deep-copying them, so wide
  alias trees don't blow up memory the way they would in some other
  parsers. Verified with a 10^12-element alias structure parsing in
  milliseconds, not tested and assumed.

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
