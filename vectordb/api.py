"""
HTTP server exposing a VectorDB over REST, so multiple processes/services
can share one local DB instead of opening the SQLite/HNSW files directly
from each process.

Run:
    DB_PATH=./mydb DIM=384 EMBED_MODEL=BAAI/bge-small-en-v1.5 \
        uvicorn vectordb.api:app --host 0.0.0.0 --port 8000

Env vars:
    DB_PATH      directory for the DB (default: ./vectordb_data)
    DIM          embedding dimension (required on first run if EMBED_MODEL
                 is not set; inferred from EMBED_MODEL or from disk otherwise)
    SPACE        "cosine" | "l2" | "ip" (default: cosine)
    EMBED_MODEL  optional fastembed model name; if set, enables the
                 text-based endpoints (/add_text, /search_text) so callers
                 can send raw text instead of vectors

Note: a single VectorDB instance holds an in-process lock, so this server
is safe for concurrent requests within itself. If you need multiple server
processes (e.g. behind a load balancer) writing to the *same* DB directory
concurrently, you'll want to route all writes through one process, or move
to a proper multi-writer store — this design is single-writer-process by
default, matching most local/embedded use.
"""
import os
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import VectorDB

DB_PATH = os.environ.get("DB_PATH", "./vectordb_data")
DIM = os.environ.get("DIM")
SPACE = os.environ.get("SPACE", "cosine")
EMBED_MODEL = os.environ.get("EMBED_MODEL")

embedder = None
if EMBED_MODEL:
    from .embeddings import TextEmbedder

    embedder = TextEmbedder(model_name=EMBED_MODEL)

db = VectorDB(
    DB_PATH,
    dim=int(DIM) if DIM else None,
    space=SPACE,
    embedder=embedder,
)

app = FastAPI(title="vectordb", version="0.1.0")


# -- schemas -----------------------------------------------------------------

class AddVectorItem(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddVectorsRequest(BaseModel):
    items: List[AddVectorItem]


class AddTextItem(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddTextsRequest(BaseModel):
    items: List[AddTextItem]


class SearchVectorRequest(BaseModel):
    vector: List[float]
    k: int = 10
    filter: Optional[Dict[str, Any]] = None


class SearchTextRequest(BaseModel):
    text: str
    k: int = 10
    filter: Optional[Dict[str, Any]] = None


class UpdateMetadataRequest(BaseModel):
    metadata: Dict[str, Any]


class DeleteRequest(BaseModel):
    ids: List[str]


# -- vector endpoints ----------------------------------------------------------

@app.post("/add")
def add_vectors(req: AddVectorsRequest):
    db.add(
        ids=[i.id for i in req.items],
        vectors=[i.vector for i in req.items],
        metadatas=[i.metadata for i in req.items],
    )
    db.save()
    return {"added": len(req.items)}


@app.post("/search")
def search_vectors(req: SearchVectorRequest):
    results = db.search(req.vector, k=req.k, filter=req.filter)
    return {"results": results}


# -- text endpoints (require EMBED_MODEL to be configured) ---------------------

def _require_embedder():
    if embedder is None:
        raise HTTPException(
            status_code=400,
            detail="No EMBED_MODEL configured on this server; use /add or "
            "/search with pre-computed vectors, or restart the server with "
            "the EMBED_MODEL env var set.",
        )


@app.post("/add_text")
def add_text(req: AddTextsRequest):
    _require_embedder()
    db.add_text(
        ids=[i.id for i in req.items],
        texts=[i.text for i in req.items],
        metadatas=[i.metadata for i in req.items],
    )
    db.save()
    return {"added": len(req.items)}


@app.post("/search_text")
def search_text(req: SearchTextRequest):
    _require_embedder()
    results = db.search_text(req.text, k=req.k, filter=req.filter)
    return {"results": results}


@app.post("/hybrid_search")
def hybrid_search(req: SearchTextRequest):
    """Semantic + keyword search fused via RRF. See VectorDB.hybrid_search
    for details. Only records added via /add_text (or add() with texts=)
    participate in the keyword side."""
    _require_embedder()
    results = db.hybrid_search(req.text, k=req.k, filter=req.filter)
    return {"results": results}


class IngestOKFRequest(BaseModel):
    bundle_path: str  # path on the server's filesystem, e.g. a mounted volume
    include_deprecated: bool = False


@app.post("/ingest_okf")
def ingest_okf(req: IngestOKFRequest):
    """Ingest an Open Knowledge Format bundle (a directory of markdown
    files on the server's filesystem) for semantic + hybrid search. See
    vectordb.okf for details. Requires EMBED_MODEL to be configured."""
    _require_embedder()
    from .okf import ingest_okf_bundle
    if not os.path.isdir(req.bundle_path):
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.bundle_path}")
    result = ingest_okf_bundle(db, req.bundle_path, include_deprecated=req.include_deprecated)
    db.save()
    return result


class GenerateOKFRequest(BaseModel):
    source_dir: str  # directory of .txt/.md files on the server's filesystem
    output_dir: str  # where to write the generated OKF bundle
    default_type: str = "Document"


@app.post("/generate_okf")
def generate_okf(req: GenerateOKFRequest):
    """Generate a valid OKF bundle from a directory of raw text/markdown
    documents (the reverse of /ingest_okf). See vectordb.okf_generate."""
    from .okf_generate import documents_to_okf
    if not os.path.isdir(req.source_dir):
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.source_dir}")
    result = documents_to_okf(req.source_dir, req.output_dir, default_type=req.default_type)
    return result


# -- record management -----------------------------------------------------

@app.get("/items/{id}")
def get_item(id: str):
    record = db.get(id)
    if record is None:
        raise HTTPException(status_code=404, detail="id not found")
    return record


@app.patch("/items/{id}/metadata")
def update_metadata(id: str, req: UpdateMetadataRequest):
    try:
        db.update_metadata(id, req.metadata)
    except KeyError:
        raise HTTPException(status_code=404, detail="id not found")
    db.save()
    return {"ok": True}


@app.post("/delete")
def delete_items(req: DeleteRequest):
    db.delete(req.ids)
    db.save()
    return {"deleted": len(req.ids)}


@app.post("/compact")
def compact():
    db.compact()
    return {"ok": True, "count": db.count()}


class AddEdgeRequest(BaseModel):
    source: str
    target: str
    relation: str = "related"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GetNeighborsRequest(BaseModel):
    id: str
    relation: Optional[str] = None
    direction: str = "both"
    with_metadata: bool = False


class TraverseRequest(BaseModel):
    start_id: str
    max_depth: int = 2
    relation: Optional[str] = None
    direction: str = "both"


class ShortestPathRequest(BaseModel):
    source: str
    target: str
    relation: Optional[str] = None
    direction: str = "both"


class ShortestPathWeightedRequest(BaseModel):
    source: str
    target: str
    relation: Optional[str] = None
    direction: str = "both"
    weight_field: str = "weight"


@app.post("/shortest_path_weighted")
def shortest_path_weighted(req: ShortestPathWeightedRequest):
    result = db.shortest_path_weighted(
        req.source, req.target, relation=req.relation, direction=req.direction,
        weight_field=req.weight_field,
    )
    return result if result is not None else {"path": None, "total_weight": None}


class GraphSearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    expand_hops: int = 1
    filter: Optional[Dict[str, Any]] = None


@app.post("/add_edge")
def add_edge(req: AddEdgeRequest):
    db.add_edge(req.source, req.target, relation=req.relation, metadata=req.metadata)
    db.save()
    return {"ok": True}


@app.post("/get_neighbors")
def get_neighbors(req: GetNeighborsRequest):
    return {"neighbors": db.get_neighbors(
        req.id, relation=req.relation, direction=req.direction, with_metadata=req.with_metadata
    )}


@app.post("/traverse")
def traverse(req: TraverseRequest):
    return db.traverse(
        req.start_id, max_depth=req.max_depth, relation=req.relation, direction=req.direction
    )


@app.post("/shortest_path")
def shortest_path(req: ShortestPathRequest):
    path = db.shortest_path(
        req.source, req.target, relation=req.relation, direction=req.direction
    )
    return {"path": path}


@app.post("/graph_search")
def graph_search(req: GraphSearchRequest):
    results = db.graph_search(req.vector, k=req.k, expand_hops=req.expand_hops, filter=req.filter)
    return {"results": results}


class QueryRequest(BaseModel):
    query: str  # e.g. "MATCH (a)-[:references*1..2]->(b) WHERE a.id = 'x' RETURN b.id, b.title"


@app.post("/query")
def query(req: QueryRequest):
    """Run a small Cypher-like query against the graph. See vectordb.query
    for the supported grammar (a real but intentionally small subset --
    not a Cypher implementation). Response includes "truncated": true if
    a variable-length pattern hit its internal traversal cap on a densely
    connected graph -- rows may be incomplete in that case."""
    from .query import run_query, QueryError
    try:
        return run_query(db, req.query)
    except QueryError as e:
        raise HTTPException(status_code=400, detail=str(e))


class AlgorithmRequest(BaseModel):
    relation: Optional[str] = None


@app.post("/graph/pagerank")
def graph_pagerank(req: AlgorithmRequest):
    from .graph_algorithms import pagerank
    return pagerank(db, relation=req.relation)


@app.post("/graph/pagerank_fast")
def graph_pagerank_fast(req: AlgorithmRequest):
    """GraphBLAS-accelerated PageRank (~3x faster on large graphs, same
    results). Requires python-graphblas to be installed on the server."""
    from .graph_algorithms import pagerank_graphblas
    try:
        return pagerank_graphblas(db, relation=req.relation)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))


@app.post("/graph/degree_centrality")
def graph_degree_centrality(req: AlgorithmRequest):
    from .graph_algorithms import degree_centrality
    return degree_centrality(db, relation=req.relation)


@app.post("/graph/connected_components")
def graph_connected_components(req: AlgorithmRequest):
    from .graph_algorithms import connected_components
    components = connected_components(db, relation=req.relation)
    return {"components": [list(c) for c in components]}


@app.get("/count")
def count():
    return {"count": db.count()}


@app.get("/health")
def health():
    return {"status": "ok", "dim": db.dim, "space": db.space, "count": db.count()}
