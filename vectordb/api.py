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


@app.get("/count")
def count():
    return {"count": db.count()}


@app.get("/health")
def health():
    return {"status": "ok", "dim": db.dim, "space": db.space, "count": db.count()}
