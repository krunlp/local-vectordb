"""
A local, embedded vector database.

Design:
  - hnswlib provides the approximate nearest-neighbor (HNSW) index, which
    scales comfortably from tens of thousands to tens of millions of vectors
    on a single machine.
  - SQLite (via MetadataStore) stores the string ID <-> integer label mapping
    plus arbitrary JSON metadata per vector, and supports metadata filtering.
  - Everything persists to a single directory: an index file + a .sqlite file.

Deletes are soft (hnswlib marks labels deleted internally and skips them in
search); labels are not reused unless you call `compact()`, which rebuilds
the index to reclaim space.
"""
import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import hnswlib
import numpy as np

from .store import MetadataStore

FilterFn = Callable[[Dict[str, Any]], bool]


class VectorDB:
    def __init__(
        self,
        path: str,
        dim: Optional[int] = None,
        space: str = "cosine",  # "cosine" | "l2" | "ip"
        max_elements: int = 200_000,
        ef_construction: int = 200,
        M: int = 16,
        embedder: Optional[Any] = None,
    ):
        """
        path: directory where the index + metadata db live (created if absent)
        dim: embedding dimension. Required the first time a DB is created;
             inferred automatically from disk on subsequent loads.
        space: distance metric. "cosine" is the right default for most text/
               image embedding models (e.g. OpenAI, Sentence-Transformers).
        max_elements: initial capacity; the index auto-grows as needed.
        ef_construction / M: HNSW build-time parameters. Higher = better
               recall, slower build, more memory. Defaults are solid for
               most use cases up to a few million vectors.
        embedder: optional object with an `.encode(texts: list[str]) -> np.ndarray`
               method (e.g. vectordb.embeddings.TextEmbedder). If provided,
               enables add_text() / search_text() and dim can be omitted
               (inferred from the embedder).
        """
        self.embedder = embedder
        if dim is None and embedder is not None:
            dim = embedder.dim
        os.makedirs(path, exist_ok=True)
        self.path = path
        self._index_path = os.path.join(path, "index.hnsw")
        self._db_path = os.path.join(path, "metadata.sqlite")
        self._lock = threading.RLock()

        self.store = MetadataStore(self._db_path)

        existing_dim = self.store.get_meta("dim")
        existing_space = self.store.get_meta("space")

        if existing_dim is not None:
            # DB already exists on disk — load it, ignoring constructor
            # args that would be inconsistent with what's stored.
            self.dim = int(existing_dim)
            self.space = existing_space or space
            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            if os.path.exists(self._index_path):
                self.index.load_index(
                    self._index_path, max_elements=max_elements
                )
            else:
                self.index.init_index(
                    max_elements=max_elements,
                    ef_construction=ef_construction,
                    M=M,
                )
        else:
            if dim is None:
                raise ValueError(
                    "dim is required when creating a new VectorDB "
                    f"(no existing DB found at {path})"
                )
            self.dim = dim
            self.space = space
            self.index = hnswlib.Index(space=space, dim=dim)
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
            )
            self.store.set_meta("dim", str(dim))
            self.store.set_meta("space", space)

        # ef controls query-time recall/speed tradeoff; can be tuned live.
        self.index.set_ef(max(50, ef_construction))

    # -- capacity management -------------------------------------------------

    def _ensure_capacity(self, extra: int):
        current_count = self.index.get_current_count()
        max_elements = self.index.get_max_elements()
        if current_count + extra > max_elements:
            new_size = max(max_elements * 2, current_count + extra)
            self.index.resize_index(new_size)

    # -- write API ------------------------------------------------------------

    def add(
        self,
        ids: Union[str, Sequence[str]],
        vectors: Union[np.ndarray, Sequence[Sequence[float]]],
        metadatas: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
        texts: Optional[Union[str, Sequence[Optional[str]]]] = None,
    ):
        """Insert or overwrite vectors by ID (upsert semantics).

        texts: optional raw text per record, indexed for keyword (BM25)
               search via SQLite FTS5. Only needed if you plan to use
               hybrid_search(). Not required for plain vector search.
        """
        single = isinstance(ids, str)
        if single:
            ids = [ids]
            vectors = [vectors]
            metadatas = [metadatas or {}]
            texts = [texts] if texts is not None else [None]
        else:
            ids = list(ids)
            vectors = np.asarray(vectors, dtype=np.float32)
            if metadatas is None:
                metadatas = [{} for _ in ids]
            if texts is None:
                texts = [None for _ in ids]
            else:
                texts = list(texts)

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} != DB dim {self.dim}"
            )

        with self._lock:
            self._ensure_capacity(len(ids))
            # Dedup within this call (last occurrence wins) — otherwise two
            # rows with the same new id would each grab a fresh label before
            # either is committed, violating the store's unique id constraint.
            dedup: Dict[str, Tuple[np.ndarray, Dict[str, Any], Optional[str]]] = {}
            for id_, vec, meta, text in zip(ids, vectors, metadatas, texts):
                dedup[id_] = (vec, meta, text)

            new_ids, new_vecs, new_labels = [], [], []
            update_ids, update_vecs, update_labels = [], [], []
            fts_records: List[Tuple[str, str]] = []
            next_label = self.store.next_label()  # only hits the DB once per add() call
            for id_, (vec, meta, text) in dedup.items():
                existing_label = self.store.get_label(id_)
                if existing_label is not None:
                    update_ids.append((id_, meta))
                    update_vecs.append(vec)
                    update_labels.append(existing_label)
                else:
                    new_ids.append((id_, meta))
                    new_vecs.append(vec)
                    new_labels.append(next_label)
                    next_label += 1  # allocate sequentially within this batch
                if text:
                    fts_records.append((id_, text))


            # One vectorized call per group instead of one Python call per
            # row — matters a lot once you're adding thousands at a time.
            if new_vecs:
                self.index.add_items(np.stack(new_vecs), np.array(new_labels))
                self.store.upsert_many(
                    [(lb, id_, meta) for (id_, meta), lb in zip(new_ids, new_labels)]
                )
            if update_vecs:
                self.index.add_items(np.stack(update_vecs), np.array(update_labels))
                self.store.upsert_many(
                    [(lb, id_, meta) for (id_, meta), lb in zip(update_ids, update_labels)]
                )
            if fts_records:
                self.store.upsert_fts_many(fts_records)

    def _require_embedder(self):
        if self.embedder is None:
            raise RuntimeError(
                "This VectorDB has no embedder configured. Pass "
                "embedder=TextEmbedder(...) when constructing it, or use "
                "add()/search() with pre-computed vectors instead."
            )

    def add_text(
        self,
        ids: Union[str, Sequence[str]],
        texts: Union[str, Sequence[str]],
        metadatas: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    ):
        """Like add(), but embeds raw text via the configured embedder AND
        indexes the text for keyword search, enabling hybrid_search()."""
        self._require_embedder()
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        vectors = self.embedder.encode(text_list)
        if single:
            self.add(ids, vectors[0], metadatas, texts=text_list[0])
        else:
            self.add(ids, vectors, metadatas, texts=text_list)

    def search_text(
        self,
        query_text: str,
        k: int = 10,
        filter: Optional[Union[FilterFn, Dict[str, Any]]] = None,
        ef: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Like search(), but embeds the query text via the configured embedder."""
        self._require_embedder()
        vector = self.embedder.encode([query_text])[0]
        return self.search(vector, k=k, filter=filter, ef=ef)

    def hybrid_search(
        self,
        query_text: str,
        k: int = 10,
        filter: Optional[Union[FilterFn, Dict[str, Any]]] = None,
        rrf_k: int = 60,
        candidate_pool: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Semantic (vector) search + keyword (BM25/FTS5) search, fused with
        Reciprocal Rank Fusion (RRF). Catches both "meaning" matches (vector
        search) and exact-term matches like names, codes, or acronyms that
        embeddings can under-rank (keyword search).

        Requires records to have been added with add_text(), or add() with
        the `texts=` argument — only records with indexed text participate
        in the keyword side (records without indexed text can still surface
        via the vector side).

        RRF score per item = sum over each ranked list it appears in of
        1 / (rrf_k + rank). rrf_k=60 is the standard default from the
        original RRF paper; larger values flatten the influence of rank
        differences, smaller values weight top ranks more heavily.

        candidate_pool: how many results to pull from each individual
        search before fusing — larger surfaces more overlap candidates but
        costs more per query.
        """
        self._require_embedder()
        pool = max(k, candidate_pool)

        query_vector = self.embedder.encode([query_text])[0]
        vector_results = self.search(query_vector, k=pool, filter=filter)

        filter_fn = self._compile_filter(filter)
        with self._lock:
            fts_hits = self.store.search_fts(query_text, limit=pool)

        # Apply the same metadata filter to keyword hits, and fetch their
        # metadata (vector_results already carry metadata; fts hits don't).
        fts_filtered: List[Tuple[str, Dict[str, Any]]] = []
        if fts_hits:
            fts_ids = [id_ for id_, _ in fts_hits]
            meta_by_id = {}
            for id_ in fts_ids:
                label = self.store.get_label(id_)
                if label is None:
                    continue
                rec = self.store.get_by_label(label)
                if rec is None:
                    continue
                _, meta = rec
                meta_by_id[id_] = meta
            for id_, _bm25 in fts_hits:
                meta = meta_by_id.get(id_)
                if meta is None:  # deleted since indexed, or filtered out below
                    continue
                if filter_fn is not None and not filter_fn(meta):
                    continue
                fts_filtered.append((id_, meta))

        # -- Reciprocal Rank Fusion --------------------------------------
        rrf_scores: Dict[str, float] = {}
        metadata_by_id: Dict[str, Dict[str, Any]] = {}
        for rank, r in enumerate(vector_results):
            rrf_scores[r["id"]] = rrf_scores.get(r["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            metadata_by_id[r["id"]] = r["metadata"]
        for rank, (id_, meta) in enumerate(fts_filtered):
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (rrf_k + rank + 1)
            metadata_by_id.setdefault(id_, meta)

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            {"id": id_, "score": score, "metadata": metadata_by_id[id_]}
            for id_, score in ranked
        ]

    def delete(self, ids: Union[str, Sequence[str]]):
        """Soft-delete by ID (batched — one commit for the whole call).
        Space is reclaimed on compact()."""
        if isinstance(ids, str):
            ids = [ids]
        with self._lock:
            labels = self.store.mark_deleted_many(ids)
            for label in labels:
                self.index.mark_deleted(label)
            self.store.delete_fts_many(ids)

    def update_metadata(self, id_: str, metadata: Dict[str, Any]):
        """Replace metadata for an existing record without touching its vector."""
        label = self.store.get_label(id_)
        if label is None:
            raise KeyError(f"id {id_!r} not found")
        self.store.upsert(label, id_, metadata)

    # -- read API ---------------------------------------------------------------

    def search(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        k: int = 10,
        filter: Optional[Union[FilterFn, Dict[str, Any]]] = None,
        ef: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the k nearest records to query_vector as a list of dicts:
            {"id": str, "score": float, "metadata": dict}

        score is distance in the configured space (lower = closer for
        "cosine"/"l2"; higher = closer for "ip").

        filter: either a dict of exact-match metadata constraints
                (e.g. {"category": "faq"}) or a callable(metadata) -> bool.
                Filtering is applied post-search by over-fetching, so very
                selective filters may require raising k or ef.
        """
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query vector dim {query_vector.shape[1]} != DB dim {self.dim}"
            )
        filter_fn = self._compile_filter(filter)

        with self._lock:
            if ef is not None:
                self.index.set_ef(max(ef, k))  # must happen inside the lock —
                # set_ef mutates shared index state; setting it before
                # acquiring the lock lets concurrent searches with different
                # ef values race and clobber each other.
            active_count = self.store.count()  # excludes soft-deleted records
            if active_count == 0:
                return []
            # Over-fetch when filtering, since hnswlib can't filter natively.
            fetch_k = k if filter_fn is None else max(k * 10, 50)
            fetch_k = min(fetch_k, active_count)
            labels, distances = self.index.knn_query(query_vector, k=fetch_k)

        results = []
        labels, distances = labels[0], distances[0]
        lookup = self.store.get_many_by_labels(labels.tolist())
        for label, dist in zip(labels, distances):
            rec = lookup.get(int(label))
            if rec is None:  # deleted since query started
                continue
            id_, meta = rec
            if filter_fn is not None and not filter_fn(meta):
                continue
            results.append({"id": id_, "score": float(dist), "metadata": meta})
            if len(results) >= k:
                break
        return results

    @staticmethod
    def _compile_filter(
        filter: Optional[Union[FilterFn, Dict[str, Any]]]
    ) -> Optional[FilterFn]:
        if filter is None:
            return None
        if callable(filter):
            return filter
        # dict -> exact-match AND filter
        conditions = dict(filter)

        def _match(meta: Dict[str, Any]) -> bool:
            return all(meta.get(k) == v for k, v in conditions.items())

        return _match

    def get(self, id_: str) -> Optional[Dict[str, Any]]:
        label = self.store.get_label(id_)
        if label is None:
            return None
        rec = self.store.get_by_label(label)
        if rec is None:
            return None
        _, meta = rec
        return {"id": id_, "metadata": meta}

    def count(self) -> int:
        return self.store.count()

    # -- persistence / maintenance ---------------------------------------------

    def save(self):
        with self._lock:
            self.index.save_index(self._index_path)

    def compact(self) -> Dict[str, int]:
        """
        Rebuild the index from scratch, dropping soft-deleted vectors.
        Use periodically after many deletes to reclaim memory/disk space.

        Also self-heals a specific inconsistency: if the process previously
        crashed (or exited) between a write and the next save(), the
        metadata store can reference labels whose vectors were never
        persisted to the index file. Those records are detected here and
        soft-deleted (their vectors are genuinely unrecoverable — this
        makes state consistent again rather than raising an error).

        Returns {"kept": N, "dropped": N} — "dropped" is normally 0; a
        nonzero value means unsaved writes were lost to an earlier crash.
        """
        with self._lock:
            active_labels = self.store.all_active_labels()
            if not active_labels:
                return {"kept": 0, "dropped": 0}

            index_labels = set(self.index.get_ids_list())
            present = [lb for lb in active_labels if lb in index_labels]
            missing = [lb for lb in active_labels if lb not in index_labels]

            if missing:
                missing_ids = []
                for lb in missing:
                    rec = self.store.get_by_label(lb)
                    if rec is not None:
                        missing_ids.append(rec[0])
                if missing_ids:
                    self.store.mark_deleted_many(missing_ids)
                    self.store.delete_fts_many(missing_ids)

            if not present:
                self.save()
                return {"kept": 0, "dropped": len(missing)}

            vectors = np.array(self.index.get_items(present), dtype=np.float32)
            new_index = hnswlib.Index(space=self.space, dim=self.dim)
            new_index.init_index(
                max_elements=max(len(present) * 2, 1000)
            )
            new_index.add_items(vectors, np.array(present))
            new_index.set_ef(self.index.ef)
            self.index = new_index
            self.save()
            return {"kept": len(present), "dropped": len(missing)}

    def __len__(self):
        return self.count()

    def __repr__(self):
        return (
            f"VectorDB(path={self.path!r}, dim={self.dim}, "
            f"space={self.space!r}, count={self.count()})"
        )
