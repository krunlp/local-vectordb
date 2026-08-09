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
    ):
        """Insert or overwrite vectors by ID (upsert semantics)."""
        single = isinstance(ids, str)
        if single:
            ids = [ids]
            vectors = [vectors]
            metadatas = [metadatas or {}]
        else:
            ids = list(ids)
            vectors = np.asarray(vectors, dtype=np.float32)
            if metadatas is None:
                metadatas = [{} for _ in ids]

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
            dedup: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
            for id_, vec, meta in zip(ids, vectors, metadatas):
                dedup[id_] = (vec, meta)

            new_ids, new_vecs, new_labels = [], [], []
            update_ids, update_vecs, update_labels = [], [], []
            next_label = self.store.next_label()  # only hits the DB once per add() call
            for id_, (vec, meta) in dedup.items():
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

            # One vectorized call per group instead of one Python call per
            # row — matters a lot once you're adding thousands at a time.
            if new_vecs:
                self.index.add_items(np.stack(new_vecs), np.array(new_labels))
                for (id_, meta), label in zip(new_ids, new_labels):
                    self.store.upsert(label, id_, meta or {})
            if update_vecs:
                self.index.add_items(np.stack(update_vecs), np.array(update_labels))
                for (id_, meta), label in zip(update_ids, update_labels):
                    self.store.upsert(label, id_, meta or {})

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
        """Like add(), but embeds raw text via the configured embedder."""
        self._require_embedder()
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        vectors = self.embedder.encode(text_list)
        self.add(ids, vectors, metadatas)

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

    def delete(self, ids: Union[str, Sequence[str]]):
        """Soft-delete by ID. Space is reclaimed on compact()."""
        if isinstance(ids, str):
            ids = [ids]
        with self._lock:
            for id_ in ids:
                label = self.store.mark_deleted(id_)
                if label is not None:
                    self.index.mark_deleted(label)

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

    def compact(self):
        """
        Rebuild the index from scratch, dropping soft-deleted vectors.
        Use periodically after many deletes to reclaim memory/disk space.
        """
        with self._lock:
            active_labels = self.store.all_active_labels()
            if not active_labels:
                return
            vectors = np.array(
                self.index.get_items(active_labels), dtype=np.float32
            )
            new_index = hnswlib.Index(space=self.space, dim=self.dim)
            new_index.init_index(
                max_elements=max(len(active_labels) * 2, 1000)
            )
            new_index.add_items(vectors, np.array(active_labels))
            new_index.set_ef(self.index.ef)
            self.index = new_index
            self.save()

    def __len__(self):
        return self.count()

    def __repr__(self):
        return (
            f"VectorDB(path={self.path!r}, dim={self.dim}, "
            f"space={self.space!r}, count={self.count()})"
        )
