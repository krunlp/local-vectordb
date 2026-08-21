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
            self.store.delete_edges_for_ids(ids)

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
            index_count = self.index.get_current_count()
            # Guard against the metadata store and the HNSW index disagreeing
            # on how many vectors actually exist -- this happens if the
            # process crashed/exited between a write and the next save()
            # (see the persistence-gotcha docs and compact()). Without this,
            # requesting more results than the index actually holds throws a
            # cryptic hnswlib error instead of degrading gracefully; run
            # compact() to repair the underlying inconsistency.
            effective_count = min(active_count, index_count)
            if effective_count == 0:
                return []
            # Over-fetch when filtering, since hnswlib can't filter natively.
            fetch_k = k if filter_fn is None else max(k * 10, 50)
            fetch_k = min(fetch_k, effective_count)
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

    def edge_count(self) -> int:
        return self.store.edge_count()

    # -- persistence / maintenance ---------------------------------------------

    def save(self):
        with self._lock:
            self.index.save_index(self._index_path)

    # -- graph: edges, traversal, paths --------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a directed, typed edge between two ids (upsert -- calling
        again with the same source/target/relation overwrites metadata).
        Edges are independent of whether either id has a vector -- you can
        link ids that don't exist yet, matching how OKF itself tolerates
        links to concepts that aren't present."""
        with self._lock:
            self.store.add_edge(source_id, target_id, relation, metadata or {})

    def add_edges(self, edges: List[Dict[str, Any]]):
        """Batch add_edge -- one commit for the whole call. Each dict needs
        'source' and 'target', optionally 'relation' (default 'related')
        and 'metadata'."""
        with self._lock:
            rows = [
                (e["source"], e["target"], e.get("relation", "related"), e.get("metadata", {}))
                for e in edges
            ]
            self.store.add_edges_many(rows)

    def delete_edge(self, source_id: str, target_id: str, relation: Optional[str] = None):
        """Delete a specific edge, or all edges between source_id and
        target_id (any relation) if relation is omitted."""
        with self._lock:
            self.store.delete_edge(source_id, target_id, relation)

    def get_neighbors(
        self,
        id_: str,
        relation: Optional[str] = None,
        direction: str = "both",
        with_metadata: bool = False,
    ) -> List[Any]:
        """
        Immediate (1-hop) neighbors of id_.

        direction: 'out' (edges where id_ is the source), 'in' (id_ is the
        target), or 'both' (default).
        relation: filter to a specific edge type (e.g. "links_to"); None
        returns neighbors across all relation types.

        Returns a list of neighbor ids, or if with_metadata=True, a list of
        {"id", "relation", "direction", "edge_metadata", "metadata"} dicts
        (where "metadata" is the neighbor node's own record metadata, and
        "edge_metadata" is metadata attached to the edge itself). A
        neighbor id with no corresponding record (a dangling edge) still
        appears, with "metadata": None.
        """
        with self._lock:
            edges = self.store.get_edges(id_, relation=relation, direction=direction)
        results = []
        seen = set()
        for source, target, rel, edge_meta in edges:
            neighbor_id = target if source == id_ else source
            neighbor_direction = "out" if source == id_ else "in"
            key = (neighbor_id, rel, neighbor_direction)
            if key in seen:
                continue
            seen.add(key)
            if not with_metadata:
                results.append(neighbor_id)
                continue
            label = self.store.get_label(neighbor_id)
            node_meta = None
            if label is not None:
                rec = self.store.get_by_label(label)
                if rec is not None:
                    node_meta = rec[1]
            results.append({
                "id": neighbor_id,
                "relation": rel,
                "direction": neighbor_direction,
                "edge_metadata": edge_meta,
                "metadata": node_meta,
            })
        return results

    def traverse(
        self,
        start_id: str,
        max_depth: int = 2,
        relation: Optional[str] = None,
        direction: str = "both",
        max_nodes: int = 1000,
    ) -> Dict[str, Any]:
        """
        Breadth-first traversal from start_id out to max_depth hops.
        Returns {"nodes": {id: depth, ...}, "edges": [(source, target, relation), ...],
        "truncated": bool} covering every node/edge visited.

        "truncated" is True if max_nodes was hit before the traversal
        would have naturally finished -- meaning the graph has MORE
        connected nodes than are reflected in "nodes" here. Always check
        this rather than assuming a returned node count reflects the
        whole reachable subgraph; a caller silently treating a truncated
        result as complete is exactly the kind of bug that's easy to miss
        (found by testing a 500-node hub graph against the default
        max_nodes=1000 -- it looked like a complete, correct result with
        no indication anything was cut off, until this flag was added).
        """
        with self._lock:
            visited = {start_id: 0}
            frontier = [start_id]
            edges_seen = []
            depth = 0
            truncated = False
            while frontier and depth < max_depth and len(visited) < max_nodes:
                next_frontier = []
                for node in frontier:
                    for source, target, rel, _meta in self.store.get_edges(
                        node, relation=relation, direction=direction
                    ):
                        neighbor = target if source == node else source
                        edges_seen.append((source, target, rel))
                        if neighbor not in visited:
                            if len(visited) >= max_nodes:
                                truncated = True
                                break
                            visited[neighbor] = depth + 1
                            next_frontier.append(neighbor)
                    if len(visited) >= max_nodes:
                        break
                # If there's still a next_frontier left unexplored because we
                # hit max_depth (not max_nodes), that's normal completion,
                # not truncation -- only flag it if nodes existed beyond the
                # cap we didn't get to add, or if the frontier itself was cut
                # off mid-iteration above.
                if next_frontier and len(visited) >= max_nodes and depth + 1 < max_depth:
                    truncated = True
                frontier = next_frontier
                depth += 1
            # dedup edges (the same edge can be re-discovered from either endpoint)
            unique_edges = sorted(set(edges_seen))
            return {"nodes": visited, "edges": unique_edges, "truncated": truncated}

    def shortest_path_weighted(
        self, source_id: str, target_id: str, relation: Optional[str] = None,
        direction: str = "both", weight_field: str = "weight", default_weight: float = 1.0,
        max_nodes: int = 100_000,
    ) -> Optional[Dict[str, Any]]:
        """
        Dijkstra's algorithm: shortest path by total edge weight, not hop
        count (contrast with shortest_path(), which is unweighted BFS).
        Edge weight is read from that edge's metadata[weight_field]
        (default key "weight"); edges missing this field use
        default_weight. All weights must be non-negative -- Dijkstra is
        not correct with negative weights (that needs Bellman-Ford, which
        this doesn't implement).

        Returns {"path": [ids...], "total_weight": float}, or None if
        unreachable within max_nodes explored.
        """
        import heapq

        if source_id == target_id:
            return {"path": [source_id], "total_weight": 0.0}

        with self._lock:
            dist = {source_id: 0.0}
            parent: Dict[str, str] = {}
            visited: set = set()
            heap = [(0.0, source_id)]
            explored = 0

            while heap and explored < max_nodes:
                d, node = heapq.heappop(heap)
                if node in visited:
                    continue
                visited.add(node)
                explored += 1
                if node == target_id:
                    path = [node]
                    while path[-1] != source_id:
                        path.append(parent[path[-1]])
                    return {"path": list(reversed(path)), "total_weight": d}

                for source, target, _rel, edge_meta in self.store.get_edges(
                    node, relation=relation, direction=direction
                ):
                    neighbor = target if source == node else source
                    if neighbor in visited:
                        continue
                    w = edge_meta.get(weight_field, default_weight)
                    if w < 0:
                        raise ValueError(
                            f"Negative edge weight ({w}) on edge {source!r}->{target!r} "
                            "-- Dijkstra requires non-negative weights"
                        )
                    new_dist = d + w
                    if neighbor not in dist or new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        parent[neighbor] = node
                        heapq.heappush(heap, (new_dist, neighbor))
            return None

    def shortest_path(
        self, source_id: str, target_id: str, relation: Optional[str] = None,
        direction: str = "both", max_depth: int = 10,
    ) -> Optional[List[str]]:
        """BFS shortest path (by hop count, not edge weight) from source_id
        to target_id. Returns the list of ids along the path (inclusive of
        both endpoints), or None if unreachable within max_depth hops."""
        if source_id == target_id:
            return [source_id]
        with self._lock:
            visited = {source_id}
            parent: Dict[str, str] = {}
            frontier = [source_id]
            depth = 0
            while frontier and depth < max_depth:
                next_frontier = []
                for node in frontier:
                    for source, target, _rel, _meta in self.store.get_edges(
                        node, relation=relation, direction=direction
                    ):
                        neighbor = target if source == node else source
                        if neighbor in visited:
                            continue
                        visited.add(neighbor)
                        parent[neighbor] = node
                        if neighbor == target_id:
                            path = [neighbor]
                            while path[-1] != source_id:
                                path.append(parent[path[-1]])
                            return list(reversed(path))
                        next_frontier.append(neighbor)
                frontier = next_frontier
                depth += 1
            return None

    def graph_search(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        k: int = 10,
        expand_hops: int = 1,
        relation: Optional[str] = None,
        filter: Optional[Union[FilterFn, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector search, then expand each hit outward through the graph by
        expand_hops and include those connected nodes too -- useful for
        "find similar things, AND whatever they're linked to" (e.g. find
        the most relevant OKF concepts, then pull in their direct
        references even if those references alone wouldn't have ranked in
        the top-k by vector similarity).

        Returns entries like search()'s, plus "via" ("vector" for a direct
        hit, or "graph" for a node pulled in through expansion) and
        "hop_distance" (0 for direct hits).
        """
        direct_hits = self.search(query_vector, k=k, filter=filter)
        results = {r["id"]: {**r, "via": "vector", "hop_distance": 0} for r in direct_hits}

        if expand_hops > 0:
            for hit in direct_hits:
                expansion = self.traverse(hit["id"], max_depth=expand_hops)
                for node_id, depth in expansion["nodes"].items():
                    if node_id in results or depth == 0:
                        continue
                    label = self.store.get_label(node_id)
                    if label is None:
                        continue  # dangling edge target with no actual record
                    rec = self.store.get_by_label(label)
                    if rec is None:
                        continue
                    _, meta = rec
                    if filter is not None:
                        filter_fn = self._compile_filter(filter)
                        if filter_fn is not None and not filter_fn(meta):
                            continue
                    results[node_id] = {
                        "id": node_id, "score": None, "metadata": meta,
                        "via": "graph", "hop_distance": depth,
                    }
        return list(results.values())

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
                    self.store.delete_edges_for_ids(missing_ids)

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
