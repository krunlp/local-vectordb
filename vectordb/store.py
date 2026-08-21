"""
SQLite-backed metadata and ID-mapping store for the vector DB.

HNSW (via hnswlib) only understands integer labels, so this store keeps:
  - a mapping between user-facing string IDs and internal integer labels
  - arbitrary JSON metadata per record
  - a free-list of reusable integer labels (for after deletes)
"""
import json
import re
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple


class MetadataStore:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        # One connection per thread to keep this safe under simple concurrent use.
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.path)
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
        return self._local.conn

    def close(self):
        """Close this thread's connection, if one was opened. Call this when
        a worker thread is being retired (e.g. in a threaded server) to
        avoid leaking connections over the life of a long-running process."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def _init_schema(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        # NORMAL is safe with WAL (durable across app crashes, only risks
        # loss on an OS crash/power loss) and is dramatically faster than
        # the default FULL for batches of small writes.
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                label INTEGER PRIMARY KEY,
                id TEXT UNIQUE NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                deleted INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_id ON records(id)"
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
                id UNINDEXED, text, tokenize='porter unicode61'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL DEFAULT 'related',
                metadata TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (source_id, target_id, relation)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    # -- label allocation -------------------------------------------------

    def next_label(self) -> int:
        cur = self.conn.execute("SELECT MAX(label) FROM records")
        row = cur.fetchone()
        return 0 if row[0] is None else row[0] + 1

    def reclaimed_labels(self, limit: int) -> List[int]:
        """Labels marked deleted that can be reused."""
        cur = self.conn.execute(
            "SELECT label FROM records WHERE deleted = 1 LIMIT ?", (limit,)
        )
        return [r[0] for r in cur.fetchall()]

    # -- CRUD ---------------------------------------------------------------

    def upsert(self, label: int, id_: str, metadata: Dict[str, Any]):
        self.upsert_many([(label, id_, metadata)])

    def upsert_many(self, records: List[Tuple[int, str, Dict[str, Any]]]):
        """Batch upsert with a single commit — much faster than calling
        upsert() in a loop, which commits (and can fsync) per row."""
        if not records:
            return
        rows = [(label, id_, json.dumps(meta)) for label, id_, meta in records]
        self.conn.executemany(
            """
            INSERT INTO records (label, id, metadata, deleted)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(label) DO UPDATE SET
                id=excluded.id, metadata=excluded.metadata, deleted=0
            """,
            rows,
        )
        self.conn.commit()

    def get_label(self, id_: str) -> Optional[int]:
        cur = self.conn.execute(
            "SELECT label FROM records WHERE id = ? AND deleted = 0", (id_,)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def get_by_label(self, label: int) -> Optional[Tuple[str, Dict[str, Any]]]:
        cur = self.conn.execute(
            "SELECT id, metadata FROM records WHERE label = ? AND deleted = 0",
            (label,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1])

    def get_many_by_labels(
        self, labels: Iterable[int]
    ) -> Dict[int, Tuple[str, Dict[str, Any]]]:
        labels = list(labels)
        if not labels:
            return {}
        placeholders = ",".join("?" for _ in labels)
        cur = self.conn.execute(
            f"SELECT label, id, metadata FROM records "
            f"WHERE label IN ({placeholders}) AND deleted = 0",
            labels,
        )
        return {r[0]: (r[1], json.loads(r[2])) for r in cur.fetchall()}

    def mark_deleted(self, id_: str) -> Optional[int]:
        labels = self.mark_deleted_many([id_])
        return labels[0] if labels else None

    def mark_deleted_many(self, ids: List[str]) -> List[int]:
        """Batch soft-delete with a single commit. Returns the labels that
        were actually found and marked (skips unknown/already-deleted ids)."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = self.conn.execute(
            f"SELECT label FROM records WHERE id IN ({placeholders}) AND deleted = 0",
            ids,
        )
        labels = [r[0] for r in cur.fetchall()]
        if labels:
            label_placeholders = ",".join("?" for _ in labels)
            self.conn.execute(
                f"UPDATE records SET deleted = 1 WHERE label IN ({label_placeholders})",
                labels,
            )
            self.conn.commit()
        return labels

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM records WHERE deleted = 0")
        return cur.fetchone()[0]

    def all_active_labels(self) -> List[int]:
        cur = self.conn.execute("SELECT label FROM records WHERE deleted = 0")
        return [r[0] for r in cur.fetchall()]

    # -- full-text (FTS5) for hybrid search --------------------------------

    def upsert_fts_many(self, records: List[Tuple[str, str]]):
        """records: list of (id, text). Upsert-by-delete-then-insert since
        FTS5 virtual tables don't support ON CONFLICT."""
        if not records:
            return
        ids = [r[0] for r in records]
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(f"DELETE FROM fts WHERE id IN ({placeholders})", ids)
        self.conn.executemany("INSERT INTO fts (id, text) VALUES (?, ?)", records)
        self.conn.commit()

    def delete_fts_many(self, ids: List[str]):
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(f"DELETE FROM fts WHERE id IN ({placeholders})", ids)
        self.conn.commit()

    def search_fts(self, query_text: str, limit: int) -> List[Tuple[str, float]]:
        """Keyword search via BM25. Returns [(id, bm25_rank), ...] ordered
        best-first (BM25's `rank` is more negative = more relevant in FTS5,
        so ORDER BY rank ascending gives best matches first)."""
        tokens = re.findall(r"\w+", query_text.lower())
        if not tokens:
            return []
        # Quote each token so FTS5 query-syntax characters in the input
        # (e.g. '-', '"', ':') can't be misinterpreted as query operators.
        fts_query = " OR ".join(f'"{t}"' for t in tokens)
        try:
            cur = self.conn.execute(
                "SELECT id, rank FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, limit),
            )
            return [(r[0], r[1]) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            # Malformed query somehow slipped through quoting — fail soft.
            return []

    # -- graph edges ---------------------------------------------------------
    # Edges reference ids by string, independent of vector labels -- this
    # deliberately allows "dangling" edges to ids that don't (yet) exist as
    # vectors (e.g. an OKF concept linking to one not yet ingested), matching
    # OKF's own tolerance for broken links rather than enforcing referential
    # integrity that would reject perfectly normal real-world graphs.

    def add_edge(self, source_id: str, target_id: str, relation: str, metadata: Dict[str, Any]):
        self.add_edges_many([(source_id, target_id, relation, metadata)])

    def add_edges_many(self, edges: List[Tuple[str, str, str, Dict[str, Any]]]):
        if not edges:
            return
        rows = [(s, t, r, json.dumps(m)) for s, t, r, m in edges]
        self.conn.executemany(
            """
            INSERT INTO edges (source_id, target_id, relation, metadata)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relation) DO UPDATE SET
                metadata=excluded.metadata
            """,
            rows,
        )
        self.conn.commit()

    def delete_edge(self, source_id: str, target_id: str, relation: Optional[str] = None):
        if relation is not None:
            self.conn.execute(
                "DELETE FROM edges WHERE source_id=? AND target_id=? AND relation=?",
                (source_id, target_id, relation),
            )
        else:
            self.conn.execute(
                "DELETE FROM edges WHERE source_id=? AND target_id=?", (source_id, target_id)
            )
        self.conn.commit()

    def delete_edges_for_ids(self, ids: List[str]):
        """Remove all edges touching any of the given ids (source or
        target) -- called when those ids are deleted from the DB, so the
        graph doesn't accumulate edges pointing at nothing."""
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(
            f"DELETE FROM edges WHERE source_id IN ({placeholders}) "
            f"OR target_id IN ({placeholders})",
            ids + ids,
        )
        self.conn.commit()

    def get_edges(
        self, id_: str, relation: Optional[str] = None, direction: str = "both"
    ) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """Returns (source_id, target_id, relation, metadata) tuples touching
        id_. direction: 'out' (id_ is source), 'in' (id_ is target), or
        'both'."""
        clauses, params = [], []
        if direction in ("out", "both"):
            clauses.append("source_id = ?")
            params.append(id_)
        if direction in ("in", "both"):
            clauses.append("target_id = ?")
            params.append(id_)
        where = " OR ".join(clauses)
        query = f"SELECT source_id, target_id, relation, metadata FROM edges WHERE ({where})"
        if relation is not None:
            query += " AND relation = ?"
            params.append(relation)
        cur = self.conn.execute(query, params)
        return [(r[0], r[1], r[2], json.loads(r[3])) for r in cur.fetchall()]

    def edge_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM edges")
        return cur.fetchone()[0]

    # -- misc key/value config store ----------------------------------------

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None
