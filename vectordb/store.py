"""
SQLite-backed metadata and ID-mapping store for the vector DB.

HNSW (via hnswlib) only understands integer labels, so this store keeps:
  - a mapping between user-facing string IDs and internal integer labels
  - arbitrary JSON metadata per record
  - a free-list of reusable integer labels (for after deletes)
"""
import json
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
        return self._local.conn

    def _init_schema(self):
        conn = sqlite3.connect(self.path)
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
        self.conn.execute(
            """
            INSERT INTO records (label, id, metadata, deleted)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(label) DO UPDATE SET
                id=excluded.id, metadata=excluded.metadata, deleted=0
            """,
            (label, id_, json.dumps(metadata)),
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
        label = self.get_label(id_)
        if label is None:
            return None
        self.conn.execute(
            "UPDATE records SET deleted = 1 WHERE label = ?", (label,)
        )
        self.conn.commit()
        return label

    def count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM records WHERE deleted = 0")
        return cur.fetchone()[0]

    def all_active_labels(self) -> List[int]:
        cur = self.conn.execute("SELECT label FROM records WHERE deleted = 0")
        return [r[0] for r in cur.fetchall()]

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
