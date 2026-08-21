"""
Bulk import of graph edges from common external formats, so you don't have
to hand-write a Python loop calling add_edge() for every row of an
existing CSV export or adjacency-list file.
"""
import csv
import json
from typing import Any, Dict, List, Optional


def import_edges_csv(
    db,
    path: str,
    source_col: str = "source",
    target_col: str = "target",
    relation_col: Optional[str] = None,
    default_relation: str = "related",
    metadata_cols: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Bulk-load edges from a CSV file with a header row.

    source_col / target_col: column names holding the endpoint ids.
    relation_col: optional column holding each row's relation type; if
      omitted, every edge uses default_relation.
    metadata_cols: optional list of column names to attach as edge
      metadata (e.g. ["weight"] so shortest_path_weighted can use it).
      Values are parsed as JSON if they look like a number/bool/null,
      otherwise kept as strings (so "0.5" becomes 0.5, "true" becomes
      True, "hello" stays "hello").

    Rows missing source_col or target_col are skipped and counted in
    "skipped" rather than raising, since real-world CSV exports often
    have a handful of malformed rows and one bad row shouldn't abort
    an otherwise-good bulk import.

    Returns {"imported": N, "skipped": N}.
    """
    edges = []
    skipped = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get(source_col)
            target = row.get(target_col)
            if not source or not target:
                skipped += 1
                continue
            relation = (row.get(relation_col) or default_relation) if relation_col else default_relation
            metadata: Dict[str, Any] = {}
            for col in metadata_cols or []:
                if col in row and row[col] not in (None, ""):
                    metadata[col] = _coerce_value(row[col])
            edges.append({"source": source, "target": target, "relation": relation, "metadata": metadata})

    if edges:
        db.add_edges(edges)
    return {"imported": len(edges), "skipped": skipped}


def import_adjacency_list(
    db,
    path: str,
    relation: str = "related",
    delimiter: str = " ",
) -> Dict[str, int]:
    """
    Bulk-load edges from a plain adjacency-list text file, one source node
    per line, followed by its targets:

        node1 node2 node3 node4
        node2 node4
        node3

    (node1 -> node2, node1 -> node3, node1 -> node4, node2 -> node4). This
    is the classic plain-text graph format used by many datasets (e.g.
    SNAP) and command-line graph tools.

    Blank lines and lines starting with '#' (a common comment convention
    in this format) are skipped. Returns {"imported": N}.
    """
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter)
            parts = [p for p in parts if p]  # tolerate repeated delimiters
            if len(parts) < 2:
                continue  # a node with no listed neighbors -- nothing to add
            source = parts[0]
            for target in parts[1:]:
                edges.append({"source": source, "target": target, "relation": relation})

    if edges:
        db.add_edges(edges)
    return {"imported": len(edges)}


def _coerce_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
