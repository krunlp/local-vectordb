"""
Graph algorithms over vectordb's edge store: PageRank, degree centrality,
and (weakly) connected components.

Honest scope: these are pure-Python implementations reading the SQLite
edges table, not vectorized/matrix-native execution (that's the actual
architectural difference from something like FalkorDB's GraphBLAS engine,
and isn't closeable without a different storage engine entirely). What
this DOES close is a real, previously-total gap: this project had no
graph algorithms at all before this module -- only BFS traversal. Verified
against known small graphs with hand-computable expected answers, not
just "it runs without erroring."
"""
from typing import Dict, List, Optional, Set

import numpy as np


def _load_adjacency(db, relation: Optional[str] = None) -> Dict[str, List[str]]:
    """out-adjacency: {node_id: [neighbor_id, ...]} across all active edges,
    including nodes that only ever appear as a target (with an empty list)."""
    adjacency: Dict[str, List[str]] = {}
    with db._lock:
        cur = db.store.conn.execute(
            "SELECT source_id, target_id, relation FROM edges"
            + (" WHERE relation = ?" if relation else "")
            , (relation,) if relation else ()
        )
        rows = cur.fetchall()
    for source, target, _rel in rows:
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])
    return adjacency


def degree_centrality(
    db, relation: Optional[str] = None, direction: str = "both"
) -> Dict[str, int]:
    """Raw degree (edge count) per node -- the simplest centrality measure:
    which nodes are most directly connected. direction: 'out', 'in', or
    'both' (default, counts each edge once per endpoint)."""
    out_deg: Dict[str, int] = {}
    in_deg: Dict[str, int] = {}
    with db._lock:
        cur = db.store.conn.execute(
            "SELECT source_id, target_id FROM edges"
            + (" WHERE relation = ?" if relation else ""),
            (relation,) if relation else (),
        )
        rows = cur.fetchall()
    for source, target in rows:
        out_deg[source] = out_deg.get(source, 0) + 1
        in_deg[target] = in_deg.get(target, 0) + 1
        out_deg.setdefault(target, out_deg.get(target, 0))
        in_deg.setdefault(source, in_deg.get(source, 0))

    if direction == "out":
        return out_deg
    if direction == "in":
        return in_deg
    all_nodes = set(out_deg) | set(in_deg)
    return {n: out_deg.get(n, 0) + in_deg.get(n, 0) for n in all_nodes}


def connected_components(db, relation: Optional[str] = None) -> List[Set[str]]:
    """Weakly connected components (treats edges as undirected for the
    purpose of grouping) via union-find. Returns a list of sets of ids.
    A node with no edges at all is NOT included (nothing to group it
    with) -- this reports graph structure, not the full record set."""
    parent: Dict[str, str] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    with db._lock:
        cur = db.store.conn.execute(
            "SELECT source_id, target_id FROM edges"
            + (" WHERE relation = ?" if relation else ""),
            (relation,) if relation else (),
        )
        rows = cur.fetchall()

    for source, target in rows:
        parent.setdefault(source, source)
        parent.setdefault(target, target)
        union(source, target)

    groups: Dict[str, Set[str]] = {}
    for node in parent:
        root = find(node)
        groups.setdefault(root, set()).add(node)
    return list(groups.values())


def pagerank(
    db,
    relation: Optional[str] = None,
    damping: float = 0.85,
    iterations: int = 50,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    PageRank via power iteration over the (directed) edge graph. Standard
    formulation: rank(n) = (1-d)/N + d * sum(rank(m)/outdeg(m) for m linking to n).
    Dangling nodes (no outgoing edges) redistribute their rank uniformly,
    the standard fix to keep total rank conserved (otherwise rank leaks
    out of the system at every node with no outgoing edges).

    Returns {id: score, ...} summing to ~1.0 across all nodes that appear
    in at least one edge. Converges when the L1 change between iterations
    drops below tol, or after `iterations` steps, whichever comes first.
    """
    adjacency = _load_adjacency(db, relation=relation)
    nodes = list(adjacency.keys())
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    outdeg = np.array([len(adjacency[node]) for node in nodes], dtype=np.float64)

    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        new_rank = np.full(n, (1 - damping) / n)
        dangling_mass = rank[outdeg == 0].sum()
        new_rank += damping * dangling_mass / n  # redistribute dangling rank uniformly
        for node in nodes:
            i = index[node]
            if outdeg[i] == 0:
                continue
            share = damping * rank[i] / outdeg[i]
            for neighbor in adjacency[node]:
                new_rank[index[neighbor]] += share
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank

    return {node: float(rank[index[node]]) for node in nodes}
