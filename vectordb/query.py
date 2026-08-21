"""
A small, honest subset of Cypher (Neo4j/FalkorDB's query language) over
vectordb's graph functionality.

This is NOT a Cypher implementation in any serious sense -- no query
planner, no indexes beyond what SQLite gives us for free, no aggregations,
no CREATE/MERGE, no full pattern language. It supports exactly one shape
of query, which is nonetheless a genuinely useful and real one:

    MATCH (a)-[:relation_type*min..max]->(b)
    WHERE a.field = 'value' AND b.field = 'value'
    RETURN a, b.field, ...

Grammar supported:
  - MATCH (var1)-[:relation]->(var2)          -- exact single hop
  - MATCH (var1)-[:relation*2]->(var2)        -- exactly N hops
  - MATCH (var1)-[:relation*1..3]->(var2)     -- variable-length path, 1-3 hops
  - MATCH (var1)-[]->(var2)                   -- any relation type
  - WHERE var.id = 'x' AND var.field = 'y'    -- exact-match only, AND-only
  - RETURN var                                 -- full {id, metadata} dict
  - RETURN var.id, var.field                   -- specific fields

What it deliberately does NOT support (be clear about this rather than
silently failing in a confusing way): OR conditions, inequality/range
conditions, undirected patterns, multiple MATCH clauses, node labels,
aggregation (COUNT, collect, etc), ORDER BY/LIMIT, CREATE/MERGE/SET/DELETE.
Unsupported syntax raises QueryError with a specific reason rather than
silently returning wrong or partial results.

Performance note: if WHERE constrains the start variable's `id` field, the
query starts from that single id and traverses outward (same cost as
calling traverse() directly). If it does NOT, every active record in the
DB is scanned as a candidate start node -- this is a real full-database
scan, same category of cost as an index-free Cypher MATCH would be
without an index, and should be expected to be slow on a large DB.
"""
import re
from typing import Any, Dict, List, Optional, Tuple


class QueryError(Exception):
    pass


_MATCH_RE = re.compile(
    r"MATCH\s*\(\s*(?P<a>\w+)\s*\)\s*-\[\s*:?\s*(?P<rel>\w+)?\s*"
    r"(?:\*\s*(?P<n1>\d+)\s*(?:\.\.\s*(?P<n2>\d+))?)?\s*\]->\s*\(\s*(?P<b>\w+)\s*\)",
    re.IGNORECASE,
)
_WHERE_RE = re.compile(r"WHERE\s+(?P<body>.+?)(?=\sRETURN\b|\Z)", re.IGNORECASE | re.DOTALL)
_RETURN_RE = re.compile(r"RETURN\s+(?P<body>.+)$", re.IGNORECASE | re.DOTALL)
_CONDITION_RE = re.compile(
    r"(?P<var>\w+)\.(?P<field>\w+)\s*=\s*(?:'(?P<sval>[^']*)'|(?P<nval>[-\w.]+))"
)


def _parse_conditions(where_body: str) -> Dict[str, Dict[str, Any]]:
    if re.search(r"\bOR\b", where_body, re.IGNORECASE):
        raise QueryError("OR is not supported in WHERE -- only AND-joined equality conditions")
    conditions: Dict[str, Dict[str, Any]] = {}
    for clause in re.split(r"\bAND\b", where_body, flags=re.IGNORECASE):
        clause = clause.strip()
        if not clause:
            continue
        m = _CONDITION_RE.fullmatch(clause)
        if not m:
            raise QueryError(
                f"Unsupported WHERE condition: {clause!r} -- only "
                "'var.field = value' equality is supported"
            )
        var, field = m.group("var"), m.group("field")
        value = m.group("sval") if m.group("sval") is not None else m.group("nval")
        conditions.setdefault(var, {})[field] = value
    return conditions


def _parse_return(return_body: str) -> List[Tuple[str, Optional[str]]]:
    items = []
    for part in return_body.split(","):
        part = part.strip()
        if not part:
            continue
        if "." in part:
            var, field = part.split(".", 1)
            items.append((var.strip(), field.strip()))
        else:
            items.append((part, None))
    return items


def parse_query(query: str) -> Dict[str, Any]:
    """Parse a query string into its structural parts. Raises QueryError
    for anything outside the supported grammar (see module docstring)."""
    m = _MATCH_RE.search(query)
    if not m:
        raise QueryError(
            "No supported MATCH pattern found. Expected shape: "
            "MATCH (a)-[:relation*min..max]->(b)"
        )
    var_a, relation, var_b = m.group("a"), m.group("rel"), m.group("b")
    n1, n2 = m.group("n1"), m.group("n2")
    if n1 is None:
        min_hops, max_hops = 1, 1
    elif n2 is None:
        min_hops = max_hops = int(n1)
    else:
        min_hops, max_hops = int(n1), int(n2)

    where_m = _WHERE_RE.search(query)
    conditions = _parse_conditions(where_m.group("body")) if where_m else {}

    return_m = _RETURN_RE.search(query)
    if not return_m:
        raise QueryError("Query must have a RETURN clause")
    return_items = _parse_return(return_m.group("body"))

    known_vars = {var_a, var_b}
    for var in conditions:
        if var not in known_vars:
            raise QueryError(f"WHERE references unknown variable {var!r} (not in MATCH pattern)")
    for var, _field in return_items:
        if var not in known_vars:
            raise QueryError(f"RETURN references unknown variable {var!r} (not in MATCH pattern)")

    return {
        "var_a": var_a, "var_b": var_b, "relation": relation,
        "min_hops": min_hops, "max_hops": max_hops,
        "conditions": conditions, "return_items": return_items,
    }


def _node_matches(node_id: str, metadata: Optional[Dict[str, Any]], conditions: Dict[str, Any]) -> bool:
    for field, value in conditions.items():
        if field == "id":
            if node_id != value:
                return False
        else:
            if metadata is None or str(metadata.get(field)) != str(value):
                return False
    return True


def run_query(db, query: str) -> List[Dict[str, Any]]:
    """Execute a query string (see module docstring for grammar) against a
    VectorDB's graph. Returns a list of result rows, each a dict keyed by
    the RETURN clause's items (e.g. {"a.id": ..., "b.title": ...} or
    {"a": {"id":..., "metadata":...}} for a bare variable)."""
    parsed = parse_query(query)
    var_a, var_b = parsed["var_a"], parsed["var_b"]
    conditions = parsed["conditions"]
    cond_a = conditions.get(var_a, {})
    cond_b = conditions.get(var_b, {})

    # Determine candidate starting ids for var_a.
    if "id" in cond_a:
        start_ids = [cond_a["id"]]
    else:
        # No id constraint -- full scan of every active record as a
        # candidate start node (see module docstring's performance note).
        all_labels = db.store.all_active_labels()
        start_ids = []
        for label in all_labels:
            rec = db.store.get_by_label(label)
            if rec is None:
                continue
            node_id, meta = rec
            if _node_matches(node_id, meta, cond_a):
                start_ids.append(node_id)

    rows = []
    for start_id in start_ids:
        traversal = db.traverse(
            start_id, max_depth=parsed["max_hops"], relation=parsed["relation"], direction="out",
        )
        for node_id, depth in traversal["nodes"].items():
            if depth < parsed["min_hops"] or depth > parsed["max_hops"]:
                continue
            if node_id == start_id and parsed["min_hops"] > 0:
                continue  # depth 0 is only valid if min_hops is 0
            label = db.store.get_label(node_id)
            meta_b = None
            if label is not None:
                rec = db.store.get_by_label(label)
                if rec is not None:
                    meta_b = rec[1]
            if not _node_matches(node_id, meta_b, cond_b):
                continue

            label_a = db.store.get_label(start_id)
            meta_a = None
            if label_a is not None:
                rec_a = db.store.get_by_label(label_a)
                if rec_a is not None:
                    meta_a = rec_a[1]

            row = {}
            for var, field in parsed["return_items"]:
                node_id_val = start_id if var == var_a else node_id
                meta_val = meta_a if var == var_a else meta_b
                if field is None:
                    row[var] = {"id": node_id_val, "metadata": meta_val}
                elif field == "id":
                    row[f"{var}.id"] = node_id_val
                else:
                    row[f"{var}.{field}"] = (meta_val or {}).get(field)
            rows.append(row)

    return rows
