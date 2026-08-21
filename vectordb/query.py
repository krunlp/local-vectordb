"""
A small, honest subset of Cypher (Neo4j/FalkorDB's query language) over
vectordb's graph functionality.

This is NOT a Cypher implementation in any serious sense -- no query
planner, no indexes beyond what SQLite gives us for free, no aggregations,
no CREATE/MERGE, no full pattern language. It supports exactly one shape
of query, which is nonetheless a genuinely useful and real one:

    MATCH (a)-[:relation_type*min..max]->(b)
    WHERE a.field = 'value' AND b.field > 10 OR a.field = 'other'
    RETURN a, b.field, ...
    ORDER BY b.field DESC
    LIMIT 10

Grammar supported:
  - MATCH (var1)-[:relation]->(var2)          -- exact single hop
  - MATCH (var1)-[:relation*2]->(var2)        -- exactly N hops
  - MATCH (var1)-[:relation*1..3]->(var2)     -- variable-length path, 1-3 hops
  - MATCH (var1)-[]->(var2)                   -- any relation type
  - WHERE with =, !=, >, <, >=, <= comparisons, joined with AND and OR
    (OR is top-level disjunction of AND-groups -- standard DNF, no
    parentheses/nesting support)
  - RETURN var                                 -- full {id, metadata} dict
  - RETURN var.id, var.field                   -- specific fields
  - ORDER BY var.field [ASC|DESC]               -- optional, single field only
  - LIMIT n                                     -- optional

What it deliberately does NOT support (be clear about this rather than
silently failing in a confusing way): parenthesized/nested boolean
expressions, undirected patterns, multiple MATCH clauses, node labels,
aggregation (COUNT, collect, etc), multi-field ORDER BY,
CREATE/MERGE/SET/DELETE. Unsupported syntax raises QueryError with a
specific reason rather than silently returning wrong or partial results.

Performance note: if every OR-group constrains the start variable's `id`
field with `=`, the query starts from just those id(s) and traverses
outward (same cost as calling traverse() directly). Otherwise, every
active record in the DB is scanned as a candidate start node -- a real
full-database scan, same cost category as an unindexed Cypher MATCH,
and should be expected to be slow on a large DB.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

_ComparisonOp = str  # one of '=', '!=', '>', '<', '>=', '<='


class QueryError(Exception):
    pass


_MATCH_RE = re.compile(
    r"MATCH\s*\(\s*(?P<a>\w+)\s*\)\s*-\[\s*:?\s*(?P<rel>\w+)?\s*"
    r"(?:\*\s*(?P<n1>\d+)\s*(?:\.\.\s*(?P<n2>\d+))?)?\s*\]->\s*\(\s*(?P<b>\w+)\s*\)",
    re.IGNORECASE,
)
_WHERE_RE = re.compile(
    r"WHERE\s+(?P<body>.+?)(?=\sRETURN\b|\Z)", re.IGNORECASE | re.DOTALL
)
_RETURN_RE = re.compile(r"RETURN\s+(?P<body>.+)$", re.IGNORECASE | re.DOTALL)
_ORDER_BY_RE = re.compile(
    r"ORDER\s+BY\s+(?P<var>\w+)\.(?P<field>\w+)\s*(?P<dir>ASC|DESC)?\b", re.IGNORECASE
)
_LIMIT_RE = re.compile(r"LIMIT\s+(?P<n>\d+)", re.IGNORECASE)
# Longer operators (>=, <=, !=) must be tried before their shorter prefixes (>, <, =).
_CONDITION_RE = re.compile(
    r"(?P<var>\w+)\.(?P<field>\w+)\s*(?P<op>>=|<=|!=|=|>|<)\s*"
    r"(?:'(?P<sval>[^']*)'|(?P<nval>[-\w.]+))"
)

Condition = Tuple[str, str, _ComparisonOp, Any]  # (var, field, op, value)


def _parse_where(where_body: str) -> List[List[Condition]]:
    """Returns a list of AND-groups (each a list of Conditions); the
    overall WHERE matches if ANY group's conditions ALL hold (OR-of-ANDs,
    i.e. disjunctive normal form -- no nested/parenthesized expressions)."""
    or_groups: List[List[Condition]] = []
    for group_text in re.split(r"\bOR\b", where_body, flags=re.IGNORECASE):
        group_text = group_text.strip()
        conditions: List[Condition] = []
        for clause in re.split(r"\bAND\b", group_text, flags=re.IGNORECASE):
            clause = clause.strip()
            if not clause:
                continue
            m = _CONDITION_RE.fullmatch(clause)
            if not m:
                raise QueryError(
                    f"Unsupported WHERE condition: {clause!r} -- expected "
                    "'var.field OP value' with OP one of =, !=, >, <, >=, <="
                )
            var, field, op = m.group("var"), m.group("field"), m.group("op")
            value = m.group("sval") if m.group("sval") is not None else m.group("nval")
            conditions.append((var, field, op, value))
        or_groups.append(conditions)
    return or_groups


def _compare(actual: Any, op: _ComparisonOp, expected: Any) -> bool:
    if op == "=":
        return str(actual) == str(expected)
    if op == "!=":
        return str(actual) != str(expected)
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        a, e = str(actual), str(expected)
    if op == ">":
        return a > e
    if op == "<":
        return a < e
    if op == ">=":
        return a >= e
    if op == "<=":
        return a <= e
    raise QueryError(f"Unsupported operator: {op!r}")


def _condition_matches(var_values: Dict[str, Tuple[str, Optional[Dict[str, Any]]]], cond: Condition) -> bool:
    var, field, op, expected = cond
    if var not in var_values:
        return False
    node_id, meta = var_values[var]
    actual = node_id if field == "id" else (meta or {}).get(field)
    if actual is None and field != "id":
        return False
    return _compare(actual, op, expected)


def _where_matches(
    or_groups: List[List[Condition]],
    var_values: Dict[str, Tuple[str, Optional[Dict[str, Any]]]],
) -> bool:
    if not or_groups or all(not group for group in or_groups):
        return True
    return any(
        all(_condition_matches(var_values, cond) for cond in group)
        for group in or_groups
    )


def _definite_equality_ids(or_groups: List[List[Condition]], var: str) -> Optional[List[str]]:
    """If EVERY OR-group constrains var.id with '=', return those id
    values (a safe, definite starting-id set). Otherwise None -> caller
    must fall back to a full scan."""
    if not or_groups:
        return None
    ids = []
    for group in or_groups:
        group_ids = [v for (gv, f, op, v) in group if gv == var and f == "id" and op == "="]
        if not group_ids:
            return None
        ids.extend(group_ids)
    return ids


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
    or_groups = _parse_where(where_m.group("body")) if where_m else []

    return_m = _RETURN_RE.search(query)
    if not return_m:
        raise QueryError("Query must have a RETURN clause")
    return_body = return_m.group("body")
    order_m = _ORDER_BY_RE.search(return_body)
    limit_m = _LIMIT_RE.search(return_body)
    return_body = _ORDER_BY_RE.sub("", return_body)
    return_body = _LIMIT_RE.sub("", return_body)
    return_items = _parse_return(return_body)

    order_by = None
    if order_m:
        order_by = {
            "var": order_m.group("var"),
            "field": order_m.group("field"),
            "desc": (order_m.group("dir") or "ASC").upper() == "DESC",
        }
    limit = int(limit_m.group("n")) if limit_m else None

    known_vars = {var_a, var_b}
    for group in or_groups:
        for var, _field, _op, _value in group:
            if var not in known_vars:
                raise QueryError(f"WHERE references unknown variable {var!r} (not in MATCH pattern)")
    for var, _field in return_items:
        if var not in known_vars:
            raise QueryError(f"RETURN references unknown variable {var!r} (not in MATCH pattern)")
    if order_by and order_by["var"] not in known_vars:
        raise QueryError(f"ORDER BY references unknown variable {order_by['var']!r}")

    return {
        "var_a": var_a, "var_b": var_b, "relation": relation,
        "min_hops": min_hops, "max_hops": max_hops,
        "or_groups": or_groups, "return_items": return_items,
        "order_by": order_by, "limit": limit,
    }


def run_query(db, query: str, max_nodes: int = 1000) -> Dict[str, Any]:
    """Execute a query string (see module docstring for grammar) against a
    VectorDB's graph. Returns {"rows": [...], "truncated": bool}."""
    parsed = parse_query(query)
    var_a, var_b = parsed["var_a"], parsed["var_b"]
    or_groups = parsed["or_groups"]

    start_ids = _definite_equality_ids(or_groups, var_a)
    if start_ids is None:
        all_labels = db.store.all_active_labels()
        start_ids = []
        for label in all_labels:
            rec = db.store.get_by_label(label)
            if rec is None:
                continue
            node_id, _meta = rec
            start_ids.append(node_id)

    rows = []
    any_truncated = False
    for start_id in start_ids:
        label_a = db.store.get_label(start_id)
        meta_a = None
        if label_a is not None:
            rec_a = db.store.get_by_label(label_a)
            if rec_a is not None:
                meta_a = rec_a[1]

        traversal = db.traverse(
            start_id, max_depth=parsed["max_hops"], relation=parsed["relation"],
            direction="out", max_nodes=max_nodes,
        )
        if traversal.get("truncated"):
            any_truncated = True

        for node_id, depth in traversal["nodes"].items():
            if depth < parsed["min_hops"] or depth > parsed["max_hops"]:
                continue
            if node_id == start_id and parsed["min_hops"] > 0:
                continue

            label_b = db.store.get_label(node_id)
            meta_b = None
            if label_b is not None:
                rec_b = db.store.get_by_label(label_b)
                if rec_b is not None:
                    meta_b = rec_b[1]

            var_values = {var_a: (start_id, meta_a), var_b: (node_id, meta_b)}
            if not _where_matches(or_groups, var_values):
                continue

            row: Dict[str, Any] = {}
            for var, field in parsed["return_items"]:
                node_id_val, meta_val = var_values[var]
                if field is None:
                    row[var] = {"id": node_id_val, "metadata": meta_val}
                elif field == "id":
                    row[f"{var}.id"] = node_id_val
                else:
                    row[f"{var}.{field}"] = (meta_val or {}).get(field)
            rows.append((row, var_values))

    order_by = parsed["order_by"]
    if order_by:
        def get_val(item):
            _row, var_values = item
            node_id, meta = var_values.get(order_by["var"], (None, None))
            return node_id if order_by["field"] == "id" else (meta or {}).get(order_by["field"])
        # Keep None values at the end regardless of ASC/DESC -- if reverse
        # were just applied to a (is_none, val) tuple key directly, DESC
        # would invert the None-last convention too (None would jump to the
        # front), which is surprising and inconsistent with SQL/Cypher's
        # usual NULLS LAST behavior. Sort non-None and None separately instead.
        with_val = [item for item in rows if get_val(item) is not None]
        without_val = [item for item in rows if get_val(item) is None]
        with_val.sort(key=get_val, reverse=order_by["desc"])
        rows = with_val + without_val

    result_rows = [r for r, _vv in rows]
    if parsed["limit"] is not None:
        result_rows = result_rows[: parsed["limit"]]

    return {"rows": result_rows, "truncated": any_truncated}
