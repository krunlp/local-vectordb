import os
import shutil
import numpy as np
from vectordb import VectorDB

np.random.seed(0)
DIM = 8

def rand_vec():
    v = np.random.randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)

shutil.rmtree("/tmp/testdb", ignore_errors=True)

# 1. Basic add + search + persistence -----------------------------------
db = VectorDB("/tmp/testdb", dim=DIM, space="cosine")

docs = {
    "doc1": ("The cat sat on the mat", {"category": "animals"}),
    "doc2": ("Dogs are loyal companions", {"category": "animals"}),
    "doc3": ("Stock markets fell today", {"category": "finance"}),
    "doc4": ("Cats and dogs are common pets", {"category": "animals"}),
    "doc5": ("Interest rates rose sharply", {"category": "finance"}),
}
vecs = {k: rand_vec() for k in docs}
for k, (text, meta) in docs.items():
    db.add(k, vecs[k], {**meta, "text": text})

assert db.count() == 5, db.count()

# search
query = vecs["doc1"] * 0.9 + rand_vec() * 0.1
query = query / np.linalg.norm(query)
results = db.search(query, k=3)
print("Unfiltered search:", [(r["id"], round(r["score"], 3)) for r in results])
assert len(results) == 3

# metadata filter
results_f = db.search(query, k=3, filter={"category": "finance"})
print("Filtered (finance):", [(r["id"], r["metadata"]["category"]) for r in results_f])
assert all(r["metadata"]["category"] == "finance" for r in results_f)

# update metadata
db.update_metadata("doc1", {"category": "animals", "text": "updated text"})
assert db.get("doc1")["metadata"]["text"] == "updated text"

# delete + recheck
db.delete("doc2")
assert db.count() == 4
results_after_delete = db.search(query, k=5)
assert all(r["id"] != "doc2" for r in results_after_delete)

# persistence round-trip
db.save()
del db
db2 = VectorDB("/tmp/testdb")  # dim/space inferred from disk
assert db2.count() == 4
assert db2.dim == DIM
results2 = db2.search(query, k=3)
print("After reload:", [(r["id"], round(r["score"], 3)) for r in results2])

# compaction
db2.compact()
assert db2.count() == 4

# 2. Recommendations-style: find items similar to a given item's vector ---
rec = db2.get("doc4")
recs = db2.search(vecs["doc4"], k=3)
print("Recs for doc4:", [(r["id"], round(r["score"], 3)) for r in recs])

# 3. Dedup: near-duplicate detection via distance threshold ---------------
near_dup_vec = vecs["doc3"] + np.random.randn(DIM).astype(np.float32) * 0.01
near_dup_vec = near_dup_vec / np.linalg.norm(near_dup_vec)
dup_check = db2.search(near_dup_vec, k=1)
print("Nearest to near-duplicate of doc3:", dup_check)
assert dup_check[0]["id"] == "doc3"
assert dup_check[0]["score"] < 0.01  # very close in cosine distance

# 4. Hybrid search (semantic + keyword, via a mock embedder) --------------
class MockEmbedder:
    dim = DIM
    def encode(self, texts):
        out = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % (2**32))
            v = rng.randn(self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)

shutil.rmtree("/tmp/testdb_hybrid", ignore_errors=True)
db3 = VectorDB("/tmp/testdb_hybrid", embedder=MockEmbedder())
db3.add_text("h1", "The quick brown fox jumps over the lazy dog")
db3.add_text("h2", "SKU-4471-X is out of stock until next month")
hybrid_results = db3.hybrid_search("SKU-4471-X", k=2)
print("Hybrid search:", [(r["id"], round(r["score"], 4)) for r in hybrid_results])
assert hybrid_results[0]["id"] == "h2"  # exact keyword match should win

# 5. Crash recovery: compact() must self-heal an index/metadata mismatch
#    caused by unsaved writes before a crash, not raise an error.
import subprocess, sys
shutil.rmtree("/tmp/testdb_crash", ignore_errors=True)
crash_script = '''
import numpy as np
from vectordb import VectorDB
db = VectorDB("/tmp/testdb_crash", dim=4, max_elements=100)
db.add(["a0","a1"], np.random.randn(2,4).astype(np.float32))
db.save()
db.add(["b0"], np.random.randn(1,4).astype(np.float32))
# exits without saving -- simulates a crash
'''
subprocess.run([sys.executable, "-c", crash_script], check=True)
db4 = VectorDB("/tmp/testdb_crash")
assert db4.get("b0") is not None  # metadata survived (SQLite commits every write)
result = db4.compact()
print("compact() after simulated crash:", result)
assert result == {"kept": 2, "dropped": 1}
assert db4.get("b0") is None  # orphaned record cleaned up
assert db4.get("a0") is not None  # real (saved) record survives

# 6. Capacity auto-growth: force multiple resize_index() calls and verify
#    zero data corruption across them (undersized on purpose).
shutil.rmtree("/tmp/testdb_resize", ignore_errors=True)
db5 = VectorDB("/tmp/testdb_resize", dim=4, max_elements=10)  # tiny on purpose
resize_vecs = {}
for batch in range(5):
    ids = [f"r{batch}_{i}" for i in range(15)]  # exceeds current capacity each time
    vecs = np.random.randn(15, 4).astype(np.float32)
    db5.add(ids, vecs)
    for id_, v in zip(ids, vecs):
        resize_vecs[id_] = v
assert db5.count() == 75
for id_, v in list(resize_vecs.items())[:15]:
    r = db5.search(v, k=1)
    assert r[0]["id"] == id_, f"corruption after resize: {id_} -> {r[0]['id']}"
print("Resize/capacity-growth: 5 forced resizes, zero corruption")

# 7. OKF (Open Knowledge Format) bundle ingestion -- self-contained synthetic
#    bundle so this test doesn't depend on cloning an external repo.
from vectordb.okf import ingest_okf_bundle, load_bundle
shutil.rmtree("/tmp/testdb_okf_bundle", ignore_errors=True)
os.makedirs("/tmp/testdb_okf_bundle/tables", exist_ok=True)
with open("/tmp/testdb_okf_bundle/index.md", "w") as f:
    f.write("# Bundle\n\n* [users](tables/users.md)\n")  # reserved, must be skipped
with open("/tmp/testdb_okf_bundle/tables/users.md", "w") as f:
    f.write(
        "---\ntype: BigQuery Table\ntitle: Users\ndescription: User accounts\n"
        "tags: [users, accounts]\n---\n\nContains user account records. "
        "See [orders](/tables/orders.md) for purchase history.\n"
    )
with open("/tmp/testdb_okf_bundle/tables/orders.md", "w") as f:
    f.write(
        "---\ntype: BigQuery Table\ntitle: Orders\nstatus: stable\n"
        "stale_after: 2026-12-31\n---\n\nContains order records.\n"
    )
with open("/tmp/testdb_okf_bundle/tables/legacy_orders.md", "w") as f:
    f.write(
        "---\ntype: BigQuery Table\ntitle: Legacy Orders\nstatus: deprecated\n---\n\n"
        "Superseded by orders.md.\n"
    )

concepts = load_bundle("/tmp/testdb_okf_bundle")
assert len(concepts) == 3  # index.md correctly excluded
ids = {c.concept_id for c in concepts}
assert ids == {"tables/users", "tables/orders", "tables/legacy_orders"}

shutil.rmtree("/tmp/testdb_okf_ingest", ignore_errors=True)
db6 = VectorDB("/tmp/testdb_okf_ingest", embedder=MockEmbedder())
result = ingest_okf_bundle(db6, "/tmp/testdb_okf_bundle")
print("OKF ingest result:", result)
assert result == {"indexed": 2, "skipped_deprecated": 1, "skipped_malformed": 0, "edges_added": 1}
assert db6.get("tables/legacy_orders") is None  # deprecated, correctly skipped
orders = db6.get("tables/orders")
assert orders["metadata"]["stale_after"] == "2026-12-31"  # date -> string, JSON-safe
users = db6.get("tables/users")
assert "/tables/orders.md" in users["metadata"]["links"]  # cross-link extracted
r = db6.hybrid_search("user accounts", k=2)
assert "tables/users" in [x["id"] for x in r]
print("OKF bundle ingestion: parsing, deprecated-skip, cross-links, search all verified")

# 8. OKF bundle GENERATION (documents -> OKF), and round-trip it back
#    through our own reader to confirm it's actually spec-conformant.
from vectordb.okf_generate import documents_to_okf
shutil.rmtree("/tmp/testdb_okf_gen_src", ignore_errors=True)
shutil.rmtree("/tmp/testdb_okf_gen_out", ignore_errors=True)
os.makedirs("/tmp/testdb_okf_gen_src/sub", exist_ok=True)
with open("/tmp/testdb_okf_gen_src/plain.txt", "w") as f:
    f.write("Plain Doc Title\n\nSome body text about this topic, spanning a couple "
            "of sentences. It should end up as the description.\n")
with open("/tmp/testdb_okf_gen_src/sub/nested.md", "w") as f:
    f.write("# Nested Concept\n\nBody content for the nested concept.\n")

gen_result = documents_to_okf("/tmp/testdb_okf_gen_src", "/tmp/testdb_okf_gen_out", default_type="Document")
print("Generation result:", gen_result)
assert gen_result == {"generated": 2}

# round-trip: our own reader must parse what our own generator wrote
roundtrip = load_bundle("/tmp/testdb_okf_gen_out")
assert len(roundtrip) == 2
rt_ids = {c.concept_id for c in roundtrip}
assert rt_ids == {"plain", "sub/nested"}
plain = next(c for c in roundtrip if c.concept_id == "plain")
assert plain.title == "Plain Doc Title"
assert plain.type == "Document"
assert "description" not in plain.title  # sanity: title not duplicated into itself
nested = next(c for c in roundtrip if c.concept_id == "sub/nested")
assert nested.title == "Nested Concept"  # pulled from markdown heading

# index.md files must exist and must themselves be excluded as concepts
assert os.path.exists("/tmp/testdb_okf_gen_out/index.md")
assert os.path.exists("/tmp/testdb_okf_gen_out/sub/index.md")
assert os.path.exists("/tmp/testdb_okf_gen_out/log.md")
print("OKF bundle generation: round-trips cleanly through our own reader")

# 9. Security: path traversal via a caller-supplied id must be rejected,
#    not silently write outside output_dir. (This was a real bug, found
#    and fixed after testing it against an adversarial input.)
shutil.rmtree("/tmp/testdb_okf_traversal", ignore_errors=True)
for bad_id in ["../../etc/evil", "/etc/passwd", "..", "a/../../b", "a//b"]:
    try:
        documents_to_okf(
            [{"text": "x", "id": bad_id, "type": "Document"}],
            "/tmp/testdb_okf_traversal",
        )
        raise AssertionError(f"path traversal id {bad_id!r} was NOT rejected")
    except ValueError:
        pass
assert not os.path.exists("/etc/evil.md")
print("OKF generation: path-traversal ids correctly rejected")

# 10. Security: OKF bundle loading must not follow symlinks pointing
#     outside the bundle (file-exfiltration risk from an untrusted bundle),
#     but must still correctly follow symlinks that stay inside it.
shutil.rmtree("/tmp/testdb_okf_symlink", ignore_errors=True)
shutil.rmtree("/tmp/testdb_okf_symlink_target", ignore_errors=True)
os.makedirs("/tmp/testdb_okf_symlink")
os.makedirs("/tmp/testdb_okf_symlink_target")
with open("/tmp/testdb_okf_symlink_target/outside.md", "w") as f:
    f.write("---\ntype: Document\n---\n\nSHOULD-NOT-BE-READABLE\n")
os.symlink("/tmp/testdb_okf_symlink_target/outside.md", "/tmp/testdb_okf_symlink/looks-normal.md")
with open("/tmp/testdb_okf_symlink/real.md", "w") as f:
    f.write("---\ntype: Document\ntitle: Real\n---\n\nreal content\n")
os.symlink("/tmp/testdb_okf_symlink/real.md", "/tmp/testdb_okf_symlink/internal-alias.md")

external_concepts = load_bundle("/tmp/testdb_okf_symlink")
external_ids = {c.concept_id for c in external_concepts}
assert "looks-normal" not in external_ids, "SYMLINK ESCAPED BUNDLE -- file exfiltration risk"
assert "real" in external_ids and "internal-alias" in external_ids  # legit internal symlinks still work
print("OKF loading: external symlinks blocked, internal symlinks still work")

# 11. Robustness: a pathologically deep (but syntactically valid) YAML
#     frontmatter must be skipped as malformed, not crash the whole
#     bundle load and take every OTHER concept down with it.
shutil.rmtree("/tmp/testdb_okf_recursion", ignore_errors=True)
os.makedirs("/tmp/testdb_okf_recursion")
depth = 3000
nested_yaml = "[" * depth + "1" + "]" * depth
with open("/tmp/testdb_okf_recursion/malicious.md", "w") as f:
    f.write(f"---\ntype: Document\nnested: {nested_yaml}\n---\n\nbody\n")
with open("/tmp/testdb_okf_recursion/sibling.md", "w") as f:
    f.write("---\ntype: Document\ntitle: Sibling\n---\n\nfine\n")

recursion_concepts = load_bundle("/tmp/testdb_okf_recursion")
assert len(recursion_concepts) == 1
assert recursion_concepts[0].concept_id == "sibling"
print("OKF loading: pathological YAML depth skipped without crashing sibling concepts")

# 12. Graph functionality: edges, neighbors, traversal, shortest path,
#     graph_search, and edge cleanup on delete.
shutil.rmtree("/tmp/testdb_graph", ignore_errors=True)
db7 = VectorDB("/tmp/testdb_graph", dim=4)
graph_vecs = {}
for gid in ["g_a", "g_b", "g_c", "g_d", "g_e"]:
    v = np.random.randn(4).astype(np.float32)
    graph_vecs[gid] = v
    db7.add(gid, v, {"name": gid})
db7.add_edge("g_a", "g_b", relation="links_to")
db7.add_edge("g_b", "g_c", relation="links_to")
db7.add_edge("g_c", "g_d", relation="links_to")
db7.add_edge("g_a", "g_e", relation="mentions")
db7.save()

assert db7.edge_count() == 4
assert set(db7.get_neighbors("g_a", direction="out")) == {"g_b", "g_e"}
assert db7.get_neighbors("g_a", relation="links_to") == ["g_b"]

path = db7.shortest_path("g_a", "g_d")
assert path == ["g_a", "g_b", "g_c", "g_d"]

trav = db7.traverse("g_a", max_depth=3)
assert trav["nodes"]["g_d"] == 3
assert trav["nodes"]["g_e"] == 1

# dangling edge: target has no record, must not crash, metadata is None
db7.add_edge("g_a", "nonexistent", relation="mentions")
dangling = [n for n in db7.get_neighbors("g_a", direction="out", with_metadata=True) if n["id"] == "nonexistent"]
assert len(dangling) == 1 and dangling[0]["metadata"] is None

# graph_search: vector hit + graph-expanded connected nodes
gs = db7.graph_search(graph_vecs["g_a"], k=1, expand_hops=2)
gs_ids = {r["id"] for r in gs}
assert "g_a" in gs_ids and "g_b" in gs_ids and "g_c" in gs_ids
assert [r for r in gs if r["id"] == "g_a"][0]["via"] == "vector"
assert [r for r in gs if r["id"] == "g_b"][0]["via"] == "graph"

# delete must clean up edges touching the deleted id
db7.delete("g_c")
assert "g_c" not in db7.get_neighbors("g_b", direction="out")
print("Graph: edges, neighbors, traversal, shortest_path, graph_search, delete-cleanup all verified")

# 12b. Weighted shortest path (Dijkstra) -- must genuinely diverge from
# unweighted BFS shortest_path when a "shorter in hops" path is more
# expensive in total weight. Cross-checked against networkx.
shutil.rmtree("/tmp/testdb_dijkstra", ignore_errors=True)
db7b = VectorDB("/tmp/testdb_dijkstra", dim=4)
for did in ["ds", "da", "db_", "dt"]:
    db7b.add(did, np.random.randn(4).astype(np.float32))
db7b.add_edge("ds", "dt", metadata={"weight": 100})
db7b.add_edge("ds", "da", metadata={"weight": 1})
db7b.add_edge("da", "db_", metadata={"weight": 1})
db7b.add_edge("db_", "dt", metadata={"weight": 1})

unweighted = db7b.shortest_path("ds", "dt")
assert unweighted == ["ds", "dt"]  # 1-hop, ignores weight
weighted = db7b.shortest_path_weighted("ds", "dt")
assert weighted["path"] == ["ds", "da", "db_", "dt"]
assert weighted["total_weight"] == 3.0

try:
    import networkx as nx
    G = nx.Graph()
    G.add_edge("ds", "dt", weight=100)
    G.add_edge("ds", "da", weight=1)
    G.add_edge("da", "db_", weight=1)
    G.add_edge("db_", "dt", weight=1)
    nx_path = nx.shortest_path(G, "ds", "dt", weight="weight")
    assert weighted["path"] == nx_path
    print("Weighted shortest_path (Dijkstra): matches networkx, correctly diverges from unweighted BFS")
except ImportError:
    print("Weighted shortest_path (Dijkstra): correct (networkx not available for cross-check)")

# unreachable and negative-weight cases
db7b.add("d_isolated", np.random.randn(4).astype(np.float32))
assert db7b.shortest_path_weighted("ds", "d_isolated") is None
db7b.add_edge("ds", "d_isolated", metadata={"weight": -1})
try:
    db7b.shortest_path_weighted("ds", "d_isolated")
    raise AssertionError("should have rejected negative edge weight")
except ValueError:
    pass
print("Weighted shortest_path: unreachable and negative-weight cases handled correctly")

# 13. Regression: search() must degrade gracefully (not crash with a
#     cryptic hnswlib error) when the metadata store and HNSW index
#     disagree on count -- e.g. from the same unsaved-write scenario
#     tested in #5, exercised through search() specifically this time.
shutil.rmtree("/tmp/testdb_search_mismatch", ignore_errors=True)
mismatch_script = '''
import numpy as np
from vectordb import VectorDB
db = VectorDB("/tmp/testdb_search_mismatch", dim=4, max_elements=100)
db.add(["m0"], np.random.randn(1,4).astype(np.float32))
# exits without saving
'''
subprocess.run([sys.executable, "-c", mismatch_script], check=True)
db8 = VectorDB("/tmp/testdb_search_mismatch")  # metadata says count=1, index has 0
r = db8.search(np.random.randn(4).astype(np.float32), k=5)
assert r == []  # must degrade gracefully, not raise
print("search() robustness: index/metadata count mismatch handled gracefully")

# 14. compact() must clean up edges for records it drops (crash-recovery
#     orphans), matching delete()'s edge cleanup -- otherwise a
#     permanently-dropped id leaves a permanently-dangling edge behind.
shutil.rmtree("/tmp/testdb_compact_edges", ignore_errors=True)
compact_edge_script = '''
import numpy as np
from vectordb import VectorDB
db = VectorDB("/tmp/testdb_compact_edges", dim=4, max_elements=100)
db.add(["ce_a", "ce_b"], np.random.randn(2,4).astype(np.float32))
db.save()
db.add(["ce_c"], np.random.randn(1,4).astype(np.float32))
db.add_edge("ce_a", "ce_c", relation="references")  # will be orphaned by the "crash"
db.add_edge("ce_a", "ce_b", relation="references")  # must survive
'''
subprocess.run([sys.executable, "-c", compact_edge_script], check=True)
db9 = VectorDB("/tmp/testdb_compact_edges")
compact_result = db9.compact()
assert compact_result == {"kept": 2, "dropped": 1}
neighbors_after_compact = db9.get_neighbors("ce_a", direction="out")
assert "ce_c" not in neighbors_after_compact  # stale edge cleaned up
assert "ce_b" in neighbors_after_compact  # legitimate edge preserved
print("compact(): edges to dropped (crash-orphaned) records cleaned up correctly")

# 15. Query language (small Cypher-like subset over the graph).
from vectordb.query import run_query, QueryError
shutil.rmtree("/tmp/testdb_query", ignore_errors=True)
db10 = VectorDB("/tmp/testdb_query", dim=4)
q_docs = {
    "q_policy": {"type": "Policy", "title": "Revenue Policy"},
    "q_metric": {"type": "Metric", "title": "Revenue"},
    "q_calc": {"type": "Computation", "title": "Revenue YTD"},
}
for qid, qmeta in q_docs.items():
    db10.add(qid, np.random.randn(4).astype(np.float32), qmeta)
db10.add_edge("q_policy", "q_metric", relation="references")
db10.add_edge("q_metric", "q_calc", relation="references")

r = run_query(db10, "MATCH (a)-[:references]->(b) WHERE a.id = 'q_policy' RETURN b.id, b.title")
assert r == {"rows": [{"b.id": "q_metric", "b.title": "Revenue"}], "truncated": False}

r2 = run_query(db10, "MATCH (a)-[:references*1..2]->(b) WHERE a.id = 'q_policy' RETURN b.id")
assert {row["b.id"] for row in r2["rows"]} == {"q_metric", "q_calc"}
assert r2["truncated"] == False

r3 = run_query(db10, "MATCH (a)-[:references*1..2]->(b) WHERE a.id = 'q_policy' AND b.type = 'Computation' RETURN b.id")
assert r3["rows"] == [{"b.id": "q_calc"}]

r4 = run_query(db10, "MATCH (a)-[:references]->(b) WHERE a.type = 'Policy' RETURN a.title, b.title")
assert r4["rows"] == [{"a.title": "Revenue Policy", "b.title": "Revenue"}]  # full-scan start (no a.id constraint)

no_match = run_query(db10, "MATCH (a)-[:references]->(b) WHERE a.id = 'nonexistent' RETURN b")
assert no_match == {"rows": [], "truncated": False}

for bad_query in [
    "MATCH (a)-[:references]->(b) WHERE a.id = 'x' AND (b.id = 'y' OR b.id = 'z') RETURN a",  # nested parens unsupported
    "MATCH (a)-[:references]->(b) WHERE c.id = 'x' RETURN a",
    "WHERE a.id = 'x' RETURN a",
    "MATCH (a)-[:references]->(b) WHERE a.id = 'x'",
]:
    try:
        run_query(db10, bad_query)
        raise AssertionError(f"should have raised QueryError: {bad_query!r}")
    except QueryError:
        pass

# truncation must propagate through run_query, not just traverse() itself --
# this was the actual gap found: run_query previously ignored traverse()'s
# own truncation flag entirely. Test it directly with a small max_nodes.
shutil.rmtree("/tmp/testdb_query_truncation", ignore_errors=True)
db11 = VectorDB("/tmp/testdb_query_truncation", dim=4, max_elements=200)
db11.add(["hub"] + [f"leaf{i}" for i in range(100)], np.random.randn(101, 4).astype(np.float32))
db11.add_edges([{"source": "hub", "target": f"leaf{i}", "relation": "r"} for i in range(100)])

capped = run_query(db11, "MATCH (a)-[:r]->(b) WHERE a.id = 'hub' RETURN b.id", max_nodes=10)
assert capped["truncated"] == True
assert len(capped["rows"]) < 100  # genuinely incomplete, and now honestly labeled as such

uncapped = run_query(db11, "MATCH (a)-[:r]->(b) WHERE a.id = 'hub' RETURN b.id", max_nodes=1000)
assert uncapped["truncated"] == False
assert len(uncapped["rows"]) == 100
print("Query language: truncation flag correctly propagates from traverse() through run_query")
print("Query language: single-hop, variable-length, filters, full-scan, and error handling all verified")

# 15b. Query language grammar expansion: OR, comparison operators,
# ORDER BY, LIMIT. Tested against hand-computed expected results.
shutil.rmtree("/tmp/testdb_query_v2", ignore_errors=True)
db10b = VectorDB("/tmp/testdb_query_v2", dim=4)
qv2_docs = {
    "qp1": {"type": "Policy", "priority": 1},
    "qp2": {"type": "Policy", "priority": 5},
    "qm1": {"type": "Metric", "priority": 3},
    "qm2": {"type": "Metric", "priority": 8},
    "qm3": {"type": "Metric", "priority": 2},
}
for qid, qmeta in qv2_docs.items():
    db10b.add(qid, np.random.randn(4).astype(np.float32), qmeta)
db10b.add_edge("qp1", "qm1")
db10b.add_edge("qp1", "qm2")
db10b.add_edge("qp1", "qm3")
db10b.add_edge("qp2", "qm1")

or_result = run_query(db10b, "MATCH (a)-[:related]->(b) WHERE a.id = 'qp1' OR a.id = 'qp2' RETURN a.id, b.id")
assert len(or_result["rows"]) == 4

gt_result = run_query(db10b, "MATCH (a)-[:related]->(b) WHERE a.id = 'qp1' AND b.priority > 2 RETURN b.id")
assert {r["b.id"] for r in gt_result["rows"]} == {"qm1", "qm2"}

range_result = run_query(db10b, "MATCH (a)-[:related]->(b) WHERE a.id = 'qp1' AND b.priority >= 2 AND b.priority <= 3 RETURN b.id")
assert {r["b.id"] for r in range_result["rows"]} == {"qm1", "qm3"}

neq_result = run_query(db10b, "MATCH (a)-[:related]->(b) WHERE a.id = 'qp1' AND b.id != 'qm1' RETURN b.id")
assert sorted(r["b.id"] for r in neq_result["rows"]) == ["qm2", "qm3"]

order_asc = run_query(db10b, "MATCH (a)-[:related]->(b) WHERE a.id = 'qp1' RETURN b.id ORDER BY b.priority ASC")
assert [r["b.id"] for r in order_asc["rows"]] == ["qm3", "qm1", "qm2"]

order_desc_limit = run_query(db10b, "MATCH (a)-[:related]->(b) WHERE a.id = 'qp1' RETURN b.id ORDER BY b.priority DESC LIMIT 2")
assert [r["b.id"] for r in order_desc_limit["rows"]] == ["qm2", "qm1"]

# unsupported: nested/parenthesized boolean expressions must still raise
try:
    run_query(db10b, "MATCH (a)-[:related]->(b) WHERE a.id = 'qp1' AND (b.id = 'qm1' OR b.id = 'qm2') RETURN a")
    raise AssertionError("nested parens should have raised QueryError")
except QueryError:
    pass
print("Query language: OR, comparison operators, ORDER BY, LIMIT all verified against hand-computed results")

# 16. Graph algorithms: PageRank, degree centrality, connected components.
# Cross-checked against networkx (an independent reference implementation)
# where available, not just checked for "runs without error."
from vectordb.graph_algorithms import degree_centrality, connected_components, pagerank
shutil.rmtree("/tmp/testdb_algo", ignore_errors=True)
db12 = VectorDB("/tmp/testdb_algo", dim=4)
for aid in ["algo_a", "algo_b", "algo_c", "algo_hub", "algo_isolated"]:
    db12.add(aid, np.random.randn(4).astype(np.float32))
db12.add_edge("algo_a", "algo_hub")
db12.add_edge("algo_b", "algo_hub")
db12.add_edge("algo_hub", "algo_c")
# algo_isolated has no edges at all -- deliberately left out of the graph

comps = connected_components(db12)
comp_sets = [frozenset(c) for c in comps]
assert frozenset({"algo_a", "algo_b", "algo_c", "algo_hub"}) in comp_sets
assert not any("algo_isolated" in c for c in comp_sets)  # correctly excluded, no edges

deg = degree_centrality(db12)
assert deg["algo_hub"] == 3  # a->hub, b->hub, hub->c

pr = pagerank(db12)
assert abs(sum(pr.values()) - 1.0) < 1e-4  # rank conservation
assert pr["algo_hub"] > pr["algo_a"]  # hub receives the most incoming rank

try:
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from([("algo_a", "algo_hub"), ("algo_b", "algo_hub"), ("algo_hub", "algo_c")])
    nx_pr = nx.pagerank(G, alpha=0.85)
    for node in ["algo_a", "algo_b", "algo_c", "algo_hub"]:
        assert abs(pr[node] - nx_pr[node]) < 0.01, f"{node}: ours={pr[node]} nx={nx_pr[node]}"
    print("Graph algorithms: PageRank verified against networkx (independent reference) -- matches")
except ImportError:
    print("Graph algorithms: PageRank, degree centrality, connected components verified (networkx not available for cross-check)")

# 17. Bulk edge import from CSV and adjacency-list formats.
from vectordb.graph_import import import_edges_csv, import_adjacency_list
import csv as csv_module
import threading

shutil.rmtree("/tmp/testdb_import", ignore_errors=True)
db13 = VectorDB("/tmp/testdb_import", dim=4)
for iid in ["i1", "i2", "i3", "i4"]:
    db13.add(iid, np.random.randn(4).astype(np.float32))

csv_path = "/tmp/testdb_import_edges.csv"
with open(csv_path, "w", newline="") as f:
    w = csv_module.writer(f)
    w.writerow(["source", "target", "rel_type", "weight"])
    w.writerow(["i1", "i2", "follows", "1.5"])
    w.writerow(["i2", "i3", "follows", "2.0"])
    w.writerow(["i1", "i3", "", ""])   # missing relation -> must use default, not empty string
    w.writerow(["", "i4", "follows", "1.0"])  # missing source -> must be skipped, not crash

csv_result = import_edges_csv(
    db13, csv_path, source_col="source", target_col="target", relation_col="rel_type",
    default_relation="related", metadata_cols=["weight"],
)
assert csv_result == {"imported": 3, "skipped": 1}
i1_neighbors = db13.get_neighbors("i1", direction="out", with_metadata=True)
i1_to_i2 = next(n for n in i1_neighbors if n["id"] == "i2")
assert i1_to_i2["relation"] == "follows"
assert i1_to_i2["edge_metadata"]["weight"] == 1.5
i1_to_i3 = next(n for n in i1_neighbors if n["id"] == "i3")
assert i1_to_i3["relation"] == "related"  # empty CSV cell correctly fell back to default, not ""

adj_path = "/tmp/testdb_import_adj.txt"
with open(adj_path, "w") as f:
    f.write("# comment\ni1 i2 i3\n\ni2 i3\ni4\n")
adj_result = import_adjacency_list(db13, adj_path, relation="linked")
assert adj_result == {"imported": 3}
assert set(db13.get_neighbors("i1", direction="out", relation="linked")) == {"i2", "i3"}
print("Bulk edge import: CSV (with default-relation and skip-malformed-row fix) and adjacency-list both verified")

# 18. High-concurrency stress test specifically on the edges table.
shutil.rmtree("/tmp/testdb_edge_concurrency", ignore_errors=True)
db14 = VectorDB("/tmp/testdb_edge_concurrency", dim=8, max_elements=5000)
for i in range(200):
    db14.add(f"cn{i}", np.random.randn(8).astype(np.float32))

concurrency_errors = []

def _edge_writer(tid):
    import random
    rng = random.Random(tid)
    try:
        for _ in range(200):
            src, tgt = f"cn{rng.randrange(200)}", f"cn{rng.randrange(200)}"
            db14.add_edge(src, tgt, relation="r")
    except Exception as e:
        concurrency_errors.append(("writer", tid, str(e)))

def _graph_reader(tid):
    import random
    rng = random.Random(tid + 1000)
    try:
        for _ in range(200):
            node = f"cn{rng.randrange(200)}"
            db14.get_neighbors(node, direction="both")
            db14.traverse(node, max_depth=2, max_nodes=50)
    except Exception as e:
        concurrency_errors.append(("reader", tid, str(e)))

threads = [threading.Thread(target=_edge_writer, args=(i,)) for i in range(16)]
threads += [threading.Thread(target=_graph_reader, args=(i,)) for i in range(16)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert len(concurrency_errors) == 0, f"concurrency errors: {concurrency_errors[:3]}"
print(f"High-concurrency edge stress test: 32 threads, edge_count={db14.edge_count()}, zero errors")

print("\nALL TESTS PASSED")
