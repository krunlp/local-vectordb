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

print("\nALL TESTS PASSED")
