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

print("\nALL TESTS PASSED")
