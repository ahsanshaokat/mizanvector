# MizanVector

A **scale-aware embedding & vector search framework** built around
**Mizan similarity / distance**.

## Quickstart (in-memory)

```python
from mizanvector import MizanMemoryStore

store = MizanMemoryStore(dim=3)
store.add_document("doc_a", [1, 2, 3])
store.add_document("doc_b", [2, 4, 6])
store.add_document("doc_c", [1, 3, 2])

query = [1, 2, 3]
results = store.search(query, top_k=3, metric="mizan")
for r in results:
    print(r.id, r.content, r.score)
```
