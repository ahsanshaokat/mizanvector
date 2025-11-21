import sys, os
import inspect
# sys.path.insert(0, "..")

import numpy as np
from mizanvector import HFEmbedder, MizanMemoryStore
from mizanvector.metrics import cosine_similarity, mizan_similarity

# Test documents
docs = [
    "Machine learning improves pattern recognition.",
    "Deep learning uses neural networks.",
    "Quranic studies analyze classical Arabic texts.",
    "The Quran is the holy book of Islam.",
    "Neural networks can learn representations."
]

embedder = HFEmbedder("intfloat/e5-large-v2")

embs = embedder.encode(docs)

store = MizanMemoryStore(dim=len(embs[0]))

for content, emb in zip(docs, embs):
    store.add_document(content=content, embedding=emb)

query = "What is the Quran?"
q_emb = embedder.encode_one(query)

print("=== MIZAN ===")
for r in store.search(q_emb, top_k=3, metric="mizan"):
    print(r.score, r.content)

print("\n=== COSINE ===")
for r in store.search(q_emb, top_k=3, metric="cosine"):
    print(r.score, r.content)


print("Norms of embeddings:")
for e in embs:
    print(np.linalg.norm(e))

# print(inspect.getsource(mm.mizan_distance))
# print("HFEmbedder loaded from:", mz_emb.__file__)
# print("\nSource code:\n")
# print(inspect.getsource(mz_emb.HFEmbedder))


print("Mizan:", mizan_similarity(embs[0], embs[1]))
print("Cosine:", cosine_similarity(embs[0], embs[1]))
