# mizanvector

**MizanVector** is the core engine behind the **Mizan Balance Function** ecosystem —  
providing *scale-aware* similarity metrics, distance functions, vector search, and  
training losses for modern embedding models.

> **Proposed & Developed by:**  
> **Ahsan Shaokat** – Computer Scientist & AI/ML Researcher  
> Creator of the **Mizan Balance Function** (2025)

---

# 🌟 Why MizanVector?

Modern embedding systems depend heavily on **cosine similarity**, but cosine has major limitations:

❌ Ignores vector **magnitude**  
❌ Fails with **multi-scale data**  
❌ Collapses when embeddings contain **outliers or noise**  
❌ Penalizes long documents unevenly  
❌ Produces unstable ranking in real RAG pipelines

**Mizan** solves these problems.

### ✔ What Mizan brings:

- **Scale-aware** similarity  
- **Proportional error** instead of absolute distance  
- **Does not require normalization** (keeps magnitude information)  
- **Stable across chunk sizes and multi-modal embeddings**  
- **Better retrieval accuracy** in large datasets  
- **Works with any embedding model**  
- Fully compatible with RAG + Vector DBs  

---

# 🔢 The Mizan Balance Function

The core similarity function:

\[
M(x,y) = 1 - \frac{\|x - y\|_p}{\|x\|_p + \|y\|_p + \epsilon}
\]

Where:

- \( x, y \) = vectors  
- \( p \ge 1 \) = strictness  
- \( \epsilon \) = numerical stability

**Interpretation:**

- If vectors are identical → Mizan = **1.0**  
- If proportional but different → high similarity  
- If very different (or noisy) → lower similarity  

Mizan is a **continuous**, **bounded**, **scale-aware**, **interpretable** metric.

---

# 📦 Features

### 1. **Mizan Similarity & Distance Metrics**
- `mizan_similarity(v1, v2, p)`
- `mizan_distance(v1, v2, p)`
- Drop-in replacements for cosine, dot-product, or L2 distance

### 2. **In-Memory Vector Store**
```python
from mizanvector import MizanMemoryStore
Features:

Store embeddings

Search with Mizan/Cosine/EUCLIDEAN

Metadata storage

Lightweight & fast

3. Postgres + pgvector Backend
python
Copy code
from mizanvector import MizanPgVectorStore
Production-ready

Mizan similarity inside SQL queries

Hybrid searching supported

4. Training Losses
MizanContrastiveLoss

MizanTripletLoss

Can replace InfoNCE or cosine losses in training your own embedding models

5. HFEmbedder Utility
Simple HuggingFace embedding wrapper:

python
Copy code
from mizanvector import HFEmbedder
emb = HFEmbedder("all-MiniLM-L6-v2")
🚀 Quickstart Usage
Install
bash
Copy code
pip install mizanvector
🔎 Example: In-Memory Search
python
Copy code
from mizanvector import MizanMemoryStore, HFEmbedder

embedder = HFEmbedder("all-MiniLM-L6-v2")
store = MizanMemoryStore(dim=384)

docs = [
    "Mizan is a scale-aware similarity metric.",
    "Cosine similarity ignores magnitude.",
    "Ahsan Shaokat created the Mizan Balance Function.",
]

embs = embedder.encode(docs)

for d, e in zip(docs, embs):
    store.add_document(content=d, embedding=e.tolist())

query = "who invented the mizan function?"
q_emb = embedder.encode_one(query).tolist()

results = store.search(q_emb, top_k=3, metric="mizan")

for r in results:
    print(r.score, "|", r.content)
🧪 Example: Compare Mizan vs Cosine
python
Copy code
from mizanvector.metrics import mizan_similarity, cosine_similarity

v1 = [1.0, 2.0, 3.0]
v2 = [1.1, 2.1, 3.1]

print("Mizan:", mizan_similarity(v1, v2))
print("Cosine:", cosine_similarity(v1, v2))
🧠 API Overview
Memory Store
python
Copy code
store = MizanMemoryStore(dim=384)

store.add_document(
    content="Some text",
    embedding=[...],
    metadata={"id": 1}
)

results = store.search(q_emb, top_k=5, metric="mizan")
Postgres Store
python
Copy code
store = MizanPgVectorStore(
    table="my_vectors",
    dsn="postgresql://user:pass@localhost:5432/db"
)
📘 How Mizan Helps (Applications)
✔ RAG Pipelines
Stable retrieval across chunk lengths.

✔ LLM Embedding Ranking
Better vector scoring for hybrid search.

✔ Outlier-Resistant Retrieval
Mizan handles noisy embeddings gracefully.

✔ Multi-Modal Search (Text + Images)
Magnitude differences become meaningful.

✔ Code Search
Detect proportional similarity even in different-length source files.

✔ Large-Scale Knowledge Bases
Reduces ranking errors caused by cosine.

🧩 Mizan Ecosystem
Component	Purpose
mizanvector	Metrics, losses, vector DB, similarity engine
mizan-embedder	Train embedding models optimized for Mizan
mizan-rag	Full Mizan-powered Retrieval-Augmented Generation
mizan-models	Published models like MizanTextEncoder-base

📜 License
MIT License
© 2025 Ahsan Shaokat

You are free to use, modify, and distribute this software.

🙌 Acknowledgements
HuggingFace Transformers

pgvector community

PyTorch

