import os
from mizanvector import HFEmbedder, MizanMemoryStore

def load_large_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "testdata_large_mizan.txt")
    text = load_large_file(path)

    print("Loaded file:", path)
    print("Length:", len(text))

    chunks = chunk_text(text, chunk_size=800, overlap=150)
    print("Total chunks:", len(chunks))

    # all-MiniLM-L6-v2
    # sentence-transformers/all-mpnet-base-v2
    embedder = HFEmbedder("intfloat/e5-large-v2")
    embeddings = embedder.encode(chunks)

    store = MizanMemoryStore(dim=len(embeddings[0]))

    # Add chunks
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        store.add_document(
            content=chunk,
            embedding=emb,
            metadata={"chunk_id": idx}
        )

    query = "what is mizan?"
    q_emb = embedder.encode_one(query)

    # --- MIZAN RESULTS ---
    print("\n==================== TOP RESULTS — MIZAN ====================")
    for r in store.search(q_emb, top_k=3, metric="mizan"):
        preview = r.content[:200].replace("\n", " ")  # first 200 chars
        print(f"\n[MIZAN] Score: {r.score:.4f} | Chunk ID: {r.id}")
        print(f"Text: {preview}...")

    # --- COSINE RESULTS ---
    print("\n==================== TOP RESULTS — COSINE ====================")
    for r in store.search(q_emb, top_k=3, metric="cosine"):
        preview = r.content[:200].replace("\n", " ")
        print(f"\n[COS]   Score: {r.score:.4f} | Chunk ID: {r.id}")
        print(f"Text: {preview}...")

if __name__ == "__main__":
    main()
